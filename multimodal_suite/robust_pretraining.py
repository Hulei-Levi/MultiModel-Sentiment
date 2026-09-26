"""Auditable feature-level adaptations of M3ER, MEmoBERT and HyCon.

These are independent implementations, not official checkpoints. See the adjacent
robust_pretraining_notes.md for source versions and differences from each paper.
In particular, MEmoBERT uses cached-feature masking and learned soft prompts, and
HyCon follows the publicly readable arXiv v1 objectives, not an unverified final
journal pair-selection implementation. All masks denote actual observations.
"""
from __future__ import annotations

from itertools import combinations
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .common import MODALITIES, ProjectedInputs, SafeAttention, masked_softmax, validate_inputs


def _options(config):
    return dict(getattr(config, "model_options", {}) or {})


def _target_tensor(targets, task):
    if targets is None or torch.is_tensor(targets):
        return targets
    keys = ("target", "labels", "classification_labels" if task == "classification" else "regression_labels")
    for key in keys:
        if key in targets:
            return targets[key]
    raise ValueError("Targets must be a tensor or a mapping containing target/labels")


class _TransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SafeAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        norm = self.norm1(x)
        attended, _ = self.attention(norm, norm, norm, mask, mask)
        x = x + self.drop(attended)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x.masked_fill(~mask[..., None], 0)


class _CCAProxyPair(nn.Module):
    """Train-only regularized CCA and paired ridge maps, fixed-size state buffers."""
    def __init__(self, left_dim, right_dim, components):
        super().__init__()
        rank = min(components, left_dim, right_dim)
        self.register_buffer("left_mean", torch.zeros(left_dim))
        self.register_buffer("right_mean", torch.zeros(right_dim))
        self.register_buffer("left_cca", torch.zeros(left_dim, rank))
        self.register_buffer("right_cca", torch.zeros(right_dim, rank))
        self.register_buffer("left_to_right", torch.zeros(left_dim, right_dim))
        self.register_buffer("right_to_left", torch.zeros(right_dim, left_dim))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("fit_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def reset_fit(self):
        """Discard every statistic and transform from a previous fitting run."""
        for buffer in self.buffers():
            buffer.zero_()

    @torch.no_grad()
    def fit(self, x, y, ridge):
        self.reset_fit()
        if len(x) < 3:
            return
        x, y = x.double().cpu(), y.double().cpu()
        mx, my = x.mean(0), y.mean(0)
        x, y = x - mx, y - my
        cxx, cyy, cxy = x.T @ x / (len(x) - 1), y.T @ y / (len(y) - 1), x.T @ y / (len(x) - 1)
        # Scale-aware ridge also regularizes constant features.
        cxx += torch.eye(cxx.shape[0], dtype=x.dtype) * ridge * cxx.diag().mean().clamp_min(1e-4)
        cyy += torch.eye(cyy.shape[0], dtype=y.dtype) * ridge * cyy.diag().mean().clamp_min(1e-4)
        ex, ux = torch.linalg.eigh(cxx)
        ey, uy = torch.linalg.eigh(cyy)
        wx = (ux * ex.clamp_min(1e-10).rsqrt()[None, :]) @ ux.T
        wy = (uy * ey.clamp_min(1e-10).rsqrt()[None, :]) @ uy.T
        u, _, vh = torch.linalg.svd(wx @ cxy @ wy, full_matrices=False)
        rank = min(self.left_cca.shape[1], len(x) - 1)
        self.left_cca.zero_()
        self.right_cca.zero_()
        self.left_cca[:, :rank].copy_((wx @ u[:, :rank]).to(self.left_cca))
        self.right_cca[:, :rank].copy_((wy @ vh.T[:, :rank]).to(self.right_cca))
        self.left_mean.copy_(mx.to(self.left_mean))
        self.right_mean.copy_(my.to(self.right_mean))
        self.left_to_right.copy_(torch.linalg.solve(cxx, cxy).to(self.left_to_right))
        self.right_to_left.copy_(torch.linalg.solve(cyy, cxy.T).to(self.right_to_left))
        self.fitted.fill_(True)
        self.fit_count.fill_(len(x))

    def correlation(self, x, y):
        x = (x - self.left_mean) @ self.left_cca
        y = (y - self.right_mean) @ self.right_cca
        x, y = x - x.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
        return F.cosine_similarity(x, y, dim=-1, eps=1e-8)

    def proxy(self, source, left_to_right):
        if left_to_right:
            return (source - self.left_mean) @ self.left_to_right + self.right_mean
        return (source - self.right_mean) @ self.right_to_left + self.left_mean


class M3ER2020Model(nn.Module):
    """CCA screening + explicitly sourced proxies + an MFN-style recurrent backbone.

    Classification uses a product of final modality hidden states alongside a
    recurrent multimodal memory. Paper Eq. (3) is returned as a separately named
    classification auxiliary objective; it is not reinterpreted as a product of
    class probabilities. The memory backbone and ridge proxy are adaptations.
    """
    default_pretrain_epochs = 0
    default_auxiliary_weights = {"m3er_multiplicative": 1.0, "m3er_unimodal_regression": 0.1}

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims, self.max_length = dict(feature_dims), max_length
        self.task = getattr(config, "task", "classification")
        self.options = opt = _options(config)
        width, dropout = config.d_model, config.dropout
        self.width = width
        self.memory_dim = int(opt.get("m3er_memory_dim", width))
        classifier_hidden = int(opt.get("m3er_classifier_hidden", width))
        if min(self.memory_dim, classifier_hidden) < 1:
            raise ValueError("M3ER memory and classifier widths must be positive")
        self.beta = float(opt.get("m3er_beta", 2.0))
        self.threshold = float(opt.get("m3er_cca_threshold", 0.0))
        self.proxy_during_training = bool(opt.get("m3er_proxy_during_training", True))
        self.pairs = nn.ModuleDict({f"{a}__{b}": _CCAProxyPair(feature_dims[a], feature_dims[b],
            int(opt.get("m3er_cca_components", 16))) for a, b in combinations(MODALITIES, 2)})
        self.register_buffer("preprocessing_fitted", torch.tensor(False))
        self.register_buffer("preprocessing_attempted", torch.tensor(False))
        self.inputs = ProjectedInputs(feature_dims, width, dropout)
        self.cells = nn.ModuleDict({m: nn.LSTMCell(width, width) for m in MODALITIES})
        self.delta_attention = nn.Sequential(nn.Linear(6 * width, 3 * width), nn.Tanh(), nn.Linear(3 * width, 6 * width))
        self.memory_candidate = nn.Linear(6 * width, self.memory_dim)
        self.memory_gates = nn.Linear(6 * width + self.memory_dim, 2 * self.memory_dim)
        self.classifier = nn.Sequential(nn.Linear(width + self.memory_dim, classifier_hidden), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(classifier_hidden, output_dim))
        self.unimodal_heads = nn.ModuleDict({m: nn.Linear(width, output_dim) for m in MODALITIES})

    @torch.no_grad()
    def fit_preprocessing(self, train_dataset):
        """Fit solely on normalized training observations, never validation/test.

        Each sample contributes a bounded random subset of co-observed aligned
        slots. Sampling is reproducible, independent of labels, and performed on
        CPU. Learned CCA and proxy parameters persist in state_dict buffers.
        """
        # Reusing the model with a new split must never retain old transforms,
        # even if the new dataset has no paired observations or fitting fails.
        self.preprocessing_fitted.fill_(False)
        self.preprocessing_attempted.fill_(True)
        for pair in self.pairs.values():
            pair.reset_fit()
        try:
            return self._fit_preprocessing_clean(train_dataset)
        finally:
            self.preprocessing_fitted.copy_(torch.stack([pair.fitted for pair in self.pairs.values()]).any())

    @torch.no_grad()
    def _fit_preprocessing_clean(self, train_dataset):
        size = len(train_dataset)
        if not size:
            raise ValueError("CCA fitting requires a nonempty training dataset")
        maximum = int(self.options.get("m3er_fit_max_slots", 8192))
        if maximum < 3:
            raise ValueError("m3er_fit_max_slots must be at least 3")
        generator = torch.Generator().manual_seed(int(self.options.get("m3er_fit_seed", 1729)))
        pair_rows = {key: ([], []) for key in self.pairs}
        per_sample = max(1, math.ceil(maximum / size))
        for index in range(size):
            item = train_dataset[index]
            features, masks = item["features"], item["masks"]
            for a, b in combinations(MODALITIES, 2):
                valid = torch.as_tensor(masks[a]).bool().cpu() & torch.as_tensor(masks[b]).bool().cpu()
                slots = valid.nonzero(as_tuple=True)[0]
                if not len(slots):
                    continue
                slots = slots[torch.randperm(len(slots), generator=generator)[:per_sample]]
                left, right = pair_rows[f"{a}__{b}"]
                left.append(torch.as_tensor(features[a]).detach().cpu()[slots])
                right.append(torch.as_tensor(features[b]).detach().cpu()[slots])
        counts = {}
        for key, (left, right) in pair_rows.items():
            if left:
                x, y = torch.cat(left), torch.cat(right)
                if len(x) > maximum:
                    ix = torch.randperm(len(x), generator=generator)[:maximum]
                    x, y = x[ix], y[ix]
                self.pairs[key].fit(x, y, float(self.options.get("m3er_ridge", 0.01)))
            counts[key] = int(self.pairs[key].fit_count)
        if not any(bool(pair.fitted) for pair in self.pairs.values()):
            warnings.warn("No modality pair has sufficient co-observed training slots: CCA/proxies disabled.")
        return {"fit_slots": counts, "source_split": "train", "proxy_method": "paired_ridge"}

    def _screen_and_fill(self, features, masks):
        observed = torch.stack([masks[m].bool() for m in MODALITIES], -1)
        checked, passed = torch.zeros_like(observed), torch.zeros_like(observed)
        reliability = features["text"].new_full(observed.shape, -2.0)
        clean = {m: features[m].masked_fill(~masks[m].bool()[..., None], 0) for m in MODALITIES}
        pair_scores = features["text"].new_zeros((*observed.shape[:2], 3))
        pair_checked = torch.zeros_like(pair_scores, dtype=torch.bool)
        for pair_index, (i, j) in enumerate(combinations(range(3), 2)):
            a, b = MODALITIES[i], MODALITIES[j]
            pair = self.pairs[f"{a}__{b}"]
            if not bool(pair.fitted):
                continue
            both = observed[..., i] & observed[..., j]
            score = pair.correlation(clean[a], clean[b])
            pair_scores[..., pair_index] = score.masked_fill(~both, 0)
            pair_checked[..., pair_index] = both
            for k in (i, j):
                checked[..., k] |= both
                passed[..., k] |= both & (score >= self.threshold)
                reliability[..., k] = torch.maximum(reliability[..., k], score.masked_fill(~both, -2.0))
        effective = observed & (passed | ~checked)
        # If all correlated pairs reject a slot, retain one original observation
        # rather than silently turn an observed time step into padding.
        fallback = observed.any(-1) & ~effective.any(-1)
        selected = reliability.masked_fill(~observed, -3.0).argmax(-1)
        effective |= F.one_hot(selected, 3).bool() & fallback[..., None]
        result, result_masks = {}, {}
        sources = torch.zeros((*observed.shape, 3), dtype=torch.bool, device=observed.device)
        for target_index, target in enumerate(MODALITIES):
            numerator = torch.zeros_like(clean[target])
            count = torch.zeros_like(observed[..., 0], dtype=clean[target].dtype)
            if not self.training or self.proxy_during_training:
                for source_index, source in enumerate(MODALITIES):
                    if source == target:
                        continue
                    lo, hi = sorted((source_index, target_index))
                    pair = self.pairs[f"{MODALITIES[lo]}__{MODALITIES[hi]}"]
                    if not bool(pair.fitted):
                        continue
                    use = effective[..., source_index] & ~effective[..., target_index]
                    proposal = pair.proxy(clean[source], source_index == lo)
                    numerator = numerator + proposal.masked_fill(~use[..., None], 0)
                    count = count + use.to(count.dtype)
                    sources[..., target_index, source_index] = use
            proxy_valid = count > 0
            proxy = numerator / count.clamp_min(1)[..., None]
            direct = effective[..., target_index]
            result[target] = torch.where(direct[..., None], clean[target], proxy)
            result_masks[target] = direct | proxy_valid
        usable = torch.stack([result_masks[m] for m in MODALITIES], -1)
        return result, result_masks, {"observed_masks": observed, "effective_masks": effective,
            "cca_pair_scores": pair_scores.detach(), "cca_pair_checked": pair_checked,
            "cca_pair_order": tuple(self.pairs.keys()), "cca_checked_masks": checked,
            "cca_passed_masks": passed, "cca_rejected_masks": observed & checked & ~passed,
            "usable_masks": usable, "proxy_masks": usable & ~effective,
            "proxy_sources": sources, "cca_fallback": fallback, "cca_fitted": self.preprocessing_fitted,
            "cca_fit_attempted": self.preprocessing_attempted,
            "cca_pair_fitted": {key: pair.fitted for key, pair in self.pairs.items()},
            "cca_pair_fit_count": {key: pair.fit_count for key, pair in self.pairs.items()}}

    def forward(self, features, masks, targets=None, return_intermediates=False):
        validate_inputs(features, masks, self.feature_dims, self.max_length)
        filled, usable, audit = self._screen_and_fill(features, masks)
        x = self.inputs(filled, usable)
        template = x["text"]
        h = {m: template.new_zeros((len(template), self.width)) for m in MODALITIES}
        c = {m: torch.zeros_like(h[m]) for m in MODALITIES}
        memory = template.new_zeros((len(template), self.memory_dim))
        sequences = {m: [] for m in MODALITIES}
        memory_sequence, delta_weights = [], []
        for step in range(template.shape[1]):
            previous = torch.cat([c[m] for m in MODALITIES], -1)
            valid = torch.stack([usable[m][:, step] for m in MODALITIES], -1)
            for m in MODALITIES:
                hh, cc = self.cells[m](x[m][:, step], (h[m], c[m]))
                use = usable[m][:, step, None]
                h[m], c[m] = torch.where(use, hh, h[m]), torch.where(use, cc, c[m])
            delta = torch.cat((previous, torch.cat([c[m] for m in MODALITIES], -1)), -1)
            feature_mask = valid.repeat(1, 2).repeat_interleave(self.width, -1)
            attention = masked_softmax(self.delta_attention(delta), feature_mask)
            attended = attention * delta
            candidate = torch.tanh(self.memory_candidate(attended))
            retain, update = torch.sigmoid(self.memory_gates(torch.cat((attended, memory), -1))).chunk(2, -1)
            updated = retain * memory + update * candidate
            memory = torch.where(valid.any(-1, keepdim=True), updated, memory)
            if return_intermediates:
                for m in MODALITIES:
                    sequences[m].append(h[m].masked_fill(~usable[m][:, step, None], 0).detach())
                memory_sequence.append(memory.masked_fill(~valid.any(-1, keepdim=True), 0).detach())
                delta_weights.append(attention.detach())
        available = torch.stack([usable[m].any(1) for m in MODALITIES], -1)
        # Neutral multiplicative identity for an entirely absent branch.
        factors = torch.stack([h[m] for m in MODALITIES], 1)
        product = torch.where(available[..., None], factors, torch.ones_like(factors)).prod(1)
        classifier_input = torch.cat((product, memory), -1)
        logits = self.classifier(classifier_input)
        branch_logits = torch.stack([self.unimodal_heads[m](h[m]) for m in MODALITIES], 1)
        branch_logits = branch_logits.masked_fill(~available[..., None], 0)
        aux = {}
        y = _target_tensor(targets, self.task)
        if self.training and y is not None:
            if self.task == "classification":
                log_probs = F.log_softmax(branch_logits, -1)
                selected = log_probs.gather(-1, y.long().reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
                # Published Eq. (3), summing the active modality terms per sample.
                term = -(selected.exp().clamp_min(1e-8).pow(self.beta / 2.0)) * selected
                aux["m3er_multiplicative"] = term.masked_fill(~available, 0).sum(1).mean()
            else:
                error = (branch_logits.squeeze(-1) - y.reshape(-1, 1)).abs()
                aux["m3er_unimodal_regression"] = error.masked_fill(~available, 0).sum() / available.sum().clamp_min(1)
        result = {"logits": logits, "aux_losses": aux, "modality_logits": branch_logits, **audit}
        if return_intermediates:
            predictions = branch_logits.softmax(-1) if self.task == "classification" else branch_logits
            result.update(
                filled_features={m: value.detach() for m, value in filled.items()},
                projected_sequences={m: value.detach() for m, value in x.items()},
                unimodal_sequences={m: torch.stack(value, dim=1) for m, value in sequences.items()},
                final_hidden_states=factors.detach(), modality_availability=available.detach(),
                multiplicative_features=product.detach(), memory=memory.detach(),
                memory_sequence=torch.stack(memory_sequence, dim=1),
                delta_attention=torch.stack(delta_weights, dim=1),
                pooled_features=classifier_input.detach(),
                branch_predictions=predictions.masked_fill(~available[..., None], 0).detach(),
            )
        return result


class MEmoBERT2022Model(nn.Module):
    """Conditional span reconstruction + joint Transformer + learned prompt query.

    This is explicitly a cached-feature adaptation: no pretrained BERT checkpoint,
    vocabulary MLM, hard textual prompt, or unavailable facial teacher is claimed.
    """
    default_pretrain_epochs = 3
    default_auxiliary_weights = {f"memobert_masked_{m}": 1.0 for m in MODALITIES}

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims, self.max_length = dict(feature_dims), max_length
        opt = _options(config)
        width = config.d_model
        self.mask_probability = float(opt.get("memobert_mask_probability", 0.15))
        self.span_length = int(opt.get("memobert_span_length", 3))
        if not 0 < self.mask_probability <= 1 or self.span_length < 1:
            raise ValueError("MEmoBERT masking probability must be in (0,1] and span length positive")
        self.inputs = ProjectedInputs(feature_dims, width, config.dropout)
        self.positions = nn.Parameter(torch.empty(max_length, width))
        self.modality_embeddings = nn.Parameter(torch.empty(3, width))
        self.soft_prompt = nn.Parameter(torch.empty(3, width))
        self.prompt_query = nn.Parameter(torch.empty(1, width))
        self.mask_embeddings = nn.Parameter(torch.empty(3, width))
        for parameter in (self.positions, self.modality_embeddings, self.soft_prompt, self.prompt_query, self.mask_embeddings):
            nn.init.normal_(parameter, std=0.02)
        self.blocks = nn.ModuleList([_TransformerBlock(width, config.nhead, config.dropout) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(width)
        self.decoders = nn.ModuleDict({m: nn.Linear(width, feature_dims[m]) for m in MODALITIES})
        # Legacy checkpoint name: this is a new class/regression projection,
        # not the paper's pretrained vocabulary MLM head or lexical verbalizer.
        self.prompt_verbalizer = nn.Linear(width, output_dim)

    def _encode(self, features, masks, corruption=None, return_intermediates=False):
        x = self.inputs(features, masks)
        projected = {m: value.detach() for m, value in x.items()} if return_intermediates else None
        batch, length = x["text"].shape[:2]
        segments, valid = [], []
        for index, m in enumerate(MODALITIES):
            representation = x[m]
            if corruption is not None:
                representation = torch.where(corruption[m][..., None], self.mask_embeddings[index], representation)
            representation = representation + self.positions[:length] + self.modality_embeddings[index]
            segments.append(representation.masked_fill(~masks[m].bool()[..., None], 0))
            valid.append(masks[m].bool())
        # Learned soft context followed by a dedicated masked-label query.
        prompt = torch.cat((self.soft_prompt, self.prompt_query), 0)[None].expand(batch, -1, -1)
        segments.append(prompt)
        valid.append(torch.ones(batch, len(self.soft_prompt) + 1, dtype=torch.bool, device=prompt.device))
        x, mask = torch.cat(segments, 1), torch.cat(valid, 1)
        if return_intermediates:
            audit = {
                "projected_sequences": projected,
                "embedded_sequences": {m: segments[i].detach() for i, m in enumerate(MODALITIES)},
                "joint_input": x.detach(), "joint_mask": mask.detach(),
            }
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.norm(x).masked_fill(~mask[..., None], 0)
        if return_intermediates:
            return hidden, audit
        return hidden

    def _conditional_span_masks(self, masks):
        available = torch.stack([masks[m].any(1) for m in MODALITIES], 1)
        # Exactly one observed modality is eligible for corruption per example.
        choice = torch.rand_like(available.float()).masked_fill(~available, -1).argmax(1)
        result = {}
        for index, m in enumerate(MODALITIES):
            valid = masks[m].bool()
            choose = choice == index
            start = (torch.rand(valid.shape, device=valid.device) < self.mask_probability / self.span_length) & valid
            corrupt = start.clone()
            continuation = start
            for _ in range(1, min(self.span_length, valid.shape[1])):
                shifted = torch.zeros_like(valid)
                shifted[:, 1:] = continuation[:, :-1] & valid[:, 1:]
                corrupt |= shifted
                continuation = shifted
            corrupt &= valid & choose[:, None]
            # Short samples still contribute at least one reconstruction target.
            absent = choose & ~corrupt.any(1)
            first = valid.long().argmax(1)
            corrupt[absent, first[absent]] = True
            result[m] = corrupt
        return result

    def _validate_corruption(self, corruption, masks):
        if not isinstance(corruption, dict) or set(corruption) != set(MODALITIES):
            raise ValueError("corruption_masks must contain text, audio and vision")
        for m in MODALITIES:
            selected = corruption[m]
            if (not torch.is_tensor(selected) or selected.dtype != torch.bool
                    or selected.shape != masks[m].shape or selected.device != masks[m].device):
                raise ValueError(f"corruption_masks[{m}] must be a same-device boolean tensor matching masks")
            if (selected & ~masks[m]).any():
                raise ValueError("Reconstruction corruption cannot select unobserved slots")
        count = torch.stack([corruption[m].any(1) for m in MODALITIES], 1).sum(1)
        if (count > 1).any():
            raise ValueError("Conditional corruption selects at most one modality per sample")
        return corruption

    def forward(self, features, masks, targets=None, return_intermediates=False,
                compute_reconstruction_losses=None, corruption_masks=None):
        """Clean-query prediction plus optional, independently masked reconstruction.

        Sentiment targets never enter this model. Explicit corruption masks enable
        deterministic eval diagnostics; default eval does not sample corruption.
        """
        validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        if corruption_masks is not None:
            if compute_reconstruction_losses is False:
                raise ValueError("Explicit corruption masks require reconstruction computation")
            corruption_masks = self._validate_corruption(corruption_masks, masks)
        compute_losses = ((self.training or corruption_masks is not None)
                          if compute_reconstruction_losses is None else bool(compute_reconstruction_losses))
        if return_intermediates:
            hidden, audit = self._encode(features, masks, return_intermediates=True)
        else:
            hidden = self._encode(features, masks)
        logits = self.prompt_verbalizer(hidden[:, -1])
        aux = {}
        reconstruction, corruption = {}, None
        length = features["text"].shape[1]
        if compute_losses:
            corruption = (self._conditional_span_masks(masks) if corruption_masks is None else corruption_masks)
            masked = self._encode(features, masks, corruption)
            for index, m in enumerate(MODALITIES):
                prediction = self.decoders[m](masked[:, index * length:(index + 1) * length])
                selected = corruption[m]
                if selected.any():
                    aux[f"memobert_masked_{m}"] = F.mse_loss(prediction[selected], features[m].detach()[selected])
                else:
                    aux[f"memobert_masked_{m}"] = prediction.sum() * 0
                if return_intermediates:
                    reconstruction[m] = prediction.masked_fill(~masks[m][..., None], 0).detach()
        result = {"logits": logits, "aux_losses": aux, "prompt_representation": hidden[:, -1]}
        if return_intermediates:
            result.update(
                **audit, joint_hidden=hidden.detach(),
                unimodal_sequences={m: hidden[:, i * length:(i + 1) * length].detach()
                                    for i, m in enumerate(MODALITIES)},
                soft_prompt_states=hidden[:, 3 * length:-1].detach(),
                prompt_representation=hidden[:, -1].detach(), pooled_features=hidden[:, -1].detach(),
            )
            if compute_losses:
                result.update(corruption_masks={m: v.detach() for m, v in corruption.items()},
                              masked_joint_hidden=masked.detach(), reconstructions=reconstruction)
        return result


class HyCon2022Model(nn.Module):
    """HyCon arXiv-v1 objectives, adapted to cached text features and valid slots.

    Implements within-modality IAMCL, across-modality IEMCL and same-sample SCL;
    SCL means semi-contrastive, not semi-supervised. Ratio objectives and positive
    refinement follow paper Eqs. (6)-(12), with nonnegative normalized embeddings.
    """
    default_pretrain_epochs = 0
    default_auxiliary_weights = {"hycon_intra": 1.0, "hycon_inter": 1.0, "hycon_semi": 1.0}

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims, self.max_length = dict(feature_dims), max_length
        self.task = getattr(config, "task", "classification")
        opt = _options(config)
        self.margin = float(opt.get("hycon_margin", 0.8))
        self.refinement_weight = float(opt.get("hycon_refinement_weight", 1.0))
        if not math.isfinite(self.margin) or not 0 < self.margin <= 1:
            raise ValueError("hycon_margin must be finite and in (0, 1]")
        if not math.isfinite(self.refinement_weight) or self.refinement_weight < 0:
            raise ValueError("hycon_refinement_weight must be finite and nonnegative")
        width = config.d_model
        self.inputs = ProjectedInputs(feature_dims, width, config.dropout)
        self.positions = nn.Parameter(torch.empty(max_length, width))
        nn.init.normal_(self.positions, std=0.02)
        self.encoders = nn.ModuleDict({m: nn.ModuleList([_TransformerBlock(width, config.nhead, config.dropout)
            for _ in range(config.num_layers)]) for m in MODALITIES})
        self.representation = nn.ModuleDict({m: nn.Linear(width, width) for m in MODALITIES})
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(width, output_dim))

    def _ratio_loss(self, similarities, candidates, positives, margin):
        positive_count = positives.sum(1)
        usable = (positive_count > 0) & candidates.any(1)
        if not usable.any():
            return similarities.sum() * 0
        positive_sum = similarities.masked_fill(~positives, 0).sum(1)
        denominator = similarities.masked_fill(~candidates, 0).sum(1).clamp_min(1e-8)
        ratio = -positive_sum / denominator
        refinement = ((similarities - margin).square().masked_fill(~positives, 0).sum(1)
                      / positive_count.clamp_min(1))
        return (ratio + self.refinement_weight * refinement)[usable].mean()

    def _contrastive(self, embeddings, present, labels, return_intermediates=False):
        batch = embeddings.shape[0]
        flat = F.normalize(embeddings.reshape(batch * 3, -1), dim=-1, eps=1e-8)
        similarities = (flat @ flat.T).clamp(0, 1)
        available = present.reshape(-1)
        sample = torch.arange(batch, device=flat.device).repeat_interleave(3)
        modality = torch.arange(3, device=flat.device).repeat(batch)
        observed_pair = available[:, None] & available[None, :]
        same_sample = sample[:, None] == sample[None, :]
        same_modality = modality[:, None] == modality[None, :]
        same_example_pair = observed_pair & same_sample & ~same_modality
        semi = ((similarities - self.margin).square()[same_example_pair].mean()
                if same_example_pair.any() else similarities.sum() * 0)
        aux = {"hycon_semi": semi}
        pair_masks = {"semi_positives": same_example_pair}
        usable_counts = {}
        if labels is not None:
            expanded = labels.reshape(-1).repeat_interleave(3)
            same_label = expanded[:, None] == expanded[None, :]
            intra = observed_pair & ~same_sample & same_modality
            inter = observed_pair & ~same_sample & ~same_modality
            aux["hycon_intra"] = self._ratio_loss(similarities, intra, intra & same_label, 1.0)
            aux["hycon_inter"] = self._ratio_loss(similarities, inter, inter & same_label, self.margin)
            if return_intermediates:
                pair_masks.update(intra_candidates=intra, intra_positives=intra & same_label,
                                  intra_negatives=intra & ~same_label,
                                  inter_candidates=inter, inter_positives=inter & same_label,
                                  inter_negatives=inter & ~same_label)
                usable_counts = {"intra": int((intra & same_label).any(1).sum()),
                                 "inter": int((inter & same_label).any(1).sum())}
        if return_intermediates:
            return aux, {
                "similarities": similarities.detach(),
                "flat_sample_indices": sample.detach(), "flat_modality_indices": modality.detach(),
                "pair_masks": {name: value.detach() for name, value in pair_masks.items()},
                "pair_counts": {name: int(value.sum()) for name, value in pair_masks.items()},
                "usable_anchor_counts": usable_counts,
                "labels": None if labels is None else labels.detach(),
            }
        return aux

    def forward(self, features, masks, targets=None, return_intermediates=False,
                compute_contrastive_losses=None):
        """Labels are used only for pair losses, never as prediction inputs.

        Normal eval avoids all batch pair construction. Explicit True enables
        deterministic diagnostics on a training mini-batch while dropout is off.
        """
        validate_inputs(features, masks, self.feature_dims, self.max_length)
        x = self.inputs(features, masks)
        vectors, present = [], []
        sequences, last_indices, last_states = {}, {}, {}
        for m in MODALITIES:
            valid = masks[m].bool()
            h = (x[m] + self.positions[:x[m].shape[1]]).masked_fill(~valid[..., None], 0)
            for block in self.encoders[m]:
                h = block(h, valid)
            position = torch.arange(h.shape[1], device=h.device)[None].expand(len(h), -1)
            last = position.masked_fill(~valid, -1).max(1).values
            vector = h[torch.arange(len(h), device=h.device), last.clamp_min(0)]
            if return_intermediates:
                sequences[m] = h.detach()
                last_indices[m] = last.detach()
                last_states[m] = vector.masked_fill((last < 0)[:, None], 0).detach()
            # L2 normalization alone does not ensure the paper's stated [0,1]
            # similarities; softplus makes that assumption explicit and stable.
            vector = F.softplus(self.representation[m](vector)).masked_fill((last < 0)[:, None], 0)
            vectors.append(vector)
            present.append(last >= 0)
        embeddings = torch.stack(vectors, 1)
        availability = torch.stack(present, 1)
        fused = embeddings.sum(1)
        logits = self.classifier(fused)
        compute_losses = self.training if compute_contrastive_losses is None else bool(compute_contrastive_losses)
        aux = {}
        if compute_losses:
            y = _target_tensor(targets, self.task)
            if y is not None:
                y = y.reshape(-1).to(embeddings.device)
                if len(y) != len(embeddings) or not torch.isfinite(y).all():
                    raise ValueError("Contrastive targets must contain one finite label per sample")
                if self.task == "classification" and not (y == y.round()).all():
                    raise ValueError("Classification contrastive targets must be integer class indices")
            labels = None if y is None else (y.long() if self.task == "classification" else (y >= 0).long())
            if return_intermediates:
                aux, contrastive_audit = self._contrastive(embeddings, availability, labels, return_intermediates=True)
            else:
                aux = self._contrastive(embeddings, availability, labels)
        result = {"logits": logits, "aux_losses": aux, "modality_representations": embeddings}
        if return_intermediates:
            result.update(
                projected_sequences={m: value.detach() for m, value in x.items()},
                unimodal_sequences=sequences, last_valid_indices=last_indices,
                last_hidden_states=last_states, modality_representations=embeddings.detach(),
                modality_availability=availability.detach(),
                normalized_representations=F.normalize(embeddings, dim=-1, eps=1e-8).detach(),
                fused_features=fused.detach(), pooled_features=fused.detach(),
            )
            if compute_losses:
                result["contrastive"] = contrastive_audit
        return result

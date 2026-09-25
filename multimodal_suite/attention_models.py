"""Mask-aware adaptations of Ren (2021), Zheng (2022), and Pan (2020).

These consume cached, aligned within-sample sequences. They are not claims of
original-data/feature-extractor replications. See attention_models_notes.md for
the verified source material, sequence-unit changes, and training differences.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    MODALITIES, MaskedRNN, ProjectedInputs, SafeAttention,
    masked_mean, masked_softmax, validate_inputs,
)


def _options(config):
    return getattr(config, "model_options", None) or {}


class _SingleStreamConversation(nn.Module):
    """Three interacting GRU states with one entity per sample.

    Ren's abstract specifies context, speaker, and emotion GRUs. Since this
    dataset supplies neither conversations nor speaker identities, this module
    has one entity state (not a fabricated set of speaker IDs). The equations
    here are an explicit engineering adaptation, not recovered IMAN equations.
    Context history is read before the entity/emotion updates; missing positions
    neither update state nor enter the history-attention normalization.
    """
    def __init__(self, d_model, dropout):
        super().__init__()
        self.context_gru = nn.GRUCell(2 * d_model, d_model)
        self.entity_gru = nn.GRUCell(2 * d_model, d_model)
        self.emotion_gru = nn.GRUCell(d_model, d_model)
        self.history_query = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask, return_states=False):
        batch, length, width = x.shape
        context = x.new_zeros(batch, width)
        entity = x.new_zeros(batch, width)
        emotion = x.new_zeros(batch, width)
        history, outputs, entity_outputs = [], [], []
        for t in range(length):
            valid = mask[:, t].bool().unsqueeze(-1)
            current = x[:, t].masked_fill(~valid, 0)
            context_candidate = self.context_gru(
                torch.cat((current, entity), dim=-1), context
            )
            context = torch.where(valid, context_candidate, context)
            if history:
                past = torch.stack(history, dim=1)
                scores = torch.einsum("bd,btd->bt", self.history_query(current), past)
                weights = masked_softmax(scores / math.sqrt(width), mask[:, :t], dim=-1)
                contextual_read = torch.einsum("bt,btd->bd", weights, past)
            else:
                contextual_read = torch.zeros_like(context)
            entity_candidate = self.entity_gru(
                torch.cat((current, contextual_read), dim=-1), entity
            )
            entity = torch.where(valid, entity_candidate, entity)
            emotion_candidate = self.emotion_gru(self.dropout(entity), emotion)
            emotion = torch.where(valid, emotion_candidate, emotion)
            history.append(context.masked_fill(~valid, 0))
            outputs.append(emotion.masked_fill(~valid, 0))
            if return_states:
                entity_outputs.append(entity.masked_fill(~valid, 0))
        emotions = torch.stack(outputs, dim=1)
        if return_states:
            return emotions, torch.stack(history, dim=1), torch.stack(entity_outputs, dim=1)
        return emotions


class Ren2021Model(nn.Module):
    """IMAN-inspired interaction + modality selection + three-state GRUs.

    Source: https://doi.org/10.1109/LSP.2021.3078698
    Verified author abstract: https://www.researchgate.net/publication/351468315
    Full equations/source code were not available. The six directional temporal
    attention blocks, residual updates, and three-state equations are declared
    adaptations of the reported components. Fifty aligned slots replace a
    conversation of utterances; there is one entity state and no speaker-ID
    claim. A masked mean produces one label per sample.
    """
    default_auxiliary_weights = {}

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        d = config.d_model
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        self.project = ProjectedInputs(feature_dims, d, config.dropout)
        self.cross_attention = nn.ModuleDict({
            f"{q}__{k}": SafeAttention(d, config.nhead, config.dropout)
            for q in MODALITIES for k in MODALITIES if q != k
        })
        self.updates = nn.ModuleDict({m: nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(config.dropout)
        ) for m in MODALITIES})
        self.norms = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.modality_score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))
        self.context_model = _SingleStreamConversation(d, config.dropout)
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(d, output_dim))

    def forward(self, features, masks, targets=None, return_intermediates=False):
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        x = self.project(features, masks)
        updated, cross_outputs = [], {}
        for query in MODALITIES:
            others = []
            for key in MODALITIES:
                if key != query:
                    attended, _ = self.cross_attention[f"{query}__{key}"](
                        x[query], x[key], x[key], masks[query], masks[key]
                    )
                    others.append(attended)
                    if return_intermediates:
                        cross_outputs[f"{query}__{key}"] = attended.detach()
            delta = self.updates[query](torch.cat([x[query], *others], dim=-1))
            updated.append(self.norms[query](x[query] + delta).masked_fill(
                ~masks[query].unsqueeze(-1), 0
            ))
        stack = torch.stack(updated, dim=2)  # [B,L,modality,D]
        modality_valid = torch.stack([masks[m] for m in MODALITIES], dim=-1)
        modality_weights = masked_softmax(
            self.modality_score(stack).squeeze(-1), modality_valid, dim=-1
        )
        fused = (modality_weights.unsqueeze(-1) * stack).sum(dim=2)
        if return_intermediates:
            emotion_states, context_states, entity_states = self.context_model(fused, union, return_states=True)
        else:
            emotion_states = self.context_model(fused, union)
        pooled = masked_mean(emotion_states, union)
        logits = self.classifier(pooled)
        result = {"logits": logits, "aux_losses": {}, "diagnostics": {
            "modality_attention": modality_weights.detach(),
            "attention_is_feature_routing_not_attribution": True,
        }}
        if return_intermediates:
            result.update(
                projected_sequences={m: value.detach() for m, value in x.items()},
                cross_attention_sequences=cross_outputs,
                unimodal_sequences={m: value.detach() for m, value in zip(MODALITIES, updated)},
                modality_attention=modality_weights.detach(),
                fusion_sequence=fused.detach(), context_sequence=context_states.detach(),
                entity_sequence=entity_states.detach(), emotion_sequence=emotion_states.detach(),
                pooled_features=pooled.detach(), fusion_mask=union.detach(),
            )
        return result


class _ResidualSequenceEncoder(nn.Module):
    """Small residual Conv1d/self-attention substitute for raw-signal encoders."""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = SafeAttention(d_model, nhead, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        valid = mask.unsqueeze(-1)
        h = F.gelu(self.conv1(x.transpose(1, 2)).transpose(1, 2)).masked_fill(~valid, 0)
        h = self.conv2(self.dropout(h).transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + self.dropout(h)).masked_fill(~valid, 0)
        attended, _ = self.self_attention(x, x, x, mask, mask)
        return self.norm2(x + self.dropout(attended)).masked_fill(~valid, 0)


class _BidirectionalPairFusion(nn.Module):
    """Two directional MHAs, concatenation, and an actual 1x1 convolution."""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.left_queries_right = SafeAttention(d_model, nhead, dropout)
        self.right_queries_left = SafeAttention(d_model, nhead, dropout)
        self.merge = nn.Conv1d(2 * d_model, d_model, kernel_size=1)

    def forward(self, left, right, left_mask, right_mask):
        lr, _ = self.left_queries_right(left, right, right, left_mask, right_mask)
        rl, _ = self.right_queries_left(right, left, left, right_mask, left_mask)
        # Explicit missing-modality extension: when an entire partner is absent,
        # forward the observed stream through the same channel-merge convolution.
        # No feature is generated for the absent stream and no mask is relabelled.
        lr = torch.where(right_mask.any(1)[:, None, None], lr, left)
        rl = torch.where(left_mask.any(1)[:, None, None], rl, right)
        union = left_mask | right_mask
        fused = self.merge(torch.cat((lr, rl), dim=-1).transpose(1, 2)).transpose(1, 2)
        return fused.masked_fill(~union.unsqueeze(-1), 0), union


class _CascadeFusion(nn.Module):
    """(A,T)->V, (A,V)->T, (T,V)->A, each with two bidirectional blocks."""
    routes = (("audio", "text", "vision"), ("audio", "vision", "text"),
              ("text", "vision", "audio"))

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.first = nn.ModuleList([
            _BidirectionalPairFusion(d_model, nhead, dropout) for _ in self.routes
        ])
        self.second = nn.ModuleList([
            _BidirectionalPairFusion(d_model, nhead, dropout) for _ in self.routes
        ])
        self.merge = nn.Conv1d(3 * d_model, d_model, kernel_size=1)

    def forward(self, x, masks, return_intermediates=False):
        paths, pairs = [], {}
        for idx, (a, b, c) in enumerate(self.routes):
            pair, pair_mask = self.first[idx](x[a], x[b], masks[a], masks[b])
            triple, union = self.second[idx](pair, x[c], pair_mask, masks[c])
            paths.append(triple)
            if return_intermediates:
                pairs[f"{a}_{b}"] = pair
        fused = self.merge(torch.cat(paths, dim=-1).transpose(1, 2)).transpose(1, 2)
        fused = fused.masked_fill(~union.unsqueeze(-1), 0)
        if return_intermediates:
            return fused, paths, pairs
        return fused, paths


class Zheng2022Model(nn.Module):
    """MCWSA-CMHA with shared fusion, three decoders, and reconstruction loss.

    Source: https://doi.org/10.1109/TMM.2022.3144885
    Author full text: https://www.researchgate.net/publication/358111733
    Equations 3--7: two directional attentions + 1x1 convolution at each stage,
    three cascade orders; equations 8--10: reconstruct each modality from one
    common fusion representation. There is exactly one fusion module shared
    by all reconstruction channels and the classification path.

    Cached vectors replace the original raw-signal encoders/decoders. Three
    train-only reconstruction warmup epochs are a small configurable default;
    the dedicated Zheng Config then uses task-only supervised fine-tuning.
    The generic suite retains its configurable auxiliary-loss defaults. Missing-stream
    fallback and masked sample pooling are explicit dataset adaptations.
    """
    default_auxiliary_weights = {"reconstruction": 0.1}
    default_pretrain_epochs = 3

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        d = config.d_model
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        self.project = ProjectedInputs(feature_dims, d, config.dropout)
        self.encoders = nn.ModuleDict({m: _ResidualSequenceEncoder(d, config.nhead, config.dropout)
                                      for m in MODALITIES})
        self.fusion = _CascadeFusion(d, config.nhead, config.dropout)
        self.decoders = nn.ModuleDict({m: nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, feature_dims[m])
        ) for m in MODALITIES})
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(config.dropout), nn.Linear(d, output_dim))

    def forward(self, features, masks, targets=None, return_intermediates=False):
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        projected = self.project(features, masks)

        x = {m: self.encoders[m](projected[m], masks[m]) for m in MODALITIES}
        # This same latent tensor supplies every decoder, so reconstruction
        # gradients really optimize the common attention/Conv fusion weights.
        if return_intermediates:
            fused, paths, pairs = self.fusion(x, masks, return_intermediates=True)
        else:
            fused, paths = self.fusion(x, masks)
            
        pooled = masked_mean(fused, union)
        logits = self.classifier(pooled)
        auxiliary, reconstructions, losses = {}, {}, {}
        if self.training or return_intermediates:
            for m in MODALITIES:
                prediction = self.decoders[m](fused)
                target = features[m].detach().masked_fill(~masks[m].unsqueeze(-1), 0)
                difference = (prediction - target).masked_fill(~masks[m].unsqueeze(-1), 0)
                denominator = (masks[m].sum() * self.feature_dims[m]).clamp_min(1)
                losses[m] = difference.square().sum() / denominator
                if return_intermediates:
                    reconstructions[m] = prediction.masked_fill(~masks[m].unsqueeze(-1), 0).detach()
            if self.training:
                auxiliary["reconstruction"] = torch.stack(list(losses.values())).sum()
        result = {"logits": logits, "aux_losses": auxiliary,
                "diagnostics": {"path_representations": torch.stack(
                    [masked_mean(p, union) for p in paths], dim=1
                ).detach()}}
        if return_intermediates:
            result.update(
                projected_sequences={m: value.detach() for m, value in projected.items()},
                encoded_sequences={m: value.detach() for m, value in x.items()},
                pair_sequences={name: value.detach() for name, value in pairs.items()},
                cascade_sequences={"_".join(route): value.detach()
                                   for route, value in zip(self.fusion.routes, paths)},
                concatenated=torch.cat(paths, dim=-1).detach(),
                fusion_sequence=fused.detach(), pooled_features=pooled.detach(),
                reconstructions=reconstructions,
                reconstruction_losses={m: value.detach() for m, value in losses.items()},
                fusion_mask=union.detach(),
            )
        return result


class _DirectionalModalityAttention(nn.Module):
    """Pan eqs. 1--4: each query attends to three modalities AT THE SAME SLOT.

    The softmax axis has length three. It is not a separate temporal attention
    softmax for each of nine modality pairs; that would implement another model.
    """
    def __init__(self, d_model, dropout, concat_skip=False, share_source_kv=False):
        super().__init__()
        self.queries = nn.ModuleDict({m: nn.Linear(d_model, d_model, bias=False) for m in MODALITIES})
        names = MODALITIES if share_source_kv else [f"{q}__{k}" for q in MODALITIES for k in MODALITIES]
        self.keys = nn.ModuleDict({name: nn.Linear(d_model, d_model, bias=False) for name in names})
        self.values = nn.ModuleDict({name: nn.Linear(d_model, d_model, bias=False) for name in names})
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        self.concat_skip = concat_skip
        self.share_source_kv = share_source_kv

    def forward(self, x, masks, return_intermediates=False):
        outputs, all_weights, attended_sequences = [], [], {}
        key_mask = torch.stack([masks[m] for m in MODALITIES], dim=-1)
        for q in MODALITIES:
            query = self.queries[q](x[q])
            keys = torch.stack([self.keys[k if self.share_source_kv else f"{q}__{k}"](x[k])
                                for k in MODALITIES], dim=2)
            values = torch.stack([self.values[k if self.share_source_kv else f"{q}__{k}"](x[k])
                                  for k in MODALITIES], dim=2)
            scores = (query.unsqueeze(2) * keys).sum(-1) / self.scale
            valid = key_mask & masks[q].unsqueeze(-1)
            weights = masked_softmax(scores, valid, dim=-1)
            weighted = (self.dropout(weights).unsqueeze(-1) * values).sum(2)
            weighted = weighted.masked_fill(~masks[q].unsqueeze(-1), 0)
            attended_sequences[q] = weighted
            outputs.append(weighted if self.concat_skip else
                           (weighted + x[q]).masked_fill(~masks[q].unsqueeze(-1), 0))
            all_weights.append(weights)
        # Figure 1 labels the skip join as concatenation: 3 attended streams
        # and 3 original projected streams. Legacy configs retain additive skips.
        if self.concat_skip:
            outputs.extend(x[m] for m in MODALITIES)
        result = torch.cat(outputs, dim=-1), torch.stack(all_weights, dim=2)
        if return_intermediates:
            return (*result, attended_sequences)
        return result


class _ContextualPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout, dense_hidden=False):
        super().__init__()
        self.rnn = MaskedRNN(input_dim, hidden_dim, kind="lstm", bidirectional=False,
                             num_layers=layers, dropout=dropout)
        self.head = (nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
                     if dense_hidden else
                     nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)))

    def forward(self, x, mask, return_intermediates=False):
        sequence = self.rnn(x, mask)
        pooled = masked_mean(sequence, mask)
        logits = self.head(pooled).masked_fill(~mask.any(1).unsqueeze(-1), 0)
        if return_intermediates:
            return logits, sequence, pooled
        return logits


class Pan2020Model(nn.Module):
    """MMAN: nine local query-key relations + cLSTM + 3 unimodal cLSTMs.

    Official full text: https://www.isca-archive.org/interspeech_2020/pan20b_interspeech.pdf
    Directional attention uses a three-modality softmax at each aligned slot,
    followed by one-layer cLSTM-MMA. Three two-layer unimodal cLSTMs retain
    modality-specific information. Four predictions feed a learned dense layer.

    Adaptations: within-sample token/segment slots replace conversation-level
    utterances; pooled sample labels replace utterance labels. The paper trains
    subnetworks separately then freezes them for late fusion. The dedicated
    Pan entry follows that schedule; the generic suite retains joint training.
    The dedicated entry also uses Figure 1's concatenated skips, two Dense
    layers and independent input projections for all four subnetworks.
    """
    default_auxiliary_weights = {"subnetwork_supervision": 1.0}
    branch_names = ("multimodal", *MODALITIES)

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        d = config.d_model
        options = _options(config)
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        self.task = config.task
        hidden = int(options.get("pan_lstm_hidden", d))
        unimodal_layers = int(options.get("pan_unimodal_layers", 2))
        if hidden < 1 or unimodal_layers < 1:
            raise ValueError("pan_lstm_hidden and pan_unimodal_layers must be positive")
        paper_layout = bool(options.get("pan_paper_layout", False))
        self.independent_branches = bool(options.get("pan_independent_branches", False))
        self.subnetworks_frozen = False
        self.project = ProjectedInputs(feature_dims, d, config.dropout)
        if self.independent_branches:
            self.unimodal_project = ProjectedInputs(feature_dims, d, config.dropout)
        self.directional_attention = _DirectionalModalityAttention(
            d, config.dropout, concat_skip=paper_layout, share_source_kv=paper_layout)
        self.multimodal_predictor = _ContextualPredictor(
            (6 if paper_layout else 3) * d, hidden, output_dim, 1, config.dropout, dense_hidden=paper_layout)
        self.unimodal_predictors = nn.ModuleDict({m: _ContextualPredictor(
            d, hidden, output_dim, unimodal_layers, config.dropout, dense_hidden=paper_layout
        ) for m in MODALITIES})
        self.late_fusion = nn.Linear(4 * output_dim, output_dim)

    def branch_modules(self, name):
        if name == "multimodal":
            return self.project, self.directional_attention, self.multimodal_predictor
        if name not in MODALITIES:
            raise ValueError(f"Unknown Pan branch: {name}")
        project = self.unimodal_project if self.independent_branches else self.project
        return project.projections[name], self.unimodal_predictors[name]

    def freeze_subnetworks(self, frozen=True):
        self.subnetworks_frozen = bool(frozen)
        for branch in self.branch_names:
            for module in self.branch_modules(branch):
                module.requires_grad_(not frozen)
                module.train(self.training and not frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.subnetworks_frozen:
            for branch in self.branch_names:
                for module in self.branch_modules(branch):
                    module.eval()
        return self

    def forward_branch(self, features, masks, branch):
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        if branch == "multimodal":
            x = self.project(features, masks)
            fused, _ = self.directional_attention(x, masks)
            return {"logits": self.multimodal_predictor(fused, union), "observed": union.any(1)}
        project, predictor = self.branch_modules(branch)
        valid = masks[branch].unsqueeze(-1)
        x = project(features[branch].masked_fill(~valid, 0)).masked_fill(~valid, 0)
        return {"logits": predictor(x, masks[branch]), "observed": masks[branch].any(1)}

    def forward(self, features, masks, targets=None, return_intermediates=False, class_weights=None):
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        x = self.project(features, masks)
        if return_intermediates:
            fused, attention, directional = self.directional_attention(x, masks, return_intermediates=True)
            prediction, context, pooled = self.multimodal_predictor(fused, union, return_intermediates=True)
            contexts, pooled_features = {"multimodal": context}, {"multimodal": pooled}
        else:
            fused, attention = self.directional_attention(x, masks)
            prediction = self.multimodal_predictor(fused, union)
        branch_logits = [prediction]
        availability = [union.any(1)]
        ux = self.unimodal_project(features, masks) if self.independent_branches else x
        for m in MODALITIES:
            if return_intermediates:
                prediction, context, pooled = self.unimodal_predictors[m](ux[m], masks[m], return_intermediates=True)
                contexts[m], pooled_features[m] = context, pooled
            else:
                prediction = self.unimodal_predictors[m](ux[m], masks[m])
            branch_logits.append(prediction)
            availability.append(masks[m].any(1))
        # The paper fuses predictions. Probabilities are only intermediate
        # classifier outputs; the final return remains logits for CrossEntropy.
        branch_predictions = []
        for prediction, observed in zip(branch_logits, availability):
            p = prediction.softmax(-1) if self.task == "classification" else prediction
            branch_predictions.append(p.masked_fill(~observed.unsqueeze(-1), 0))
        late_input = torch.cat(branch_predictions, dim=-1)
        logits = self.late_fusion(late_input)
        auxiliary = {}
        if self.training and targets is not None:
            losses = []
            for prediction, observed in zip(branch_logits, availability):
                if observed.any():
                    if self.task == "classification":
                        losses.append(F.cross_entropy(prediction[observed], targets[observed].long(), weight=class_weights))
                    else:
                        losses.append(F.l1_loss(prediction[observed].squeeze(-1), targets[observed].reshape(-1)))
            auxiliary["subnetwork_supervision"] = torch.stack(losses).mean()
        result = {"logits": logits, "aux_losses": auxiliary, "diagnostics": {
            "branch_logits": torch.stack(branch_logits, dim=1).detach(),
            "branch_order": ("multimodal", *MODALITIES),
            "directional_modality_attention": attention.detach(),
            "attention_is_feature_routing_not_attribution": True,
        }}
        if return_intermediates:
            result.update(
                projected_sequences={m: v.detach() for m, v in x.items()},
                unimodal_projected_sequences={m: v.detach() for m, v in ux.items()},
                directional_sequences={m: v.detach() for m, v in directional.items()},
                modality_attention=attention.detach(), fusion_sequence=fused.detach(),
                context_sequences={m: v.detach() for m, v in contexts.items()},
                branch_pooled_features={m: v.detach() for m, v in pooled_features.items()},
                pooled_features=torch.cat(list(pooled_features.values()), dim=-1).detach(),
                branch_logits=torch.stack(branch_logits, dim=1).detach(),
                branch_predictions=torch.stack(branch_predictions, dim=1).detach(),
                branch_availability=torch.stack(availability, dim=1).detach(),
                late_fusion_input=late_input.detach(), fusion_mask=union.detach(),
            )
        return result

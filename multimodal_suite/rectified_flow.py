"""CaReFlow released-code flow core adapted to cached, masked sequences.

References: https://arxiv.org/html/2602.19140v1, the CVPR 2026 supplement, and
https://github.com/TmacMai/CaReFlow at commit
5c9f9c7a0bb3f1202ebb3258052da2b99710c565. This is a feature-level adaptation,
not a checkpoint-compatible reproduction. The flow core follows that code where
it differs from the paper's notation: feature-MEAN squared error; live source and
detached language endpoints in forward matching; detached velocity input in
Euler integration; and a one-step, randomly timed forward estimate as the cyclic
source. Sinusoidal time features are a sine block followed by a cosine block.

Bias-free text projection and AV kernel-three convolution, per-stream LayerNorm,
mean pooling, the four-layer drift MLP, and fusion/predictor shapes follow the
release. Observed-slot mean and mask-aware pre-norm AV Transformers are explicit
adaptations. Cached BERT replaces the released fine-tuned DeBERTa backbone.
Cross-pair proposals sample with replacement and discard self-pairs; unavailable
streams are excluded. Prediction needs neither paired target text inside its
flow, labels, random interpolation times nor other batch samples.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from .common import MODALITIES, masked_mean, validate_inputs
from .robust_pretraining import _TransformerBlock, _target_tensor


class _TimeConditionedDrift(nn.Module):
    """Released sinusoidal time embedding, then 2d -> d -> d -> d -> d."""

    def __init__(self, width):
        super().__init__()
        frequency = torch.pow(10000.0, torch.linspace(0.0, 1.0, width // 2))
        self.register_buffer("time_frequency", frequency)
        self.width = width
        self.network = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width),
        )

    def time_embedding(self, times):
        if times.ndim != 2 or times.shape[1] != 1:
            raise ValueError("Flow times must have shape [pairs, 1]")
        phase = times * 1000.0 / self.time_frequency
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def forward(self, features, times):
        if times.shape != (len(features), 1):
            raise ValueError("One interpolation time is required per feature vector")
        return self.network(torch.cat((features, self.time_embedding(times)), dim=-1))


class CaReFlow2026Model(nn.Module):
    """Joint supervised fusion with adaptive forward and cyclic backward flow.

    Feature concatenation is text/audio/vision order. Flow's t is a latent
    evolution coordinate in [0, 1], unrelated to media timestamps. Reverse Euler
    trajectories start from the deterministic prediction endpoint and are only
    diagnostics. The actual cyclic training source is a random-time, one-step
    estimate; its objective is backward velocity matching, not an endpoint loss.
    """

    default_pretrain_epochs = 0
    default_auxiliary_weights = {
        "careflow_forward_audio": 0.1,
        "careflow_forward_vision": 0.1,
        "careflow_backward_audio": 0.1,
        "careflow_backward_vision": 0.1,
    }
    _default_options = {
        "careflow_steps": 2,
        "careflow_pair_ratio": 4,
        "careflow_margin": 1e-4,
        "careflow_fusion_hidden": 128,
    }

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        self.output_dim = output_dim
        self.task = getattr(config, "task", "classification")
        if self.task not in ("classification", "regression"):
            raise ValueError("CaReFlow task must be classification or regression")
        if set(feature_dims) != set(MODALITIES):
            raise ValueError("CaReFlow requires text, audio and vision feature dimensions")
        for name, value in {**feature_dims, "output_dim": output_dim,
                            "max_length": max_length, "d_model": config.d_model,
                            "nhead": config.nhead, "num_layers": config.num_layers}.items():
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"CaReFlow {name} must be a positive integer")
        if config.d_model % config.nhead or config.d_model % 2:
            raise ValueError("CaReFlow d_model must be even and divisible by nhead")
        if self.task == "regression" and output_dim != 1:
            raise ValueError("CaReFlow regression requires output_dim=1")
        if (not isinstance(config.dropout, Real) or isinstance(config.dropout, bool)
                or not math.isfinite(config.dropout) or not 0 <= config.dropout < 1):
            raise ValueError("CaReFlow dropout must be finite and in [0, 1)")
        options = getattr(config, "model_options", {})
        if not isinstance(options, dict):
            raise ValueError("CaReFlow model_options must be a dictionary")
        unknown = set(options) - set(self._default_options)
        if unknown:
            raise ValueError(f"Unknown CaReFlow model_options: {sorted(unknown)!r}")
        self.options = {**self._default_options, **options}
        for name in ("careflow_steps", "careflow_pair_ratio", "careflow_fusion_hidden"):
            value = self.options[name]
            minimum = 0 if name == "careflow_pair_ratio" else 1
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        margin = self.options["careflow_margin"]
        if (not isinstance(margin, Real) or isinstance(margin, bool)
                or not math.isfinite(margin) or margin < 0):
            raise ValueError("careflow_margin must be finite and nonnegative")
        self.steps = int(self.options["careflow_steps"])
        self.pair_ratio = int(self.options["careflow_pair_ratio"])
        self.margin = float(margin)
        self.width = width = int(config.d_model)
        fusion_hidden = int(self.options["careflow_fusion_hidden"])

        self.text_projection = nn.Linear(feature_dims["text"], width, bias=False)
        self.av_projections = nn.ModuleDict({m: nn.Conv1d(feature_dims[m], width, 3, padding=1, bias=False)
                                           for m in ("audio", "vision")})
        self.encoders = nn.ModuleDict({m: nn.ModuleList([
            _TransformerBlock(width, config.nhead, config.dropout)
            for _ in range(config.num_layers)]) for m in ("audio", "vision")})
        positions = torch.arange(max_length, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                              * (-math.log(10000.0) / width))
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = (positions * frequency).sin()
        encoding[:, 1::2] = (positions * frequency).cos()
        self.register_buffer("positional_encoding", encoding)
        self.pool_norms = nn.ModuleDict({m: nn.LayerNorm(width) for m in MODALITIES})
        self.forward_flows = nn.ModuleDict({m: _TimeConditionedDrift(width)
                                           for m in ("audio", "vision")})
        self.backward_flows = nn.ModuleDict({m: _TimeConditionedDrift(width)
                                            for m in ("audio", "vision")})
        self.fusion = nn.Sequential(nn.Linear(3 * width, fusion_hidden), nn.ReLU(),
                                    nn.Linear(fusion_hidden, width))
        self.predictor = nn.Sequential(nn.Linear(width, fusion_hidden), nn.ReLU(),
                                       nn.Linear(fusion_hidden, output_dim))

    def _labels(self, targets, reference):
        labels = _target_tensor(targets, self.task)
        if not torch.is_tensor(labels):
            raise ValueError("CaReFlow targets must resolve to a tensor")
        labels = labels.detach().to(device=reference.device).reshape(-1)
        if len(labels) != len(reference) or not torch.isfinite(labels).all():
            raise ValueError("CaReFlow requires one finite target per sample")
        if self.task == "classification":
            if ((labels != labels.long()).any() or (labels < 0).any()
                    or (labels >= self.output_dim).any()):
                raise ValueError("CaReFlow targets must be valid integer class indices")
            return labels.long()
        return labels.to(dtype=reference.dtype)

    def integrate_flow(self, flow, source, available):
        """Released Euler gradient: live residual state, detached drift input."""
        state = source.masked_fill(~available[:, None], 0)
        trajectory = [state]
        for step in range(self.steps):
            times = state.new_full((len(state), 1), step / self.steps)
            state = (state + flow(state.detach(), times) / self.steps).masked_fill(~available[:, None], 0)
            trajectory.append(state)
        return state, torch.stack(trajectory, dim=1)

    def _sample_pairs(self, source_available, target_available, include_cross=True):
        """All diagonal observations plus beta*n independent pair proposals.

        If there are no observed diagonal pairs, the requested cross count is
        zero. Endpoints are sampled uniformly with replacement from observed
        source and target streams, then same-index proposals are removed without
        resampling, as in released code. Thus cross count can be less than beta*n.
        """
        same = (source_available & target_available).nonzero(as_tuple=True)[0]
        source_indices, target_indices = same, same
        count = self.pair_ratio * len(same) if include_cross else 0
        if count:
            source_candidates = source_available.nonzero(as_tuple=True)[0]
            target_candidates = target_available.nonzero(as_tuple=True)[0]
            src_draws = torch.randint(len(source_candidates), (count,), device=same.device)
            tgt_draws = torch.randint(len(target_candidates), (count,), device=same.device)
            cross_source, cross_target = source_candidates[src_draws], target_candidates[tgt_draws]
            keep = cross_source != cross_target
            source_indices = torch.cat((same, cross_source[keep]))
            target_indices = torch.cat((same, cross_target[keep]))
        return source_indices, target_indices

    @staticmethod
    def _empty_loss(flow, reference):
        # A differentiable finite zero is useful for all-missing branches and
        # standalone auxiliary backward calls, without connecting the encoders.
        return next(flow.parameters()).sum() * 0 + reference.detach().sum() * 0

    def _forward_loss(self, modality, source, target, source_available,
                      target_available, labels=None):
        """Released forward objective: live source, detached target, mean-d error."""
        src_idx, tgt_idx = self._sample_pairs(source_available, target_available,
                                               include_cross=labels is not None)
        source_pair = source[src_idx]
        target_pair = target[tgt_idx].detach()
        same = src_idx == tgt_idx
        margins = source.new_zeros(len(src_idx))
        if labels is not None:
            difference = ((labels[src_idx] != labels[tgt_idx]).to(source.dtype)
                          if self.task == "classification"
                          else (labels[src_idx] - labels[tgt_idx]).square())
            margins = torch.where(same, margins, difference + self.margin)
        times = torch.rand((len(src_idx), 1), device=source.device, dtype=source.dtype)
        interpolation = (1 - times) * source_pair + times * target_pair
        target_velocity = target_pair - source_pair
        velocity = self.forward_flows[modality](interpolation, times)
        squared_errors = (velocity - target_velocity).square().mean(-1)
        loss = (F.relu(squared_errors - margins).mean() if len(src_idx)
                else self._empty_loss(self.forward_flows[modality], source))
        diagnostics = {
            "source_indices": src_idx, "target_indices": tgt_idx,
            "same_sample": same, "margins": margins, "times": times,
            "source_features": source_pair, "target_features": target_pair,
            "interpolated": interpolation, "velocity": velocity,
            "target_velocity": target_velocity, "squared_errors": squared_errors,
        }
        same_count = int(same.sum())
        # Diagonal pairs are first; the release uses these random-time velocity
        # predictions for its cyclic source, separately from Euler inference.
        cyclic_estimate = source[src_idx[:same_count]].detach() + velocity[:same_count]
        return loss, diagnostics, cyclic_estimate, src_idx[:same_count]

    def _backward_loss(self, modality, source, cyclic_estimate, indices):
        """Cyclic source stays live; original source detaches (released Eq. 11)."""
        start = cyclic_estimate
        endpoint = source[indices].detach()
        times = torch.rand((len(indices), 1), device=source.device, dtype=source.dtype)
        interpolation = (1 - times) * start + times * endpoint
        target_velocity = endpoint - start
        velocity = self.backward_flows[modality](interpolation, times)
        squared_errors = (velocity - target_velocity).square().mean(-1)
        loss = (squared_errors.mean() if len(indices)
                else self._empty_loss(self.backward_flows[modality], source))
        diagnostics = {
            "source_indices": indices, "target_indices": indices, "times": times,
            "source_features": start, "target_features": endpoint,
            "interpolated": interpolation, "velocity": velocity,
            "target_velocity": target_velocity, "squared_errors": squared_errors,
        }
        return loss, diagnostics

    def forward(self, features, masks, targets=None, return_intermediates=False,
                compute_auxiliary_losses=None):
        validate_inputs(features, masks, self.feature_dims, self.max_length)
        if compute_auxiliary_losses is not None and not isinstance(compute_auxiliary_losses, bool):
            raise ValueError("compute_auxiliary_losses must be bool or None")
        compute_auxiliary_losses = self.training if compute_auxiliary_losses is None else compute_auxiliary_losses
        valid = {m: masks[m].bool() for m in MODALITIES}
        clean = {m: features[m].masked_fill(~valid[m][..., None], 0) for m in MODALITIES}
        if any(not torch.isfinite(clean[m]).all() for m in MODALITIES):
            raise ValueError("CaReFlow observed features must be finite")
        available = {m: valid[m].any(1) for m in MODALITIES}
        length = clean["text"].shape[1]
        projected, sequences, pooled = {}, {}, {}
        projected["text"] = self.text_projection(clean["text"]).masked_fill(~valid["text"][..., None], 0)
        sequences["text"] = projected["text"]
        for m in ("audio", "vision"):
            projected[m] = self.av_projections[m](clean[m].transpose(1, 2)).transpose(1, 2)
            projected[m] = projected[m].masked_fill(~valid[m][..., None], 0)
            sequence = (projected[m] + self.positional_encoding[:length]).masked_fill(~valid[m][..., None], 0)
            for block in self.encoders[m]:
                sequence = block(sequence, valid[m])
            sequences[m] = sequence
        for m in MODALITIES:
            sequences[m] = self.pool_norms[m](sequences[m]).masked_fill(~valid[m][..., None], 0)
            pooled[m] = masked_mean(sequences[m], valid[m])
        mapped = {"text": pooled["text"]}
        trajectories = {}
        for m in ("audio", "vision"):
            mapped[m], trajectories[m] = self.integrate_flow(self.forward_flows[m], pooled[m], available[m])
        concatenated = torch.cat([mapped[m] for m in MODALITIES], dim=-1)
        fused = self.fusion(concatenated)
        logits = self.predictor(fused)
        auxiliary, forward_diagnostics, backward_diagnostics = {}, {}, {}
        cyclic_estimates, cyclic_indices = {}, {}
        if compute_auxiliary_losses:
            labels = self._labels(targets, logits) if targets is not None else None
            for m in ("audio", "vision"):
                (auxiliary[f"careflow_forward_{m}"], forward_diagnostics[m],
                 cyclic_estimates[m], cyclic_indices[m]) = self._forward_loss(
                    m, pooled[m], pooled["text"], available[m], available["text"], labels)
                auxiliary[f"careflow_backward_{m}"], backward_diagnostics[m] = self._backward_loss(
                    m, pooled[m], cyclic_estimates[m], cyclic_indices[m])
        output = {"logits": logits, "aux_losses": auxiliary,
                  "modality_availability": torch.stack([available[m] for m in MODALITIES], dim=1)}
        if return_intermediates:
            reconstructed, reverse_trajectories = {}, {}
            for m in ("audio", "vision"):
                reconstructed[m], reverse_trajectories[m] = self.integrate_flow(
                    self.backward_flows[m], mapped[m], available[m])
            output.update({
                "projected_sequences": projected, "unimodal_sequences": sequences,
                "pooled_modalities": pooled, "mapped_modalities": mapped,
                "forward_trajectories": trajectories,
                "flow_times": logits.new_tensor([step / self.steps for step in range(self.steps + 1)]),
                "concatenated": concatenated, "fused_features": fused,
                "pooled_features": fused, "reconstructed_modalities": reconstructed,
                "reverse_trajectories": reverse_trajectories,
                "forward_diagnostics": forward_diagnostics,
                "backward_diagnostics": backward_diagnostics,
                "cyclic_estimates": cyclic_estimates, "cyclic_indices": cyclic_indices,
            })
        return output

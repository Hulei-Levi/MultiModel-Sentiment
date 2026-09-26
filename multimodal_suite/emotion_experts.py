"""EMOE sum-fusion architecture and losses adapted to cached aligned features.

The official EMOE model, router, trainer and utility implementations were checked
against the local source snapshots in ``work/emoe_reference``. Preserved details
include the shared pointwise encoder, independent unimodal Transformers, residual
prediction heads, dense temperature-scaled router, inverse-error routing targets,
negative entropy term and residual-feature distillation with a detached teacher.

Adaptations are explicit: inputs are cached features; kernel-one, bias-free Linear
projections retain every slot; the suite's independent mask-aware Transformer
blocks use sinusoidal slot positions; routing hidden width is configurable rather
than fixed to flattened_input_dim / 8. All tensors use text/audio/vision order
(the official code uses text/vision/audio). Missing streams are masked, and short
sequences are right-padded only for the fixed-width router. Classification uses
per-expert cross entropy and Brier routing error; the official task is scalar
sentiment regression with per-expert MAE and squared routing error. This is not a
checkpoint-compatible reproduction of the original encoders or training loop.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from .common import MODALITIES, masked_softmax, validate_inputs
from .robust_pretraining import _TransformerBlock, _target_tensor


class _ResidualPredictionHead(nn.Module):
    """h + W2(dropout(ReLU(W1(h)))), followed by an output projection."""

    def __init__(self, width, output_dim, dropout):
        super().__init__()
        self.proj1 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(width, width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, features, observed=None):
        residual = features + self.proj2(self.dropout(F.relu(self.proj1(features))))
        if observed is not None:
            residual = residual.masked_fill(~observed[:, None], 0)
        logits = self.output(residual)
        if observed is not None:
            logits = logits.masked_fill(~observed[:, None], 0)
        return residual, logits


class EMOE2025Model(nn.Module):
    """Dense three-expert sum fusion with joint task/routing/distillation losses.

    ``routing_weights`` and ``modality_logits`` always use text/audio/vision order.
    Weights describe the model's gating operation, not causal contribution shares.
    ``targets`` only construct auxiliary losses; prediction never uses labels.
    The shared runtime supplies the separate fused supervised task loss.
    """

    default_pretrain_epochs = 0
    default_auxiliary_weights = {
        "emoe_unimodal": 1.0,
        "emoe_router_fit": 0.01,
        "emoe_router_entropy": 0.1,
        "emoe_distillation": 0.1,
    }
    _default_options = {
        "emoe_router_hidden": 128,
        "emoe_temperature": 0.1,
        "emoe_reliability_epsilon": 0.1,
    }

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        self.output_dim = output_dim
        self.task = getattr(config, "task", "classification")
        if self.task not in ("classification", "regression"):
            raise ValueError("EMOE task must be classification or regression")
        if set(self.feature_dims) != set(MODALITIES):
            raise ValueError("EMOE feature_dims must contain text, audio and vision")
        for name, value in {**self.feature_dims, "output_dim": output_dim,
                            "max_length": max_length, "d_model": config.d_model,
                            "nhead": config.nhead, "num_layers": config.num_layers}.items():
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"EMOE {name} must be a positive integer")
        if self.task == "regression" and output_dim != 1:
            raise ValueError("EMOE regression requires output_dim=1")
        if config.d_model % config.nhead:
            raise ValueError("EMOE d_model must be divisible by nhead")
        if (not isinstance(config.dropout, Real) or isinstance(config.dropout, bool)
                or not math.isfinite(config.dropout) or not 0 <= config.dropout < 1):
            raise ValueError("EMOE dropout must be finite and in [0, 1)")
        options = getattr(config, "model_options", {})
        if not isinstance(options, dict):
            raise ValueError("EMOE model_options must be a dictionary")
        unknown = set(options) - set(self._default_options)
        if unknown:
            raise ValueError(f"Unknown EMOE model_options: {list(unknown)!r}")
        options = {**self._default_options, **options}
        hidden = options["emoe_router_hidden"]
        if not isinstance(hidden, Integral) or isinstance(hidden, bool) or hidden <= 0:
            raise ValueError("emoe_router_hidden must be a positive integer")
        for name in ("emoe_temperature", "emoe_reliability_epsilon"):
            value = options[name]
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.temperature = float(options["emoe_temperature"])
        self.reliability_epsilon = float(options["emoe_reliability_epsilon"])
        width = int(config.d_model)
        self.projections = nn.ModuleDict({m: nn.Linear(self.feature_dims[m], width, bias=False)
                                         for m in MODALITIES})
        self.shared_encoder = nn.Linear(width, width, bias=False)
        # Independent parameters per modality; only shared_encoder is shared.
        self.encoders = nn.ModuleDict({m: nn.ModuleList([
            _TransformerBlock(width, config.nhead, config.dropout)
            for _ in range(config.num_layers)]) for m in MODALITIES})
        position = torch.arange(max_length, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                              * (-math.log(10000.0) / width))
        positional_encoding = torch.zeros(max_length, width)
        positional_encoding[:, 0::2] = torch.sin(position * frequency)
        positional_encoding[:, 1::2] = torch.cos(position * frequency[:width // 2])
        self.register_buffer("positional_encoding", positional_encoding)
        self.unimodal_heads = nn.ModuleDict({m: _ResidualPredictionHead(width, output_dim, config.dropout)
                                            for m in MODALITIES})
        self.fusion_head = _ResidualPredictionHead(width, output_dim, config.dropout)
        self.router_input = nn.Linear(max_length * sum(self.feature_dims.values()), int(hidden))
        self.router_output = nn.Linear(int(hidden), len(MODALITIES))

    def _label_auxiliaries(self, modality_logits, routing_weights, available, targets):
        """Available-expert means per sample, then batch mean; targets detach."""
        labels = _target_tensor(targets, self.task)
        if not torch.is_tensor(labels):
            raise ValueError("EMOE targets must resolve to a tensor")
        labels = labels.to(device=modality_logits.device).reshape(-1)
        if len(labels) != len(modality_logits) or not torch.isfinite(labels).all():
            raise ValueError("EMOE requires one finite target per sample")
        if self.task == "classification":
            if ((labels != labels.long()).any() or (labels < 0).any()
                    or (labels >= self.output_dim).any()):
                raise ValueError("EMOE classification targets must be valid integer class indices")
            labels = labels.long()
            repeated = labels[:, None].expand(-1, len(MODALITIES))
            losses = F.cross_entropy(modality_logits.reshape(-1, self.output_dim),
                                     repeated.reshape(-1), reduction="none").reshape_as(available)
            probabilities = torch.softmax(modality_logits, dim=-1)
            one_hot = F.one_hot(labels, num_classes=self.output_dim).to(probabilities.dtype)
            errors = (probabilities - one_hot[:, None]).square().mean(-1)
        else:
            labels = labels.to(dtype=modality_logits.dtype)
            errors = (modality_logits.squeeze(-1) - labels[:, None]).square()
            losses = (modality_logits.squeeze(-1) - labels[:, None]).abs()
        count = available.sum(1).to(modality_logits.dtype)
        unimodal = (losses.masked_fill(~available, 0).sum(1) / count).mean()
        # Detach the complete normalized target, including errors from all experts.
        inverse_error = (1.0 / (errors + self.reliability_epsilon)).masked_fill(~available, 0)
        importance = (inverse_error / inverse_error.sum(1, keepdim=True)).detach()
        router_fit = ((routing_weights - importance).square().masked_fill(~available, 0).sum(1)
                      / count).mean()
        return unimodal, router_fit, importance, errors.detach()

    def forward(self, features, masks, targets=None, return_intermediates=False,
                compute_auxiliary_losses=None):
        validate_inputs(features, masks, self.feature_dims, self.max_length)
        if compute_auxiliary_losses is not None and not isinstance(compute_auxiliary_losses, bool):
            raise ValueError("compute_auxiliary_losses must be bool or None")
        compute_auxiliary_losses = self.training if compute_auxiliary_losses is None else compute_auxiliary_losses
        valid = {m: masks[m].bool() for m in MODALITIES}
        clean = {m: features[m].masked_fill(~valid[m][..., None], 0) for m in MODALITIES}
        if any(not torch.isfinite(clean[m]).all() for m in MODALITIES):
            raise ValueError("EMOE observed features must be finite")
        available = torch.stack([valid[m].any(1) for m in MODALITIES], dim=1)
        batch, length = clean["text"].shape[:2]
        slot_inputs = torch.cat([clean[m] for m in MODALITIES], dim=-1)
        router_features = F.pad(slot_inputs, (0, 0, 0, self.max_length - length)).reshape(batch, -1)
        router_hidden = F.relu(F.normalize(self.router_input(router_features), p=2, dim=-1))
        raw_router_logits = self.router_output(router_hidden)
        router_logits = raw_router_logits / self.temperature
        routing_weights = masked_softmax(router_logits, available, dim=-1)

        projected, shared, encoded, last_indices = {}, {}, {}, {}
        experts, residual_experts, predictions = [], [], []
        for index, m in enumerate(MODALITIES):
            projected[m] = self.projections[m](clean[m]).masked_fill(~valid[m][..., None], 0)
            shared[m] = self.shared_encoder(projected[m]).masked_fill(~valid[m][..., None], 0)
            sequence = (shared[m] + self.positional_encoding[:length]).masked_fill(~valid[m][..., None], 0)
            for block in self.encoders[m]:
                sequence = block(sequence, valid[m])
            encoded[m] = sequence
            slots = torch.arange(length, device=sequence.device)[None].expand(batch, -1)
            last_indices[m] = slots.masked_fill(~valid[m], -1).max(dim=1).values
            last = sequence[torch.arange(batch, device=sequence.device), last_indices[m].clamp_min(0)]
            last = last.masked_fill(~available[:, index, None], 0)
            residual, logits = self.unimodal_heads[m](last, available[:, index])
            experts.append(last)
            residual_experts.append(residual)
            predictions.append(logits)
        expert_representations = torch.stack(experts, dim=1)
        unimodal_residual_features = torch.stack(residual_experts, dim=1)
        modality_logits = torch.stack(predictions, dim=1)
        weighted_experts = routing_weights[..., None] * expert_representations
        fused_features = weighted_experts.sum(1)
        fusion_residual_features, logits = self.fusion_head(fused_features)
        # Official uni_distill operates on residual FEATURES, not class logits.
        # The whole teacher includes routing weights before detachment.
        distillation_target = (routing_weights[..., None] * unimodal_residual_features).sum(1).detach()
        target_probabilities = torch.softmax(distillation_target, dim=-1)
        auxiliary = {}
        importance_targets = routing_errors = None
        if compute_auxiliary_losses:
            auxiliary["emoe_distillation"] = F.mse_loss(
                torch.softmax(fusion_residual_features, dim=-1), target_probabilities)
            entropy_terms = routing_weights * routing_weights.clamp_min(1e-9).log()
            auxiliary["emoe_router_entropy"] = (available.sum(1) * entropy_terms.sum(1)).mean()
            if targets is not None:
                unimodal, router_fit, importance_targets, routing_errors = self._label_auxiliaries(
                    modality_logits, routing_weights, available, targets)
                auxiliary["emoe_unimodal"] = unimodal
                auxiliary["emoe_router_fit"] = router_fit
        output = {"logits": logits, "aux_losses": auxiliary,
                  "routing_weights": routing_weights, "modality_logits": modality_logits,
                  "modality_availability": available}
        if return_intermediates:
            output.update({
                "projected_sequences": projected,
                "shared_sequences": shared,
                "unimodal_sequences": encoded,
                "expert_representations": expert_representations,
                "unimodal_residual_features": unimodal_residual_features,
                "last_valid_indices": last_indices,
                "router_features": router_features,
                "router_hidden": router_hidden,
                "router_raw_logits": raw_router_logits,
                "router_logits": router_logits,
                "weighted_experts": weighted_experts,
                "fused_features": fused_features,
                "fusion_residual_features": fusion_residual_features,
                "pooled_features": fusion_residual_features,
                "distillation_target": distillation_target,
                "distillation_target_probabilities": target_probabilities,
            })
            if importance_targets is not None:
                output["importance_targets"] = importance_targets
                output["routing_errors"] = routing_errors
        return output

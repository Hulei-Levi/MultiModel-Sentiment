"""Aligned-feature adaptations of MFRM and TransModality.

TransModality follows Sections 3.2--3.3 of arXiv:2009.02902: two
text-primary forward/backward Transformer cells, encoder feature concatenation,
and four MAE reconstruction objectives. Its original utterance sequence is
adapted here to within-sample aligned slots with sample-level pooling.

MFRM is explicitly a *principle-level adaptation*: the author repository has
only a README, and the full RMN/EIA equations could not be verified. We retain
same-slot interaction, acoustic intensity/change selection, and additive memory
carry; the exact fusion channels and recurrence below are our documented design.
See memory_translation_notes.md for sources and all material deviations.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .common import (
    MODALITIES, MaskedRNN, ProjectedInputs, SafeAttention,
    masked_mean, masked_softmax, validate_inputs,
)


def _zero_invalid(x, mask):
    return x.masked_fill(~mask.bool().unsqueeze(-1), 0)


class _ContextEncoder(nn.Module):
    """Independent bidirectional GRU and a tanh dense projection."""
    def __init__(self, input_dim, d_model, layers, dropout):
        super().__init__()
        self.gru = MaskedRNN(input_dim, max(1, d_model // 2), kind="gru",
                             bidirectional=True, num_layers=layers, dropout=dropout)
        self.projection = nn.Sequential(nn.Linear(self.gru.output_dim, d_model),
                                        nn.Tanh(), nn.Dropout(dropout))

    def forward(self, x, mask):
        return _zero_invalid(self.projection(self.gru(x, mask)), mask)


class _ResidualMemory(nn.Module):
    """Adapted residual accumulator, not a claimed copy of the paper's RMN.

    s_t = s_(t-1) + sigmoid(g([x_t, s_(t-1)]))*tanh(u([x_t, s_(t-1)])).
    Missing positions do not update state; returned sequences retain slot IDs.
    A normalization is applied to emitted states, not to the direct carry path.
    """
    def __init__(self, dimension, dropout):
        super().__init__()
        self.write_gate = nn.Linear(2 * dimension, dimension)
        self.update = nn.Linear(2 * dimension, dimension)
        self.read_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.write_gate.bias, -2.0)

    def forward(self, sequence, valid, return_intermediates=False):
        sequence = _zero_invalid(sequence, valid)
        state = sequence.new_zeros((len(sequence), sequence.shape[-1]))
        outputs = []
        raw_states, write_gates, candidates, updates = [], [], [], []
        for time_index in range(sequence.shape[1]):
            joint = torch.cat((sequence[:, time_index], state), dim=-1)
            gate = torch.sigmoid(self.write_gate(joint))
            candidate = torch.tanh(self.update(joint))
            delta = self.dropout(gate * candidate)
            proposal = state + delta
            state = torch.where(valid[:, time_index, None], proposal, state)
            outputs.append(_zero_invalid(self.read_norm(state), valid[:, time_index]))
            if return_intermediates:
                raw_states.append(state.detach())  # Includes unchanged carry at missing slots.
                write_gates.append(_zero_invalid(gate, valid[:, time_index]).detach())
                candidates.append(_zero_invalid(candidate, valid[:, time_index]).detach())
                updates.append(_zero_invalid(delta, valid[:, time_index]).detach())
        if return_intermediates:
            return torch.stack(outputs, dim=1), self.read_norm(state), {
                "memory_state_sequence": torch.stack(raw_states, dim=1),
                "memory_write_gates": torch.stack(write_gates, dim=1),
                "memory_candidates": torch.stack(candidates, dim=1),
                "memory_updates": torch.stack(updates, dim=1),
                "final_raw_memory": state.detach(),
            }
        return torch.stack(outputs, dim=1), self.read_norm(state)


class MFRM2022Model(nn.Module):
    """MFRM-inspired slot fusion + acoustic selection + residual memory.

    This is not an equation-level MFRM reproduction. In particular, seven
    Hadamard interaction channels and the residual accumulator are explicit
    implementation choices. It is a runnable controlled adaptation for cached
    text/audio/vision features, requiring no claimed acoustic intensity labels.
    """
    default_auxiliary_weights = {}

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        d = config.d_model
        options = dict(getattr(config, "model_options", {}) or {})
        self.intensity_gain = float(options.get("mfrm_intensity_gain", 1.0))
        if not math.isfinite(self.intensity_gain) or self.intensity_gain < 0:
            raise ValueError("mfrm_intensity_gain must be finite and nonnegative")
        self.inputs = ProjectedInputs(feature_dims, d, config.dropout)
        self.context = nn.ModuleDict({m: _ContextEncoder(d, d, config.num_layers,
                                        config.dropout) for m in MODALITIES})
        # Three unimodal, three pairwise, and one trimodal interaction channel.
        self.slot_fusion = nn.Sequential(nn.Linear(7 * d, d), nn.GELU(),
                                        nn.LayerNorm(d), nn.Dropout(config.dropout))
        # Audio strength AND change: zero changes for the first observed slot.
        self.intensity = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.memory = _ResidualMemory(d, config.dropout)
        self.classifier = nn.Sequential(nn.Linear(5 * d, d), nn.GELU(),
                                        nn.Dropout(config.dropout), nn.Linear(d, output_dim))

    def forward(self, features, masks, return_intermediates=False):
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        projected = self.inputs(features, masks)
        context = {m: self.context[m](projected[m], masks[m]) for m in MODALITIES}
        text, audio, vision = (context[m] for m in MODALITIES)
        interactions = [text, audio, vision, text * audio, text * vision,
                        audio * vision, text * audio * vision]
        fused = _zero_invalid(self.slot_fusion(torch.cat(interactions, dim=-1)), union)

        # Use the previous *observed audio* value; padding/holes never masquerade
        # as an emotional change. This tracks slot order, not elapsed wall time.
        previous = torch.zeros_like(audio[:, 0])
        has_previous = torch.zeros(len(audio), dtype=torch.bool, device=audio.device)
        changes = []
        for time_index in range(audio.shape[1]):
            valid = masks["audio"][:, time_index]
            current = audio[:, time_index]
            changes.append((current - previous).abs().masked_fill(
                ~(valid & has_previous)[:, None], 0))
            previous = torch.where(valid[:, None], current, previous)
            has_previous = has_previous | valid
        change = torch.stack(changes, dim=1)
        scores = self.intensity(torch.cat((audio.abs(), change), dim=-1)).squeeze(-1)
        alpha = masked_softmax(scores, masks["audio"], dim=1)
        # A neutral factor of one preserves non-audio evidence at missing slots.
        # Normalizing by audio length keeps the gate on a comparable scale.
        emphasis = 1 + self.intensity_gain * alpha * masks["audio"].sum(1, keepdim=True)
        attended = _zero_invalid(fused * emphasis.unsqueeze(-1), union)
        if return_intermediates:
            memory_sequence, final_memory, memory_audit = self.memory(attended, union, return_intermediates=True)
        else:
            memory_sequence, final_memory = self.memory(attended, union)
        utterance_features = [masked_mean(context[m], masks[m]) for m in MODALITIES]
        representation = torch.cat([masked_mean(memory_sequence, union), final_memory,
                                    *utterance_features], dim=-1)
        result = {"logits": self.classifier(representation), "aux_losses": {},
                "emotion_intensity_weights": alpha,
                "fused_sequence": fused, "memory_sequence": memory_sequence}
        if return_intermediates:
            interaction_names = ("text", "audio", "vision", "text_audio", "text_vision", "audio_vision", "text_audio_vision")
            interaction_masks = [masks["text"], masks["audio"], masks["vision"],
                                 masks["text"] & masks["audio"], masks["text"] & masks["vision"],
                                 masks["audio"] & masks["vision"],
                                 masks["text"] & masks["audio"] & masks["vision"]]
            result.update(
                projected_sequences={m: value.detach() for m, value in projected.items()},
                unimodal_sequences={m: value.detach() for m, value in context.items()},
                interaction_sequences={name: value.detach() for name, value in zip(interaction_names, interactions)},
                interaction_masks={name: value.detach() for name, value in zip(interaction_names, interaction_masks)},
                concatenated=torch.cat(interactions, dim=-1).detach(),
                audio_strength=audio.abs().detach(), audio_change=change.detach(),
                intensity_scores=scores.masked_fill(~masks["audio"], 0).detach(),
                emotion_intensity_weights=alpha.detach(), emphasis_factors=emphasis.detach(),
                fused_sequence=fused.detach(), fusion_sequence=fused.detach(),
                attended_sequence=attended.detach(), memory_sequence=memory_sequence.detach(),
                final_memory=final_memory.detach(),
                unimodal_pooled_features={m: value.detach() for m, value in zip(MODALITIES, utterance_features)},
                pooled_features=representation.detach(), fusion_mask=union.detach(),
                **memory_audit,
            )
        return result


class _PositionEncoding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                                * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x, mask):
        return _zero_invalid(x + self.encoding[:, :x.shape[1]].to(dtype=x.dtype), mask)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, ff_dim):
        super().__init__()
        self.attention = SafeAttention(d_model, nhead, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_dim, d_model))

    def forward(self, x, mask):
        attended, _ = self.attention(x, x, x, mask, mask)
        x = _zero_invalid(self.norm1(x + self.dropout(attended)), mask)
        return _zero_invalid(self.norm2(x + self.dropout(self.ff(x))), mask)


class _DecoderLayer(nn.Module):
    """Non-autoregressive observed-target decoder, with safe masked attention."""
    def __init__(self, d_model, nhead, dropout, ff_dim):
        super().__init__()
        self.self_attention = SafeAttention(d_model, nhead, dropout)
        self.cross_attention = SafeAttention(d_model, nhead, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_dim, d_model))

    def forward(self, query, source, query_mask, source_mask):
        query_mask = query_mask & source_mask.any(1, keepdim=True)
        attended, _ = self.self_attention(query, query, query, query_mask, query_mask)
        query = _zero_invalid(self.norms[0](query + self.dropout(attended)), query_mask)
        translated, _ = self.cross_attention(query, source, source, query_mask, source_mask)
        query = _zero_invalid(self.norms[1](query + self.dropout(translated)), query_mask)
        return _zero_invalid(self.norms[2](query + self.dropout(self.ff(query))), query_mask)


class _TranslationTransformer(nn.Module):
    def __init__(self, d_model, nhead, layers, dropout, ff_dim, max_length):
        super().__init__()
        self.positions = _PositionEncoding(max_length, d_model)
        self.encoder = nn.ModuleList([_EncoderLayer(d_model, nhead, dropout, ff_dim)
                                      for _ in range(layers)])
        self.decoder = nn.ModuleList([_DecoderLayer(d_model, nhead, dropout, ff_dim)
                                      for _ in range(layers)])

    def forward(self, source, target_context, source_mask, target_mask):
        encoded = self.positions(source, source_mask)
        for layer in self.encoder:
            encoded = layer(encoded, source_mask)
        decode_mask = target_mask & source_mask.any(1, keepdim=True)
        decoded = self.positions(target_context, decode_mask)
        for layer in self.decoder:
            decoded = layer(decoded, encoded, decode_mask, source_mask)
        return encoded, decoded, decode_mask


class _ModalityFusionCell(nn.Module):
    """alpha -> beta -> alpha; backward input really is forward decoder output."""
    def __init__(self, d_model, alpha_dim, beta_dim, config, max_length):
        super().__init__()
        options = dict(getattr(config, "model_options", {}) or {})
        ff_dim = int(options.get("translation_ff_dim", 2 * d_model))
        if ff_dim <= 0:
            raise ValueError("translation_ff_dim must be positive")
        args = (d_model, config.nhead, config.num_layers, config.dropout, ff_dim, max_length)
        self.forward_transformer = _TranslationTransformer(*args)
        self.backward_transformer = _TranslationTransformer(*args)
        self.to_beta = nn.Linear(d_model, beta_dim)
        self.to_alpha = nn.Linear(d_model, alpha_dim)

    def forward(self, alpha, beta, alpha_mask, beta_mask):
        encoded_forward, decoded_forward, forward_mask = self.forward_transformer(
            alpha, beta, alpha_mask, beta_mask)
        encoded_backward, decoded_backward, backward_mask = self.backward_transformer(
            decoded_forward, alpha, forward_mask, alpha_mask)
        return {"encoded_forward": encoded_forward, "encoded_backward": encoded_backward,
                "decoded_forward": decoded_forward, "decoded_backward": decoded_backward,
                "forward_mask": forward_mask, "backward_mask": backward_mask,
                "predicted_beta": _zero_invalid(self.to_beta(decoded_forward), forward_mask),
                "reconstructed_alpha": _zero_invalid(self.to_alpha(decoded_backward), backward_mask)}


def _masked_mae(prediction, target, valid):
    # Mask BEFORE subtraction: NaN values at padded targets must not poison loss.
    target = _zero_invalid(target, valid)
    error = _zero_invalid((prediction - target).abs(), valid)
    return error.sum() / (valid.sum() * prediction.shape[-1]).clamp_min(1)


class TransModality2020Model(nn.Module):
    """Two parallel forward/backward text-primary translation fusion cells.

    Equation (4)'s four encoder streams and three contextual streams are
    concatenated at aligned slots. Masked sample pooling replaces the paper's
    per-utterance classification because this dataset supplies one sample label.
    Translation decoders condition on observed target context; reconstructions
    are auxiliary tasks, not a guarantee of missing-modality generation.
    """
    default_auxiliary_weights = {
        "translation_text_audio": 0.1, "cycle_audio_text": 0.1,
        "translation_text_vision": 0.1, "cycle_vision_text": 0.1,
    }

    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.max_length = max_length
        d = config.d_model
        self.context = nn.ModuleDict({m: _ContextEncoder(feature_dims[m], d,
                                     config.num_layers, config.dropout) for m in MODALITIES})
        self.cells = nn.ModuleDict({m: _ModalityFusionCell(d, feature_dims["text"],
                                   feature_dims[m], config, max_length) for m in ("audio", "vision")})
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(7 * d, output_dim)

    def forward(self, features, masks, return_intermediates=False,
                compute_reconstruction_losses=None):
        """Predict without sentiment labels; optionally inspect translation paths.

        Reconstruction losses are computed during training by default. Explicit
        True enables an eval-mode reconstruction check without dropout or updates.
        Observed target features still condition the decoder at inference time.
        """
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        masks = {m: masks[m].bool() for m in MODALITIES}
        context = {m: self.context[m](features[m], masks[m]) for m in MODALITIES}
        encoded_streams, losses, reconstruction = [], {}, {}
        encoder_sequences, decoder_sequences, encoder_masks, decoder_masks = {}, {}, {}, {}
        compute_losses = self.training if compute_reconstruction_losses is None else bool(compute_reconstruction_losses)
        for other in ("audio", "vision"):
            output = self.cells[other](context["text"], context[other],
                                       masks["text"], masks[other])
            encoded_streams.extend([output["encoded_forward"], output["encoded_backward"]])
            if compute_losses:
                losses[f"translation_text_{other}"] = _masked_mae(
                    output["predicted_beta"], features[other], output["forward_mask"])
                losses[f"cycle_{other}_text"] = _masked_mae(
                    output["reconstructed_alpha"], features["text"], output["backward_mask"])
            reconstruction[other] = {"predicted_beta": output["predicted_beta"],
                                      "reconstructed_text": output["reconstructed_alpha"],
                                      "forward_mask": output["forward_mask"],
                                      "backward_mask": output["backward_mask"]}
            if return_intermediates:
                forward_name, backward_name = f"text_to_{other}", f"{other}_to_text"
                encoder_sequences[forward_name] = output["encoded_forward"].detach()
                encoder_sequences[backward_name] = output["encoded_backward"].detach()
                decoder_sequences[forward_name] = output["decoded_forward"].detach()
                decoder_sequences[backward_name] = output["decoded_backward"].detach()
                encoder_masks[forward_name] = masks["text"].detach()
                encoder_masks[backward_name] = output["forward_mask"].detach()
                decoder_masks[forward_name] = output["forward_mask"].detach()
                decoder_masks[backward_name] = output["backward_mask"].detach()
        # Order: t->a, a->t, t->v, v->t, t, a, v (permutation of Eq. 4).
        fused = torch.cat([*encoded_streams, *(context[m] for m in MODALITIES)], dim=-1)
        representation = masked_mean(fused, union)
        result = {"logits": self.classifier(self.dropout(representation)), "aux_losses": losses,
                  "fused_sequence": fused, "reconstructions": reconstruction}
        if return_intermediates:
            result.update(
                unimodal_sequences={m: value.detach() for m, value in context.items()},
                encoder_sequences=encoder_sequences, decoder_sequences=decoder_sequences,
                encoder_masks=encoder_masks, decoder_masks=decoder_masks,
                concatenated=fused.detach(), fused_sequence=fused.detach(),
                pooled_features=representation.detach(), fusion_mask=union.detach(),
                reconstructions={m: {k: v.detach() for k, v in values.items()}
                                 for m, values in reconstruction.items()},
            )
        return result

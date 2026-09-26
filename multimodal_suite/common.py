"""Shared mask-aware layers; masks mean observed slots, never inferred from zeros."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODALITIES = ("text", "audio", "vision")


def validate_inputs(features, masks, feature_dims, max_length):
    shape = features["text"].shape[:2]
    if len(shape) != 2 or not 0 < shape[1] <= max_length:
        raise ValueError("Expected a sequence length between 1 and max_length")
    for m in MODALITIES:
        if features[m].shape != (*shape, feature_dims[m]) or masks[m].shape != shape:
            raise ValueError(f"Invalid features or mask for {m}")
    union = torch.stack([masks[m].bool() for m in MODALITIES], -1).any(-1)
    if not union.any(1).all():
        raise ValueError("Cannot predict a sample with all three modalities unobserved")
    return union


def masked_mean(x, mask):
    mask = mask.bool()
    return x.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def masked_softmax(scores, mask, dim=-1):
    mask = mask.bool()
    safe = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(safe, dim=dim).masked_fill(~mask, 0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)


class SafeAttention(nn.Module):
    """MHA with zero outputs for absent queries or an entirely absent key source."""
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, query, key, value, query_mask, key_mask, need_weights=False):
        qm, km = query_mask.bool(), key_mask.bool()
        query = query.masked_fill(~qm.unsqueeze(-1), 0)
        key = key.masked_fill(~km.unsqueeze(-1), 0)
        value = value.masked_fill(~km.unsqueeze(-1), 0)
        has_key = km.any(1)
        # An always-present tensor slot is enabled only for examples without keys.
        sentinel = key.new_zeros((len(key), 1, key.shape[-1]))
        key, value = torch.cat((key, sentinel), 1), torch.cat((value, sentinel), 1)
        key_valid = torch.cat((km, ~has_key[:, None]), 1)
        output, weights = self.attention(query, key, value, key_padding_mask=~key_valid,
                                         need_weights=need_weights, average_attn_weights=False)
        observed = qm & has_key[:, None]
        output = output.masked_fill(~observed.unsqueeze(-1), 0)
        if weights is not None:
            weights = weights[..., :-1].masked_fill(~observed[:, None, :, None], 0)
        return output, weights


class ProjectedInputs(nn.Module):
    def __init__(self, feature_dims, d_model, dropout=0.0):
        super().__init__()
        self.projections = nn.ModuleDict({m: nn.Sequential(nn.Linear(feature_dims[m], d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)) for m in MODALITIES})

    def forward(self, features, masks):
        result = {}
        for m in MODALITIES:
            valid = masks[m].bool().unsqueeze(-1)
            clean = features[m].masked_fill(~valid, 0)
            result[m] = self.projections[m](clean).masked_fill(~valid, 0)
        return result


class MaskedRNN(nn.Module):
    """Pack observed inputs, encode, scatter back to the original time slots."""
    def __init__(self, input_dim, hidden_dim, kind="gru", bidirectional=True, num_layers=1, dropout=0.0):
        super().__init__()
        if kind not in ("gru", "lstm"):
            raise ValueError("kind must be gru or lstm")
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                       num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x, mask):
        valid = mask.bool()
        clean = x.masked_fill(~valid.unsqueeze(-1), 0)
        result = clean.new_zeros((*clean.shape[:2], self.output_dim)) + clean.sum(-1, keepdim=True) * 0
        lengths = valid.sum(1)
        active = lengths.nonzero(as_tuple=True)[0]
        if not len(active):
            return result
        xx, mm = clean[active], valid[active]
        position = torch.arange(x.shape[1], device=x.device).expand(len(active), -1)
        order = position.masked_fill(~mm, x.shape[1]).argsort(1)
        compact = xx.gather(1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        packed = pack_padded_sequence(compact, lengths[active].cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        dense, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        restored = torch.zeros_like(dense).scatter(1, order.unsqueeze(-1).expand(-1, -1, self.output_dim), dense)
        restored = restored.masked_fill(~mm.unsqueeze(-1), 0)
        return result.index_copy(0, active, restored)

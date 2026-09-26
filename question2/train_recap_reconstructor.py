"""Single-file reconstruction methods: RECAP, MTSIT, BRITS, Centaur, NAOMI and CSDI adaptations.

这个代码用于重建缺失特征，对缺失的特征进行复原

Only reconstruction is trained/evaluated. No emotion labels, sentiment model,
adversarial loss, FID module, or reconstructed dataset export is involved.
RECAP uses multiscale convolutions and within-sample cross-modal attention.
MTSIT uses a temporal Transformer encoder and linear decoder, with geometric
training masks by default. BRITS uses bidirectional recurrent imputation with
observed-value estimation and consistency losses. All methods retain the common
clean-target reconstruction metric and input/output contract.
Centaur adds a joint Conv2d denoising autoencoder with a dense bottleneck.
NAOMI adds deterministic bidirectional multiresolution recurrent imputation.
CSDI adds conditional score diffusion with a noise-prediction training objective.
Fixed p010/p020/p030 validation chooses a checkpoint;
the four test conditions are evaluated only after selection.

Run from MultiModel-Sentiment:
  .venv-align/bin/python -B question2/train_recap_reconstructor.py self-test
  .venv-align/bin/python -B question2/train_recap_reconstructor.py train --device cuda:0
  .venv-align/bin/python -B question2/train_recap_reconstructor.py train --method mtsit --d-model 16 --heads 4 --layers 3 --ffn-dim 16 --output-root results/second_question/mtsit_reconstruction

Import load_reconstructor(checkpoint) to obtain a ReconstructionBundle.
ReconstructionBundle.from_checkpoint(checkpoint) is the equivalent class API.
Config.method / --method selects the network; data, loss and evaluation remain
shared. Add a BaseReconstructor subclass and register it in RECONSTRUCTOR_METHODS
below to compare another method in this same file.
bundle.reconstruct(features, valid_mask, corruption_mask) accepts raw-unit
numpy arrays and returns the same three shapes. The text input MUST have been
encoded from the corrupted tokens, not from the original unmasked sentence.
corruption_mask means artificial erasure, never padding or natural zero rows.
Only erased audio/video rows are replaced. Text can change over the complete
valid sequence. Uncorrupted samples pass through exactly; padding stays zero.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MODS = ("text", "audio", "vision")
DIMS = {"text": 768, "audio": 74, "vision": 35}
PROBS = (0.1, 0.2, 0.3)
SPLITS = ("train", "valid", "test")


@dataclass
class Config:
    data_dir: str = str(ROOT / "datasets/附件2-同步扰动特征")
    adapter_module: str = str(SCRIPT_DIR / "bert_feature_adapter.py")
    adapter_checkpoint: str = str(ROOT / "experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt")
    output_root: str = str(ROOT / "results/second_question/recap_reconstruction")
    device: str = "cuda:0"
    seed: int = 42
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    max_length: int = 50
    kernels: tuple = (3, 5, 9)
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0
    std_floor: float = 1e-4
    cpu_threads: int = 4
    method: str = "recap"
    ffn_dim: int = 16  # MTSIT feed-forward width; RECAP retains 2*d_model.
    mask_distribution: str = "auto"  # RECAP: Bernoulli; MTSIT: geometric.
    mean_mask_length: float = 3.0
    brits_estimation_weight: float = 1.0
    brits_consistency_weight: float = 0.1
    centaur_latent_dim: int = 200
    naomi_highest: int = 8
    naomi_decoder_dim: int = 128
    csdi_steps: int = 50
    csdi_samples: int = 3
    csdi_sampling_seed: int = 314159
    csdi_microbatch: int = 32
    csdi_time_dim: int = 32
    csdi_feature_dim: int = 16
    csdi_beta_start: float = 1e-4
    csdi_beta_end: float = 0.5
    csdi_amp: bool = True
    csdi_compile: bool = True
    csdi_validation_interval: int = 5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_pickle(path):
    # These are the user's trusted locally generated pickle files.
    with Path(path).open("rb") as f:
        return pickle.load(f)


def eligible_slots(text_bert):
    return text_bert[:, 1].astype(bool) & ~np.isin(text_bert[:, 0], (0, 101, 102))


def select_fields(split):
    """Neither labels nor raw_text are retained in the training data view."""
    keys = (*MODS, "text_bert", "id", "corruption_mask")
    return {key: split[key] for key in keys}


def validate_clean(split, name):
    n = len(split["id"])
    tb = np.asarray(split["text_bert"])
    if tb.shape != (n, 3, 50) or not np.isfinite(tb).all():
        raise ValueError(f"{name}: invalid text_bert")
    if not np.array_equal(tb, tb.astype(np.int64)) or not np.isin(tb[:, 1], [0, 1]).all():
        raise ValueError(f"{name}: invalid token IDs or attention mask")
    valid = tb[:, 1].astype(bool)
    if not valid.any(1).all() or np.any((tb[:, 0] == 100) & valid):
        raise ValueError(f"{name}: empty sequence or pre-existing UNK; inspect before training")
    if split["corruption_mask"].shape != (n, 50) or split["corruption_mask"].any():
        raise ValueError(f"{name}: p000 must have no artificial corruption")
    for m in MODS:
        x = np.asarray(split[m])
        if x.shape != (n, 50, DIMS[m]) or not np.isfinite(x).all():
            raise ValueError(f"{name}/{m}: invalid shape or non-finite values")
        if np.any(x[~valid] != 0):
            raise ValueError(f"{name}/{m}: nonzero padding conflicts with exact passthrough contract")


def validate_pair(clean, corrupted, name):
    if list(clean["id"]) != list(corrupted["id"]):
        raise ValueError(f"{name}: sample order/IDs differ")
    mask = np.asarray(corrupted["corruption_mask"])
    if mask.dtype != np.bool_ or mask.shape != clean["text_bert"][:, 0].shape:
        raise ValueError(f"{name}: invalid explicit corruption mask")
    if np.any(mask & ~eligible_slots(clean["text_bert"])):
        raise ValueError(f"{name}: corruption outside eligible content slots")
    expected = clean["text_bert"].copy()
    expected[:, 0][mask] = 100
    if not np.array_equal(expected, corrupted["text_bert"]):
        raise ValueError(f"{name}: token corruption does not match recorded mask")
    valid = expected[:, 1].astype(bool)
    for m in MODS:
        x = corrupted[m]
        if x.shape != clean[m].shape or not np.isfinite(x).all():
            raise ValueError(f"{name}/{m}: invalid features")
        if np.any(x[~valid] != 0):
            raise ValueError(f"{name}/{m}: nonzero padding")
        if m != "text":
            if np.any(x[mask] != 0) or not np.array_equal(x[~mask], clean[m][~mask]):
                raise ValueError(f"{name}/{m}: observed values changed or erased values nonzero")


def audit_splits(data):
    sets, report = {}, {"splits": {}, "cross_split": []}
    for name in SPLITS:
        s = data[name]
        validate_clean(s, name)
        ids = [str(x) for x in s["id"]]
        if any("$_$" not in x for x in ids):
            raise ValueError("Unknown original-video ID format; audit manually")
        videos = {x.split("$_$")[0] for x in ids}
        text = {}
        fingerprints = []
        for i, raw in enumerate(s["raw_text"]):
            text.setdefault(" ".join(str(raw).lower().split()), []).append(ids[i])
            h = hashlib.sha256()
            for key in ("text_bert", "audio", "vision"):
                h.update(np.ascontiguousarray(s[key][i]).tobytes())
            fingerprints.append(h.hexdigest())
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs within {name}")
        sets[name] = (set(ids), videos, text, set(fingerprints))
        report["splits"][name] = {"samples": len(ids), "original_videos": len(videos),
                                  "unique_multimodal_fingerprints": len(set(fingerprints))}
    for ai, a in enumerate(SPLITS):
        for b in SPLITS[ai + 1:]:
            sa, sb = sets[a], sets[b]
            entry = {"splits": [a, b], "shared_ids": sorted(sa[0] & sb[0]),
                     "shared_videos": sorted(sa[1] & sb[1]),
                     "shared_multimodal_samples": len(sa[3] & sb[3]),
                     "shared_normalized_text": [{"text": t, a: sa[2][t], b: sb[2][t]}
                                                for t in sorted(set(sa[2]) & set(sb[2]))]}
            report["cross_split"].append(entry)
            if entry["shared_ids"] or entry["shared_videos"] or entry["shared_multimodal_samples"]:
                raise ValueError("Split leakage risk: " + json.dumps(entry, ensure_ascii=False))
    report["interpretation"] = (
        "No duplicate IDs, original videos, or full multimodal samples across splits. "
        "Generic phrases okay/alright occur in train and valid, with distinct videos and AV. "
        "Splits are preserved; identical textual phrases are disclosed, not removed. "
        "Dynamic masks of a training sample never cross splits. Clean targets are loss-only. "
        "The existing frozen BERT adapter was previously fit on train and selected on valid. "
        "This reconstructor does not refit the adapter or use test to select parameters.")
    return report


class Normalizer:
    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, clean_train, floor=1e-4):
        valid = clean_train["text_bert"][:, 1].astype(bool)
        stats = {}
        for m in MODS:
            # Accumulate in float64 without a large persistent float64 copy.
            total = np.zeros(DIMS[m], np.float64)
            total2 = np.zeros_like(total)
            count = 0
            for start in range(0, len(valid), 64):
                x = clean_train[m][start:start + 64][valid[start:start + 64]].astype(np.float64)
                total += x.sum(0)
                total2 += np.square(x).sum(0)
                count += len(x)
            mean = total / count
            raw_std = np.sqrt(np.maximum(total2 / count - mean ** 2, 0))
            stats[m] = {"mean": mean.astype(np.float32), "std": np.maximum(raw_std, floor).astype(np.float32),
                        "valid_slots": count, "floored_dimensions": int((raw_std < floor).sum())}
        return cls(stats)

    def normalize(self, features, valid, device):
        out = {}
        for m in MODS:
            x = np.asarray(features[m], dtype=np.float32)
            y = (x - self.stats[m]["mean"]) / self.stats[m]["std"]
            out[m] = torch.from_numpy(np.where(valid[..., None], y, 0).astype(np.float32)).to(device)
        return out

    def checkpoint(self):
        return {m: {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                    for k, v in stat.items()} for m, stat in self.stats.items()}

    @classmethod
    def from_checkpoint(cls, value):
        return cls({m: {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in stat.items()}
                    for m, stat in value.items()})


class TemporalBlock(nn.Module):
    def __init__(self, d, kernels, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.branches = nn.ModuleList(nn.Conv1d(d, d, k, padding=k // 2) for k in kernels)
        self.merge = nn.Linear(d * len(kernels), d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, valid):
        x = self.norm(h).masked_fill(~valid[..., None], 0).transpose(1, 2)
        y = torch.cat([torch.nn.functional.gelu(conv(x)).transpose(1, 2) for conv in self.branches], -1)
        return (h + self.dropout(self.merge(y))).masked_fill(~valid[..., None], 0)


class BaseReconstructor(nn.Module):
    """Shared normalized-feature reconstruction contract for every method.

    Subclasses implement predict_residuals(features, valid_mask, corruption_mask)
    and return {'text': [B,L,768], 'audio': [B,L,74], 'vision': [B,L,35]}.
    Residuals use normalized input units and preserve the batch dimension.
    Neither clean targets nor emotion labels are network inputs.

    The shared forward applies residuals to the original feature sequence:
    text may change at all valid slots of corrupted samples; audio/vision only
    at explicitly erased slots. Uncorrupted samples and observed AV are copied.
    Register a new subclass in RECONSTRUCTOR_METHODS, then use Config(method=...).
    """

    description = "Sequence feature reconstruction"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def predict_residuals(self, features, valid_mask, corruption_mask):
        raise NotImplementedError("Implement the reconstruction method's residual predictor")

    def forward(self, features, valid_mask, corruption_mask):
        _, length = valid_mask.shape
        if length > self.cfg.max_length:
            raise ValueError("Input exceeds configured sequence length")
        residuals = self.predict_residuals(features, valid_mask, corruption_mask)
        return self.apply_residuals(features, valid_mask, corruption_mask, residuals)

    def training_objective(self, features, target, valid, missing):
        return reconstruction_loss(self(features, valid, missing), target, valid, missing)

    def apply_residuals(self, features, valid_mask, corruption_mask, residuals):
        """Apply the unchanged shared output contract to normalized residuals."""
        if set(residuals) != set(MODS):
            raise ValueError("A reconstruction method must return all three residuals")
        any_missing = corruption_mask.any(1, keepdim=True)
        outputs = {}
        for m in MODS:
            delta = residuals[m]
            if delta.shape != features[m].shape:
                raise ValueError(f"Invalid {m} residual shape: {tuple(delta.shape)}")
            change = valid_mask & (any_missing if m == "text" else corruption_mask)
            outputs[m] = torch.where(change[..., None], features[m] + delta, features[m])
            outputs[m] = outputs[m].masked_fill(~valid_mask[..., None], 0)
        return outputs


class RECAPReconstructor(BaseReconstructor):
    """Multiscale temporal convolutions plus within-sample joint attention."""

    description = "RECAP-inspired lightweight feature reconstruction; not full RECAP"

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.d_model
        self.project = nn.ModuleDict({m: nn.Linear(DIMS[m], d) for m in MODS})
        self.temporal = nn.ModuleDict({m: TemporalBlock(d, cfg.kernels, cfg.dropout) for m in MODS})
        self.position = nn.Parameter(torch.empty(1, cfg.max_length, d))
        self.modality = nn.Parameter(torch.empty(3, 1, d))
        self.erasure = nn.Embedding(2, d)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.modality, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, dim_feedforward=2 * d,
                                           dropout=cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, cfg.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.decode = nn.ModuleDict({m: nn.Linear(d, DIMS[m]) for m in MODS})
        # The initial model is the corrupted-input baseline, avoiding random
        # destructive changes to the contextual text representation.
        for layer in self.decode.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def predict_residuals(self, features, valid_mask, corruption_mask):
        """Only corrupted features and masks enter this function; no targets."""
        _, length = valid_mask.shape
        h = []
        for mi, m in enumerate(MODS):
            z = self.project[m](features[m]) + self.position[:, :length] + self.modality[mi]
            z = z + self.erasure(corruption_mask.long())
            z = z.masked_fill(~valid_mask[..., None], 0)
            h.append(self.temporal[m](z, valid_mask))
        # B is always the batch dimension. Attention spans only the 3L slots
        # of one sample; examples cannot attend to each other.
        z = self.interaction(torch.cat(h, dim=1), src_key_padding_mask=~valid_mask.repeat(1, 3))
        
        return {m: self.decode[m](z[:, mi * length:(mi + 1) * length])
                for mi, m in enumerate(MODS)}


class MaskedSequenceBatchNorm(nn.Module):
    """MTSIT BatchNorm across valid batch/time entries, excluding padding."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, eps=1e-5)

    def forward(self, sequence, valid):
        observed = sequence[valid]
        if not len(observed):
            return torch.zeros_like(sequence)
        if self.training and len(observed) == 1:
            # PyTorch training BatchNorm needs at least two entries.
            normalized = torch.nn.functional.batch_norm(
                observed, self.norm.running_mean, self.norm.running_var,
                self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps)
        else:
            normalized = self.norm(observed)
        result = torch.zeros_like(sequence)
        result[valid] = normalized
        return result


class MTSITEncoderBlock(nn.Module):
    """Post-residual BatchNorm encoder with bidirectional temporal attention."""

    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm1 = MaskedSequenceBatchNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = MaskedSequenceBatchNorm(dim)

    def forward(self, sequence, valid):
        attended, _ = self.attention(sequence, sequence, sequence,
                                      key_padding_mask=~valid, need_weights=False)
        sequence = self.norm1(sequence + self.attention_dropout(attended), valid)
        return self.norm2(sequence + self.ffn_dropout(self.ffn(sequence)), valid)


class MTSITReconstructor(BaseReconstructor):
    """MTSIT adapted to the shared multimodal data/loss/output contract.

    Paper: https://doi.org/10.1109/LSP.2022.3224880
    Author code: https://github.com/koc-lab/MTSIT
    Same-slot features form [B,L,877], so attention spans L time steps, not 3L.
    Uses learnable positions, post-BatchNorm encoder blocks, GELU and a linear
    decoder producing absolute normalized estimates. Subtracting the input
    only adapts that direct decoder to BaseReconstructor's residual interface.

    Adaptations: train-only feature normalization, padding-excluded BatchNorm,
    token-first contextual BERT corruption, shared modality-balanced loss
    (all valid text / missing AV), p=.1/.2/.3, and existing validation/optimizer.
    Common Config model dimensions remain explicit; paper-size settings are
    d_model=16, heads=4, layers=3, ffn_dim=16.
    """

    description = "MTSIT adapted to aligned multimodal reconstruction; shared benchmark loss"

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.d_model < 1 or cfg.heads < 1 or cfg.layers < 1 or cfg.ffn_dim < 1:
            raise ValueError("MTSIT dimensions, heads and layers must be positive")
        if cfg.d_model % cfg.heads:
            raise ValueError("MTSIT d_model must be divisible by heads")
        self.project = nn.Linear(sum(DIMS.values()), cfg.d_model)
        self.position = nn.Parameter(torch.empty(1, cfg.max_length, cfg.d_model))
        nn.init.uniform_(self.position, -.02, .02)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList(
            MTSITEncoderBlock(cfg.d_model, cfg.heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.layers))
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.decode = nn.Linear(cfg.d_model, sum(DIMS.values()))

    def predict_residuals(self, features, valid_mask, corruption_mask):
        if not valid_mask.any(1).all():
            raise ValueError("MTSIT needs at least one valid slot in each sample")
        joined = torch.cat([features[m] for m in MODS], dim=-1)
        # Follow masked-autoencoder input semantics, without reading clean text.
        # Retained text slots still contain the context of the corrupted sentence.
        joined = joined.masked_fill((~valid_mask | corruption_mask)[..., None], 0)
        hidden = self.project(joined) * self.cfg.d_model ** .5
        hidden = self.input_dropout(hidden + self.position[:, :joined.shape[1]])
        hidden = hidden.masked_fill(~valid_mask[..., None], 0)
        for block in self.encoder:
            hidden = block(hidden, valid_mask)
        estimate = self.decode(self.output_dropout(torch.nn.functional.gelu(hidden)))
        estimates = estimate.split([DIMS[m] for m in MODS], dim=-1)
        return {m: value - features[m] for m, value in zip(MODS, estimates)}



class BRITSTemporalDecay(nn.Module):
    """exp(-ReLU(W delta + b)); feature decay has a diagonal W."""

    def __init__(self, input_dim, output_dim, diagonal=False):
        super().__init__()
        self.diagonal = diagonal
        if diagonal and input_dim != output_dim:
            raise ValueError("Diagonal decay needs equal input/output dimensions")
        shape = (input_dim,) if diagonal else (output_dim, input_dim)
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(output_dim))
        bound = input_dim ** -.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, delta):
        value = (delta * self.weight + self.bias if self.diagonal else
                 torch.nn.functional.linear(delta, self.weight, self.bias))
        return torch.exp(-torch.relu(value))


class BRITSFeatureRegression(nn.Module):
    """Regress each feature from other dimensions, with an exact zero diagonal."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.empty(dim))
        self.register_buffer("off_diagonal", torch.ones(dim, dim) - torch.eye(dim))
        nn.init.uniform_(self.weight, -dim ** -.5, dim ** -.5)
        nn.init.uniform_(self.bias, -dim ** -.5, dim ** -.5)

    def forward(self, values):
        return torch.nn.functional.linear(values, self.weight * self.off_diagonal, self.bias)


def brits_time_gaps(observed, valid):
    """Paper Eq. 1 in valid-slot units, computed separately for each direction.

    First valid step has delta=0; later delta=1+(1-m_previous)*delta_previous.
    Padding neither advances the clock nor supplies an observation.
    """
    positions = valid.long().cumsum(1) - 1
    observation_positions = torch.where(observed & valid, positions, 0)
    latest = observation_positions.cummax(1).values
    previous = torch.nn.functional.pad(latest[:, :-1], (1, 0), value=0)
    return (positions - previous).masked_fill(~valid, 0)


class BRITSDirection(nn.Module):
    """One RITS: history/feature estimates -> sigmoid blend -> LSTMCell.

    Imputations remain in the recurrent computation graph. Only hidden h,
    not cell c, is decayed. Observed entries always feed their input values.
    """

    def __init__(self, modality_dims, hidden_dim):
        super().__init__()
        dim = sum(modality_dims)
        self.hidden_dim = hidden_dim
        self.temporal_decay_h = BRITSTemporalDecay(dim, hidden_dim)
        self.temporal_decay_x = BRITSTemporalDecay(dim, dim, diagonal=True)
        self.history = nn.Linear(hidden_dim, dim)
        self.feature_regression = BRITSFeatureRegression(dim)
        self.combine = nn.Linear(2 * dim, dim)
        self.cell = nn.LSTMCell(2 * dim, hidden_dim)
        # Equal modality weighting, rather than letting 768 text dimensions dominate.
        self.register_buffer("feature_weights", torch.cat([
            torch.full((d,), 1.0 / (len(modality_dims) * d)) for d in modality_dims]))

    def forward(self, values, valid, observed, collect_loss=False):
        batch, length, dim = values.shape
        h = values.new_zeros(batch, self.hidden_dim)
        c = torch.zeros_like(h)
        deltas = brits_time_gaps(observed, valid).to(values.dtype)
        imputations, observation_error = [], values.new_zeros(())
        # This masked weight is time independent; keep its autograd graph intact.
        feature_weight = self.feature_regression.weight * self.feature_regression.off_diagonal
        for slot in range(length):
            active = valid[:, slot:slot + 1]
            mask = observed[:, slot:slot + 1].expand(-1, dim)
            numeric_mask = mask.to(values.dtype)
            x = values[:, slot]
            delta = deltas[:, slot:slot + 1].expand(-1, dim)
            gamma_h = self.temporal_decay_h(delta)
            gamma_x = self.temporal_decay_x(delta)
            decayed_h = h * gamma_h
            historical = self.history(decayed_h)
            history_completed = torch.where(mask, x, historical)
            feature_estimate = torch.nn.functional.linear(
                history_completed, feature_weight, self.feature_regression.bias)
            # Follow paper Eq. 8 (the original code omitted this sigmoid).
            alpha = torch.sigmoid(self.combine(torch.cat([gamma_x, numeric_mask], -1)))
            combined = alpha * feature_estimate + (1 - alpha) * historical
            completed = torch.where(mask, x, combined)
            next_h, next_c = self.cell(torch.cat([completed, numeric_mask], -1), (decayed_h, c))
            h = torch.where(active, next_h, h)
            c = torch.where(active, next_c, c)
            imputations.append(completed.masked_fill(~active, 0))
            if collect_loss:
                absolute_errors = ((historical - x).abs() + (feature_estimate - x).abs()
                                   + (combined - x).abs()) / 3
                per_sample = (absolute_errors * self.feature_weights).sum(-1)
                observation_error = observation_error + (per_sample * observed[:, slot]).sum()
        estimation_loss = observation_error / observed.sum().clamp_min(1)
        return {"imputations": torch.stack(imputations, 1),
                "observed_estimation_loss": estimation_loss}


class BRITSReconstructor(BaseReconstructor):
    """BRITS adapted to the common clean-target benchmark.

    Paper: https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
    Author code: https://github.com/caow13/BRITS
    Two independently parameterized RITS operate directly on the aligned 877D
    vectors. Reverse-direction gaps are recomputed after reversing input/masks.
    Following BRITS, all observed entries (including text) are copied exactly;
    therefore contextual text contamination outside erased slots is not denoised.

    The common text-all-valid / AV-missing clean-target MSE remains the primary
    loss and checkpoint criterion. Training adds BRITS observed-value estimation
    MAE (history, feature and blend; both directions averaged) and 0.1-weighted
    consistency MAE. Both auxiliary terms are modality balanced and exclude
    padding. Consistency averages over all valid slots; observed differences
    are exactly zero. No downstream labels enter reconstruction training.
    """

    description = "BRITS adapted to aligned multimodal reconstruction; shared benchmark plus BRITS auxiliary losses"

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.d_model < 1 or cfg.max_length < 1:
            raise ValueError("BRITS needs positive hidden size and maximum length")
        if not (np.isfinite(cfg.brits_estimation_weight) and cfg.brits_estimation_weight >= 0
                and np.isfinite(cfg.brits_consistency_weight) and cfg.brits_consistency_weight >= 0):
            raise ValueError("BRITS auxiliary weights must be finite and nonnegative")
        dims = tuple(DIMS[m] for m in MODS)
        self.forward_rits = BRITSDirection(dims, cfg.d_model)
        self.backward_rits = BRITSDirection(dims, cfg.d_model)

    def directional_outputs(self, features, valid, missing, collect_loss=False):
        if valid.shape[1] > self.cfg.max_length or not valid.any(1).all():
            raise ValueError("BRITS needs a nonempty sequence within configured length")
        observed = valid & ~missing
        values = torch.cat([features[m] for m in MODS], -1)
        # Erased placeholders and padding must not become observed evidence.
        values = values.masked_fill(~observed[..., None], 0)
        forward = self.forward_rits(values, valid, observed, collect_loss)
        backward = self.backward_rits(values.flip(1), valid.flip(1), observed.flip(1), collect_loss)
        backward["imputations"] = backward["imputations"].flip(1)
        return forward, backward

    def residuals_from_directions(self, features, forward, backward):
        mean = (forward["imputations"] + backward["imputations"]) * .5
        return {m: part - features[m] for m, part in
                zip(MODS, mean.split([DIMS[m] for m in MODS], -1))}

    def predict_residuals(self, features, valid_mask, corruption_mask):
        forward, backward = self.directional_outputs(features, valid_mask, corruption_mask)
        return self.residuals_from_directions(features, forward, backward)

    def training_objective(self, features, target, valid, missing):
        forward, backward = self.directional_outputs(features, valid, missing, collect_loss=True)
        residuals = self.residuals_from_directions(features, forward, backward)
        pred = self.apply_residuals(features, valid, missing, residuals)
        primary, terms = reconstruction_loss(pred, target, valid, missing)
        estimation = (forward["observed_estimation_loss"] + backward["observed_estimation_loss"]) * .5
        difference = (forward["imputations"] - backward["imputations"]).abs()
        per_slot = sum(part.mean(-1) for part in difference.split(
            [DIMS[m] for m in MODS], -1)) / len(MODS)
        consistency = (per_slot * valid).sum() / valid.sum().clamp_min(1)
        total = (primary + self.cfg.brits_estimation_weight * estimation
                 + self.cfg.brits_consistency_weight * consistency)
        return total, {**terms, "primary_mse": primary,
                       "observed_estimation_mae": estimation, "consistency_mae": consistency}



class CentaurReconstructor(BaseReconstructor):
    """Centaur convolutional DAE adapted to aligned BERT/audio/vision features.

    Paper: https://doi.org/10.1109/JSEN.2024.3388893
    Author code: https://github.com/sustainable-computing/Centaur
    Joint input [B,1,877,L] follows the paper's sensor-by-time matrix layout.
    Four Conv2d(k5,s2,p2)+ReLU stages -> flat Linear latent -> Linear+ReLU
    -> four ConvTranspose2d stages, with ReLU except at the final output.
    No pooling, normalization, dropout, skip connection or residual bypass.

    Adaptations: input dimensions and decoder sizes, unbounded output instead
    of sigmoid for standardized signed features, Bernoulli slot corruption,
    shared clean-target loss and outer preservation of observed AV/intact data.
    Adjacent embedding dimensions are not physical neighboring sensor channels.
    This implements the DAE only; the existing frozen ThisWork remains the head.
    """

    description = "Centaur convolutional denoising autoencoder adapted to aligned multimodal features"

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.d_model < 1 or cfg.centaur_latent_dim < 1 or cfg.max_length < 1:
            raise ValueError("Centaur requires positive channels, latent width and sequence length")
        channels = (1, cfg.d_model, 2 * cfg.d_model, 4 * cfg.d_model, 8 * cfg.d_model)
        self.spatial_shapes = [(sum(DIMS.values()), cfg.max_length)]
        self.encoder = nn.ModuleList()
        for source, target in zip(channels[:-1], channels[1:]):
            self.encoder.append(nn.Conv2d(source, target, 5, stride=2, padding=2))
            height, width = self.spatial_shapes[-1]
            self.spatial_shapes.append(((height + 1) // 2, (width + 1) // 2))
        self.encoded_channels = channels[-1]
        flat_dim = channels[-1] * self.spatial_shapes[-1][0] * self.spatial_shapes[-1][1]
        # As in the author implementation, latent Linear has no activation.
        self.to_latent = nn.Linear(flat_dim, cfg.centaur_latent_dim)
        self.from_latent = nn.Linear(cfg.centaur_latent_dim, flat_dim)
        self.decoder = nn.ModuleList()
        reverse_channels = channels[::-1]
        self.decoder_kernels = (3, 2, 3, 2)  # Author PAMAP2 decoder schedule.
        for stage, kernel in enumerate(self.decoder_kernels):
            before, after = self.spatial_shapes[-1 - stage], self.spatial_shapes[-2 - stage]
            if kernel == 3:
                padding = (1, 1)
                output_padding = tuple(target - (2 * source - 1)
                                       for source, target in zip(before, after))
            else:
                # k2: p0/op0 produces 2n; p1/op1 produces 2n-1.
                padding = tuple(2 * source - target for source, target in zip(before, after))
                output_padding = padding
            if any(value not in (0, 1) for value in (*padding, *output_padding)):
                raise ValueError("Cannot invert Centaur encoder spatial dimensions")
            self.decoder.append(nn.ConvTranspose2d(
                reverse_channels[stage], reverse_channels[stage + 1], kernel,
                stride=2, padding=padding, output_padding=output_padding))

    def normalized_estimate(self, features, valid, missing):
        length = valid.shape[1]
        if length < 1 or length > self.cfg.max_length or not valid.any(1).all():
            raise ValueError("Centaur requires a nonempty sequence within configured length")
        joined = torch.cat([features[m] for m in MODS], -1)
        # Artificially erased slots are mean-filled in normalized coordinates.
        # Remaining text still contains the context of the corrupted sentence.
        joined = joined.masked_fill((~valid | missing)[..., None], 0)
        matrix = joined.transpose(1, 2).unsqueeze(1)
        if length < self.cfg.max_length:
            matrix = torch.nn.functional.pad(matrix, (0, self.cfg.max_length - length))
        h = matrix
        for layer in self.encoder:
            h = torch.relu(layer(h))
        latent = self.to_latent(h.flatten(1))
        h = torch.relu(self.from_latent(latent))
        h = h.reshape(len(matrix), self.encoded_channels, *self.spatial_shapes[-1])
        for index, layer in enumerate(self.decoder):
            h = layer(h)
            if index < len(self.decoder) - 1:
                h = torch.relu(h)
        if tuple(h.shape[2:]) != self.spatial_shapes[0]:
            raise RuntimeError("Centaur decoder failed to restore original feature/time axes")
        # Linear output: applying sigmoid would clip signed standardized targets.
        return h[:, 0, :, :length].transpose(1, 2)

    def predict_residuals(self, features, valid_mask, corruption_mask):
        estimate = self.normalized_estimate(features, valid_mask, corruption_mask)
        # This subtraction is an API bridge; the decoder itself is absolute.
        return {m: value - features[m] for m, value in
                zip(MODS, estimate.split([DIMS[m] for m in MODS], -1))}



class NAOMIReconstructor(BaseReconstructor):
    """Deterministic, free-running NAOMI adapted to aligned feature sequences.

    https://arxiv.org/abs/1901.10946
    https://github.com/felixykliu/NAOMI/blob/master/model.py
    Forward GRU consumes completed values; backward GRU consumes mask+values.
    A separate MLP for every power-of-two step imputes left+step using the
    forward state at left and backward state at left+2*step (author code).
    Generated values update both states and remain attached to the graph.

    Unlike the author's shared-batch-mask sampler, schedules are per sample.
    Zero hidden states at virtual endpoints extend the sampler to leading,
    trailing and fully missing valid sequences. No padding is an observation.
    Only erased slots change, including text. This is imputation, so it cannot
    denoise contextual contamination in the remaining observed BERT tokens.
    """

    description = "NAOMI deterministic multiresolution recurrent imputation (free-running adaptation)"

    def __init__(self, cfg):
        super().__init__(cfg)
        highest = cfg.naomi_highest
        if (cfg.d_model < 1 or cfg.layers < 1 or cfg.naomi_decoder_dim < 1
                or highest < 1 or highest & (highest - 1)):
            raise ValueError("NAOMI requires positive widths/layers and a power-of-two highest step")
        self.steps = tuple(2**i for i in range(highest.bit_length()))
        width = sum(DIMS.values())
        # Original NAOMI GRUs have no dropout; cfg.dropout belongs to other methods.
        self.forward_gru = nn.GRU(width, cfg.d_model, cfg.layers)
        self.backward_gru = nn.GRU(width + 1, cfg.d_model, cfg.layers)
        self.decoders = nn.ModuleDict({
            str(step): nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.naomi_decoder_dim),
                                     nn.ReLU(), nn.Linear(cfg.naomi_decoder_dim, width))
            for step in self.steps})
        self.resolution_counts = {str(step): 0 for step in self.steps}

    def schedule(self, valid, missing):
        """Mask-only schedule: (left, step, sample indices); no feature/target access."""
        valid = np.asarray(valid, dtype=bool)
        missing = np.asarray(missing, dtype=bool)
        if valid.ndim != 2 or missing.shape != valid.shape or np.any(missing & ~valid):
            raise ValueError("NAOMI needs matching valid/missing masks; padding cannot be missing")
        lengths = valid.sum(1)
        if (np.any(lengths == 0)
                or not np.array_equal(valid, np.arange(valid.shape[1])[None] < lengths[:, None])):
            raise ValueError("NAOMI requires nonempty, right-padded contiguous valid slots")
        known = valid & ~missing
        operations = {}
        counts = {str(step): 0 for step in self.steps}
        for left in range(-1, int(lengths.max()) - 1):
            current = []
            # After a coarse insertion the next insertion at this left is finer.
            for step in reversed(self.steps):
                rows = []
                for row in np.flatnonzero(valid[:, left + 1] & ~known[:, left + 1]):
                    right = left + 1
                    while right < lengths[row] and not known[row, right]:
                        right += 1
                    selected = min(self.cfg.naomi_highest, 1 << (((right - left) // 2).bit_length() - 1))
                    if selected == step:
                        rows.append(int(row))
                if rows:
                    known[rows, left + step] = True
                    current.append((step, rows))
                    counts[str(step)] += len(rows)
            if current:
                operations[left] = current
        if not np.array_equal(known, valid):
            raise AssertionError("NAOMI schedule did not fill every missing valid slot")
        return operations, counts

    def normalized_estimate(self, features, valid, missing):
        operations, counts = self.schedule(valid.detach().cpu().numpy(),
                                            missing.detach().cpu().numpy())
        for key, value in counts.items():
            self.resolution_counts[key] += value
        joined = torch.cat([features[m] for m in MODS], -1)
        # Missing values (including contextual UNK embeddings) are not observations.
        masked = joined.masked_fill((missing | ~valid)[..., None], 0)
        values = list(masked.unbind(1))
        batch, length, width = masked.shape
        zero_state = masked.new_zeros(self.cfg.layers, batch, self.cfg.d_model)
        backward = [None] * (length + 1)
        backward[length] = zero_state
        for slot in range(length - 1, -1, -1):
            indicator = (valid[:, slot] & ~missing[:, slot]).to(masked.dtype)[:, None]
            inputs = torch.cat((indicator, values[slot]), -1).unsqueeze(0)
            _, candidate = self.backward_gru(inputs, backward[slot + 1])
            backward[slot] = torch.where(valid[:, slot][None, :, None],
                                         candidate, backward[slot + 1])
        forward = zero_state
        for left in range(-1, length - 1):
            if left >= 0:
                _, candidate = self.forward_gru(values[left].unsqueeze(0), forward)
                forward = torch.where(valid[:, left][None, :, None], candidate, forward)
            for step, rows in operations.get(left, ()):
                index = torch.as_tensor(rows, dtype=torch.long, device=masked.device)
                right = left + step
                context = torch.cat((forward[-1].index_select(0, index),
                                     backward[left + 2 * step][-1].index_select(0, index)), -1)
                generated = self.decoders[str(step)](context)
                values[right] = values[right].index_copy(0, index, generated)
                if step > 1:
                    state = backward[right + 1].index_select(1, index)
                    inputs = torch.cat((generated.new_ones(len(rows), 1), generated), -1)
                    _, state = self.backward_gru(inputs.unsqueeze(0), state)
                    backward[right] = backward[right].index_copy(1, index, state)
                    # Slots between the new pivot and its next finer left midpoint
                    # are still missing. Cache their refreshed backward states.
                    zeros = generated.new_zeros(1, len(rows), width + 1)
                    for slot in range(right - 1, left + step // 2 - 1, -1):
                        _, state = self.backward_gru(zeros, state)
                        backward[slot] = backward[slot].index_copy(1, index, state)
        estimate = torch.stack(values, 1)
        return torch.where(missing[..., None], estimate, joined).masked_fill(~valid[..., None], 0)

    def predict_residuals(self, features, valid_mask, corruption_mask):
        estimate = self.normalized_estimate(features, valid_mask, corruption_mask)
        return {m: (value - features[m]).masked_fill(~corruption_mask[..., None], 0)
                for m, value in zip(MODS, estimate.split([DIMS[m] for m in MODS], -1))}

    def training_objective(self, features, target, valid, missing):
        loss, terms = super().training_objective(features, target, valid, missing)
        if not loss.requires_grad:
            # An entirely intact training batch is an identity mapping. Retain
            # a zero derivative so the shared loop can call backward safely.
            loss = loss + self.forward_gru.weight_ih_l0.sum() * 0
        return loss, terms


class CSDILinearAttention(nn.Module):
    """Global noncausal linear Transformer used by the author's high-D option.

    Matches the dependency's default PreNorm attention + PreNorm FF(4*C):
    softmax(Q, channels), softmax(K, tokens), Q(K^T V)/sqrt(head_dim).
    This is global attention, not independent feature chunks or a PCA bottleneck.
    """
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.out = nn.Linear(channels, channels)
        self.ff = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(),
                                nn.Linear(4 * channels, channels))

    def forward(self, x, valid=None):
        batch, length, channels = x.shape
        normalized = self.norm1(x)
        def split(layer):
            return layer(normalized).reshape(batch, length, self.heads, -1).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        if valid is not None:
            k = k.masked_fill(~valid[:, None, :, None], -torch.finfo(k.dtype).max)
            v = v.masked_fill(~valid[:, None, :, None], 0)
        # Softmax in float32 also protects mixed-precision normalization.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float().softmax(-1) / (channels // self.heads) ** .5
            k = k.float().softmax(-2)
            context = k.transpose(-1, -2) @ v.float()
            attended = (q @ context).transpose(1, 2).reshape(batch, length, channels)
        attended = attended.to(x.dtype)
        x = x + self.out(attended)
        x = x + self.ff(self.norm2(x))
        if valid is not None:
            x = x.masked_fill(~valid[..., None], 0)
        return x


class CSDIResidualBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg.d_model
        self.diffusion_projection = nn.Linear(128, width)
        self.time_attention = CSDILinearAttention(width, cfg.heads)
        self.feature_attention = CSDILinearAttention(width, cfg.heads)
        self.mid_projection = nn.Linear(width, 2 * width)
        self.side_projection = nn.Linear(cfg.csdi_time_dim + cfg.csdi_feature_dim + 1, 2 * width)
        self.output_projection = nn.Linear(width, 2 * width)
        for layer in (self.mid_projection, self.side_projection, self.output_projection):
            nn.init.kaiming_normal_(layer.weight)

    def side_context(self, time_embedding, feature_embedding, observed):
        # Exactly factor a 1x1 projection of [time, feature-ID, cond-mask].
        # Avoid materializing a [B,L,877,time_dim+feature_dim+1] tensor.
        nt, nf = time_embedding.shape[-1], feature_embedding.shape[-1]
        weight = self.side_projection.weight
        time = torch.nn.functional.linear(time_embedding, weight[:, :nt],
                                           self.side_projection.bias)[None, :, None]
        feature = torch.nn.functional.linear(feature_embedding, weight[:, nt:nt+nf])[None, None]
        mask = observed[..., None, None] * weight[:, -1]
        return time + feature + mask

    def forward(self, x, diffusion, side, valid):
        batch, length, features, channels = x.shape
        y = x + self.diffusion_projection(diffusion)[:, None, None]
        if length > 1:
            sequence = y.permute(0, 2, 1, 3).reshape(batch * features, length, channels)
            mask = valid[:, None].expand(-1, features, -1).reshape(batch * features, length)
            y = self.time_attention(sequence, mask).reshape(batch, features, length, channels).permute(0, 2, 1, 3)
        if features > 1:
            y = self.feature_attention(y.reshape(batch * length, features, channels)).reshape(batch, length, features, channels)
        gate, filt = (self.mid_projection(y) + side).chunk(2, -1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.output_projection(y).chunk(2, -1)
        keep = valid[:, :, None, None]
        return ((x + residual) / 2**.5).masked_fill(~keep, 0), skip.masked_fill(~keep, 0)


class CSDIReconstructor(BaseReconstructor):
    """Conditional DDPM imputation with CSDI's global linear-attention option.

    Sources: https://github.com/ermongroup/CSDI (main_model.py/diff_models.py,
    config/base_forecasting.yaml), and its linear-attention-transformer dependency.
    Complete 877 scalar features are retained. Clean hidden values only construct
    noisy TRAINING states and noise targets; they are never inference conditions.
    The public forward performs reverse diffusion without clean targets.
    Observed text/AV pass through; padding is neither condition nor loss.
    """
    description = "CSDI conditional diffusion with global linear attention (feature-level adaptation)"

    def __init__(self, cfg):
        super().__init__(cfg)
        if (cfg.d_model < 1 or cfg.heads < 1 or cfg.d_model % cfg.heads
                or cfg.layers < 1 or cfg.csdi_steps < 2 or cfg.csdi_samples < 1
                or cfg.csdi_microbatch < 1 or cfg.csdi_time_dim < 2
                or cfg.csdi_time_dim % 2 or cfg.csdi_feature_dim < 1
                or not 0 < cfg.csdi_beta_start < cfg.csdi_beta_end < 1):
            raise ValueError("Invalid CSDI dimensions, schedule or sampling settings")
        width = sum(DIMS.values())
        beta = np.linspace(cfg.csdi_beta_start**.5, cfg.csdi_beta_end**.5, cfg.csdi_steps)**2
        alpha = np.cumprod(1 - beta)
        previous = np.concatenate(([1.], alpha[:-1]))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("alpha_bar", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("posterior_std", torch.tensor(
            (beta * (1 - previous) / (1 - alpha))**.5, dtype=torch.float32))
        self.feature_embedding = nn.Embedding(width, cfg.csdi_feature_dim)
        pos = torch.arange(cfg.max_length, dtype=torch.float32)[:, None]
        scale = 10000 ** (-torch.arange(0, cfg.csdi_time_dim, 2).float() / cfg.csdi_time_dim)
        time_emb = torch.zeros(cfg.max_length, cfg.csdi_time_dim)
        time_emb[:, 0::2], time_emb[:, 1::2] = (pos * scale).sin(), (pos * scale).cos()
        self.register_buffer("time_embedding", time_emb, persistent=False)
        step = torch.arange(cfg.csdi_steps).float()[:, None]
        frequency = 10 ** (torch.arange(64).float()[None] / 63 * 4)
        self.register_buffer("diffusion_table", torch.cat(((step * frequency).sin(),
                                                         (step * frequency).cos()), -1), persistent=False)
        self.diffusion_projection = nn.Sequential(nn.Linear(128, 128), nn.SiLU(),
                                                 nn.Linear(128, 128), nn.SiLU())
        self.input_projection = nn.Linear(2, cfg.d_model)
        self.blocks = nn.ModuleList([CSDIResidualBlock(cfg) for _ in range(cfg.layers)])
        self.output_projection1 = nn.Linear(cfg.d_model, cfg.d_model)
        self.output_projection2 = nn.Linear(cfg.d_model, 1)
        for layer in (self.input_projection, self.output_projection1):
            nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(self.output_projection2.weight)
        # Common random numbers for deterministic checkpoint comparison and
        # batch/order-invariant inference. Three (configurable) independent
        # trajectories are shared across examples, never selected using labels.
        generator = torch.Generator(device="cpu").manual_seed(cfg.csdi_sampling_seed)
        bank = torch.randn(cfg.csdi_samples, cfg.csdi_steps + 1,
                           cfg.max_length, width, generator=generator)
        self.register_buffer("sampling_noise", bank, persistent=False)
        self._compiled_predictor = None

    def _autocast(self, device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=self.cfg.csdi_amp and device.type == "cuda")

    def side_contexts(self, valid, missing):
        observed = (valid & ~missing).float()
        embedding = self.feature_embedding.weight
        return [block.side_context(self.time_embedding[:valid.shape[1]], embedding, observed)
                for block in self.blocks]

    def predict_noise(self, observed_values, noisy_state, valid, missing, steps, sides=None):
        if self.cfg.csdi_compile and observed_values.device.type == "cuda":
            if self._compiled_predictor is None:
                self._compiled_predictor = torch.compile(
                    self._predict_noise_impl, dynamic=True, fullgraph=True, mode="default")
            return self._compiled_predictor(observed_values, noisy_state, valid, missing, steps, sides)
        return self._predict_noise_impl(observed_values, noisy_state, valid, missing, steps, sides)

    def _predict_noise_impl(self, observed_values, noisy_state, valid, missing, steps, sides=None):
        if sides is None:
            sides = self.side_contexts(valid, missing)
        conditions = observed_values.masked_fill((~valid | missing)[..., None], 0)
        noisy = noisy_state.masked_fill(~missing[..., None], 0)
        with self._autocast(observed_values.device):
            x = torch.relu(self.input_projection(torch.stack((conditions, noisy), -1)))
            x = x.masked_fill(~valid[:, :, None, None], 0)
            diffusion = self.diffusion_projection(self.diffusion_table[steps])
            skips = []
            for block, side in zip(self.blocks, sides):
                x, skip = block(x, diffusion, side, valid)
                skips.append(skip)
            x = torch.stack(skips).sum(0) / len(skips)**.5
            noise = self.output_projection2(torch.relu(self.output_projection1(x))).squeeze(-1)
        return noise.float().masked_fill(~valid[..., None], 0)

    def training_objective(self, features, target, valid, missing):
        observed = torch.cat([features[m] for m in MODS], -1)
        clean = torch.cat([target[m] for m in MODS], -1).masked_fill(~missing[..., None], 0)
        steps = torch.randint(self.cfg.csdi_steps, (len(valid),), device=observed.device)
        noise = torch.randn_like(clean)
        alpha = self.alpha_bar[steps, None, None]
        state = (alpha.sqrt() * clean + (1 - alpha).sqrt() * noise).masked_fill(~missing[..., None], 0)
        predicted = self.predict_noise(observed, state, valid, missing, steps)
        terms = {}
        for m, error in zip(MODS, (predicted - noise).split(list(DIMS.values()), -1)):
            per_slot = error.square().mean(-1)
            terms["noise_" + m] = (per_slot * missing).sum() / missing.sum().clamp_min(1)
        return sum(terms.values()) / len(MODS), terms

    def reverse_step(self, state, predicted_noise, step, innovation):
        mean = (state - self.beta[step] / (1 - self.alpha_bar[step]).sqrt() * predicted_noise) / (1 - self.beta[step]).sqrt()
        return mean + self.posterior_std[step] * innovation if step > 0 else mean

    @torch.no_grad()
    def normalized_samples(self, features, valid, missing):
        observed = torch.cat([features[m] for m in MODS], -1)
        if (valid.ndim != 2 or missing.shape != valid.shape or (missing & ~valid).any()
                or not valid.any(1).all() or valid.shape[1] > self.cfg.max_length):
            raise ValueError("Invalid CSDI validity or corruption mask")
        batch, length, width = observed.shape
        count = self.cfg.csdi_samples
        result = observed[:, None].expand(-1, count, -1, -1).clone()
        active = torch.nonzero(missing.any(1), as_tuple=False).flatten()
        for begin in range(0, len(active), self.cfg.csdi_microbatch):
            rows = active[begin:begin + self.cfg.csdi_microbatch]
            n = len(rows)
            values = observed[rows, None].expand(-1, count, -1, -1).reshape(n * count, length, width)
            v = valid[rows, None].expand(-1, count, -1).reshape(n * count, length)
            k = missing[rows, None].expand(-1, count, -1).reshape(n * count, length)
            def bank_at(index):
                return self.sampling_noise[:, index, :length][None].expand(n, -1, -1, -1).reshape(n * count, length, width)
            state = bank_at(0).masked_fill(~k[..., None], 0)
            sides = self.side_contexts(v, k)
            for step in range(self.cfg.csdi_steps - 1, -1, -1):
                steps = torch.full((n * count,), step, device=observed.device, dtype=torch.long)
                predicted = self.predict_noise(values, state, v, k, steps, sides)
                state = self.reverse_step(state, predicted, step, bank_at(step + 1))
                # Conditioning is always supplied separately without noise.
                state = state.masked_fill(~k[..., None], 0)
            result[rows] = torch.where(k[..., None], state, values).reshape(n, count, length, width)
        return result.masked_fill(~valid[:, None, :, None], 0)

    def predict_residuals(self, features, valid_mask, corruption_mask):
        # Mean estimates the conditional expectation targeted by MSE; official
        # MAE evaluation instead uses a median over 100 trajectories.
        estimate = self.normalized_samples(features, valid_mask, corruption_mask).mean(1)
        return {m: (value - features[m]).masked_fill(~corruption_mask[..., None], 0)
                for m, value in zip(MODS, estimate.split(list(DIMS.values()), -1))}


# Preserve imports and state_dict parameter names from existing experiments.
Reconstructor = RECAPReconstructor
RECONSTRUCTOR_METHODS = {"recap": RECAPReconstructor, "mtsit": MTSITReconstructor,
                         "brits": BRITSReconstructor, "centaur": CentaurReconstructor,
                         "naomi": NAOMIReconstructor, "csdi": CSDIReconstructor}


def build_reconstructor(cfg):
    """Build a registered method without changing the surrounding experiment."""
    try:
        model_class = RECONSTRUCTOR_METHODS[cfg.method]
    except KeyError as exc:
        raise ValueError(f"Unknown reconstruction method {cfg.method!r}; "
                         f"choose from {sorted(RECONSTRUCTOR_METHODS)}") from exc
    if not isinstance(model_class, type) or not issubclass(model_class, BaseReconstructor):
        raise TypeError("Registered reconstruction methods must inherit BaseReconstructor")
    return model_class(cfg)


def reconstruction_loss(pred, target, valid, missing):
    terms = {}
    for m in MODS:
        region = valid if m == "text" else missing
        per_slot = (pred[m] - target[m]).square().mean(-1)
        terms[m] = (per_slot * region).sum() / region.sum().clamp_min(1)
    return sum(terms.values()) / len(MODS), terms


def resolve_mask_distribution(cfg):
    distribution = cfg.mask_distribution
    if distribution == "auto":
        return "geometric" if cfg.method == "mtsit" else "bernoulli"
    if distribution not in ("bernoulli", "geometric"):
        raise ValueError("mask_distribution must be auto, bernoulli or geometric")
    return distribution


def geometric_corruption_mask(eligible, rate, mean_length, rng):
    """Stationary two-state chain; mean erased run length is mean_length.

    Missing -> observed has probability 1/mean_length. Observed -> missing
    has probability rate / ((1-rate)*mean_length). Thus the stationary erased
    proportion is rate, with geometrically distributed erased and kept runs.
    Special tokens, padding and gaps terminate a run; no fixed count is forced.
    """
    if not np.isfinite(mean_length) or mean_length < 1 or not 0 < rate < 1:
        raise ValueError("Need mean_mask_length >= 1 and 0 < rate < 1")
    leave_missing = 1.0 / mean_length
    enter_missing = leave_missing * rate / (1.0 - rate)
    if enter_missing > 1:
        raise ValueError("mean_mask_length is too short for the requested masking rate")
    result = np.zeros_like(eligible, dtype=bool)
    state = None
    for slot, allowed in enumerate(eligible):
        if not allowed:
            state = None
            continue
        if state is None:
            state = bool(rng.random() < rate)
        result[slot] = state
        if rng.random() < (leave_missing if state else enter_missing):
            state = not state
    return result


def reconstruction_protocol(cfg):
    distribution = resolve_mask_distribution(cfg)
    protocol = {
        "method": cfg.method, "mask_distribution": distribution,
        "nominal_training_rates": list(PROBS),
        "mean_mask_length": cfg.mean_mask_length if distribution == "geometric" else None,
        "mask_scope": "Shared content-slot mask across all three modalities; excludes padding/CLS/SEP",
        "text_encoding": "Replace masked tokens with UNK, then re-encode the whole sentence",
        "loss": "Equal mean of three standardized modality MSEs; text all valid, audio/vision erased only",
        "evaluation": "Unchanged fixed p000/p010/p020/p030 sets; only p010/p020/p030 valid select weights",
    }
    if cfg.method == "mtsit":
        protocol.update({
            "paper": "https://doi.org/10.1109/LSP.2022.3224880",
            "author_code": "https://github.com/koc-lab/MTSIT",
            "architecture": "Same-slot 877D input; learned position; temporal post-BN encoder; linear decoder",
            "paper_hyperparameters": {"d_model": 16, "heads": 4, "layers": 3, "ffn_dim": 16,
                                     "masking_rate": .15, "mean_mask_length": 3},
            "actual_hyperparameters": {k: getattr(cfg, k) for k in
                                      ("d_model", "heads", "layers", "ffn_dim", "dropout")},
            "adaptations": [
                "877 aligned multimodal features instead of the original 35/36-variable datasets.",
                "Train-only feature standardization and BatchNorm statistics excluding padded slots.",
                "Existing text-all-valid / AV-missing, equal-modality reconstruction objective.",
                "Existing .1/.2/.3 corruption rates, fixed validation sets and AdamW training schedule.",
                "Corrupted text is re-encoded before reconstruction; explicit erased slots are then zeroed in normalized encoder input.",
                "Paper describes a concurrent time mask; author code also supports separate-variable masking. This framework uses concurrent slots.",
            ],
        })
    if cfg.method == "brits":
        protocol.update({
            "paper": "https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html",
            "author_code": "https://github.com/caow13/BRITS",
            "architecture": "Independent forward/backward RITS; time decay, zero-diagonal feature regression, sigmoid blend, LSTMCell",
            "actual_hyperparameters": {"input_dim": sum(DIMS.values()), "hidden_dim": cfg.d_model,
                                      "directions": 2, "layers_per_direction": 1,
                                      "estimation_weight": cfg.brits_estimation_weight,
                                      "consistency_weight": cfg.brits_consistency_weight},
            "training_loss": "Shared clean-target balanced MSE + estimation_weight * observed-estimation MAE + consistency_weight * bidirectional MAE",
            "adaptations": [
                "Full 877D aligned multimodal vectors; no low-dimensional input projection or low-rank feature regression.",
                "Paper sigmoid gate and previous-observation time-gap recurrence; first valid gap is zero.",
                "Unit spacing means valid aligned slots, not physical seconds. Reverse gaps are recomputed.",
                "Observed entries, including text, pass through. Remaining contextual BERT contamination is measured but not denoised.",
                "Primary clean-target MSE and checkpoint selection use the unchanged shared benchmark.",
                "Observed-estimation MAE uses corrupted observed inputs, not clean hidden values; three predictors and both directions averaged.",
                "Auxiliary errors average equally across modalities and exclude padding. Consistency includes all valid slots (zero on observed slots).",
                "No classification/regression head or labels; existing Bernoulli rates, AdamW and train-only normalization.",
                "Config heads/layers/ffn_dim/dropout belong to other methods; BRITS uses one LSTMCell per direction and no dropout.",
            ],
        })
    if cfg.method == "centaur":
        protocol.update({
            "paper": "https://doi.org/10.1109/JSEN.2024.3388893",
            "author_code": "https://github.com/sustainable-computing/Centaur",
            "architecture": "Joint sensor-by-time Conv2d DAE; four strided encoder layers, dense latent, four transposed convolutions",
            "actual_hyperparameters": {
                "input_height": sum(DIMS.values()), "input_width": cfg.max_length,
                "encoder_channels": [cfg.d_model * 2**i for i in range(4)],
                "latent_dim": cfg.centaur_latent_dim,
                "encoder_kernel": 5, "encoder_stride": 2, "encoder_padding": 2,
                "decoder_kernels": [3, 2, 3, 2], "decoder_stride": 2,
                "output_activation": "linear"},
            "adaptations": [
                "877 embedding/features dimensions by 50 aligned slots instead of original IMU sensor channels.",
                "Four Conv2d layers with ReLU, a linear latent, and four ConvTranspose2d layers; no skip/residual, BN, pooling or dropout.",
                "Original PAMAP2 channel widths (base 64) and latent size 200 are configurable; decoder padding/output_padding are derived to invert the input shape exactly.",
                "Train-only standardization and linear output replace original [0,1] scaling and sigmoid output.",
                "Artificially erased slots are zero-filled in normalized coordinates; remaining text is from the corrupted-sentence BERT encoding.",
                "Only configured external Bernoulli corruption is used in this experiment; no internal random corruption or Gaussian noise.",
                "The existing text-all-valid / AV-missing equal-modality MSE, optimizer and validation selection are retained.",
                "The DAE estimates all entries; shared outer API preserves observed AV, intact samples and zero padding.",
                "The original separate HAR network is not used; downstream evaluation uses the frozen best ThisWork.",
                "Embedding-axis convolution is an adaptation; adjacent BERT dimensions are not physical neighboring sensors.",
                "Config heads/layers/ffn_dim/dropout/kernels belong to other methods and do not alter this four-stage DAE.",
            ],
        })
    if cfg.method == "naomi":
        protocol.update({
            "paper": "https://arxiv.org/abs/1901.10946",
            "author_code": "https://github.com/felixykliu/NAOMI",
            "architecture": "Forward/backward GRUs with independent power-of-two MLP decoders; coarse-to-fine differentiable imputation",
            "actual_hyperparameters": {"input_dim": sum(DIMS.values()), "rnn_dim": cfg.d_model,
                "rnn_layers": cfg.layers, "decoder_dim": cfg.naomi_decoder_dim,
                "highest_step": cfg.naomi_highest, "stochastic": False, "teacher_forcing": False},
            "adaptations": [
                "Deterministic MSE version; no discriminator, GAN, stochastic sampling or sentiment labels.",
                "Free-running from epoch one: only generated values are fed back; clean targets are loss-only.",
                "Author sampler's context at left+2*step and selective backward-state refresh are retained.",
                "Per-sample schedules replace the author implementation's shared mask for the whole batch.",
                "Zero hidden states at virtual endpoints support leading/trailing/all-missing cases; padding is skipped.",
                "The benchmark preserves CLS/SEP; virtual endpoint extrapolation is a generic API fallback.",
                "Observed values, including text, remain unchanged; remaining BERT contextual contamination is not denoised.",
                "877D aligned features, train-only normalization, existing Bernoulli rates, equal-modality MSE and AdamW schedule.",
                "No extra block masks are introduced to exercise coarse decoders; actual resolution counts are recorded.",
                "GRUs have no dropout as in author code. heads/ffn_dim/dropout/kernels apply to other methods.",
                "A supervised deterministic adaptation, not reproduction of the paper's adversarial basketball experiment.",
            ],
        })
    if cfg.method == "csdi":
        protocol.update({
            "paper": "https://arxiv.org/abs/2107.03502",
            "author_code": "https://github.com/ermongroup/CSDI",
            "linear_attention_source": "https://github.com/lucidrains/linear-attention-transformer",
            "architecture": "Conditional two-channel noise network, diffusion-step embedding, temporal and feature global linear attention, gated residual/skip blocks",
            "loss": "Training: equal-modality epsilon-prediction MSE on artificial missing slots only",
            "validation_loss": "Unchanged clean-target reconstruction MSE after genuine reverse-DDPM mean imputation",
            "validation_interval_epochs": cfg.csdi_validation_interval,
            "actual_hyperparameters": {"input_dim":sum(DIMS.values()),"channels":cfg.d_model,
                "residual_layers":cfg.layers,"heads":cfg.heads,"time_embedding_dim":cfg.csdi_time_dim,
                "feature_embedding_dim":cfg.csdi_feature_dim,"diffusion_embedding_dim":128,
                "diffusion_steps":cfg.csdi_steps,"beta_start":cfg.csdi_beta_start,
                "beta_end":cfg.csdi_beta_end,"beta_schedule":"quad","sample_count":cfg.csdi_samples,
                "point_estimator":"mean","sampling_seed":cfg.csdi_sampling_seed,
                "microbatch_samples":cfg.csdi_microbatch,"network_bfloat16":cfg.csdi_amp,
                "torch_compile_cuda":cfg.csdi_compile},
            "adaptations": [
                "Uses the author's global linear-attention option for high-dimensional forecasting, adapted to imputation; it is not the dense-attention Physio experiment.",
                "All 877 scalar features are retained during both training and inference; no PCA, latent dimensionality reduction, feature sampling or independent feature chunks.",
                "Small channel/layer counts, slot indices as time positions, train-only normalization and the existing AdamW budget.",
                "Same .1/.2/.3 Bernoulli synchronized slot masks; no extra random/block corruption.",
                "During TRAINING, clean erased values construct noisy diffusion states and epsilon targets; they are never supplied as unnoised conditions. Inference never reads targets.",
                "Conditions are corrupted-sentence BERT at remaining slots plus observed AV. Only padding is excluded from attention; missing slots remain valid tokens.",
                "Training epsilon loss is equally weighted across modalities and is not directly comparable to other methods' reconstruction training losses.",
                "Every configured validation interval (and the final epoch), all three complete validation conditions are sampled; patience counts validation checks only.",
                "Validation and test use identical reverse schedules, trajectory counts, aggregation and fixed Gaussian bank. Test is used only after checkpoint selection.",
                "A fixed bank shares Gaussian paths across examples for batch/order-independent reproducibility; these are common random numbers, not independent draws across examples.",
                "Point estimate is the mean of configured trajectories for MSE, rather than the official median of 100. No probabilistic calibration or CRPS claim is made.",
                "Observed text as well as AV is preserved. Remaining BERT contextual contamination is measured but not denoised.",
                "Only neural-network operations use optional bfloat16; attention softmax/accumulation, losses and DDPM state updates use float32.",
                "Optional CUDA torch.compile fuses the same noise network; CPU loading uses eager execution. Floating-point rounding can differ across execution backends.",
                "No extra dependency or Python file is needed; the global linear attention branch is implemented in this file.",
            ],
        })
    return protocol



def dynamic_masks(clean_tb, indices, epoch, seed, distribution="bernoulli", mean_mask_length=3.0):
    if distribution not in ("bernoulli", "geometric"):
        raise ValueError("Unknown resolved masking distribution")
    eligible = eligible_slots(clean_tb)
    missing = np.zeros_like(eligible)
    rates = np.zeros(len(indices), np.float32)
    for i, index in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, int(index), 9167]))
        rates[i] = PROBS[int(rng.integers(len(PROBS)))]
        if distribution == "bernoulli":
            # Keep the original RNG calls unchanged for RECAP compatibility.
            missing[i] = (rng.random(eligible.shape[1]) < float(rates[i])) & eligible[i]
        else:
            missing[i] = geometric_corruption_mask(eligible[i], float(rates[i]), mean_mask_length, rng)
    return missing, rates


def corrupted_training_batch(clean, indices, epoch, cfg, adapter):
    tb = clean["text_bert"][indices].copy()
    missing, rates = dynamic_masks(tb, indices, epoch, cfg.seed,
                                  distribution=resolve_mask_distribution(cfg),
                                  mean_mask_length=cfg.mean_mask_length)
    tb[:, 0][missing] = 100
    # Do not use clean['text'] as generator input, including at unmasked slots.
    features = {"text": adapter.encode_text_bert(tb, batch_size=cfg.batch_size,
                                                zero_padding=True, apply_adapter=True)}
    for m in ("audio", "vision"):
        features[m] = clean[m][indices].copy()
        features[m][missing] = 0
    return features, tb[:, 1].astype(bool), missing, rates


class ReconstructionBundle:
    def __init__(self, model, normalizer, device):
        self.model, self.normalizer, self.device = model, normalizer, device

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        """Load old RECAP checkpoints or a registered common-interface method."""
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") not in ("recap_inspired_reconstructor_v1", "sequence_reconstructor_v1"):
            raise ValueError("Unsupported checkpoint")
        cfg = Config(**saved["config"])  # Old checkpoints default to method="recap".
        if saved["format"] == "recap_inspired_reconstructor_v1" and cfg.method != "recap":
            raise ValueError("Legacy RECAP checkpoint cannot select a different method")
        model = build_reconstructor(cfg).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        return cls(model, Normalizer.from_checkpoint(saved["normalizer"]), device)

    @torch.no_grad()
    def reconstruct(self, features, valid_mask, corruption_mask, batch_size=32):
        """Raw-unit output, observed AV and fully intact samples preserved exactly."""
        valid = np.asarray(valid_mask, dtype=bool)
        missing = np.asarray(corruption_mask, dtype=bool)
        if valid.ndim != 2 or missing.shape != valid.shape or (missing & ~valid).any() or not valid.any(1).all():
            raise ValueError("Invalid valid/corruption masks")
        for m in MODS:
            x = np.asarray(features[m])
            if x.shape != (*valid.shape, DIMS[m]) or not np.isfinite(x).all() or x.dtype.kind != "f":
                raise ValueError(f"Invalid {m} features")
            if np.any(x[~valid] != 0):
                raise ValueError("Expected zero padding")
        self.model.eval()
        result = {m: np.asarray(features[m]).copy() for m in MODS}
        for start in range(0, len(valid), batch_size):
            end = start + batch_size
            v, k = valid[start:end], missing[start:end]
            if not k.any():
                continue
            raw = {m: features[m][start:end] for m in MODS}
            x = self.normalizer.normalize(raw, v, self.device)
            pred = self.model(x, torch.as_tensor(v, device=self.device), torch.as_tensor(k, device=self.device))
            for m in MODS:
                change = v & k.any(1, keepdims=True) if m == "text" else k
                stat = self.normalizer.stats[m]
                # Apply only the learned delta in original units, avoiding
                # normalize/denormalize roundoff when that delta is zero.
                delta = (pred[m] - x[m]).cpu().numpy() * stat["std"]
                restored = raw[m] + delta
                result[m][start:end][change] = restored[change].astype(result[m].dtype)
        return result


def load_reconstructor(checkpoint, device="cpu"):
    """Backward-compatible entry point; class and function APIs are equivalent."""
    return ReconstructionBundle.from_checkpoint(checkpoint, device)


def sample_error(pred, target, std, region):
    # float64 accumulation/reporting; training itself is float32.
    difference = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    raw = np.mean(difference ** 2, -1)
    norm = np.mean((difference / std) ** 2, -1)
    count = region.sum(1)
    return {"raw_sum": (raw * region).sum(1), "normalized_sum": (norm * region).sum(1), "count": count}


def evaluate(bundle, clean, corrupted, batch_size, csv_path=None):
    """Targets are read here solely to measure error, after model execution."""
    bundle.model.eval()
    valid = corrupted["text_bert"][:, 1].astype(bool)
    missing = corrupted["corruption_mask"]
    totals = {}
    rows = []
    unchanged_av = True
    intact_samples = True
    for start in range(0, len(valid), batch_size):
        end = start + batch_size
        v, k = valid[start:end], missing[start:end]
        raw = {m: corrupted[m][start:end] for m in MODS}
        reconstructed = bundle.reconstruct(raw, v, k, batch_size)
        for m in ("audio", "vision"):
            unchanged_av &= np.array_equal(raw[m][~k], reconstructed[m][~k])
        for m in MODS:
            intact_samples &= np.array_equal(raw[m][~k.any(1)], reconstructed[m][~k.any(1)])
        region_map = {"text": v, "audio": k, "vision": k, "text_missing": k, "text_remaining": v & ~k}
        batch_rows = [{"id": str(x), "missing_slots": int(k[i].sum()), "valid_slots": int(v[i].sum())}
                      for i, x in enumerate(clean["id"][start:end])]
        for method, features in (("baseline", raw), ("reconstructed", reconstructed)):
            for region_name, region in region_map.items():
                m = region_name.split("_")[0]
                part = sample_error(features[m], clean[m][start:end], bundle.normalizer.stats[m]["std"], region)
                dest = totals.setdefault(method, {}).setdefault(region_name, {"raw_sum": 0., "normalized_sum": 0., "count": 0})
                for key in dest:
                    dest[key] += float(part[key].sum())
                for i, row in enumerate(batch_rows):
                    count = int(part["count"][i])
                    row[f"{method}_{region_name}_normalized_mse"] = float(part["normalized_sum"][i] / count) if count else None
                    row[f"{method}_{region_name}_raw_mse"] = float(part["raw_sum"][i] / count) if count else None
        rows.extend(batch_rows)
        if bundle.model.cfg.method == "csdi" and ((start // batch_size + 1) % 4 == 0 or end == len(valid)):
            print(json.dumps({"event": "csdi_evaluation_progress", "completed_samples": end,
                              "total_samples": len(valid), "condition_missing_slots": int(missing.sum())}), flush=True)
    result = {"samples": len(valid), "artificial_missing_slots": int(missing.sum()),
              "unchanged_observed_av": bool(unchanged_av), "unchanged_intact_samples": bool(intact_samples)}
    if not unchanged_av or not intact_samples:
        raise AssertionError("Passthrough contract violated")
    for method, regions in totals.items():
        result[method] = {}
        for region, stat in regions.items():
            count = int(stat["count"])
            result[method][region] = {"slots": count, "raw_mse": stat["raw_sum"] / count if count else None,
                                     "normalized_mse": stat["normalized_sum"] / count if count else None}
        if missing.any():
            result[method]["balanced_normalized_mse"] = float(np.mean([result[method][m]["normalized_mse"] for m in MODS]))
        else:
            result[method]["balanced_normalized_mse"] = 0.0
    if csv_path is not None:
        with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def fixed_split(path, name, clean, reference_metadata, protected):
    data = load_pickle(path)
    meta = data["_metadata"]
    for key in ("source_sha256", "adapter_module_sha256", "checkpoint_sha256", "apply_adapter", "zero_padding", "bert_model_sha256"):
        if meta[key] != reference_metadata[key]:
            raise ValueError(f"{path}: incompatible encoding provenance ({key})")
    result = select_fields(data[name])
    validate_pair(clean, result, f"{Path(path).stem}/{name}")
    del data
    gc.collect()
    return result


def train(cfg):
    if (cfg.epochs < 1 or cfg.patience < 1 or cfg.heads < 1 or cfg.d_model < 1
            or cfg.layers < 1 or (cfg.method in ("recap", "mtsit", "csdi") and cfg.d_model % cfg.heads)
            or (cfg.method == "csdi" and cfg.csdi_validation_interval < 1)):
        raise ValueError("Invalid training/model configuration")
    protocol = reconstruction_protocol(cfg)
    if protocol["mask_distribution"] == "geometric":
        for rate in PROBS:
            geometric_corruption_mask(np.ones(1, bool), rate, cfg.mean_mask_length,
                                      np.random.default_rng(0))
    torch.set_num_threads(cfg.cpu_threads)
    seed_all(cfg.seed)
    files = {rate: Path(cfg.data_dir) / f"aligned_50_p{rate:03d}.pkl" for rate in (0, 10, 20, 30)}
    protected = [*files.values(), Path(cfg.adapter_module), Path(cfg.adapter_checkpoint)]
    hashes_before = {str(p): sha256(p) for p in protected}
    data = load_pickle(files[0])
    metadata = data["_metadata"]
    for key, path in (("adapter_module_sha256", cfg.adapter_module), ("checkpoint_sha256", cfg.adapter_checkpoint)):
        if metadata[key] != hashes_before[str(Path(path))]:
            raise ValueError(f"Current encoder differs from dataset provenance: {key}")
    if not metadata["apply_adapter"] or not metadata["zero_padding"]:
        raise ValueError("Unexpected text encoding settings")
    audit = audit_splits(data)
    if cfg.method == "csdi":
        audit["interpretation"] = audit["interpretation"].replace(
            "Clean targets are loss-only.",
            "CSDI clean erased values construct noisy training diffusion states and epsilon supervision; they are never unnoised conditions or inference inputs.")
    clean = {name: select_fields(data[name]) for name in SPLITS}
    del data
    gc.collect()
    normalizer = Normalizer.fit(clean["train"], cfg.std_floor)
    fixed_valid = {rate: fixed_split(files[rate], "valid", clean["valid"], metadata, protected) for rate in (10, 20, 30)}
    spec = importlib.util.spec_from_file_location("recap_existing_bert_adapter", cfg.adapter_module)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter = module.load_adapter(cfg.adapter_checkpoint, device=cfg.device)
    if any(p.requires_grad for p in adapter.parameters()):
        raise AssertionError("Encoder is not frozen")
    # Confirm all fixed conditions and dynamic inputs use the same encoder.
    encoding_checks = {}
    for rate, split in [(0, clean["train"]), *fixed_valid.items()]:
        indices = np.array([0, len(split["id"]) // 2, len(split["id"]) - 1])
        encoded = adapter.encode_text_bert(split["text_bert"][indices], zero_padding=True, apply_adapter=True)
        error = float(np.abs(encoded - split["text"][indices]).max())
        encoding_checks[str(rate)] = error
        if not np.allclose(encoded, split["text"][indices], rtol=2e-4, atol=1e-4):
            raise ValueError(f"BERT encoding mismatch for condition {rate}: {error}")
    audit["reconstruction_protocol"] = protocol
    audit["encoding_spot_check_max_absolute_error"] = encoding_checks
    audit["normalization"] = {m: {"fit_split": "train", "valid_slots": normalizer.stats[m]["valid_slots"],
                                    "floored_dimensions": normalizer.stats[m]["floored_dimensions"]} for m in MODS}
    run_dir = Path(cfg.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"event": "run_started", "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    write_json(run_dir / "config.json", asdict(cfg))
    write_json(run_dir / "audit.json", audit)
    write_json(run_dir / "provenance.json", {"protected_sha256_before": hashes_before, "script_sha256": sha256(__file__),
                                           "p000_metadata": metadata, "torch": str(torch.__version__), "numpy": str(np.__version__)})
    np.savez(run_dir / "normalization.npz", **{f"{m}_{k}": normalizer.stats[m][k] for m in MODS for k in ("mean", "std")})
    model = build_reconstructor(cfg).to(cfg.device)
    bundle = ReconstructionBundle(model, normalizer, cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"event": "model", "trainable_parameters": params}), flush=True)
    history, best_score, best_epoch, bad_epochs = [], float("inf"), 0, 0
    training_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        model.train()
        adapter.eval()
        order = np.random.default_rng(np.random.SeedSequence([cfg.seed, epoch, 701])).permutation(len(clean["train"]["id"]))
        loss_total, samples, missing_count, eligible_count = 0., 0, 0, 0
        loss_components = {}
        if cfg.method == "naomi":
            model.resolution_counts = {str(step): 0 for step in model.steps}
        rate_counts = {str(p): 0 for p in PROBS}
        mask_hash = hashlib.sha256()
        for start in range(0, len(order), cfg.batch_size):
            indices = order[start:start + cfg.batch_size]
            features, valid, missing, rates = corrupted_training_batch(clean["train"], indices, epoch, cfg, adapter)
            x = normalizer.normalize(features, valid, cfg.device)
            # Targets are prepared independently. CSDI uses erased clean values
            # to construct noisy training states; inference never reads targets.
            target = normalizer.normalize({m: clean["train"][m][indices] for m in MODS}, valid, cfg.device)
            v, k = torch.as_tensor(valid, device=cfg.device), torch.as_tensor(missing, device=cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_terms = model.training_objective(x, target, v, k)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
            for name, value in loss_terms.items():
                loss_components[name] = loss_components.get(name, 0.) + float(value.detach()) * len(indices)
            samples += len(indices)
            missing_count += int(missing.sum())
            eligible_count += int(eligible_slots(clean["train"]["text_bert"][indices]).sum())
            for rate in PROBS:
                rate_counts[str(rate)] += int(np.isclose(rates, rate).sum())
            mask_hash.update(indices.tobytes())
            mask_hash.update(missing.tobytes())
        resolution_counts = dict(model.resolution_counts) if cfg.method == "naomi" else None
        validate_now = (cfg.method != "csdi" or epoch % cfg.csdi_validation_interval == 0
                        or epoch == cfg.epochs)
        if validate_now:
            print(json.dumps({"event": "validation_started", "epoch": epoch}), flush=True)
            valid_metrics = {str(rate): evaluate(bundle, clean["valid"], fixed_valid[rate], cfg.batch_size) for rate in (10, 20, 30)}
            score = float(np.mean([x["reconstructed"]["balanced_normalized_mse"] for x in valid_metrics.values()]))
        else:
            valid_metrics, score = {}, None
        record = {"epoch": epoch, "train_loss": loss_total / samples, "validation_score": score,
                  "train_loss_components": {name: value / samples for name, value in loss_components.items()},
                  "valid": valid_metrics, "seconds": time.time() - started,
                  "mask_sha256": mask_hash.hexdigest(), "nominal_rate_sample_counts": rate_counts,
                  "realized_corruption_fraction": missing_count / eligible_count}
        if resolution_counts is not None:
            record["naomi_resolution_counts"] = resolution_counts
        history.append(record)
        write_json(run_dir / "history.json", history)
        improved = score is not None and score < best_score - 1e-8
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"format": ("recap_inspired_reconstructor_v1" if cfg.method == "recap"
                                  else "sequence_reconstructor_v1"), "config": asdict(cfg),
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "normalizer": normalizer.checkpoint(), "epoch": epoch,
                        "validation_score": score, "training_only_reconstruction": True,
                        "provenance_hashes": hashes_before}, run_dir / "best.pt")
        elif score is not None:
            bad_epochs += 1
        print(json.dumps({"epoch": epoch, "train_loss": record["train_loss"], "validation_score": score,
                          "best_epoch": best_epoch, "seconds": round(record["seconds"], 2)}), flush=True)
        if bad_epochs >= cfg.patience:
            break
    # No further updates occur after this point. Test never affects selection.
    del optimizer, adapter, model, bundle, fixed_valid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bundle = load_reconstructor(run_dir / "best.pt", cfg.device)
    results = {"method": bundle.model.description, "reconstruction_protocol": protocol,
               "best_epoch": best_epoch, "epochs_run": len(history), "trainable_parameters": params,
               "best_validation_score": best_score, "training_seconds": time.time() - training_start,
               "selection": "Mean of p010/p020/p030 validation MSE; modalities equally weighted after train-only standardization",
               "loss_regions": {"text": "all valid positions", "audio": "artificially erased slots", "vision": "artificially erased slots"},
               "test": {}}
    for rate in (0, 10, 20, 30):
        corrupted = clean["test"] if rate == 0 else fixed_split(files[rate], "test", clean["test"], metadata, protected)
        if cfg.method == "naomi":
            bundle.model.resolution_counts = {str(step): 0 for step in bundle.model.steps}
        result = evaluate(bundle, clean["test"], corrupted, cfg.batch_size, run_dir / f"test_p{rate:03d}_errors.csv")
        if cfg.method == "naomi":
            result["naomi_resolution_counts"] = dict(bundle.model.resolution_counts)
        baseline = result["baseline"]["balanced_normalized_mse"]
        result["relative_mse_reduction"] = 1 - result["reconstructed"]["balanced_normalized_mse"] / baseline if baseline > 0 else None
        results["test"][str(rate)] = result
        print(json.dumps({"event": "test_reconstruction", "rate": rate, "baseline_mse": baseline,
                          "reconstructed_mse": result["reconstructed"]["balanced_normalized_mse"]}), flush=True)
        if rate:
            del corrupted
            gc.collect()
    hashes_after = {str(p): sha256(p) for p in protected}
    results["protected_inputs_unchanged"] = hashes_before == hashes_after
    write_json(run_dir / "integrity.json", {"before": hashes_before, "after": hashes_after, "unchanged": hashes_before == hashes_after})
    write_json(run_dir / "metrics.json", results)
    write_report(run_dir, cfg, results)
    if hashes_before != hashes_after:
        raise RuntimeError("A protected input changed during the run; inspect integrity.json")
    print(json.dumps({"event": "completed", "run_dir": str(run_dir), "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
    return run_dir


def write_report(run_dir, cfg, results):
    title = "RECAP-inspired feature reconstruction" if cfg.method == "recap" else f"{cfg.method} feature reconstruction"
    method_note = ("This is a lightweight adaptation, not a reproduction of the full RECAP architecture."
                   if cfg.method == "recap" else results["method"])
    protocol = reconstruction_protocol(cfg)
    mask_note = (f"Training uses {protocol['mask_distribution']} masks with nominal p=0.1/0.2/0.3, "
                 "drawn independently per example each epoch; the same mask is applied across modalities.")
    if protocol["mask_distribution"] == "geometric":
        mask_note += f" Mean erased segment length is {cfg.mean_mask_length:g}; finite samples vary."
    lines = ["# " + title, "", method_note,
             "No sentiment labels, classification/regression model, GAN, FID, or distillation is used.", "",
             f"Best epoch: {results['best_epoch']}; epochs run: {results['epochs_run']}; trainable parameters: {results['trainable_parameters']}.",
             "Normalization is fitted on clean train only; the frozen existing BERT adapter is reused.",
             ("The same sample can have different epoch masks. CSDI clean erased targets construct noisy training states and epsilon supervision; no clean target enters inference or the unnoised condition channel."
              if cfg.method == "csdi" else
              "The same training example can have different masks across epochs. Its clean target never enters the reconstructor forward call."),
             "Masks do not cross original train/valid/test boundaries. No attention crosses the batch dimension.",
             "Repeated generic phrases okay/alright across train/valid are disclosed in audit.json; original video IDs and full multimodal samples do not overlap.", "",
             mask_note + " Text tokens are corrupted before whole-sequence BERT encoding.",
             "Validation uses the fixed three corrupted datasets. Their mean balanced standardized MSE selects the checkpoint; test is evaluated only afterwards.", "",
             "| Test corruption | Corrupted input MSE | Reconstructed MSE | Relative reduction |",
             "|---|---:|---:|---:|"]
    for rate, result in results["test"].items():
        old = result["baseline"]["balanced_normalized_mse"]
        new = result["reconstructed"]["balanced_normalized_mse"]
        change = result["relative_mse_reduction"]
        change_str = f"{100 * change:.2f}%" if change is not None else "N/A (exact passthrough)"
        lines.append(f"| {rate}% | {old:.8f} | {new:.8f} | {change_str} |")
    lines += ["", "MSE is first averaged within each modality and then equally across modalities. Text includes all valid slots; audio/vision include erased slots only.",
              "Compare before/after within one rate. Different rates select different missing slots, so rate-to-rate MSE need not be monotonic.",
              "metrics.json also separates text_missing/text_remaining and reports raw-unit MSE per modality. Per-sample error CSV files contain no reconstructed feature arrays.",
              "Observed audio/video values and fully uncorrupted samples pass through exactly, in the original dtype. Padding stays zero.",
              "A better feature MSE does not establish better Macro-F1 or MAE; downstream sentiment evaluation has intentionally not been run.",
              "Because all three modalities lose the same slots, the lost fine-grained information may be ambiguous. Output is an estimate, not verified original evidence.", "",
              "## Reuse", "", "```python", "from question2.train_recap_reconstructor import load_reconstructor",
              f"reconstructor = load_reconstructor({str(run_dir / 'best.pt')!r}, device='cuda:0')",
              "# features = {'text': corrupted_text, 'audio': corrupted_audio, 'vision': corrupted_vision}",
              "restored = reconstructor.reconstruct(features, valid_mask, corruption_mask)", "```", "",
              "Inputs must use the same BERT+adapter encoding and raw audio/video units as this run. valid_mask comes from text_bert[:,1,:]. corruption_mask is explicit provenance, not inferred from zero vectors.",
              (f"{cfg.method.upper()} fills erased text slots only and preserves remaining observed text; contextual BERT contamination at those positions remains."
               if cfg.method in ("naomi", "csdi") else
               "Text can change throughout a corrupted sample because BERT contextual encoding propagates token corruption beyond the masked slots."),
              "No reconstructed datasets are written. best.pt contains the model, config, normalization, selected epoch and provenance hashes.", "",
              f"Protected data, adapter source and adapter weights unchanged: {results['protected_inputs_unchanged']}.",
              "Shared RECAP benchmark reference: https://ojs.aaai.org/index.php/AAAI/article/view/39349/43310", ""]
    if cfg.method == "mtsit":
        lines += ["## MTSIT adaptation", "",
                  "Source: https://doi.org/10.1109/LSP.2022.3224880",
                  "Author code: https://github.com/koc-lab/MTSIT", "",
                  "This is an adaptation under the shared reconstruction benchmark, not an exact reproduction.",
                  "Encoder: same-slot feature concatenation, scaled linear projection, learnable positions, "
                  "post-BatchNorm temporal attention blocks, GELU/dropout and a linear decoder.",
                  "The decoder estimates absolute normalized features; estimate-minus-input only bridges the common residual API.",
                  f"Actual model dimensions: {protocol['actual_hyperparameters']}.",
                  "Paper settings: d_model=16, heads=4, layers=3, FFN=16, geometric mean run=3, rate=.15.", ""]
        lines += ["- " + note for note in protocol["adaptations"]]
        lines += ["", "For an architecture comparison with existing RECAP masking, explicitly use --mask-distribution bernoulli.",
                  "Validation/test corruptions are never regenerated by the new masking option.", ""]
    if cfg.method == "brits":
        lines += ["## BRITS adaptation", "", "Source: " + protocol["paper"],
                  "Author code: " + protocol["author_code"], "",
                  "This is a shared-benchmark adaptation, not an exact reproduction of the original training loss.",
                  "Training objective: " + protocol["training_loss"],
                  f"Actual model settings: {protocol['actual_hyperparameters']}.", ""]
        lines += ["- " + note for note in protocol["adaptations"]]
        lines += ["", "The primary MSE column above retains the original RECAP metric regions. "
                  "It is not the full-sequence MSE; text covers all valid positions, audio/vision only erased positions.", ""]
    if cfg.method == "centaur":
        lines += ["## Centaur DAE adaptation", "", "Source: " + protocol["paper"],
                  "Author code: " + protocol["author_code"], "",
                  "Only the convolutional data-cleaning module is adapted, not the original HAR network.",
                  f"Actual model settings: {protocol['actual_hyperparameters']}.", ""]
        lines += ["- " + note for note in protocol["adaptations"]]
        lines += ["", "The primary MSE above retains the historical mixed-region metric. "
                  "Full-sequence and missing-only errors must be reported separately.", ""]
    if cfg.method == "naomi":
        lines += ["## NAOMI adaptation", "", "Source: " + protocol["paper"],
                  "Author code: " + protocol["author_code"], "",
                  f"Actual model settings: {protocol['actual_hyperparameters']}.", ""]
        lines += ["- " + note for note in protocol["adaptations"]]
        lines += ["", "This method preserves observed text as well as audio/video. "
                  "It does not implement the optional full-text denoising allowed by the shared API.",
                  "Per-epoch history and test metrics record the number of imputed slots at each resolution.", ""]
    if cfg.method == "csdi":
        lines += ["## CSDI adaptation", "", "Source: " + protocol["paper"],
                  "Author code: " + protocol["author_code"], "",
                  f"Actual model settings: {protocol['actual_hyperparameters']}.",
                  f"All three complete validation conditions are sampled every {cfg.csdi_validation_interval} epochs and at the final epoch. Early stopping counts validation checks, not skipped epochs.",
                  "Training loss is epsilon MSE; checkpoint selection still uses clean-target reconstruction MSE after full reverse sampling.", ""]
        lines += ["- " + note for note in protocol["adaptations"]]
        lines += ["", "Primary test MSE retains the original text-all-valid / AV-missing metric. Full-sequence and erased-only metrics must be reported separately.", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def self_test():
    torch.set_num_threads(2)
    seed_all(42)
    cfg = Config(device="cpu", d_model=16, heads=2, layers=1, max_length=8, dropout=0.)
    rng = np.random.default_rng(42)
    valid = np.ones((4, 8), bool)
    valid[:, -2:] = False
    missing = np.zeros_like(valid)
    missing[1:, 2:4] = True
    features = {m: rng.normal(size=(4, 8, DIMS[m])).astype(np.float64 if m != "text" else np.float32) for m in MODS}
    for x in features.values():
        x[~valid] = 0
    stats = {m: {"mean": np.zeros(DIMS[m], np.float32), "std": np.ones(DIMS[m], np.float32)} for m in MODS}
    norm = Normalizer(stats)
    model = build_reconstructor(cfg)
    assert isinstance(model, RECAPReconstructor) and Reconstructor is RECAPReconstructor
    bundle = ReconstructionBundle(model, norm, "cpu")
    x = norm.normalize(features, valid, "cpu")
    v, k = torch.as_tensor(valid), torch.as_tensor(missing)
    model.eval()
    first = bundle.reconstruct(features, valid, missing)
    for m in MODS:
        assert np.array_equal(first[m], features[m]), "Zero initialized model must be input identity"
    # Exercise nonzero reconstruction to make passthrough checks meaningful.
    for layer in model.decode.values():
        nn.init.normal_(layer.weight, std=.02)
    restored = bundle.reconstruct(features, valid, missing)
    for m in MODS:
        assert np.array_equal(restored[m][0], features[m][0])
        assert np.array_equal(restored[m][~valid], features[m][~valid])
    for m in ("audio", "vision"):
        assert np.array_equal(restored[m][~missing], features[m][~missing])
    # Changing a different batch item must not change item zero.
    with torch.no_grad():
        p = model(x, v, k)
        other = {m: z.clone() for m, z in x.items()}
        for z in other.values():
            z[3] = 100 * torch.randn_like(z[3])
        q = model(other, v, k)
        for m in MODS:
            torch.testing.assert_close(p[m][1], q[m][1], atol=1e-6, rtol=1e-6)
    # Targets outside AV erased regions and at padding must not affect loss.
    target = {m: z.clone() for m, z in x.items()}
    modified = {m: z.clone() for m, z in target.items()}
    for m in MODS:
        modified[m][~v] = 1e4
    for m in ("audio", "vision"):
        modified[m][~k] = 1e4
    a, _ = reconstruction_loss(p, target, v, k)
    b, _ = reconstruction_loss(p, modified, v, k)
    torch.testing.assert_close(a, b)
    # Verify token-first masking without depending on BERT or real datasets.
    tb = np.zeros((4, 3, 8), np.int64)
    tb[:, 0] = [101, 200, 201, 202, 203, 102, 0, 0]
    tb[:, 1] = valid
    clean = {**features, "text_bert": tb}
    class StubEncoder:
        def encode_text_bert(self, tokens, **kwargs):
            self.last = tokens.copy()
            return np.repeat(tokens[:, 0, :, None], 768, axis=2).astype(np.float32) * tokens[:, 1, :, None]
    stub = StubEncoder()
    f, vv, kk, _ = corrupted_training_batch(clean, np.arange(4), 1, cfg, stub)
    assert np.all(stub.last[:, 0][kk] == 100)
    assert not np.any(kk & ~eligible_slots(tb))
    poisoned = {**clean, "text": np.full_like(clean["text"], np.nan)}
    f2, _, _, _ = corrupted_training_batch(poisoned, np.arange(4), 1, cfg, stub)
    for m in MODS:
        np.testing.assert_array_equal(f[m], f2[m])
    mask1, _ = dynamic_masks(np.tile(tb, (20, 1, 1)), np.arange(80), 1, 42)
    mask2, _ = dynamic_masks(np.tile(tb, (20, 1, 1)), np.arange(80), 2, 42)
    assert not np.array_equal(mask1, mask2)
    # A learnable synthetic residual checks optimizer/gradient and masking paths.
    target = {m: z + ((v & k.any(1, keepdim=True)) if m == "text" else k)[..., None] * .3 for m, z in x.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = float(reconstruction_loss(model(x, v, k), target, v, k)[0].detach())
    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        loss, _ = reconstruction_loss(model(x, v, k), target, v, k)
        loss.backward()
        optimizer.step()
    after = float(reconstruction_loss(model(x, v, k), target, v, k)[0].detach())
    assert after < before * .25, (before, after)

    # New methods share the same full-sequence I/O and preservation contract.
    # Registration is local to this self-test; it does not add a benchmark model.
    import io
    class ConstantResidualReconstructor(BaseReconstructor):
        def __init__(self, config):
            super().__init__(config)
            self.scale = nn.Parameter(torch.tensor(.25))

        def predict_residuals(self, inputs, valid_mask, corruption_mask):
            return {m: torch.ones_like(inputs[m]) * self.scale for m in MODS}

    method = "_self_test_constant"
    RECONSTRUCTOR_METHODS[method] = ConstantResidualReconstructor
    try:
        test_cfg = Config(device="cpu", d_model=16, heads=2, layers=1,
                          max_length=8, dropout=0., method=method)
        alternative = build_reconstructor(test_cfg)
        alternative_bundle = ReconstructionBundle(alternative, norm, "cpu")
        result = alternative_bundle.reconstruct(features, valid, missing)
        for m in MODS:
            allowed = valid & missing.any(1, keepdims=True) if m == "text" else missing
            assert result[m].shape == features[m].shape and result[m].dtype == features[m].dtype
            np.testing.assert_array_equal(result[m][~allowed], features[m][~allowed])
            np.testing.assert_allclose(result[m][allowed], features[m][allowed] + .25,
                                       rtol=1e-6, atol=1e-6)
        buffer = io.BytesIO()
        torch.save({"format": "sequence_reconstructor_v1", "config": asdict(test_cfg),
                    "model": alternative.state_dict(), "normalizer": norm.checkpoint()}, buffer)
        buffer.seek(0)
        reloaded = ReconstructionBundle.from_checkpoint(buffer)
        result2 = reloaded.reconstruct(features, valid, missing)
        for m in MODS:
            np.testing.assert_array_equal(result[m], result2[m])
    finally:
        del RECONSTRUCTOR_METHODS[method]
    try:
        build_reconstructor(Config(method="_not_registered"))
    except ValueError as exc:
        assert "Unknown reconstruction method" in str(exc)
    else:
        raise AssertionError("Unregistered methods must fail clearly")
    # Legacy configs omitted 'method'; no historical path needs to exist at load.
    legacy_cfg = asdict(cfg)
    legacy_cfg.pop("method")
    buffer = io.BytesIO()
    torch.save({"format": "recap_inspired_reconstructor_v1", "config": legacy_cfg,
                "model": model.state_dict(), "normalizer": norm.checkpoint()}, buffer)
    buffer.seek(0)
    loaded = load_reconstructor(buffer)
    expected = bundle.reconstruct(features, valid, missing)
    observed = loaded.reconstruct(features, valid, missing)
    for m in MODS:
        np.testing.assert_array_equal(expected[m], observed[m])
    self_test_mtsit()
    self_test_brits()
    self_test_centaur()
    self_test_naomi()
    self_test_csdi()
    print(json.dumps({"self_test": "passed", "synthetic_loss_before": before, "synthetic_loss_after": after,
                      "checked": ["identity", "observed_AV_preservation", "intact_sample_preservation", "zero_padding",
                                  "no_cross_sample_attention", "loss_masks", "token_first_encoding", "clean_text_not_input",
                                  "fresh_epoch_masks", "optimizer_learns", "method_factory",
                                  "alternate_method_contract", "new_checkpoint_roundtrip", "legacy_checkpoint_compatibility"]}), flush=True)


def self_test_mtsit():
    """Behavior tests only; no dataset files or experiment artifacts are written."""
    import copy
    import io
    seed_all(71)
    cfg = Config(method="mtsit", device="cpu", d_model=8, heads=2, layers=1,
                 ffn_dim=8, max_length=8, dropout=0.)
    model = build_reconstructor(cfg)
    assert isinstance(model, MTSITReconstructor)
    valid = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 0, 0]], dtype=bool)
    missing = np.zeros_like(valid)
    missing[1, 2:4] = True
    missing[2, 1:3] = True
    rng = np.random.default_rng(71)
    features = {m: rng.normal(size=(*valid.shape, DIMS[m])).astype(
        np.float32 if m == "text" else np.float64) for m in MODS}
    for x in features.values():
        x[~valid] = 0
    stats = {m: {"mean": np.zeros(DIMS[m], np.float32),
                 "std": np.ones(DIMS[m], np.float32)} for m in MODS}
    normalizer = Normalizer(stats)
    bundle = ReconstructionBundle(model, normalizer, "cpu")
    untouched = {m: x.copy() for m, x in features.items()}
    out = bundle.reconstruct(features, valid, missing)
    single = bundle.reconstruct(features, valid, missing, batch_size=1)
    for m in MODS:
        allowed = valid & missing.any(1, keepdims=True) if m == "text" else missing
        assert out[m].shape == features[m].shape and out[m].dtype == features[m].dtype
        assert np.isfinite(out[m]).all()
        np.testing.assert_array_equal(out[m][~allowed], features[m][~allowed])
        np.testing.assert_array_equal(features[m], untouched[m])
        np.testing.assert_allclose(out[m], single[m], atol=2e-6, rtol=2e-6)
    x = normalizer.normalize(features, valid, "cpu")
    v, k = torch.as_tensor(valid), torch.as_tensor(missing)
    encoder_inputs = []
    hook = model.project.register_forward_pre_hook(
        lambda module, args: encoder_inputs.append(args[0].detach().clone()))
    with torch.no_grad():
        original = model(x, v, k)
    hook.remove()
    assert encoder_inputs[0].shape == (3, 8, 877)
    assert (encoder_inputs[0][k | ~v] == 0).all()
    changed = {m: a.clone() for m, a in x.items()}
    for a in changed.values():
        a[0] = 100
        a[~v] = 1e5
    with torch.no_grad():
        altered = model(changed, v, k)
    for m in MODS:
        torch.testing.assert_close(original[m][1:], altered[m][1:], atol=1e-6, rtol=1e-6)
    # Padding is excluded even from training BatchNorm statistics.
    bn = MaskedSequenceBatchNorm(8)
    bn2 = copy.deepcopy(bn)
    hidden = torch.randn(3, 8, 8)
    noisy = hidden.clone()
    noisy[~v] = 1e6
    torch.testing.assert_close(bn(hidden, v), bn2(noisy, v), atol=0, rtol=0)
    for key in bn.state_dict():
        torch.testing.assert_close(bn.state_dict()[key], bn2.state_dict()[key], atol=0, rtol=0)
    assert torch.isfinite(bn(torch.ones(1, 1, 8), torch.ones(1, 1, dtype=torch.bool))).all()
    # The absolute linear decoder has an optimizer path to reconstruction targets.
    model.train()
    train_x = {m: value[1:] for m, value in x.items()}
    train_v, train_k = v[1:], k[1:]
    targets = {m: torch.zeros_like(value) for m, value in train_x.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = float(reconstruction_loss(model(train_x, train_v, train_k),
                                       targets, train_v, train_k)[0].detach())
    for _ in range(35):
        optimizer.zero_grad()
        loss, _ = reconstruction_loss(model(train_x, train_v, train_k),
                                      targets, train_v, train_k)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    after = float(reconstruction_loss(model(train_x, train_v, train_k),
                                      targets, train_v, train_k)[0].detach())
    assert after < before * .5, (before, after)
    buffer = io.BytesIO()
    torch.save({"format": "sequence_reconstructor_v1", "config": asdict(cfg),
                "model": model.state_dict(), "normalizer": normalizer.checkpoint()}, buffer)
    buffer.seek(0)
    loaded = load_reconstructor(buffer)
    expected = bundle.reconstruct(features, valid, missing)
    actual = loaded.reconstruct(features, valid, missing)
    for m in MODS:
        np.testing.assert_array_equal(expected[m], actual[m])
    # Geometric masks reproduce by sample ID/epoch, not by batch order.
    tb = np.zeros((80, 3, 50), np.int64)
    tb[:, 0, 0], tb[:, 0, 1:41], tb[:, 0, 41] = 101, 200, 102
    tb[:, 1, :42] = 1
    ids = np.arange(80)
    masks, rates = dynamic_masks(tb, ids, 1, 42, "geometric", 3.)
    again, _ = dynamic_masks(tb, ids, 1, 42, "geometric", 3.)
    other_epoch, _ = dynamic_masks(tb, ids, 2, 42, "geometric", 3.)
    assert np.array_equal(masks, again) and not np.array_equal(masks, other_epoch)
    perm = np.random.default_rng(9).permutation(len(ids))
    shuffled, _ = dynamic_masks(tb[perm], ids[perm], 1, 42, "geometric", 3.)
    np.testing.assert_array_equal(masks[perm], shuffled)
    assert not np.any(masks & ~eligible_slots(tb))
    # The Bernoulli branch remains byte-for-byte equivalent to its old RNG rule.
    bernoulli, old_rates = dynamic_masks(tb, ids, 1, 42)
    expected_mask = np.zeros_like(bernoulli)
    for i in ids:
        local_rng = np.random.default_rng(np.random.SeedSequence([42, 1, int(i), 9167]))
        rate = np.float32(PROBS[int(local_rng.integers(len(PROBS)))])
        expected_mask[i] = (local_rng.random(50) < float(rate)) & eligible_slots(tb)[i]
        assert old_rates[i] == rate
    np.testing.assert_array_equal(bernoulli, expected_mask)
    long_mask = geometric_corruption_mask(np.ones(100000, bool), .2, 3.,
                                          np.random.default_rng(713))
    transitions = np.flatnonzero(np.diff(np.r_[False, long_mask, False].astype(int)))
    runs = transitions[1::2] - transitions[::2]
    assert abs(long_mask.mean() - .2) < .01
    assert abs(runs.mean() - 3.) < .15
    assert resolve_mask_distribution(Config()) == "bernoulli"
    assert resolve_mask_distribution(cfg) == "geometric"
    print(json.dumps({"mtsit_self_test": "passed",
                      "synthetic_loss_before": before, "synthetic_loss_after": after,
                      "geometric_realized_rate": float(long_mask.mean()),
                      "geometric_mean_run": float(runs.mean()),
                      "checked": ["full_sequence_contract", "unchanged_observed_AV_and_intact_samples",
                                  "padding_zero", "input_immutable", "eval_batch_invariance",
                                  "same_slot_877D_input", "normalized_erased_input_zero",
                                  "padding_excluded_from_BatchNorm", "finite_learning",
                                  "checkpoint_roundtrip", "geometric_mask_statistics",
                                  "per_sample_epoch_reproducibility", "legacy_Bernoulli_unchanged"]}))



def self_test_brits():
    """BRITS behavior checks; uses memory only and creates no files."""
    import copy
    import io
    seed_all(91)
    observed = torch.tensor([[1, 0, 0, 1, 0]], dtype=torch.bool)
    valid0 = torch.ones_like(observed)
    assert brits_time_gaps(observed, valid0).tolist() == [[0, 1, 2, 3, 1]]
    assert brits_time_gaps(observed.flip(1), valid0).tolist() == [[0, 1, 1, 2, 3]]
    padded_valid = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.bool)
    padded_obs = torch.tensor([[0, 1, 0, 0, 0, 1, 0, 0]], dtype=torch.bool)
    assert brits_time_gaps(padded_obs, padded_valid).tolist() == [[0, 0, 1, 0, 2, 3, 1, 0]]

    feature = BRITSFeatureRegression(4)
    small_x = torch.randn(2, 4)
    changed = small_x.clone()
    changed[:, 2] += 100
    torch.testing.assert_close(feature(small_x)[:, 2], feature(changed)[:, 2])
    feature(small_x).sum().backward()
    assert torch.count_nonzero(feature.weight.grad.diag()) == 0
    decay = BRITSTemporalDecay(4, 4, diagonal=True)
    with torch.no_grad():
        decay.weight.fill_(1)
        decay.bias.zero_()
    torch.testing.assert_close(decay(torch.ones(2, 4)), torch.full((2, 4), float(np.exp(-1))))

    # A later missing estimate must send gradients through an earlier imputation.
    core = BRITSDirection((3, 2, 1), 5)
    values = torch.randn(2, 5, 6)
    slots = torch.ones(2, 5, dtype=torch.bool)
    observations = slots.clone()
    observations[:, (1, 3)] = False
    retained_inputs = []
    def retain_cell_input(module, args):
        value = args[0]
        value.retain_grad()
        retained_inputs.append(value)
    hook = core.cell.register_forward_pre_hook(retain_cell_input)
    result = core(values.masked_fill(~observations[..., None], 0), slots, observations, True)
    result["imputations"][:, 3].square().sum().backward()
    hook.remove()
    assert retained_inputs[1].grad is not None
    assert retained_inputs[1].grad[:, :6].abs().sum() > 0

    cfg = Config(method="brits", device="cpu", d_model=8, max_length=6,
                 brits_estimation_weight=.1, brits_consistency_weight=.1)
    model = build_reconstructor(cfg)
    assert isinstance(model, BRITSReconstructor)
    assert model.forward_rits is not model.backward_rits
    rng = np.random.default_rng(37)
    valid = np.ones((3, 6), bool)
    valid[0, 4:] = False
    valid[1, 5:] = False
    missing = np.zeros_like(valid)
    missing[1, [1, 3]] = True
    missing[2, [1, 2, 4]] = True
    features = {m: rng.normal(size=(3, 6, DIMS[m])).astype(
        np.float32 if m == "text" else np.float64) for m in MODS}
    for value in features.values():
        value[~valid] = 0
    source_copy = {m: value.copy() for m, value in features.items()}
    stats = {m: {"mean": np.zeros(DIMS[m], np.float32),
                 "std": np.ones(DIMS[m], np.float32)} for m in MODS}
    norm = Normalizer(stats)
    bundle = ReconstructionBundle(model, norm, "cpu")
    v, k = torch.as_tensor(valid), torch.as_tensor(missing)
    x = norm.normalize(features, valid, "cpu")
    model.eval()
    restored = bundle.reconstruct(features, valid, missing)
    for m in MODS:
        assert restored[m].shape == features[m].shape and restored[m].dtype == features[m].dtype
        np.testing.assert_array_equal(restored[m][~missing], features[m][~missing])
        np.testing.assert_array_equal(features[m], source_copy[m])
        assert np.isfinite(restored[m]).all()
    with torch.no_grad():
        f, b = model.directional_outputs(x, v, k)
        pred = model(x, v, k)
        poisoned = {m: value.clone() for m, value in x.items()}
        for value in poisoned.values():
            value[k | ~v] = 1234.
        pf, pb = model.directional_outputs(poisoned, v, k)
        torch.testing.assert_close(f["imputations"], pf["imputations"], atol=0, rtol=0)
        torch.testing.assert_close(b["imputations"], pb["imputations"], atol=0, rtol=0)
        single = model({m: value[1:2] for m, value in x.items()}, v[1:2], k[1:2])
        for m in MODS:
            torch.testing.assert_close(pred[m][1:2], single[m], atol=2e-6, rtol=2e-6)
        swapped = copy.deepcopy(model)
        swapped.forward_rits, swapped.backward_rits = swapped.backward_rits, swapped.forward_rits
        reversed_pred = swapped({m: value.flip(1) for m, value in x.items()}, v.flip(1), k.flip(1))
        for m in MODS:
            torch.testing.assert_close(pred[m], reversed_pred[m].flip(1), atol=2e-6, rtol=2e-6)
        equal_directions = copy.deepcopy(model)
        equal_directions.backward_rits = copy.deepcopy(equal_directions.forward_rits)
        # With exactly one valid erased slot, both identical directions see the same input.
        one_v = torch.ones(2, 1, dtype=torch.bool)
        one_x = {m: torch.zeros(2, 1, DIMS[m]) for m in MODS}
        _, one_terms = equal_directions.training_objective(one_x, one_x, one_v, one_v)
        assert one_terms["consistency_mae"].item() == 0.
        assert one_terms["observed_estimation_mae"].item() == 0.

    target = {m: torch.zeros_like(value) for m, value in x.items()}
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    losses = []
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = model.training_objective(x, target, v, k)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], losses
    model.eval()
    with torch.no_grad():
        loss, terms = model.training_objective(x, target, v, k)
        modified = {m: value.clone() for m, value in target.items()}
        for value in modified.values():
            value[~v] = 1e4
        for m in ("audio", "vision"):
            modified[m][~k] = 1e4
        other_loss, _ = model.training_objective(x, modified, v, k)
        torch.testing.assert_close(loss, other_loss)
    buffer = io.BytesIO()
    torch.save({"format": "sequence_reconstructor_v1", "config": asdict(cfg),
                "model": model.state_dict(), "normalizer": norm.checkpoint()}, buffer)
    buffer.seek(0)
    reloaded = load_reconstructor(buffer)
    expected = bundle.reconstruct(features, valid, missing)
    actual = reloaded.reconstruct(features, valid, missing)
    for m in MODS:
        np.testing.assert_array_equal(expected[m], actual[m])
    assert resolve_mask_distribution(Config(method="brits")) == "bernoulli"
    print(json.dumps({"brits_self_test": "passed", "synthetic_total_loss_before": losses[0],
                      "synthetic_total_loss_after": losses[-1],
                      "checked": ["paper_directional_time_gaps", "padding_skips_clock_and_state",
                                  "zero_diagonal_feature_regression", "exponential_decay",
                                  "gradients_through_earlier_imputation", "independent_RITS",
                                  "all_observed_values_preserved", "input_immutable",
                                  "missing_placeholder_invariance", "batch_invariance",
                                  "time_reversal_with_direction_swap", "all_missing_finite",
                                  "exact_zero_consistency", "finite_learning",
                                  "shared_loss_regions", "checkpoint_roundtrip"]}), flush=True)



def self_test_centaur():
    """In-memory behavioral tests for the Centaur DAE and shared contract."""
    import copy
    import io
    seed_all(113)
    cfg = Config(method="centaur", device="cpu", d_model=8,
                 centaur_latent_dim=24, max_length=8)
    model = build_reconstructor(cfg)
    assert isinstance(model, CentaurReconstructor)
    assert len(model.encoder) == len(model.decoder) == 4
    assert not any(isinstance(layer, (nn.BatchNorm2d, nn.Dropout, nn.Sigmoid,
                                     nn.MaxPool2d, nn.AvgPool2d)) for layer in model.modules())
    # Verify the exact inverse schedule for multiple odd/even lengths.
    for length in (1, 7, 8, 50):
        candidate = CentaurReconstructor(Config(method="centaur", d_model=1,
                                               centaur_latent_dim=2, max_length=length))
        shape = candidate.spatial_shapes[-1]
        for layer, expected in zip(candidate.decoder, candidate.spatial_shapes[-2::-1]):
            shape = tuple((n - 1) * s - 2 * p + k + op for n,s,p,k,op in
                          zip(shape, layer.stride, layer.padding, layer.kernel_size, layer.output_padding))
            assert shape == expected, (length, shape, expected)

    rng = np.random.default_rng(13)
    valid = np.ones((3, 8), bool)
    valid[:, 5:] = False
    missing = np.zeros_like(valid)
    missing[1, [1, 3]] = True
    missing[2, [0, 2, 4]] = True
    features = {m: rng.normal(size=(3, 8, DIMS[m])).astype(
        np.float32 if m == "text" else np.float64) for m in MODS}
    for value in features.values():
        value[~valid] = 0
    original = {m: value.copy() for m,value in features.items()}
    stats = {m: {"mean": np.zeros(DIMS[m], np.float32), "std": np.ones(DIMS[m], np.float32)}
             for m in MODS}
    norm = Normalizer(stats)
    bundle = ReconstructionBundle(model, norm, "cpu")
    x = norm.normalize(features, valid, "cpu")
    v, k = torch.as_tensor(valid), torch.as_tensor(missing)
    recorded = []
    hook = model.encoder[0].register_forward_pre_hook(lambda module,args: recorded.append(args[0].detach().clone()))
    model.eval()
    with torch.no_grad():
        prediction = model(x, v, k)
    hook.remove()
    assert recorded[0].shape == (3, 1, sum(DIMS.values()), 8)
    expected_input = torch.cat([x[m] for m in MODS], -1).masked_fill((~v | k)[..., None], 0)
    torch.testing.assert_close(recorded[0][:, 0].transpose(1, 2), expected_input, atol=0, rtol=0)
    restored = bundle.reconstruct(features, valid, missing)
    for m in MODS:
        assert restored[m].shape == features[m].shape and restored[m].dtype == features[m].dtype
        allowed = valid & missing.any(1, keepdims=True) if m == "text" else missing
        np.testing.assert_array_equal(restored[m][~allowed], features[m][~allowed])
        np.testing.assert_array_equal(features[m], original[m])
        assert np.isfinite(restored[m]).all()
    with torch.no_grad():
        single = model({m: z[1:2] for m,z in x.items()}, v[1:2], k[1:2])
        shorter = model({m: z[:, :5] for m,z in x.items()}, v[:, :5], k[:, :5])
        altered = {m: z.clone() for m,z in x.items()}
        for z in altered.values():
            z[k | ~v] = 1e4
        reference_estimate = model.normalized_estimate(x, v, k)
        altered_estimate = model.normalized_estimate(altered, v, k)
        torch.testing.assert_close(reference_estimate, altered_estimate, atol=0, rtol=0)
        all_missing = model.normalized_estimate(x, v, v)
        assert torch.isfinite(all_missing).all()
        for m in MODS:
            torch.testing.assert_close(prediction[m][1:2], single[m], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(prediction[m][:, :5], shorter[m], atol=2e-6, rtol=2e-6)
        signed = copy.deepcopy(model)
        for parameter in signed.parameters():
            parameter.zero_()
        signed.decoder[-1].bias.fill_(-.75)
        negative = signed.normalized_estimate(x, v, k)
        torch.testing.assert_close(negative, torch.full_like(negative, -.75), atol=0, rtol=0)
        negative_output = signed(x, v, k)
        torch.testing.assert_close(negative_output["text"][v & k.any(1, keepdim=True)],
                                   torch.full_like(negative_output["text"][v & k.any(1, keepdim=True)], -.75),
                                   atol=2e-6, rtol=2e-6)
    target = {m: torch.full_like(z, .4) for m,z in x.items()}
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    losses = []
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.training_objective(x, target, v, k)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if step == 0:
            assert model.encoder[0].weight.grad.abs().sum() > 0
            assert model.to_latent.weight.grad.abs().sum() > 0
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], losses
    model.eval()
    with torch.no_grad():
        loss, _ = model.training_objective(x, target, v, k)
        changed_target = {m: z.clone() for m,z in target.items()}
        for m in MODS:
            changed_target[m][~v] = 1e4
        for m in ("audio", "vision"):
            changed_target[m][~k] = 1e4
        changed_loss, _ = model.training_objective(x, changed_target, v, k)
        torch.testing.assert_close(loss, changed_loss)
    buffer = io.BytesIO()
    torch.save({"format": "sequence_reconstructor_v1", "config": asdict(cfg),
                "model": model.state_dict(), "normalizer": norm.checkpoint()}, buffer)
    buffer.seek(0)
    loaded = load_reconstructor(buffer)
    expected = bundle.reconstruct(features, valid, missing)
    actual = loaded.reconstruct(features, valid, missing)
    for m in MODS:
        np.testing.assert_array_equal(expected[m], actual[m])
    assert resolve_mask_distribution(cfg) == "bernoulli"
    print(json.dumps({"centaur_self_test": "passed", "synthetic_loss_before": losses[0],
                      "synthetic_loss_after": losses[-1],
                      "checked": ["joint_feature_time_layout", "four_stage_conv_DAE",
                                  "exact_odd_even_decoder_shapes", "padding_and_erasure_input_mask",
                                  "observed_AV_and_intact_preservation", "raw_shape_dtype",
                                  "signed_linear_absolute_output", "batch_invariance",
                                  "padding_length_invariance", "all_missing_finite",
                                  "input_immutable", "encoder_and_latent_gradients",
                                  "finite_learning", "shared_loss_regions",
                                  "checkpoint_roundtrip", "Bernoulli_default"]}), flush=True)


def self_test_naomi():
    """Exercise multiresolution dependencies, ragged masks and the public API."""
    import io
    seed_all(211)
    torch.set_num_threads(2)
    cfg = Config(method="naomi", device="cpu", d_model=8, layers=1,
                 naomi_decoder_dim=16, naomi_highest=8, max_length=18)
    model = build_reconstructor(cfg)
    valid = np.zeros((5, 18), bool)
    for row, length in enumerate((17, 10, 5, 1, 8)):
        valid[row, :length] = True
    missing = np.zeros_like(valid)
    missing[0, 1:16] = True  # Forces 8 -> 4 -> 2 -> 1 refinement.
    missing[1, [0, 2, 3, 7, 8, 9]] = True  # Both boundaries and a short interior gap.
    missing[2, :5] = True
    missing[3, 0] = True
    # Row 4 is intact.
    operations, counts = model.schedule(valid, missing)
    sequence0 = [(left, step) for left, items in operations.items()
                 for step, rows in items if 0 in rows]
    assert sequence0[:4] == [(0,8), (0,4), (0,2), (0,1)], sequence0
    assert sum(counts.values()) == int(missing.sum())
    rng = np.random.default_rng(9)
    features = {m: rng.normal(size=(5, 18, DIMS[m])).astype(
        np.float32 if m == "text" else np.float64) for m in MODS}
    for value in features.values():
        value[~valid] = 0
    stats = {m: {"mean": np.full(DIMS[m], .3, np.float32),
                 "std": np.full(DIMS[m], 1.7, np.float32)} for m in MODS}
    norm = Normalizer(stats)
    bundle = ReconstructionBundle(model, norm, "cpu")
    x = norm.normalize(features, valid, "cpu")
    v, k = torch.as_tensor(valid), torch.as_tensor(missing)
    model.eval()
    with torch.no_grad():
        output = model(x, v, k)
        estimate = model.normalized_estimate(x, v, k)
        altered = {m: z.clone() for m,z in x.items()}
        for value in altered.values():
            value[k | ~v] = 1e4
        alternate = model.normalized_estimate(altered, v, k)
        torch.testing.assert_close(estimate, alternate, atol=0, rtol=0)
        for row in range(len(v)):
            length = int(v[row].sum())
            single = model({m:z[row:row+1,:length] for m,z in x.items()},
                           v[row:row+1,:length], k[row:row+1,:length])
            for m in MODS:
                torch.testing.assert_close(output[m][row:row+1,:length],
                                           single[m], atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(output[m][~k], x[m][~k], atol=0, rtol=0)
    restored = bundle.reconstruct(features, valid, missing)
    for m in MODS:
        assert restored[m].shape == features[m].shape and restored[m].dtype == features[m].dtype
        assert np.isfinite(restored[m]).all()
        np.testing.assert_array_equal(restored[m][~missing], features[m][~missing])
    # Fine outputs must depend differentiably on both future observations and
    # earlier coarse predictions, rather than detached interpolation.
    inputs = {m:z.clone().requires_grad_() for m,z in x.items()}
    captured = []
    def keep_coarse(module, args, output):
        output.retain_grad()
        captured.append(output)
    hook = model.decoders["8"].register_forward_hook(keep_coarse)
    prediction = model(inputs, v, k)
    prediction["audio"][0, 1].square().mean().backward()
    hook.remove()
    assert captured and captured[0].grad.abs().sum() > 0
    assert inputs["audio"].grad[0, 16].abs().sum() > 0
    assert model.forward_gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.backward_gru.weight_ih_l0.grad.abs().sum() > 0
    for value in inputs.values():
        assert value.grad[~v].abs().sum() == 0
    # Shared loss regions, loss-only targets, finite optimization.
    target = {m: torch.full_like(z, .25) for m,z in x.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    losses = []
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.training_objective(x, target, v, k)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], losses
    changed_target = {m:z.clone() for m,z in target.items()}
    for m in MODS:
        changed_target[m][~v] = 1e4
    for m in ("audio", "vision"):
        changed_target[m][~k] = 1e4
    torch.testing.assert_close(model.training_objective(x,target,v,k)[0],
                               model.training_objective(x,changed_target,v,k)[0])
    buffer = io.BytesIO()
    torch.save({"format":"sequence_reconstructor_v1", "config":asdict(cfg),
                "model":model.state_dict(), "normalizer":norm.checkpoint()},buffer)
    buffer.seek(0)
    reloaded = load_reconstructor(buffer)
    expected = bundle.reconstruct(features, valid, missing)
    actual = reloaded.reconstruct(features, valid, missing)
    for m in MODS:
        np.testing.assert_array_equal(expected[m], actual[m])
    invalid = valid.copy()
    invalid[0, 2] = False
    try:
        model.schedule(invalid, missing & invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Noncontiguous valid slots must fail explicitly")
    model.zero_grad(set_to_none=True)
    intact_loss, _ = model.training_objective(x, x, v, torch.zeros_like(k))
    assert intact_loss.requires_grad and float(intact_loss.detach()) == 0
    intact_loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    assert resolve_mask_distribution(cfg) == "bernoulli"
    print(json.dumps({"naomi_self_test":"passed", "synthetic_loss_before":losses[0],
                      "synthetic_loss_after":losses[-1], "checked":[
        "per_sample_coarse_to_fine_schedule", "all_scales_covered", "batch_and_length_invariance",
        "leading_trailing_all_missing", "observed_text_AV_passthrough", "raw_shape_dtype",
        "missing_placeholder_and_padding_invariance", "future_observation_dependency",
        "coarse_to_fine_gradient", "both_GRU_gradients", "finite_learning",
        "shared_loss_regions", "checkpoint_roundtrip", "Bernoulli_default"]}), flush=True)


def self_test_csdi():
    """In-memory tests: conditional loss, DDPM math and probabilistic sampling."""
    import io
    import types
    seed_all(313)
    torch.set_num_threads(2)
    cfg = Config(method="csdi", device="cpu", d_model=4, heads=1, layers=1,
                 max_length=6, csdi_steps=4, csdi_samples=3, csdi_microbatch=2,
                 csdi_time_dim=4, csdi_feature_dim=2, csdi_amp=False)
    model = build_reconstructor(cfg)
    valid = np.array([[1,1,1,1,0,0],[1,1,1,0,0,0],[1,1,1,1,1,0]],bool)
    missing = np.array([[0,1,0,0,0,0],[1,1,1,0,0,0],[0,0,0,0,0,0]],bool)
    rng = np.random.default_rng(3)
    raw = {m:rng.normal(size=(3,6,DIMS[m])).astype(np.float32 if m=="text" else np.float64)
           for m in MODS}
    for x in raw.values():x[~valid]=0
    stats = {m:{"mean":np.full(DIMS[m],.2,np.float32),"std":np.full(DIMS[m],1.3,np.float32)} for m in MODS}
    norm = Normalizer(stats)
    features = norm.normalize(raw,valid,"cpu")
    v,k = torch.as_tensor(valid),torch.as_tensor(missing)
    model.eval()
    # Fixed bank: batch composition/order cannot change the sampled paths.
    with torch.no_grad():
        samples = model.normalized_samples(features,v,k)
        prediction = model(features,v,k)
        assert samples.shape == (3,3,6,sum(DIMS.values()))
        assert torch.isfinite(samples).all()
        assert (samples[0,0,1]-samples[0,1,1]).abs().sum()>0
        for row in range(3):
            length=int(v[row].sum())
            single=model({m:x[row:row+1,:length] for m,x in features.items()},
                         v[row:row+1,:length],k[row:row+1,:length])
            for m in MODS:torch.testing.assert_close(single[m],prediction[m][row:row+1,:length],atol=2e-5,rtol=2e-5)
        perm=torch.tensor([2,0,1])
        reordered=model({m:x[perm] for m,x in features.items()},v[perm],k[perm])
        for m in MODS:
            torch.testing.assert_close(reordered[m],prediction[m][perm],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(prediction[m][~k],features[m][~k],atol=0,rtol=0)
        poisoned={m:x.clone() for m,x in features.items()}
        for x in poisoned.values():x[k | ~v]=1e4
        alternate=model.normalized_samples(poisoned,v,k)
        torch.testing.assert_close(samples,alternate,atol=0,rtol=0)
    bundle=ReconstructionBundle(model,norm,"cpu")
    restored=bundle.reconstruct(raw,valid,missing)
    for m in MODS:
        assert restored[m].shape==raw[m].shape and restored[m].dtype==raw[m].dtype
        np.testing.assert_array_equal(restored[m][~missing],raw[m][~missing])
    # Clean observed targets must not become conditions or affect loss.
    target={m:torch.full_like(x,.4) for m,x in features.items()}
    modified={m:x.clone() for m,x in target.items()}
    for x in modified.values():x[~k]=1e4
    seed_all(7);a,terms=model.training_objective(features,target,v,k)
    seed_all(7);b,_=model.training_objective(features,modified,v,k)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert set(terms)=={"noise_text","noise_audio","noise_vision"}
    torch.testing.assert_close(a,sum(terms.values())/3)
    a.backward()
    assert torch.isfinite(model.output_projection2.weight.grad).all()
    assert model.output_projection2.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    zero,_=model.training_objective(features,target,v,torch.zeros_like(k))
    assert float(zero.detach())==0 and zero.requires_grad
    zero.backward()
    # At t=0 no innovation is allowed; check DDPM coefficients directly.
    state=torch.randn(2,3,5);epsilon=torch.randn_like(state);innovation=torch.randn_like(state)
    for step in range(cfg.csdi_steps):
        expected=(state-model.beta[step]/(1-model.alpha_bar[step]).sqrt()*epsilon)/(1-model.beta[step]).sqrt()
        if step:expected=expected+model.posterior_std[step]*innovation
        torch.testing.assert_close(model.reverse_step(state,epsilon,step,innovation),expected,atol=0,rtol=0)
    assert float(model.posterior_std[0])==0
    # An oracle epsilon for a point-mass conditional distribution must generate
    # that point at all erased entries, even with stochastic intermediate steps.
    actual_predict=model.predict_noise
    def oracle(self, observed, state, valid, missing, steps, sides=None):
        alpha=self.alpha_bar[steps,None,None]
        return (state-alpha.sqrt()*.4)/(1-alpha).sqrt()
    model.predict_noise=types.MethodType(oracle,model)
    oracle_samples=model.normalized_samples(features,v,k)
    selected=k[:,None,:,None].expand_as(oracle_samples)
    torch.testing.assert_close(oracle_samples[selected],torch.full_like(oracle_samples[selected],.4),atol=2e-5,rtol=2e-5)
    model.predict_noise=actual_predict
    # Finite learning with a fixed training noise draw on a small synthetic case.
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    losses=[]
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        seed_all(27)
        loss,_=model.training_objective(features,target,v,k)
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0],losses
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    buffer=io.BytesIO()
    torch.save({"format":"sequence_reconstructor_v1","config":asdict(cfg),"model":model.state_dict(),
                "normalizer":norm.checkpoint()},buffer);buffer.seek(0)
    reloaded=load_reconstructor(buffer)
    expected=bundle.reconstruct(raw,valid,missing);actual=reloaded.reconstruct(raw,valid,missing)
    for m in MODS:np.testing.assert_array_equal(expected[m],actual[m])
    assert resolve_mask_distribution(cfg)=="bernoulli"
    print(json.dumps({"csdi_self_test":"passed","loss_before":losses[0],"loss_after":losses[-1],
      "checked":["stochastic_trajectories","batch_order_length_invariance","all_missing_finite",
        "observed_text_AV_passthrough","raw_shape_dtype","missing_placeholder_and_padding_invariance",
        "no_clean_observed_target_leakage","equal_modality_noise_loss","intact_batch_backward",
        "DDPM_mean_variance_formula","oracle_reverse_chain","finite_learning",
        "checkpoint_roundtrip","Bernoulli_default"]}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    p = sub.add_parser("train")
    defaults = Config()
    p.add_argument("--method", choices=sorted(RECONSTRUCTOR_METHODS), default=defaults.method)
    for field in ("data_dir", "adapter_module", "adapter_checkpoint", "output_root", "device"):
        p.add_argument("--" + field.replace("_", "-"), default=getattr(defaults, field))
    for field in ("seed", "d_model", "batch_size", "epochs", "patience", "cpu_threads",
                  "heads", "layers", "ffn_dim", "centaur_latent_dim", "naomi_highest", "naomi_decoder_dim",
                  "csdi_steps", "csdi_samples", "csdi_sampling_seed", "csdi_microbatch",
                  "csdi_time_dim", "csdi_feature_dim", "csdi_validation_interval"):
        p.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    for field in ("brits_estimation_weight", "brits_consistency_weight", "csdi_beta_start", "csdi_beta_end"):
        p.add_argument("--" + field.replace("_", "-"), type=float, default=getattr(defaults, field))
    p.add_argument("--csdi-amp", action=argparse.BooleanOptionalAction, default=defaults.csdi_amp)
    p.add_argument("--csdi-compile", action=argparse.BooleanOptionalAction, default=defaults.csdi_compile)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--mask-distribution", choices=("auto", "bernoulli", "geometric"),
                   default=defaults.mask_distribution,
                   help="auto: recap/brits/centaur/naomi/csdi use Bernoulli; mtsit uses geometric contiguous masks")
    p.add_argument("--mean-mask-length", type=float, default=defaults.mean_mask_length)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        kwargs = vars(args)
        kwargs.pop("command")
        train(Config(**kwargs))


if __name__ == "__main__":
    main()

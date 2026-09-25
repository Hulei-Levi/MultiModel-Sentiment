"""Train a small RECAP-inspired feature reconstructor, NOT a RECAP reproduction.

这个代码用于重建缺失特征，对缺失的特征进行复原

Only reconstruction is trained/evaluated. No emotion labels, sentiment model,
adversarial loss, FID module, or reconstructed dataset export is involved.
Multiscale temporal convolutions and within-sample cross-modal attention are
the borrowed ideas. Fixed p010/p020/p030 validation chooses a checkpoint;
the four test conditions are evaluated only after selection.

Run from MultiModel-Sentiment:
  .venv-align/bin/python -B train_recap_reconstructor.py self-test
  .venv-align/bin/python -B train_recap_reconstructor.py train --device cuda:0

Import load_reconstructor(checkpoint) to obtain a ReconstructionBundle.
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

ROOT = Path(__file__).resolve().parent
MODS = ("text", "audio", "vision")
DIMS = {"text": 768, "audio": 74, "vision": 35}
PROBS = (0.1, 0.2, 0.3)
SPLITS = ("train", "valid", "test")


@dataclass
class Config:
    data_dir: str = str(ROOT / "datasets/附件2-同步扰动特征")
    adapter_module: str = str(ROOT / "bert_feature_adapter.py")
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


class Reconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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

    def forward(self, features, valid_mask, corruption_mask):
        """Only corrupted features and masks enter this function; no targets."""
        b, length = valid_mask.shape
        if length > self.cfg.max_length:
            raise ValueError("Input exceeds configured sequence length")
        h = []
        for mi, m in enumerate(MODS):
            z = self.project[m](features[m]) + self.position[:, :length] + self.modality[mi]
            z = z + self.erasure(corruption_mask.long())
            z = z.masked_fill(~valid_mask[..., None], 0)
            h.append(self.temporal[m](z, valid_mask))
        # B is always the batch dimension. Attention spans only the 3L slots
        # of one sample; examples cannot attend to each other.
        z = self.interaction(torch.cat(h, dim=1), src_key_padding_mask=~valid_mask.repeat(1, 3))
        outputs = {}
        any_missing = corruption_mask.any(1, keepdim=True)
        for mi, m in enumerate(MODS):
            delta = self.decode[m](z[:, mi * length:(mi + 1) * length])
            change = valid_mask & (any_missing if m == "text" else corruption_mask)
            outputs[m] = torch.where(change[..., None], features[m] + delta, features[m])
            outputs[m] = outputs[m].masked_fill(~valid_mask[..., None], 0)
        return outputs


def reconstruction_loss(pred, target, valid, missing):
    terms = {}
    for m in MODS:
        region = valid if m == "text" else missing
        per_slot = (pred[m] - target[m]).square().mean(-1)
        terms[m] = (per_slot * region).sum() / region.sum().clamp_min(1)
    return sum(terms.values()) / len(MODS), terms


def dynamic_masks(clean_tb, indices, epoch, seed):
    eligible = eligible_slots(clean_tb)
    missing = np.zeros_like(eligible)
    rates = np.zeros(len(indices), np.float32)
    for i, index in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, int(index), 9167]))
        rates[i] = PROBS[int(rng.integers(len(PROBS)))]
        missing[i] = (rng.random(eligible.shape[1]) < float(rates[i])) & eligible[i]
    return missing, rates


def corrupted_training_batch(clean, indices, epoch, cfg, adapter):
    tb = clean["text_bert"][indices].copy()
    missing, rates = dynamic_masks(tb, indices, epoch, cfg.seed)
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
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "recap_inspired_reconstructor_v1":
        raise ValueError("Unsupported checkpoint")
    cfg = Config(**saved["config"])
    model = Reconstructor(cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return ReconstructionBundle(model, Normalizer.from_checkpoint(saved["normalizer"]), device)


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
    if cfg.epochs < 1 or cfg.patience < 1 or cfg.d_model % cfg.heads:
        raise ValueError("Invalid training/model configuration")
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
    model = Reconstructor(cfg).to(cfg.device)
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
        rate_counts = {str(p): 0 for p in PROBS}
        mask_hash = hashlib.sha256()
        for start in range(0, len(order), cfg.batch_size):
            indices = order[start:start + cfg.batch_size]
            features, valid, missing, rates = corrupted_training_batch(clean["train"], indices, epoch, cfg, adapter)
            x = normalizer.normalize(features, valid, cfg.device)
            # Targets are prepared independently and never passed to forward.
            target = normalizer.normalize({m: clean["train"][m][indices] for m in MODS}, valid, cfg.device)
            v, k = torch.as_tensor(valid, device=cfg.device), torch.as_tensor(missing, device=cfg.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, v, k)
            loss, _ = reconstruction_loss(pred, target, v, k)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
            samples += len(indices)
            missing_count += int(missing.sum())
            eligible_count += int(eligible_slots(clean["train"]["text_bert"][indices]).sum())
            for rate in PROBS:
                rate_counts[str(rate)] += int(np.isclose(rates, rate).sum())
            mask_hash.update(indices.tobytes())
            mask_hash.update(missing.tobytes())
        valid_metrics = {str(rate): evaluate(bundle, clean["valid"], fixed_valid[rate], cfg.batch_size) for rate in (10, 20, 30)}
        score = float(np.mean([x["reconstructed"]["balanced_normalized_mse"] for x in valid_metrics.values()]))
        record = {"epoch": epoch, "train_loss": loss_total / samples, "validation_score": score,
                  "valid": valid_metrics, "seconds": time.time() - started,
                  "mask_sha256": mask_hash.hexdigest(), "nominal_rate_sample_counts": rate_counts,
                  "realized_corruption_fraction": missing_count / eligible_count}
        history.append(record)
        write_json(run_dir / "history.json", history)
        improved = score < best_score - 1e-8
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"format": "recap_inspired_reconstructor_v1", "config": asdict(cfg),
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "normalizer": normalizer.checkpoint(), "epoch": epoch,
                        "validation_score": score, "training_only_reconstruction": True,
                        "provenance_hashes": hashes_before}, run_dir / "best.pt")
        else:
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
    results = {"method": "RECAP-inspired lightweight feature reconstruction; not full RECAP",
               "best_epoch": best_epoch, "epochs_run": len(history), "trainable_parameters": params,
               "best_validation_score": best_score, "training_seconds": time.time() - training_start,
               "selection": "Mean of p010/p020/p030 validation MSE; modalities equally weighted after train-only standardization",
               "loss_regions": {"text": "all valid positions", "audio": "artificially erased slots", "vision": "artificially erased slots"},
               "test": {}}
    for rate in (0, 10, 20, 30):
        corrupted = clean["test"] if rate == 0 else fixed_split(files[rate], "test", clean["test"], metadata, protected)
        result = evaluate(bundle, clean["test"], corrupted, cfg.batch_size, run_dir / f"test_p{rate:03d}_errors.csv")
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
    lines = ["# RECAP-inspired feature reconstruction", "",
             "This is a lightweight adaptation, not a reproduction of the full RECAP architecture.",
             "No sentiment labels, classification/regression model, GAN, FID, or distillation is used.", "",
             f"Best epoch: {results['best_epoch']}; epochs run: {results['epochs_run']}; trainable parameters: {results['trainable_parameters']}.",
             "Normalization is fitted on clean train only; the frozen existing BERT adapter is reused.",
             "The same training example can have different masks across epochs. Its clean target never enters the reconstructor forward call.",
             "Masks do not cross original train/valid/test boundaries. No attention crosses the batch dimension.",
             "Repeated generic phrases okay/alright across train/valid are disclosed in audit.json; original video IDs and full multimodal samples do not overlap.", "",
             "Training draws nominal Bernoulli p=0.1/0.2/0.3 independently per example each epoch. Text tokens are corrupted before whole-sequence BERT encoding.",
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
              "## Reuse", "", "```python", "from train_recap_reconstructor import load_reconstructor",
              f"reconstructor = load_reconstructor({str(run_dir / 'best.pt')!r}, device='cuda:0')",
              "# features = {'text': corrupted_text, 'audio': corrupted_audio, 'vision': corrupted_vision}",
              "restored = reconstructor.reconstruct(features, valid_mask, corruption_mask)", "```", "",
              "Inputs must use the same BERT+adapter encoding and raw audio/video units as this run. valid_mask comes from text_bert[:,1,:]. corruption_mask is explicit provenance, not inferred from zero vectors.",
              "Text can change throughout a corrupted sample because BERT contextual encoding propagates token corruption beyond the masked slots.",
              "No reconstructed datasets are written. best.pt contains the model, config, normalization, selected epoch and provenance hashes.", "",
              f"Protected data, adapter source and adapter weights unchanged: {results['protected_inputs_unchanged']}.",
              "Reference: https://ojs.aaai.org/index.php/AAAI/article/view/39349/43310", ""]
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
    model = Reconstructor(cfg)
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
    print(json.dumps({"self_test": "passed", "synthetic_loss_before": before, "synthetic_loss_after": after,
                      "checked": ["identity", "observed_AV_preservation", "intact_sample_preservation", "zero_padding",
                                  "no_cross_sample_attention", "loss_masks", "token_first_encoding", "clean_text_not_input",
                                  "fresh_epoch_masks", "optimizer_learns"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    p = sub.add_parser("train")
    defaults = Config()
    for field in ("data_dir", "adapter_module", "adapter_checkpoint", "output_root", "device"):
        p.add_argument("--" + field.replace("_", "-"), default=getattr(defaults, field))
    for field in ("seed", "d_model", "batch_size", "epochs", "patience", "cpu_threads"):
        p.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    p.add_argument("--lr", type=float, default=defaults.lr)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        kwargs = vars(args)
        kwargs.pop("command")
        train(Config(**kwargs))


if __name__ == "__main__":
    main()

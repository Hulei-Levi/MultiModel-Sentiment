"""Dai-style learned logit fusion for precomputed, aligned T/A/V features.

The fusion operator follows the authors' nn.Linear(3, 1, bias=False).
The feature encoders, masks, equal initialization and single-label CE are
adaptations, NOT a reproduction of the end-to-end sparse MESM model.
Requires Python >= 3.10, numpy and PyTorch >= 2.0. No model download is needed.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

MODALITIES = ("text", "audio", "vision")  # Explicit order, unlike paper's T/V/A.
SOURCE = "https://github.com/wenliangdai/Multimodal-End2end-Sparse/blob/main/src/models/sparse_e2e.py"


@dataclass
class Config:
    task: str = "classification"  # or "regression"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 1
    ff_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    standardize_av: bool = True
    class_weighted_loss: bool = False
    auxiliary_loss_weight: float = 0.0  # >0 changes the training objective.
    device: str | None = None
    output_root: str = "weighted_sum_runs"

    def validate(self):
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        if min(self.d_model, self.nhead, self.num_layers, self.ff_dim,
               self.batch_size, self.epochs, self.patience) <= 0:
            raise ValueError("Model sizes, batch size, epochs and patience must be positive")
        if self.d_model % self.nhead or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by nhead")
        if not 0 <= self.dropout < 1 or self.auxiliary_loss_weight < 0:
            raise ValueError("Invalid dropout or auxiliary loss weight")
        if self.lr <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimization settings")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _binary_mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or not np.isin(a, [0, 1]).all():
        raise ValueError(f"{name}: expected a binary mask of shape {shape}, got {a.shape}")
    return a.astype(bool, copy=True)


def resolve_masks(split, overrides=None):
    """Default assumes the SAME 50-slot layout, including special-token slots.

    overrides: {modality: [N,L] binary mask}. A supplied mask is authoritative,
    and should combine that modality's padding validity and known availability.
    All-zero features are never automatically considered missing.
    """
    n, length = np.asarray(split["text"]).shape[:2]
    tb = np.asarray(split["text_bert"])
    if tb.shape != (n, 3, length):
        raise ValueError(f"text_bert must have shape {(n, 3, length)}, got {tb.shape}")
    text_mask = _binary_mask(tb[:, 1, :], (n, length), "text_bert attention_mask")
    overrides = overrides or {}
    if set(overrides) - set(MODALITIES):
        raise ValueError("Mask override keys must be text/audio/vision")
    masks = {m: _binary_mask(overrides.get(m, text_mask), (n, length), m)
             for m in MODALITIES}
    available = np.stack([masks[m].any(axis=1) for m in MODALITIES], axis=1)
    if not available.any(axis=1).all():
        bad = np.flatnonzero(~available.any(axis=1))[:5]
        raise ValueError(f"Samples have no observed positions in any modality: {bad.tolist()}")
    return masks


def audit_data(data, mask_overrides=None):
    """Return diagnostics, not an inferred missing-data mask."""
    mask_overrides = mask_overrides or {}
    report = {}
    for name in ("train", "valid", "test"):
        s = data[name]
        masks = resolve_masks(s, mask_overrides.get(name))
        token_mask = np.asarray(s["text_bert"])[:, 1, :] == 1
        record = {"samples": len(s["text"]), "modalities": {}}
        for m in MODALITIES:
            x = np.asarray(s[m])
            if x.ndim != 3 or x.shape[:2] != token_mask.shape:
                raise ValueError(f"{name}/{m}: incompatible feature shape {x.shape}")
            zero = np.all(x == 0, axis=-1)
            nonfinite_observed = 0
            for start in range(0, len(x), 128):
                xx = x[start:start + 128]
                mm = masks[m][start:start + 128]
                nonfinite_observed += int((~np.isfinite(xx[mm])).sum())
            record["modalities"][m] = {
                "shape": list(x.shape), "dtype": str(x.dtype),
                "observed_positions": int(masks[m].sum()),
                "all_zero_positions": int(zero.sum()),
                "zero_in_text_valid_region": int((zero & token_mask).sum()),
                "nonzero_in_text_padding_region": int((~zero & ~token_mask).sum()),
                "nonfinite_observed_values": nonfinite_observed,
                "fully_unobserved_samples": int((~masks[m].any(axis=1)).sum()),
                "mask_source": "override" if m in mask_overrides.get(name, {}) else "text_bert[:,1,:]",
            }
            if nonfinite_observed:
                raise ValueError(f"{name}/{m}: nonfinite observed values; supply a justified mask or clean first")
        report[name] = record
    return report


def fit_normalizers(train, masks, standardize_av=True):
    """Fit on TRAIN ONLY. Zero rows count as observations unless explicitly masked."""
    stats = {}
    for m in MODALITIES:
        x = np.asarray(train[m])
        dim = x.shape[-1]
        mean, std = np.zeros(dim, np.float32), np.ones(dim, np.float32)
        count = int(masks[m].sum())
        if m != "text" and standardize_av and count:
            sums, sums2 = np.zeros(dim, np.float64), np.zeros(dim, np.float64)
            for start in range(0, len(x), 128):
                values = x[start:start + 128][masks[m][start:start + 128]].astype(np.float64)
                sums += values.sum(axis=0)
                sums2 += np.square(values).sum(axis=0)
            mu = sums / count
            sigma = np.sqrt(np.maximum(sums2 / count - mu ** 2, 0))
            sigma[sigma < 1e-6] = 1.0
            mean, std = mu.astype(np.float32), sigma.astype(np.float32)
        stats[m] = {"mean": mean, "std": std, "observed_count": count}
    return stats


class FeatureDataset(Dataset):
    def __init__(self, split, masks, stats, task="classification", class_values=None,
                 require_labels=True):
        # Keep references to existing arrays; cast/normalize per sample to avoid
        # duplicating the entire [N,50,768] text tensor in host memory.
        self.split = split
        self.features = {m: np.asarray(split[m]) for m in MODALITIES}
        self.masks, self.stats, self.task = masks, stats, task
        self.n = len(self.features["text"])
        self.ids = [str(v) for v in split.get("id", range(self.n))]
        if len(self.ids) != self.n:
            raise ValueError("id length differs from sample count")
        for m in MODALITIES:
            x = self.features[m]
            if x.ndim != 3 or x.shape[:2] != masks[m].shape:
                raise ValueError(f"Bad {m} shape: {x.shape}")
            if x.shape[-1] != len(stats[m]["mean"]):
                raise ValueError(f"Feature dimension changed for {m}")
        key = "classification_labels" if task == "classification" else "regression_labels"
        self.targets = None
        if key not in split:
            if require_labels:
                raise ValueError(f"Missing target field: {key}")
        else:
            y = np.asarray(split[key])
            if y.shape not in ((self.n,), (self.n, 1)) or not np.isfinite(y).all():
                raise ValueError(f"{key}: expected one finite target per sample")
            y = y.reshape(-1)
            if task == "classification":
                mapping = {float(v): i for i, v in enumerate(class_values)}
                if any(float(v) not in mapping for v in np.unique(y)):
                    raise ValueError("A validation/test class is absent from the training class mapping")
                self.targets = np.array([mapping[float(v)] for v in y], dtype=np.int64)
            else:
                self.targets = y.astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        features, masks = {}, {}
        for m in MODALITIES:
            valid = self.masks[m][index]
            x = np.array(self.features[m][index], dtype=np.float32, copy=True)
            x[~valid] = 0.0
            x = (x - self.stats[m]["mean"]) / self.stats[m]["std"]
            x[~valid] = 0.0
            features[m], masks[m] = torch.from_numpy(x), torch.from_numpy(valid.copy())
        item = {"features": features, "masks": masks, "id": self.ids[index], "index": index}
        if self.targets is not None:
            dtype = torch.long if self.task == "classification" else torch.float32
            item["target"] = torch.tensor(self.targets[index], dtype=dtype)
        return item


class SequenceBranch(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, config: Config):
        super().__init__()
        d = config.d_model
        self.project = nn.Sequential(nn.Linear(input_dim, d), nn.LayerNorm(d), nn.GELU())
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        position = torch.arange(max_length + 1).unsqueeze(1).float()
        scale = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(max_length + 1, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("position", pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.nhead, dim_feedforward=config.ff_dim,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        # TransformerEncoder clones a layer; initialize attention/FFN matrices independently.
        for block in self.encoder.layers:
            for param in block.parameters():
                if param.ndim > 1:
                    nn.init.xavier_uniform_(param)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(config.dropout), nn.Linear(d, output_dim))

    def forward(self, x, valid):
        if x.ndim != 3 or valid.shape != x.shape[:2]:
            raise ValueError("Expected features [B,L,D] and mask [B,L]")
        if x.shape[1] + 1 > self.position.shape[1]:
            raise ValueError("Sequence exceeds configured maximum length")
        valid = valid.bool()
        # Padding values cannot contaminate projection, including masked NaNs.
        x = self.project(x.masked_fill(~valid.unsqueeze(-1), 0.0))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        x = x + self.position[:, :x.shape[1]]
        cls_valid = torch.ones((len(x), 1), dtype=torch.bool, device=x.device)
        padding = ~torch.cat([cls_valid, valid], dim=1)
        h = self.encoder(x, src_key_padding_mask=padding)
        logits = self.head(h[:, 0])
        # Missing-modality adaptation: no observations -> no evidence contribution.
        # CLS remains unmasked internally so an empty branch never produces NaNs.
        return logits * valid.any(dim=1, keepdim=True).to(logits.dtype)


class WeightedSumFusion(nn.Module):
    def __init__(self, feature_dims: Mapping[str, int], output_dim: int,
                 max_length: int, config: Config):
        super().__init__()
        config.validate()
        self.branches = nn.ModuleDict({
            m: SequenceBranch(feature_dims[m], output_dim, max_length, config) for m in MODALITIES
        })
        self.weighted_fusion = nn.Linear(3, 1, bias=False)
        nn.init.constant_(self.weighted_fusion.weight, 1.0 / 3.0)

    def forward(self, features, masks):
        available = torch.stack([masks[m].bool().any(dim=1) for m in MODALITIES], dim=-1)
        if not available.any(dim=1).all():
            raise ValueError("Cannot predict a sample with all three modalities unobserved")
        per_modality = torch.stack([self.branches[m](features[m], masks[m]) for m in MODALITIES], dim=-1)
        # [B,C,3] -> [B,C,1] -> [B,C]. These are logits, not probabilities.
        fused = self.weighted_fusion(per_modality).squeeze(-1)
        weights = self.weighted_fusion.weight[0]
        return {"logits": fused, "modality_logits": per_modality,
                "weighted_logits": per_modality * weights.view(1, 1, 3),
                "weights": weights}


def classification_metrics(y, pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y.astype(int), pred.astype(int)), 1)
    tp = cm.diagonal().astype(float)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    support = cm.sum(axis=1)
    return {"accuracy": float(tp.sum() / len(y)), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.dot(f1, support) / len(y)),
            "per_class_f1": f1.tolist(), "confusion_matrix": cm.tolist()}


def regression_metrics(y, pred):
    corr = None
    if len(y) > 1 and np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {"mae": float(np.mean(np.abs(pred - y))),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "pearson": corr}


def _to_device(batch, device):
    return ({m: batch["features"][m].to(device) for m in MODALITIES},
            {m: batch["masks"][m].to(device) for m in MODALITIES})


def _loss(logits, target, criterion, task):
    # Explicit squeeze avoids [B,1] vs [B] broadcasting in regression.
    return criterion(logits if task == "classification" else logits.squeeze(-1), target)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    chunks = {k: [] for k in ("logits", "modality_logits", "weighted_logits")}
    ids, indices, targets = [], [], []
    for batch in loader:
        features, masks = _to_device(batch, device)
        output = model(features, masks)
        for k in chunks:
            chunks[k].append(output[k].cpu().numpy())
        ids.extend(batch["id"])
        indices.extend(batch["index"].tolist())
        if "target" in batch:
            targets.append(batch["target"].numpy())
    if not ids:
        raise ValueError("Cannot evaluate an empty split")
    result = {k: np.concatenate(v) for k, v in chunks.items()}
    result.update(ids=np.asarray(ids, dtype=str), indices=np.asarray(indices),
                  weights=model.weighted_fusion.weight[0].detach().cpu().numpy().copy())
    if targets:
        result["targets"] = np.concatenate(targets)
    return result


def _score_prediction(prediction, config, n_classes, criterion, device):
    y, logits = prediction["targets"], prediction["logits"]
    if config.task == "classification":
        metrics = classification_metrics(y, logits.argmax(axis=1), n_classes)
    else:
        metrics = regression_metrics(y, logits[:, 0])
    target = torch.as_tensor(y, device=device)
    metrics["loss"] = float(_loss(torch.as_tensor(logits, device=device), target, criterion, config.task))
    return metrics


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export_predictions(prediction, run_dir, task, class_values):
    """Exact score decomposition, NOT modality importance percentages."""
    run_dir = Path(run_dir)
    np.savez_compressed(run_dir / "test_predictions.npz", **prediction,
                        modality_order=np.asarray(MODALITIES), class_values=np.asarray(class_values))
    logits, weighted = prediction["logits"], prediction["weighted_logits"]
    rows = []
    for i, sample_id in enumerate(prediction["ids"]):
        row = {"id": sample_id, "index": int(prediction["indices"][i])}
        if task == "classification":
            order = np.argsort(logits[i])
            winner, runner = int(order[-1]), int(order[-2])
            row.update(predicted_label=class_values[winner],
                       true_label=class_values[int(prediction["targets"][i])],
                       reference_label=class_values[runner],
                       fused_margin=float(logits[i, winner] - logits[i, runner]))
            for j, m in enumerate(MODALITIES):
                row[f"{m}_margin_contribution"] = float(weighted[i, winner, j] - weighted[i, runner, j])
        else:
            row.update(prediction=float(logits[i, 0]), target=float(prediction["targets"][i]))
            for j, m in enumerate(MODALITIES):
                row[f"{m}_score_contribution"] = float(weighted[i, 0, j])
        rows.append(row)
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Experiment:
    model: WeightedSumFusion
    config: Config
    run_dir: Path
    datasets: dict[str, FeatureDataset]
    stats: dict
    class_values: list
    history: list
    valid_metrics: dict
    test_metrics: dict
    test_predictions: dict


def fit_experiment(data, config=None, mask_overrides=None):
    """Call after the user's existing `data = ...` loading code.

    Only TRAIN updates parameters/statistics. VALID selects the checkpoint.
    TEST is evaluated once after loading the selected checkpoint.
    """
    config = config or Config()
    config.validate()
    seed_everything(config.seed)

    #### ==============================开始统计数据信息=================================================
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    mask_overrides = mask_overrides or {}
    report = audit_data(data, mask_overrides)
    masks = {s: resolve_masks(data[s], mask_overrides.get(s)) for s in ("train", "valid", "test")}
    stats = fit_normalizers(data["train"], masks["train"], config.standardize_av)
    class_values = []
    if config.task == "classification":
        y = np.asarray(data["train"]["classification_labels"])
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite training labels")
        class_values = [float(v) for v in np.unique(y)]
        if len(class_values) < 2:
            raise ValueError("Classification needs at least two training classes")
    output_dim = len(class_values) if config.task == "classification" else 1

    # ============= 这里开始把原始数据变成 tensor，然后构建成 DataLoader 形式=================================
    datasets = {s: FeatureDataset(data[s], masks[s], stats, config.task, class_values)
                for s in ("train", "valid", "test")}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError("train, valid and test must all be nonempty")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {s: DataLoader(ds, batch_size=config.batch_size, shuffle=s == "train",
                            num_workers=0, generator=generator if s == "train" else None)
               for s, ds in datasets.items()}
    dims = {m: int(np.asarray(data["train"][m]).shape[-1]) for m in MODALITIES}
    max_length = int(np.asarray(data["train"]["text"]).shape[1])
    
    for s in ("valid", "test"):
        if np.asarray(data[s]["text"]).shape[1] > max_length:
            raise ValueError("Validation/test sequence length exceeds training max_length")
    
    model = WeightedSumFusion(dims, output_dim, max_length, config).to(device)
    class_weights = None
    if config.task == "classification" and config.class_weighted_loss:
        counts = np.bincount(datasets["train"].targets, minlength=output_dim)
        class_weights = torch.tensor(len(datasets["train"]) / (output_dim * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if config.task == "classification" else nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    run_dir = Path(config.output_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / "config.json", {**asdict(config), "actual_device": str(device),
          "feature_dims": dims, "class_values": class_values, "modality_order": list(MODALITIES),
          "fusion_source": SOURCE, "mask_assumption": "shared text slot layout unless overridden"})
    _json(run_dir / "data_audit.json", report)
    history, best_score, stale = [], -float("inf"), 0
    print(f"device={device}; task={config.task}; classes={class_values}; output={run_dir}")
    print("Fusion order: text/audio/vision. Raw global weights are NOT importance percentages.")
    print("Default A/V masks share text positions; zero rows remain observations unless explicitly masked.")
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loaders["train"]:
            features, valid = _to_device(batch, device)
            
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(features, valid)
            loss = _loss(output["logits"], target, criterion, config.task)

            if config.auxiliary_loss_weight:
                terms = []
                for j, m in enumerate(MODALITIES):
                    observed = valid[m].any(dim=1)
                    if observed.any():
                        terms.append(_loss(output["modality_logits"][observed, :, j], target[observed], criterion, config.task))
                if terms:
                    loss = loss + config.auxiliary_loss_weight * torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inspect feature scale")
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        
        val_prediction = predict_loader(model, loaders["valid"], device)
        val_metrics = _score_prediction(val_prediction, config, output_dim, criterion, device)
        
        score = val_metrics["macro_f1"] if config.task == "classification" else -val_metrics["mae"]
        record = {"epoch": epoch, "train_loss": total / seen, "valid": val_metrics,
                  "weights": model.weighted_fusion.weight[0].detach().cpu().tolist()}
        history.append(record)
        name = "macro_f1" if config.task == "classification" else "mae"
        print(f"epoch {epoch:02d} | train_loss={record['train_loss']:.4f} | valid_{name}={val_metrics[name]:.4f} | w={np.round(record['weights'], 3)}")
        
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            checkpoint = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(config), "feature_dims": dims, "output_dim": output_dim,
                "max_length": max_length, "class_values": class_values,
                "normalizers": {m: {"mean": torch.from_numpy(stats[m]["mean"]),
                                    "std": torch.from_numpy(stats[m]["std"]),
                                    "observed_count": stats[m]["observed_count"]} for m in MODALITIES},
                "epoch": epoch, "valid_metrics": val_metrics,
                "modality_order": list(MODALITIES), "fusion_source": SOURCE,
            }
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale += 1
        if stale >= config.patience:
            print(f"Early stopping after {epoch} epochs")
            break
    
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)

    model.load_state_dict(best["state_dict"])
    model.eval()
    
    # The test set is not consulted inside the training/selection loop.
    test_prediction = predict_loader(model, loaders["test"], device)
    test_metrics = _score_prediction(test_prediction, config, output_dim, criterion, device)
    
    _json(run_dir / "history.json", history)
    _json(run_dir / "metrics.json", {"selected_epoch": best["epoch"],
          "valid": best["valid_metrics"], "test": test_metrics})
    export_predictions(test_prediction, run_dir, config.task, class_values)
    print("Selected epoch:", best["epoch"], "Test:", test_metrics)
    
    return Experiment(model, config, run_dir, datasets, stats, class_values, history,
                      best["valid_metrics"], test_metrics, test_prediction)


def load_experiment_model(checkpoint_path, device=None):
    """Reload model plus training normalization; no optimizer/training resume."""
    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config(**meta["config"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = WeightedSumFusion(meta["feature_dims"], meta["output_dim"], meta["max_length"], config)
    model.load_state_dict(meta["state_dict"])
    model.to(device).eval()
    meta["normalizers"] = {m: {"mean": v["mean"].numpy(), "std": v["std"].numpy(),
                                "observed_count": v["observed_count"]}
                           for m, v in meta["normalizers"].items()}
    return model, meta


def predict_split(model, split, metadata, masks=None, batch_size=32):
    """Inference may omit labels. Feature preprocessing matches the checkpoint."""
    ds = FeatureDataset(split, resolve_masks(split, masks), metadata["normalizers"],
                        metadata["config"]["task"], metadata["class_values"], require_labels=False)
    return predict_loader(model, DataLoader(ds, batch_size=batch_size, num_workers=0), next(model.parameters()).device)


@torch.no_grad()
def explain_feature_groups(experiment: Experiment, split="test", index=0,
                           target_class=None, baseline=0.0, perturb_batch_size=32):
    """Feature-space group ablation; NOT a raw-word/audio/pixel causal claim.

    Each group is one modality at one original time index. Mask is unchanged:
    replace content, not sequence length. baseline=0 means zero in MODEL input
    space (training mean for standardized A/V; zero latent vector for text).
    Classification uses a fixed target-vs-reference logit margin. BERT raw-word
    validation requires editing raw text and re-encoding, outside this function.
    """
    if perturb_batch_size <= 0:
        raise ValueError("perturb_batch_size must be positive")
    model, ds = experiment.model, experiment.datasets[split]
    device = next(model.parameters()).device
    item = ds[index]
    features = {m: item["features"][m].unsqueeze(0).to(device) for m in MODALITIES}
    masks = {m: item["masks"][m].unsqueeze(0).to(device) for m in MODALITIES}
    was_training = model.training
    model.eval()
    try:
        original = model(features, masks)
        if experiment.config.task == "classification":
            scores = original["logits"][0]
            target = int(scores.argmax()) if target_class is None else experiment.class_values.index(float(target_class))
            candidates = scores.clone()
            candidates[target] = -torch.inf
            reference = int(candidates.argmax())
            def score(output):
                return output["logits"][:, target] - output["logits"][:, reference]
        else:
            target, reference = 0, None
            def score(output):
                return output["logits"][:, 0]
        original_score = float(score(original)[0])
        groups = [(m, t) for m in MODALITIES for t in range(features[m].shape[1]) if bool(masks[m][0, t])]
        deltas = np.full((features["text"].shape[1], 3), np.nan, dtype=np.float32)
        rows = []
        token_ids = np.asarray(ds.split["text_bert"])[index, 0].astype(int)
        for start in range(0, len(groups), perturb_batch_size):
            chunk = groups[start:start + perturb_batch_size]
            size = len(chunk)
            xx = {m: features[m].expand(size, -1, -1).clone() for m in MODALITIES}
            mm = {m: masks[m].expand(size, -1) for m in MODALITIES}
            for j, (m, t) in enumerate(chunk):
                xx[m][j, t, :] = baseline
            changed = score(model(xx, mm)).cpu().numpy()
            for j, (m, t) in enumerate(chunk):
                delta = original_score - float(changed[j])
                deltas[t, MODALITIES.index(m)] = delta
                rows.append({"id": item["id"], "time_index": t, "modality": m,
                             "aligned_text_token_id": int(token_ids[t]), "score_drop": delta,
                             "original_score": original_score, "perturbed_score": float(changed[j])})
        rows.sort(key=lambda row: abs(row["score_drop"]), reverse=True)
        return {"id": item["id"], "index": index, "split": split,
                "target_class": experiment.class_values[target] if reference is not None else None,
                "reference_class": experiment.class_values[reference] if reference is not None else None,
                "baseline_in_model_space": baseline, "original_score": original_score,
                "modality_order": MODALITIES, "scores": deltas, "rows": rows,
                "scope": "cached_feature_group_ablation"}
    finally:
        model.train(was_training)

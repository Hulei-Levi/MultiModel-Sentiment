"""Baseline A: the same small feature backbone trained separately per task.

This is a controlled baseline, not a paper reproduction or a joint-task model.
Each modality uses a projection and one masked GRU. Outputs are restored to
the original aligned slots, concatenated, projected, and mean pooled. The
classification and regression runs differ only in task supervision, prediction
head size, and validation selection metric (Macro-F1 versus MAE by default).

Data handling and feature-group ablation reuse the existing project contract.
No raw feature extraction, pretrained-model download, or new dependency is
needed. Requires the existing multi_fusion_model and multimodal_suite packages.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multi_fusion_model.weighted_sum_fusion import (
    MODALITIES, FeatureDataset, audit_data, classification_metrics,
    explain_feature_groups as _explain_feature_groups,
    fit_normalizers, regression_metrics, resolve_masks, seed_everything,
)
from multimodal_suite.common import MaskedRNN, masked_mean, validate_inputs

MODEL_KIND = "lightweight_aligned_single_task_baseline_v1"


@dataclass
class Config:
    task: str = "classification"
    d_model: int = 64
    hidden_dim: int = 32
    fusion_dim: int = 64
    bidirectional: bool = True
    dropout: float = 0.2
    pooling: str = "mean"
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    standardize_av: bool = True
    class_weighted_loss: bool = False
    selection_metric: str = "auto"
    device: str | None = None
    output_root: str = "results/second_question/lightweight_baseline"

    @property
    def resolved_selection_metric(self):
        if self.selection_metric == "auto":
            return "macro_f1" if self.task == "classification" else "mae"
        return self.selection_metric

    def validate(self):
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        for name in ("d_model", "hidden_dim", "fusion_dim", "batch_size", "epochs", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("bidirectional", "standardize_av", "class_weighted_loss"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("dropout", "lr", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.dropout < 1 or self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid dropout or optimizer settings")
        if self.pooling != "mean":
            raise ValueError("Baseline A supports pooling='mean' only")
        allowed = {"auto", "accuracy", "macro_f1"} if self.task == "classification" else {"auto", "mae"}
        if self.selection_metric not in allowed:
            raise ValueError(f"selection_metric for {self.task} must be one of {sorted(allowed)}")
        if self.task == "regression" and self.class_weighted_loss:
            raise ValueError("class_weighted_loss applies only to classification")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be a nonempty path string")
        if self.device is not None and not isinstance(self.device, str):
            raise ValueError("device must be a device string or None")


class SequenceBranch(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, config.d_model), nn.LayerNorm(config.d_model),
            nn.GELU(), nn.Dropout(config.dropout),
        )
        self.encoder = MaskedRNN(config.d_model, config.hidden_dim, kind="gru",
                                 bidirectional=config.bidirectional, num_layers=1)

    def forward(self, x, mask):
        valid = mask.bool().unsqueeze(-1)
        projected = self.project(x.masked_fill(~valid, 0)).masked_fill(~valid, 0)
        return self.encoder(projected, mask)


class LightweightBaseline(nn.Module):
    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        config.validate()
        if set(feature_dims) != set(MODALITIES) or any(int(v) <= 0 for v in feature_dims.values()):
            raise ValueError("feature_dims must contain positive text/audio/vision dimensions")
        if max_length <= 0 or output_dim <= 0:
            raise ValueError("max_length and output_dim must be positive")
        if config.task == "regression" and output_dim != 1:
            raise ValueError("Regression requires exactly one unbounded output")
        if config.task == "classification" and output_dim < 2:
            raise ValueError("Classification requires at least two outputs")
        self.feature_dims, self.max_length = dict(feature_dims), max_length
        # All backbone layers are constructed before the task-dependent head.
        self.branches = nn.ModuleDict({m: SequenceBranch(feature_dims[m], config) for m in MODALITIES})
        width = config.hidden_dim * (2 if config.bidirectional else 1)
        self.fusion = nn.Sequential(nn.Linear(3 * width, config.fusion_dim),
                                    nn.GELU(), nn.Dropout(config.dropout))
        self.head = nn.Linear(config.fusion_dim, output_dim)

    def forward(self, features, masks):
        union = validate_inputs(features, masks, self.feature_dims, self.max_length)
        sequences = {m: self.branches[m](features[m], masks[m]) for m in MODALITIES}
        concatenated = torch.cat([sequences[m] for m in MODALITIES], dim=-1)
        fused = self.fusion(concatenated).masked_fill(~union.unsqueeze(-1), 0)
        pooled = masked_mean(fused, union)
        return {"logits": self.head(pooled), "unimodal_sequences": sequences,
                "concatenated": concatenated, "fusion_sequence": fused,
                "fusion_mask": union, "pooled_features": pooled}


Model = LightweightBaseline


def backbone_state_hash(model):
    """SHA256 of all backbone parameters/buffers, excluding the prediction head."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("head."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return {"total": total, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "backbone": total - head, "head": head}


def _to_device(batch, device):
    return ({m: batch["features"][m].to(device) for m in MODALITIES},
            {m: batch["masks"][m].to(device) for m in MODALITIES})


def _loss(logits, target, criterion, task):
    # Explicit squeezing avoids accidental [B,1] - [B] -> [B,B] broadcasting.
    return criterion(logits if task == "classification" else logits.squeeze(-1), target)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    logits, pooled, targets, ids, indices = [], [], [], [], []
    for batch in loader:
        xx, mm = _to_device(batch, device)
        output = model(xx, mm)
        if not torch.isfinite(output["logits"]).all():
            raise FloatingPointError("Nonfinite model predictions")
        logits.append(output["logits"].cpu().numpy())
        pooled.append(output["pooled_features"].cpu().numpy())
        ids.extend(batch["id"])
        indices.extend(batch["index"].tolist())
        if "target" in batch:
            targets.append(batch["target"].numpy())
    if not logits:
        raise ValueError("Cannot evaluate an empty split")
    result = {"logits": np.concatenate(logits), "pooled_features": np.concatenate(pooled),
              "ids": np.asarray(ids, dtype=str), "indices": np.asarray(indices, dtype=np.int64)}
    if targets:
        result["targets"] = np.concatenate(targets)
    return result


def _metrics(prediction, config, output_dim, criterion, device):
    y, z = prediction["targets"], prediction["logits"]
    result = (classification_metrics(y, z.argmax(1), output_dim) if config.task == "classification"
              else regression_metrics(y, z[:, 0]))
    result["loss"] = float(_loss(torch.as_tensor(z, device=device), torch.as_tensor(y, device=device),
                                 criterion, config.task).detach().cpu())
    return result


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _export_predictions(prediction, run_dir, split_name, task, class_values):
    np.savez_compressed(run_dir / f"{split_name}_predictions.npz", **prediction,
                        class_values=np.asarray(class_values), modality_order=np.asarray(MODALITIES))
    rows = []
    for i, sample_id in enumerate(prediction["ids"]):
        row = {"id": sample_id, "index": int(prediction["indices"][i])}
        z = prediction["logits"][i]
        if task == "classification":
            order = np.argsort(z)
            winner, runner = int(order[-1]), int(order[-2])
            row.update(predicted_label=class_values[winner], reference_label=class_values[runner],
                       margin=float(z[winner] - z[runner]))
            if "targets" in prediction:
                row["true_label"] = class_values[int(prediction["targets"][i])]
            row.update({f"logit_class_{j}": float(z[j]) for j in range(len(class_values))})
        else:
            row["prediction"] = float(z[0])
            if "targets" in prediction:
                row["target"] = float(prediction["targets"][i])
        rows.append(row)
    with (run_dir / f"{split_name}_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Experiment:
    model: LightweightBaseline
    config: Config
    run_dir: Path
    datasets: dict
    stats: dict
    class_values: list
    history: list
    train_metrics: dict
    valid_metrics: dict
    test_metrics: dict
    test_predictions: dict
    selected_epoch: int
    initial_backbone_sha256: str
    parameter_counts: dict


def fit_experiment(data, config=None, mask_overrides=None):
    """One task per run. Train-only preprocessing, validation selection, test once.

    Use the same architecture/seed settings in two calls with different `task`
    values for Baseline A. The backbone initialization hashes should match;
    fitted weights and selected epochs are expected to differ between tasks.
    """
    config = config or Config()
    config.validate()
    seed_everything(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    mask_overrides = mask_overrides or {}
    if set(mask_overrides) - {"train", "valid", "test"}:
        raise ValueError("Unknown split in mask_overrides")
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
            raise ValueError("Classification requires at least two training classes")
    output_dim = len(class_values) if config.task == "classification" else 1
    datasets = {s: FeatureDataset(data[s], masks[s], stats, config.task, class_values)
                for s in ("train", "valid", "test")}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError("train, valid and test must be nonempty")
    dims = {m: int(np.asarray(data["train"][m]).shape[-1]) for m in MODALITIES}
    max_length = int(np.asarray(data["train"]["text"]).shape[1])
    if any(np.asarray(data[s]["text"]).shape[1] > max_length for s in ("valid", "test")):
        raise ValueError("Validation/test sequence length exceeds training max_length")
    model = LightweightBaseline(dims, output_dim, max_length, config).to(device)
    initial_hash = backbone_state_hash(model)
    counts = parameter_counts(model)
    # Head sizes consume different numbers of random draws during construction.
    # Reset before training; each loader also has its own independent RNG.
    seed_everything(config.seed)
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(config.seed))
    evaluation_loaders = {
        s: DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed + i + 1))
        for i, (s, ds) in enumerate(datasets.items())
    }
    class_weights = None
    if config.task == "classification" and config.class_weighted_loss:
        frequencies = np.bincount(datasets["train"].targets, minlength=output_dim)
        class_weights = torch.tensor(len(datasets["train"]) / (output_dim * frequencies),
                                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if config.task == "classification" else nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    run_dir = (Path(config.output_root).expanduser().resolve() / config.task /
               datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    common_metadata = {
        "model_kind": MODEL_KIND, "feature_dims": dims, "output_dim": output_dim,
        "max_length": max_length, "class_values": class_values, "modality_order": list(MODALITIES),
        "initial_backbone_sha256": initial_hash, "parameter_counts": counts,
        "resolved_selection_metric": config.resolved_selection_metric,
        "sequence_unit": "within_sample_aligned_slot",
        "mask_assumption": "shared text slot layout including special tokens unless overridden",
    }
    _json(run_dir / "config.json", {**asdict(config), **common_metadata, "actual_device": str(device)})
    _json(run_dir / "data_audit.json", report)
    print(f"Baseline A | device={device} | task={config.task} | parameters={counts['total']:,}")
    print(f"Validation selection: {config.resolved_selection_metric}; output={run_dir}")
    print("A/V masks follow text slots unless overridden; zero rows remain observations.")
    history, best_score, stale = [], -float("inf"), 0
    best_state, selected = None, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            xx, mm = _to_device(batch, device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(xx, mm)["logits"], target, criterion, config.task)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inspect observed inputs and scales")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        valid_prediction = predict_loader(model, evaluation_loaders["valid"], device)
        valid_metrics = _metrics(valid_prediction, config, output_dim, criterion, device)
        name = config.resolved_selection_metric
        score = -valid_metrics[name] if name == "mae" else valid_metrics[name]
        record = {"stage": "single_task", "epoch": epoch, "train_loss": total / seen,
                  "train": {"total": total / seen}, "valid": valid_metrics}
        history.append(record)
        print(f"epoch {epoch:02d} | train_loss={record['train_loss']:.4f} | valid_{name}={valid_metrics[name]:.4f}")
        if score > best_score:
            best_score, stale = score, 0
            best_state = _cpu_state(model)
            selected = {"epoch": epoch, "metric": name, "valid_metrics": valid_metrics}
        else:
            stale += 1
        _json(run_dir / "history.json", history)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {
        **common_metadata, "state_dict": best_state, "config": asdict(config), "selected": selected,
        "normalizers": {m: {"mean": torch.from_numpy(stats[m]["mean"].copy()),
                            "std": torch.from_numpy(stats[m]["std"].copy()),
                            "observed_count": stats[m]["observed_count"]} for m in MODALITIES},
    }
    torch.save(checkpoint, run_dir / "best.pt")
    # Train metrics below use eval mode and the selected weights, not a mixture
    # of changing/dropout-active weights from the training-loss curve.
    train_prediction = predict_loader(model, evaluation_loaders["train"], device)
    train_metrics = _metrics(train_prediction, config, output_dim, criterion, device)
    valid_prediction = predict_loader(model, evaluation_loaders["valid"], device)
    valid_metrics = _metrics(valid_prediction, config, output_dim, criterion, device)
    # The test loader is traversed exactly once, after all model selection.
    test_prediction = predict_loader(model, evaluation_loaders["test"], device)
    test_metrics = _metrics(test_prediction, config, output_dim, criterion, device)
    _json(run_dir / "metrics.json", {
        "selected_epoch": selected["epoch"], "selection_metric": config.resolved_selection_metric,
        "train": train_metrics, "valid": valid_metrics, "test": test_metrics,
        "initial_backbone_sha256": initial_hash, "parameter_counts": counts,
    })
    _export_predictions(valid_prediction, run_dir, "valid", config.task, class_values)
    _export_predictions(test_prediction, run_dir, "test", config.task, class_values)
    print(f"Selected epoch: {selected['epoch']} | Test: {test_metrics}")
    return Experiment(model=model, config=config, run_dir=run_dir, datasets=datasets, stats=stats,
                      class_values=class_values, history=history, train_metrics=train_metrics,
                      valid_metrics=valid_metrics, test_metrics=test_metrics, test_predictions=test_prediction,
                      selected_epoch=selected["epoch"], initial_backbone_sha256=initial_hash,
                      parameter_counts=counts)


def load_experiment_model(checkpoint_path, device=None):
    """Load this module's complete best.pt; returns (model, metadata)."""
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if metadata.get("model_kind") != MODEL_KIND:
        raise ValueError("Expected a complete lightweight_baseline best.pt checkpoint")
    config = Config(**metadata["config"])
    model = LightweightBaseline(metadata["feature_dims"], metadata["output_dim"], metadata["max_length"], config)
    model.load_state_dict(metadata["state_dict"])
    model.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))).eval()
    metadata["normalizers"] = {
        m: {"mean": values["mean"].numpy(), "std": values["std"].numpy(),
            "observed_count": values["observed_count"]}
        for m, values in metadata["normalizers"].items()
    }
    return model, metadata


def predict_split(model, split, metadata, masks=None, batch_size=32):
    """Predict in original sample order; labels are optional and never model inputs.

    Pass custom masks again if they were used in training. No statistics are
    fitted on this split; all normalization comes from the saved train split.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    valid = resolve_masks(split, masks)
    for m in MODALITIES:
        x = np.asarray(split[m])
        if x.ndim != 3 or x.shape[:2] != valid[m].shape:
            raise ValueError(f"Bad {m} feature shape")
        for start in range(0, len(x), 128):
            if not np.isfinite(x[start:start + 128][valid[m][start:start + 128]]).all():
                raise ValueError(f"Nonfinite observed input in {m}")
    ds = FeatureDataset(split, valid, metadata["normalizers"], metadata["config"]["task"],
                        metadata["class_values"], require_labels=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    return predict_loader(model, loader, next(model.parameters()).device)


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32):
    """Original-slot cached feature ablation, not raw-input causal attribution.

    Classification uses a fixed target-versus-reference logit margin; regression
    uses the scalar output. Deltas are not additive fusion contributions.
    Zero baseline is the training mean for standardized A/V and a zero latent
    vector for text. Sequence masks and original indices are held unchanged.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)


__all__ = ["Config", "Experiment", "Model", "LightweightBaseline", "fit_experiment",
           "load_experiment_model", "predict_split", "explain_feature_groups",
           "audit_data", "backbone_state_hash", "parameter_counts"]

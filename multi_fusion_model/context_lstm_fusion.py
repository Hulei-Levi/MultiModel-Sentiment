"""Poria-style hierarchical contextual LSTM adapted to aligned feature sequences.

Paper: https://aclanthology.org/P17-1081/ (Sections 3.2 and 3.3.2).
The paper's time steps are utterances within a video. Here they are the user's
50 within-sample slots, with ONE label per sample. Frozen feature inputs,
masked pooling and a neural prediction head are adaptations; this is not a
reproduction of the paper's utterance-context/SVM evaluation pipeline.

Keep weighted_sum_fusion.py alongside this file: its data preprocessing,
metrics and cached-feature ablation code are shared, not its fusion model.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from multi_fusion_model.weighted_sum_fusion import (
    MODALITIES, FeatureDataset, audit_data, classification_metrics,
    explain_feature_groups as _explain_feature_groups,
    fit_normalizers, regression_metrics, resolve_masks, seed_everything,
)

SOURCE = "https://aclanthology.org/P17-1081/"
MODEL_KIND = "poria_hierarchical_lstm_within_sample_v1"


@dataclass
class Config:
    task: str = "classification"
    hidden_dim: int = 128
    context_dim: int = 128
    fusion_hidden_dim: int = 128
    fusion_context_dim: int = 128
    unimodal_layers: int = 1
    fusion_layers: int = 1
    bidirectional: bool = True  # bc-LSTM; False selects sc-LSTM style.
    dropout: float = 0.2
    pooling: str = "mean"  # masked mean, or last observed contextual output
    training_mode: str = "staged"  # independent unimodal training -> frozen -> fusion
    pretrain_epochs: int = 20
    pretrain_patience: int = 5
    epochs: int = 40  # fusion stage (or the single joint stage)
    patience: int = 8
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    standardize_av: bool = True
    class_weighted_loss: bool = False
    device: str | None = None
    output_root: str = "context_lstm_runs"

    def validate(self):
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        if self.training_mode not in ("staged", "joint"):
            raise ValueError("training_mode must be staged or joint")
        if self.pooling not in ("mean", "last"):
            raise ValueError("pooling must be mean or last")
        if min(self.hidden_dim, self.context_dim, self.fusion_hidden_dim,
               self.fusion_context_dim, self.unimodal_layers, self.fusion_layers,
               self.epochs, self.patience, self.batch_size) <= 0:
            raise ValueError("Model sizes, layers, epochs, patience and batch_size must be positive")
        if self.training_mode == "staged" and min(self.pretrain_epochs, self.pretrain_patience) <= 0:
            raise ValueError("Staged training requires positive pretrain_epochs/pretrain_patience")
        if not 0 <= self.dropout < 1 or self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid dropout or optimizer settings")


class MaskedLSTM(nn.Module):
    """Run an LSTM over observed slots, then restore the ORIGINAL slot indices.

    Unlike passing zeros into an ordinary LSTM, invalid slots never update
    recurrent states. Arbitrary interior gaps and entirely absent branches are
    supported. Skipping a gap does not model the duration of that gap.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x, valid):
        if x.ndim != 3 or valid.shape != x.shape[:2] or x.shape[1] == 0:
            raise ValueError("Expected nonempty features [B,L,D] and mask [B,L]")
        valid = valid.bool()
        clean = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        # Keep a zero-gradient path to even an entirely absent input branch.
        output = clean.new_zeros((*clean.shape[:2], self.output_dim)) + clean.sum(-1, keepdim=True) * 0.0
        lengths = valid.sum(1)
        active = (lengths > 0).nonzero(as_tuple=True)[0]
        if len(active) == 0:
            return output
        xx, vv = clean.index_select(0, active), valid.index_select(0, active)
        positions = torch.arange(x.shape[1], device=x.device).expand(len(active), -1)
        # Valid indices are unique and ordered; invalid indices sort to the end.
        order = positions.masked_fill(~vv, x.shape[1]).argsort(dim=1)
        compact = xx.gather(1, order.unsqueeze(-1).expand(-1, -1, xx.shape[-1]))
        packed = pack_padded_sequence(compact, lengths[active].cpu(), batch_first=True,
                                      enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        dense, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        restored = torch.zeros_like(dense).scatter(
            1, order.unsqueeze(-1).expand(-1, -1, self.output_dim), dense)
        restored = restored.masked_fill(~vv.unsqueeze(-1), 0.0)
        return output.index_copy(0, active, restored)


def masked_pool(sequence, valid, pooling="mean"):
    valid = valid.bool()
    if pooling == "mean":
        return sequence.masked_fill(~valid.unsqueeze(-1), 0.0).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    if pooling == "last":
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand(valid.shape)
        last = positions.masked_fill(~valid, -1).max(1).values.clamp_min(0)
        pooled = sequence[torch.arange(len(sequence), device=sequence.device), last]
        return pooled * valid.any(1, keepdim=True).to(sequence.dtype)
    raise ValueError("Unknown pooling")


class ContextBranch(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.encoder = MaskedLSTM(input_dim, config.hidden_dim, config.unimodal_layers,
                                  config.bidirectional, config.dropout)
        # The paper forwards the dense ReLU activations, not class scores.
        self.context = nn.Sequential(nn.Dropout(config.dropout),
                                     nn.Linear(self.encoder.output_dim, config.context_dim), nn.ReLU())
        self.head = nn.Linear(config.context_dim, output_dim)
        self.pooling = config.pooling

    def forward(self, x, valid):
        sequence = self.context(self.encoder(x, valid))
        return sequence.masked_fill(~valid.bool().unsqueeze(-1), 0.0)

    def predict(self, x, valid):
        sequence = self(x, valid)
        logits = self.head(masked_pool(sequence, valid, self.pooling))
        return logits * valid.bool().any(1, keepdim=True).to(logits.dtype)


class ContextLSTMFusion(nn.Module):
    def __init__(self, feature_dims, output_dim, max_length, config):
        super().__init__()
        config.validate()
        self.feature_dims, self.max_length, self.pooling = dict(feature_dims), max_length, config.pooling
        self.branches = nn.ModuleDict({m: ContextBranch(feature_dims[m], output_dim, config)
                                       for m in MODALITIES})
        self.fusion_encoder = MaskedLSTM(3 * config.context_dim, config.fusion_hidden_dim,
                                         config.fusion_layers, config.bidirectional, config.dropout)
        self.fusion_context = nn.Sequential(nn.Dropout(config.dropout),
                                            nn.Linear(self.fusion_encoder.output_dim, config.fusion_context_dim),
                                            nn.ReLU())
        self.head = nn.Linear(config.fusion_context_dim, output_dim)
        self.unimodal_frozen = False

    def freeze_unimodal(self, freeze=True):
        self.unimodal_frozen = freeze
        self.branches.requires_grad_(not freeze)
        self.branches.train(False if freeze else self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.unimodal_frozen:
            self.branches.eval()  # Fixed Level-1 representations, including dropout.
        return self

    def forward(self, features, masks):
        shape = features["text"].shape[:2]
        if len(shape) != 2 or not 0 < shape[1] <= self.max_length:
            raise ValueError("Sequence length must be between 1 and max_length")
        for m in MODALITIES:
            if features[m].shape != (*shape, self.feature_dims[m]) or masks[m].shape != shape:
                raise ValueError(f"Inconsistent feature/mask shape for {m}")
        valid = {m: masks[m].bool() for m in MODALITIES}
        fusion_mask = torch.stack([valid[m] for m in MODALITIES], dim=-1).any(-1)
        if not fusion_mask.any(1).all():
            raise ValueError("Cannot predict a sample with all three modalities unobserved")
        sequences = {m: self.branches[m](features[m], valid[m]) for m in MODALITIES}
        # All sequences are restored to the original slots before concatenation.
        concatenated = torch.cat([sequences[m] for m in MODALITIES], dim=-1)
        fused = self.fusion_context(self.fusion_encoder(concatenated, fusion_mask))
        fused = fused.masked_fill(~fusion_mask.unsqueeze(-1), 0.0)
        pooled = masked_pool(fused, fusion_mask, self.pooling)
        return {"logits": self.head(pooled), "unimodal_sequences": sequences,
                "concatenated": concatenated, "fusion_sequence": fused,
                "fusion_mask": fusion_mask, "pooled_features": pooled}


def _to_device(batch, device):
    return ({m: batch["features"][m].to(device) for m in MODALITIES},
            {m: batch["masks"][m].to(device) for m in MODALITIES})


def _loss(logits, target, criterion, task):
    return criterion(logits if task == "classification" else logits.squeeze(-1), target)


@torch.no_grad()
def predict_loader(model, loader, device, branch=None):
    model.eval()
    logits, pooled, targets, ids, indices = [], [], [], [], []
    for batch in loader:
        xx, mm = _to_device(batch, device)
        keep = mm[branch].any(1) if branch is not None else torch.ones(len(batch["index"]), device=device, dtype=torch.bool)
        if not keep.any():
            continue
        if branch is None:
            output = model(xx, mm)
            zz = output["logits"]
            pooled.append(output["pooled_features"].cpu().numpy())
        else:
            zz = model.branches[branch].predict(xx[branch], mm[branch])
        logits.append(zz[keep].cpu().numpy())
        selection = keep.cpu().numpy()
        ids.extend(np.asarray(batch["id"])[selection].tolist())
        indices.extend(batch["index"][keep.cpu()].tolist())
        if "target" in batch:
            targets.append(batch["target"][keep.cpu()].numpy())
    if not logits:
        raise ValueError(f"No observed samples to evaluate for {branch or 'fusion'}")
    result = {"logits": np.concatenate(logits), "ids": np.asarray(ids, dtype=str),
              "indices": np.asarray(indices)}
    if pooled:
        result["pooled_features"] = np.concatenate(pooled)
    if targets:
        result["targets"] = np.concatenate(targets)
    return result


def _metrics(prediction, config, output_dim, criterion, device):
    y, z = prediction["targets"], prediction["logits"]
    result = (classification_metrics(y, z.argmax(1), output_dim) if config.task == "classification"
              else regression_metrics(y, z[:, 0]))
    result["loss"] = float(_loss(torch.as_tensor(z, device=device), torch.as_tensor(y, device=device),
                                 criterion, config.task))
    return result


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _run_stage(model, loaders, criterion, config, output_dim, device, run_dir, branch=None):
    stage = f"unimodal_{branch}" if branch is not None else "fusion"
    module = model.branches[branch] if branch is not None else model
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    epochs = config.pretrain_epochs if branch is not None else config.epochs
    patience = config.pretrain_patience if branch is not None else config.patience
    path = run_dir / f"best_{stage}.pt"
    history, best_score, stale = [], -float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loaders["train"]:
            xx, mm = _to_device(batch, device)
            target = batch["target"].to(device)
            keep = mm[branch].any(1) if branch is not None else torch.ones(len(target), device=device, dtype=torch.bool)
            if not keep.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            zz = (model.branches[branch].predict(xx[branch], mm[branch]) if branch is not None
                  else model(xx, mm)["logits"])
            loss = _loss(zz[keep], target[keep], criterion, config.task)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inspect observed inputs and scales")
            loss.backward()
            nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            count = int(keep.sum())
            total += float(loss.detach()) * count
            seen += count
        if not seen:
            raise ValueError(f"No observed training samples for {stage}")
        pred = predict_loader(model, loaders["valid"], device, branch)
        metrics = _metrics(pred, config, output_dim, criterion, device)
        score = metrics["macro_f1"] if config.task == "classification" else -metrics["mae"]
        record = {"stage": stage, "epoch": epoch, "train_loss": total / seen, "valid": metrics}
        history.append(record)
        metric_name = "macro_f1" if config.task == "classification" else "mae"
        print(f"{stage} epoch {epoch:02d} | train_loss={record['train_loss']:.4f} | valid_{metric_name}={metrics[metric_name]:.4f}")
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                        "stage": stage, "epoch": epoch, "valid_metrics": metrics}, path)
        else:
            stale += 1
        if stale >= patience:
            break
    best = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(best["state_dict"])
    model.eval()
    return history, {k: best[k] for k in ("stage", "epoch", "valid_metrics")}


def export_predictions(prediction, run_dir, task, class_values):
    run_dir = Path(run_dir)
    np.savez_compressed(run_dir / "test_predictions.npz", **prediction,
                        class_values=np.asarray(class_values), modality_order=np.asarray(MODALITIES))
    rows = []
    for i, sample_id in enumerate(prediction["ids"]):
        row = {"id": sample_id, "index": int(prediction["indices"][i])}
        z = prediction["logits"][i]
        if task == "classification":
            order = np.argsort(z)
            winner, runner = int(order[-1]), int(order[-2])
            row.update(predicted_label=class_values[winner],
                       true_label=class_values[int(prediction["targets"][i])],
                       reference_label=class_values[runner], margin=float(z[winner] - z[runner]))
            for j in range(len(class_values)):
                row[f"logit_class_{j}"] = float(z[j])
        else:
            row.update(prediction=float(z[0]), target=float(prediction["targets"][i]))
        rows.append(row)
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Experiment:
    model: ContextLSTMFusion
    config: Config
    run_dir: Path
    datasets: dict
    stats: dict
    class_values: list
    history: list
    valid_metrics: dict
    test_metrics: dict
    test_predictions: dict
    pretraining: dict


def fit_experiment(data, config=None, mask_overrides=None):
    """Train on train, select all stages on valid, evaluate test once at the end."""
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
    if config.training_mode == "staged":
        for s in ("train", "valid"):
            for m in MODALITIES:
                if not masks[s][m].any():
                    raise ValueError(f"Staged training requires observed {m} samples in {s}")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {s: DataLoader(ds, batch_size=config.batch_size, shuffle=s == "train", num_workers=0,
                            generator=generator if s == "train" else None) for s, ds in datasets.items()}
    dims = {m: int(np.asarray(data["train"][m]).shape[-1]) for m in MODALITIES}
    max_length = int(np.asarray(data["train"]["text"]).shape[1])
    if any(np.asarray(data[s]["text"]).shape[1] > max_length for s in ("valid", "test")):
        raise ValueError("Validation/test sequence length exceeds training max_length")
    model = ContextLSTMFusion(dims, output_dim, max_length, config).to(device)
    class_weights = None
    if config.task == "classification" and config.class_weighted_loss:
        counts = np.bincount(datasets["train"].targets, minlength=output_dim)
        class_weights = torch.tensor(len(datasets["train"]) / (output_dim * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if config.task == "classification" else nn.L1Loss()
    run_dir = Path(config.output_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / "config.json", {**asdict(config), "model_kind": MODEL_KIND, "actual_device": str(device),
          "feature_dims": dims, "class_values": class_values, "modality_order": list(MODALITIES),
          "source": SOURCE, "sequence_unit": "within_sample_aligned_slot",
          "mask_assumption": "shared text slot layout including special tokens unless overridden"})
    _json(run_dir / "data_audit.json", report)
    print(f"device={device}; mode={config.training_mode}; task={config.task}; output={run_dir}")
    print("Adaptation: within-sample time slots, masked pooling and a neural head; NOT inter-utterance context/SVM.")
    print("Default A/V masks share text positions; zero rows remain observations unless explicitly masked.")
    history, pretraining = [], {}
    if config.training_mode == "staged":
        for m in MODALITIES:
            records, selected = _run_stage(model, loaders, criterion, config, output_dim, device, run_dir, m)
            history.extend(records)
            pretraining[m] = selected
        model.freeze_unimodal()
    records, selected = _run_stage(model, loaders, criterion, config, output_dim, device, run_dir)
    history.extend(records)
    checkpoint = {
        "model_kind": MODEL_KIND, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "config": asdict(config), "feature_dims": dims, "output_dim": output_dim, "max_length": max_length,
        "class_values": class_values, "modality_order": list(MODALITIES), "source": SOURCE,
        "unimodal_frozen": model.unimodal_frozen, "selected": selected, "pretraining": pretraining,
        "normalizers": {m: {"mean": torch.from_numpy(stats[m]["mean"]), "std": torch.from_numpy(stats[m]["std"]),
                            "observed_count": stats[m]["observed_count"]} for m in MODALITIES},
    }
    torch.save(checkpoint, run_dir / "best.pt")
    # No test predictions are made in the preceding optimization/selection stages.
    test_prediction = predict_loader(model, loaders["test"], device)
    test_metrics = _metrics(test_prediction, config, output_dim, criterion, device)
    _json(run_dir / "history.json", history)
    _json(run_dir / "metrics.json", {"pretraining": pretraining, "selected_epoch": selected["epoch"],
          "valid": selected["valid_metrics"], "test": test_metrics})
    export_predictions(test_prediction, run_dir, config.task, class_values)
    print("Selected fusion epoch:", selected["epoch"], "Test:", test_metrics)
    return Experiment(model, config, run_dir, datasets, stats, class_values, history,
                      selected["valid_metrics"], test_metrics, test_prediction, pretraining)


def load_experiment_model(checkpoint_path, device=None):
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if metadata.get("model_kind") != MODEL_KIND:
        raise ValueError("Expected this model's complete best.pt, not a stage checkpoint or Weighted Sum checkpoint")
    config = Config(**metadata["config"])
    model = ContextLSTMFusion(metadata["feature_dims"], metadata["output_dim"], metadata["max_length"], config)
    model.load_state_dict(metadata["state_dict"])
    model.freeze_unimodal(metadata["unimodal_frozen"])
    model.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))).eval()
    metadata["normalizers"] = {m: {"mean": v["mean"].numpy(), "std": v["std"].numpy(),
                                   "observed_count": v["observed_count"]} for m, v in metadata["normalizers"].items()}
    return model, metadata


def predict_split(model, split, metadata, masks=None, batch_size=32):
    """Labels optional; custom masks must be provided again using training conventions."""
    valid = resolve_masks(split, masks)
    for m in MODALITIES:
        x = np.asarray(split[m])
        for start in range(0, len(x), 128):
            if not np.isfinite(x[start:start + 128][valid[m][start:start + 128]]).all():
                raise ValueError(f"Nonfinite observed input in {m}")
    ds = FeatureDataset(split, valid, metadata["normalizers"], metadata["config"]["task"],
                        metadata["class_values"], require_labels=False)
    return predict_loader(model, DataLoader(ds, batch_size=batch_size, num_workers=0), next(model.parameters()).device)


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32):
    """One cached modality/time feature group at a time, with a fixed class margin.

    This nonadditive perturbation result is NOT an exact modality decomposition,
    attention score, or raw-word/audio/frame causal attribution. baseline=0 is
    the training mean for standardized A/V and a zero latent vector for text.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)

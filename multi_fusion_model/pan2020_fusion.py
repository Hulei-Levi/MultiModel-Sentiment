"""Dedicated Pan MMAN training: four independent subnetworks, then late fusion.

The paper's utterance-level context is adapted to aligned within-sample slots
and one target per sample. The dedicated entry enables concatenated skip paths,
two dense prediction layers, and independent input projections. Staged training
selects each subnetwork on validation data, freezes it (including dropout), then
trains only the final fusion layer. Joint training is an explicit alternative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multimodal_suite.attention_models import Pan2020Model
from multi_fusion_model.context_lstm_fusion import (
    _json, _loss, _metrics, _to_device, export_predictions,
)
from multi_fusion_model.weighted_sum_fusion import (
    MODALITIES, FeatureDataset, audit_data,
    explain_feature_groups as _explain_feature_groups, fit_normalizers,
    resolve_masks, seed_everything,
)

SOURCE = "https://www.isca-archive.org/interspeech_2020/pan20b_interspeech.pdf"
MODEL_KIND = "pan_mman_independent_staged_within_sample_v1"
BRANCH_NAMES = ("multimodal", "text", "audio", "vision")


@dataclass
class Config:
    task: str = "classification"
    d_model: int = 128
    dropout: float = 0.2
    lstm_hidden: int = 128
    unimodal_layers: int = 2
    training_mode: str = "staged"
    pretrain_epochs: int = 20
    pretrain_patience: int = 5
    epochs: int = 40
    patience: int = 8
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    branch_loss_weight: float = 1.0
    seed: int = 42
    standardize_av: bool = True
    class_weighted_loss: bool = False
    device: str | None = None
    output_root: str = "results/second_question/pan2020_fusion"

    def validate(self):
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        if self.training_mode not in ("staged", "joint"):
            raise ValueError("training_mode must be staged or joint")
        if min(self.d_model, self.lstm_hidden, self.unimodal_layers, self.epochs,
               self.patience, self.batch_size) <= 0:
            raise ValueError("Model widths, layers, epochs, patience and batch_size must be positive")
        if self.training_mode == "staged" and min(self.pretrain_epochs, self.pretrain_patience) <= 0:
            raise ValueError("Staged training requires positive pretrain_epochs/pretrain_patience")
        numeric = (self.dropout, self.lr, self.weight_decay, self.grad_clip, self.branch_loss_weight)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Dropout, optimizer and loss settings must be finite")
        if not 0 <= self.dropout < 1 or self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid dropout or optimizer settings")
        if self.branch_loss_weight < 0:
            raise ValueError("branch_loss_weight must be nonnegative")


def _backend_config(config):
    return SimpleNamespace(task=config.task, d_model=config.d_model, dropout=config.dropout,
        model_options={"pan_lstm_hidden": config.lstm_hidden,
                       "pan_unimodal_layers": config.unimodal_layers,
                       "pan_independent_branches": True, "pan_paper_layout": True})


class Model(Pan2020Model):
    """Paper-layout backend configured only through the dedicated Pan options."""
    def __init__(self, feature_dims, output_dim, max_length, config):
        config.validate()
        super().__init__(feature_dims, output_dim, max_length, _backend_config(config))
        identities = {}
        for branch in BRANCH_NAMES:
            current = {id(p) for module in self.branch_modules(branch) for p in module.parameters()}
            for previous, seen in identities.items():
                if current & seen:
                    raise RuntimeError(f"Pan independent branches unexpectedly share parameters: {previous}, {branch}")
            identities[branch] = current


def _stage_modules(model, config, branch):
    if branch is not None:
        if branch not in BRANCH_NAMES:
            raise ValueError(f"Unknown Pan branch: {branch}")
        return model.branch_modules(branch)
    return (model.late_fusion,) if config.training_mode == "staged" else (model,)


def _configure_stage(model, config, branch):
    """Select both trainable parameters and module training/dropout modes."""
    if branch is not None:
        model.freeze_subnetworks(False)
        model.requires_grad_(False)
        model.eval()
        for module in model.branch_modules(branch):
            module.requires_grad_(True)
            module.train()
    elif config.training_mode == "staged":
        model.freeze_subnetworks(True)
        model.late_fusion.requires_grad_(True)
        model.train()
    else:
        model.freeze_subnetworks(False)
        model.requires_grad_(True)
        model.train()


@torch.no_grad()
def predict_loader(model, loader, device, branch=None):
    """Evaluate one branch on observed samples, or final predictions on all samples."""
    model.eval()
    logits, targets, ids, indices, branch_outputs = [], [], [], [], []
    for batch in loader:
        xx, mm = _to_device(batch, device)
        if branch is None:
            output = model(xx, mm)
            keep = torch.ones(len(batch["index"]), device=device, dtype=torch.bool)
            diagnostic = output.get("diagnostics", {}).get("branch_logits")
            if diagnostic is not None:
                branch_outputs.append(diagnostic.detach().cpu().numpy())
        else:
            output = model.forward_branch(xx, mm, branch)
            keep = output["observed"].bool()
        if not keep.any():
            continue
        selected_logits = output["logits"][keep]
        if not torch.isfinite(selected_logits).all():
            raise FloatingPointError(f"Nonfinite evaluation logits for {branch or 'final fusion'}")
        logits.append(selected_logits.cpu().numpy())
        selected = keep.cpu()
        ids.extend(np.asarray(batch["id"])[selected.numpy()].tolist())
        indices.extend(batch["index"][selected].tolist())
        if "target" in batch:
            targets.append(batch["target"][selected].numpy())
    if not logits:
        raise ValueError(f"No observed samples to evaluate for {branch or 'final fusion'}")
    result = {"logits": np.concatenate(logits), "ids": np.asarray(ids, dtype=str),
              "indices": np.asarray(indices)}
    if targets:
        result["targets"] = np.concatenate(targets)
    if branch_outputs:
        result["branch_logits"] = np.concatenate(branch_outputs)
    return result


def _run_stage(model, loaders, criterion, config, output_dim, device, run_dir, branch=None):
    stage = f"subnetwork_{branch}" if branch is not None else (
        "late_fusion" if config.training_mode == "staged" else "joint")
    _configure_stage(model, config, branch)
    modules = _stage_modules(model, config, branch)
    parameters = list({id(p): p for module in modules for p in module.parameters() if p.requires_grad}.values())
    if not parameters:
        raise RuntimeError(f"No trainable parameters for {stage}")
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    epochs = config.pretrain_epochs if branch is not None else config.epochs
    patience = config.pretrain_patience if branch is not None else config.patience
    path = run_dir / f"best_{stage}.pt"
    history, best_score, stale = [], -float("inf"), 0
    for epoch in range(1, epochs + 1):
        _configure_stage(model, config, branch)
        total, task_total, branch_total, seen = 0.0, 0.0, 0.0, 0
        for batch in loaders["train"]:
            xx, mm = _to_device(batch, device)
            target = batch["target"].to(device)
            model.zero_grad(set_to_none=True)
            if branch is not None:
                output = model.forward_branch(xx, mm, branch)
                keep = output["observed"].bool()
                if not keep.any():
                    continue
                task_loss = _loss(output["logits"][keep], target[keep], criterion, config.task)
                branch_loss = task_loss.new_zeros(())
            else:
                keep = torch.ones(len(target), device=device, dtype=torch.bool)
                # Labels are passed only in joint training, never during evaluation.
                output = (model(xx, mm, targets=target,
                                class_weights=criterion.weight if config.task == "classification" else None)
                          if config.training_mode == "joint" else model(xx, mm))
                task_loss = _loss(output["logits"], target, criterion, config.task)
                if config.training_mode == "joint":
                    branch_loss = output["aux_losses"]["subnetwork_supervision"]
                else:
                    branch_loss = task_loss.new_zeros(())
            loss = task_loss + config.branch_loss_weight * branch_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {stage}; inspect observed features and labels")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            count = int(keep.sum())
            seen += count
            total += float(loss.detach()) * count
            task_total += float(task_loss.detach()) * count
            branch_total += float(branch_loss.detach()) * count
        if not seen:
            raise ValueError(f"No observed training samples for {stage}")
        prediction = predict_loader(model, loaders["valid"], device, branch)
        valid = _metrics(prediction, config, output_dim, criterion, device)
        score = valid["macro_f1"] if config.task == "classification" else -valid["mae"]
        if not math.isfinite(score):
            raise FloatingPointError(f"Nonfinite validation selection metric in {stage}")
        row = {"stage": stage, "epoch": epoch, "train_loss": total / seen,
               "train_task_loss": task_total / seen, "train_branch_loss": branch_total / seen,
               "train_samples": seen, "valid": valid}
        history.append(row)
        metric_name = "macro_f1" if config.task == "classification" else "mae"
        print(f"{stage} epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | valid_{metric_name}={valid[metric_name]:.4f}")
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            torch.save({"module_states": [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                                          for module in modules],
                        "stage": stage, "epoch": epoch, "valid_metrics": valid}, path)
        else:
            stale += 1
        if stale >= patience:
            break
    selected = torch.load(path, map_location="cpu", weights_only=True)
    for module, state in zip(modules, selected["module_states"]):
        module.load_state_dict(state)
    model.eval()
    return history, {key: selected[key] for key in ("stage", "epoch", "valid_metrics")}


@dataclass
class Experiment:
    model: Model
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
    """Fit training data; select four branches and final head on valid; test once."""
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
    if config.training_mode == "staged":
        for split in ("train", "valid"):
            for m in MODALITIES:
                if not masks[split][m].any():
                    raise ValueError(f"Pan staged training requires observed {m} samples in {split}")
    stats = fit_normalizers(data["train"], masks["train"], config.standardize_av)
    class_values = []
    if config.task == "classification":
        labels = np.asarray(data["train"]["classification_labels"])
        if not np.isfinite(labels).all():
            raise ValueError("Nonfinite training labels")
        class_values = [float(value) for value in np.unique(labels)]
        if len(class_values) < 2:
            raise ValueError("Classification requires at least two training classes")
    output_dim = len(class_values) if config.task == "classification" else 1
    datasets = {s: FeatureDataset(data[s], masks[s], stats, config.task, class_values)
                for s in ("train", "valid", "test")}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("train, valid and test must be nonempty")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {s: DataLoader(dataset, batch_size=config.batch_size, shuffle=s == "train", num_workers=0,
                            generator=generator if s == "train" else None) for s, dataset in datasets.items()}
    dims = {m: int(np.asarray(data["train"][m]).shape[-1]) for m in MODALITIES}
    max_length = int(np.asarray(data["train"]["text"]).shape[1])
    if any(np.asarray(data[s]["text"]).shape[1] > max_length for s in ("valid", "test")):
        raise ValueError("Validation/test sequence length exceeds training max_length")
    model = Model(dims, output_dim, max_length, config).to(device)
    class_weights = None
    if config.task == "classification" and config.class_weighted_loss:
        counts = np.bincount(datasets["train"].targets, minlength=output_dim)
        class_weights = torch.tensor(len(datasets["train"]) / (output_dim * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if config.task == "classification" else nn.L1Loss()
    run_dir = Path(config.output_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / "config.json", {**asdict(config), "model_kind": MODEL_KIND, "actual_device": str(device),
          "feature_dims": dims, "class_values": class_values, "branch_order": list(BRANCH_NAMES),
          "modality_order": list(MODALITIES), "source": SOURCE,
          "sequence_unit": "within_sample_aligned_slot", "pan_independent_branches": True,
          "pan_paper_layout": True, "mask_assumption": "shared text slot layout unless explicitly overridden"})
    _json(run_dir / "data_audit.json", report)
    print(f"device={device}; Pan mode={config.training_mode}; task={config.task}; output={run_dir}")
    print("Adaptation: aligned within-sample slots and one sample target; not the paper's utterance-context evaluation.")
    print("A/V masks default to text positions; zero-valued features remain observations unless explicitly masked.")
    history, pretraining = [], {}
    if config.training_mode == "staged":
        for branch in BRANCH_NAMES:
            records, selected = _run_stage(model, loaders, criterion, config, output_dim, device, run_dir, branch)
            history.extend(records)
            pretraining[branch] = selected
        model.freeze_subnetworks(True)
    records, selected = _run_stage(model, loaders, criterion, config, output_dim, device, run_dir)
    history.extend(records)
    checkpoint = {"model_kind": MODEL_KIND, "checkpoint_version": 1,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "config": asdict(config), "feature_dims": dims, "output_dim": output_dim, "max_length": max_length,
        "class_values": class_values, "modality_order": list(MODALITIES), "branch_order": list(BRANCH_NAMES),
        "source": SOURCE, "subnetworks_frozen": config.training_mode == "staged",
        "selected": selected, "pretraining": pretraining,
        "normalizers": {m: {"mean": torch.from_numpy(stats[m]["mean"]), "std": torch.from_numpy(stats[m]["std"]),
                            "observed_count": stats[m]["observed_count"]} for m in MODALITIES}}
    torch.save(checkpoint, run_dir / "best.pt")
    # Test data is evaluated only after every branch and final model is selected.
    test_prediction = predict_loader(model, loaders["test"], device)
    test_metrics = _metrics(test_prediction, config, output_dim, criterion, device)
    _json(run_dir / "history.json", history)
    _json(run_dir / "metrics.json", {"pretraining": pretraining, "selected_epoch": selected["epoch"],
                                   "valid": selected["valid_metrics"], "test": test_metrics})
    export_predictions(test_prediction, run_dir, config.task, class_values)
    print("Selected final epoch:", selected["epoch"], "Test:", test_metrics)
    return Experiment(model, config, run_dir, datasets, stats, class_values, history,
                      selected["valid_metrics"], test_metrics, test_prediction, pretraining)


def load_experiment_model(checkpoint_path, device=None):
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if metadata.get("model_kind") != MODEL_KIND:
        raise ValueError("Expected the dedicated Pan complete best.pt, not a stage or generic-suite checkpoint")
    config = Config(**metadata["config"])
    config.validate()
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = Model(metadata["feature_dims"], metadata["output_dim"], metadata["max_length"], config)
    model.load_state_dict(metadata["state_dict"])
    model.freeze_subnetworks(metadata["subnetworks_frozen"])
    model.to(selected_device).eval()
    metadata["normalizers"] = {m: {"mean": value["mean"].numpy(), "std": value["std"].numpy(),
                                  "observed_count": value["observed_count"]}
                               for m, value in metadata["normalizers"].items()}
    return model, metadata


def predict_split(model, split, metadata, masks=None, batch_size=32):
    """Predict using saved normalization/class mapping; labels are optional."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    valid = resolve_masks(split, masks)
    for m in MODALITIES:
        values = np.asarray(split[m])
        for start in range(0, len(values), 128):
            if not np.isfinite(values[start:start + 128][valid[m][start:start + 128]]).all():
                raise ValueError(f"Nonfinite observed input in {m}")
    dataset = FeatureDataset(split, valid, metadata["normalizers"], metadata["config"]["task"],
                             metadata["class_values"], require_labels=False)
    return predict_loader(model, DataLoader(dataset, batch_size=batch_size, num_workers=0),
                          next(model.parameters()).device)


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32):
    """Cached-feature ablation with fixed class margin, not raw-input causality."""
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)

"""Common training/inference for feature-space adaptations.

The existing project's preprocessing contract is preserved. Only training data
fits normalization and optional CCA; validation selects checkpoints; test is
evaluated after selection. Model-specific auxiliary objectives are explicit.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multi_fusion_model.weighted_sum_fusion import (FeatureDataset, audit_data, classification_metrics,
    explain_feature_groups as _explain, fit_normalizers, regression_metrics,
    resolve_masks, seed_everything)
from multi_fusion_model.context_lstm_fusion import export_predictions
from .common import MODALITIES
from .model_info import MODEL_INFO

MODEL_NAMES = ("ren2021", "zheng2022", "pan2020", "m3er2020", "mfrm2022",
               "transmodality2020", "memobert2022", "hycon2022", "emoe2025")
FORMAT = "aligned_multimodal_suite_v1"


@dataclass
class Config:
    model_name: str = "pan2020"
    task: str = "classification"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    pretrain_epochs: int | None = None  # model default, usually zero
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    standardize_av: bool = True
    class_weighted_loss: bool = False
    auxiliary_weights: dict = field(default_factory=dict)  # overrides model defaults
    pretrain_auxiliary_weights: dict = field(default_factory=dict)  # overrides supervised-stage weights for warmup
    model_options: dict = field(default_factory=dict)
    device: str | None = None
    output_root: str = "results/second_question/multimodal_suite"

    def validate(self):
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {self.model_name}; choose {MODEL_NAMES}")
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        if min(self.d_model, self.nhead, self.num_layers, self.batch_size, self.epochs, self.patience) <= 0:
            raise ValueError("Sizes, epochs, patience and batch_size must be positive")
        if self.d_model % self.nhead or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by nhead")
        if not 0 <= self.dropout < 1 or self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid dropout or optimizer settings")
        if self.pretrain_epochs is not None and self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs cannot be negative")
        for field_name in ("auxiliary_weights", "pretrain_auxiliary_weights"):
            weights = getattr(self, field_name)
            if not isinstance(weights, dict):
                raise ValueError(f"{field_name} must be a dictionary")
            for name, value in weights.items():
                try:
                    valid = isinstance(name, str) and bool(np.isfinite(value)) and value >= 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(f"{field_name} values must be finite and nonnegative")
        json.dumps(asdict(self), allow_nan=False)


def create_model(feature_dims, output_dim, max_length, config):
    config.validate()
    from .attention_models import Ren2021Model, Zheng2022Model, Pan2020Model
    from .memory_translation import MFRM2022Model, TransModality2020Model
    from .robust_pretraining import M3ER2020Model, MEmoBERT2022Model, HyCon2022Model
    from .emotion_experts import EMOE2025Model
    classes = dict(zip(MODEL_NAMES, (Ren2021Model, Zheng2022Model, Pan2020Model,
                   M3ER2020Model, MFRM2022Model, TransModality2020Model,
                   MEmoBERT2022Model, HyCon2022Model, EMOE2025Model)))
    return classes[config.model_name](feature_dims, output_dim, max_length, config)


def _batch(batch, device):
    return ({m: batch["features"][m].to(device) for m in MODALITIES},
            {m: batch["masks"][m].to(device) for m in MODALITIES})


def _supervised(z, y, criterion, task):
    return criterion(z if task == "classification" else z.squeeze(-1), y)


def _resolve_auxiliary_weights(model, config):
    """Resolve each phase independently while preserving legacy defaults."""
    defaults = dict(getattr(model, "default_auxiliary_weights", {}))
    for field_name in ("auxiliary_weights", "pretrain_auxiliary_weights"):
        unknown = set(getattr(config, field_name)) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown auxiliary objectives in {field_name}: {unknown}")
    supervised = {**defaults, **config.auxiliary_weights}
    pretrain = {**supervised, **config.pretrain_auxiliary_weights}
    return supervised, pretrain


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    logits, ids, indices, targets = [], [], [], []
    routing_audit = {name: [] for name in ("routing_weights", "modality_logits", "modality_availability")}
    for batch in loader:
        xx, mm = _batch(batch, device)
        output = model(xx, mm)  # Labels NEVER enter prediction.
        if not torch.isfinite(output["logits"]).all():
            raise FloatingPointError("Nonfinite prediction")
        logits.append(output["logits"].cpu().numpy())
        if "routing_weights" in output:
            for name in routing_audit:
                if name in output:
                    routing_audit[name].append(output[name].cpu().numpy())
        ids.extend(batch["id"])
        indices.extend(batch["index"].tolist())
        if "target" in batch:
            targets.append(batch["target"].numpy())
    if not logits:
        raise ValueError("Cannot evaluate an empty split")
    result = {"logits": np.concatenate(logits), "ids": np.asarray(ids, dtype=str),
              "indices": np.asarray(indices)}
    if targets:
        result["targets"] = np.concatenate(targets)
    for name, values in routing_audit.items():
        if values:
            result[name] = np.concatenate(values)
    return result


def _metrics(pred, config, output_dim, criterion, device):
    y, z = pred["targets"], pred["logits"]
    metrics = (classification_metrics(y, z.argmax(1), output_dim) if config.task == "classification"
               else regression_metrics(y, z[:, 0]))
    metrics["loss"] = float(_supervised(torch.as_tensor(z, device=device), torch.as_tensor(y, device=device),
                                       criterion, config.task))
    return metrics


@dataclass
class Experiment:
    model: nn.Module
    config: Config
    run_dir: Path
    datasets: dict
    stats: dict
    class_values: list
    history: list
    valid_metrics: dict
    test_metrics: dict
    test_predictions: dict


def fit_experiment(data, config=None, mask_overrides=None):
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
        labels = np.asarray(data["train"]["classification_labels"])
        if not np.isfinite(labels).all():
            raise ValueError("Nonfinite training labels")
        class_values = [float(v) for v in np.unique(labels)]
        if len(class_values) < 2:
            raise ValueError("Classification needs at least two training classes")
    output_dim = len(class_values) if config.task == "classification" else 1
    datasets = {s: FeatureDataset(data[s], masks[s], stats, config.task, class_values)
                for s in ("train", "valid", "test")}
    if any(not len(ds) for ds in datasets.values()):
        raise ValueError("All splits must be nonempty")
    dims = {m: int(data["train"][m].shape[-1]) for m in MODALITIES}
    max_length = int(data["train"]["text"].shape[1])
    if any(data[s]["text"].shape[1] > max_length for s in ("valid", "test")):
        raise ValueError("Validation/test exceeds training maximum length")
    model = create_model(dims, output_dim, max_length, config).to(device)
    # Data-dependent reliability/proxy fitting has access to TRAIN ONLY.
    preprocessing_report = None
    if hasattr(model, "fit_preprocessing"):
        preprocessing_report = model.fit_preprocessing(datasets["train"])
    pretrain_epochs = (int(getattr(model, "default_pretrain_epochs", 0)) if config.pretrain_epochs is None
                       else config.pretrain_epochs)
    auxiliary_weights, pretrain_auxiliary_weights = _resolve_auxiliary_weights(model, config)
    if pretrain_epochs and not any(weight > 0 for weight in pretrain_auxiliary_weights.values()):
        raise ValueError("Auxiliary pretraining requested but no active auxiliary objective exists")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {s: DataLoader(ds, batch_size=config.batch_size, shuffle=s == "train", num_workers=0,
                            generator=generator if s == "train" else None) for s, ds in datasets.items()}
    class_weights = None
    if config.task == "classification" and config.class_weighted_loss:
        counts = np.bincount(datasets["train"].targets, minlength=output_dim)
        class_weights = torch.tensor(len(datasets["train"]) / (output_dim * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if config.task == "classification" else nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    run_dir = Path(config.output_root).expanduser().resolve() / config.model_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / "config.json", {**asdict(config), "actual_device": str(device),
          "feature_dims": dims, "class_values": class_values, "modality_order": list(MODALITIES),
          "effective_auxiliary_weights": auxiliary_weights,
          "effective_pretrain_auxiliary_weights": pretrain_auxiliary_weights,
          "effective_pretrain_epochs": pretrain_epochs,
          "scope": "cached_feature_within_sample_adaptation", "implementation": MODEL_INFO[config.model_name],
          "preprocessing_report": preprocessing_report})
    _json(run_dir / "data_audit.json", report)
    print(f"model={config.model_name}; device={device}; output={run_dir}")
    print("Implementation scope:", MODEL_INFO[config.model_name]["label"])
    print(f"supervised_auxiliary_weights={auxiliary_weights}; "
          f"pretrain_auxiliary_weights={pretrain_auxiliary_weights}; auxiliary_pretrain_epochs={pretrain_epochs}")
    print("Aligned cached features; shared text mask unless overridden. Zero rows do not imply missingness.")
    accepts_targets = "targets" in inspect.signature(model.forward).parameters
    history = []

    def train_epoch(pretraining):
        model.train()
        phase_weights = pretrain_auxiliary_weights if pretraining else auxiliary_weights
        sums, seen = {}, 0
        for batch in loaders["train"]:
            xx, mm = _batch(batch, device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = (model(xx, mm, targets=target) if accepts_targets and not pretraining else model(xx, mm))
            # A disconnected scalar keeps classifier grad=None in pretraining;
            # zero-valued classifier gradients would still trigger AdamW decay.
            supervised = (output["logits"].new_zeros(()) if pretraining
                          else _supervised(output["logits"], target, criterion, config.task))
            aux = output.get("aux_losses", {})
            if set(aux) - set(phase_weights):
                raise ValueError(f"Model returned unregistered auxiliary loss: {set(aux) - set(phase_weights)}")
            loss = supervised
            terms = {}
            for name, value in aux.items():
                if value.ndim != 0 or not torch.isfinite(value):
                    raise FloatingPointError(f"Invalid auxiliary loss: {name}")
                # Skipping is different from multiplying by zero: unused
                # decoder parameters must retain grad=None during task-only
                # fine-tuning, including when weight decay is enabled.
                if phase_weights[name] > 0:
                    loss = loss + phase_weights[name] * value
                terms[name] = float(value.detach())
            if pretraining and not any(phase_weights[k] > 0 for k in aux):
                raise ValueError("Model returned no active objective during auxiliary pretraining")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            n = len(target)
            seen += n
            terms.update(total=float(loss.detach()), supervised=float(supervised.detach()))
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + n * value
        return {name: value / seen for name, value in sums.items()}

    # Fixed training-only auxiliary warmup; validation labels never enter it.
    for epoch in range(1, pretrain_epochs + 1):
        record = {"stage": "auxiliary_pretrain", "epoch": epoch, "train": train_epoch(True)}
        history.append(record)
        print(f"pretrain {epoch:02d} | loss={record['train']['total']:.4f}")
    if pretrain_epochs:
        torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "model_name": config.model_name, "epochs": pretrain_epochs,
                    "config": asdict(config), "effective_auxiliary_weights": auxiliary_weights,
                    "effective_pretrain_auxiliary_weights": pretrain_auxiliary_weights}, run_dir / "pretrained.pt")
        # Fresh optimizer for supervised fine-tuning; no test-dependent decision.
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_score, stale = -float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        training = train_epoch(False)
        val = _metrics(predict_loader(model, loaders["valid"], device), config, output_dim, criterion, device)
        record = {"stage": "supervised", "epoch": epoch, "train": training, "valid": val}
        history.append(record)
        score = val["macro_f1"] if config.task == "classification" else -val["mae"]
        name = "macro_f1" if config.task == "classification" else "mae"
        print(f"epoch {epoch:02d} | loss={training['total']:.4f} | valid_{name}={val[name]:.4f}")
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            checkpoint = {"format": FORMAT, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": asdict(config), "feature_dims": dims, "output_dim": output_dim, "max_length": max_length,
                "class_values": class_values, "modality_order": list(MODALITIES), "selected_epoch": epoch,
                "valid_metrics": val, "effective_pretrain_epochs": pretrain_epochs,
                "implementation": MODEL_INFO[config.model_name], "preprocessing_report": preprocessing_report,
                "effective_auxiliary_weights": auxiliary_weights,
                "effective_pretrain_auxiliary_weights": pretrain_auxiliary_weights,
                "normalizers": {m: {"mean": torch.from_numpy(stats[m]["mean"]), "std": torch.from_numpy(stats[m]["std"]),
                                    "observed_count": stats[m]["observed_count"]} for m in MODALITIES}}
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale += 1
        _json(run_dir / "history.json", history)
        if stale >= config.patience:
            break
    selected = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    prediction = predict_loader(model, loaders["test"], device)
    metrics = _metrics(prediction, config, output_dim, criterion, device)
    _json(run_dir / "metrics.json", {"model_name": config.model_name, "selected_epoch": selected["selected_epoch"],
          "valid": selected["valid_metrics"], "test": metrics})
    export_predictions(prediction, run_dir, config.task, class_values)
    print("Selected epoch:", selected["selected_epoch"], "Test:", metrics)
    return Experiment(model, config, run_dir, datasets, stats, class_values, history,
                      selected["valid_metrics"], metrics, prediction)


def load_experiment_model(checkpoint_path, device=None):
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if metadata.get("format") != FORMAT:
        raise ValueError("Use a complete multimodal_suite best.pt checkpoint")
    config = Config(**metadata["config"])
    model = create_model(metadata["feature_dims"], metadata["output_dim"], metadata["max_length"], config)
    supervised_weights, pretrain_weights = _resolve_auxiliary_weights(model, config)
    metadata.setdefault("effective_auxiliary_weights", supervised_weights)
    metadata.setdefault("effective_pretrain_auxiliary_weights", pretrain_weights)
    model.load_state_dict(metadata["state_dict"])
    model.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))).eval()
    metadata["normalizers"] = {m: {"mean": v["mean"].numpy(), "std": v["std"].numpy(),
                                   "observed_count": v["observed_count"]} for m, v in metadata["normalizers"].items()}
    return model, metadata


def predict_split(model, split, metadata, masks=None, batch_size=32):
    valid = resolve_masks(split, masks)
    for m in MODALITIES:
        x = np.asarray(split[m])
        for start in range(0, len(x), 128):
            if not np.isfinite(x[start:start + 128][valid[m][start:start + 128]]).all():
                raise ValueError(f"Nonfinite observed features in {m}")
    ds = FeatureDataset(split, valid, metadata["normalizers"], metadata["config"]["task"],
                        metadata["class_values"], require_labels=False)
    return predict_loader(model, DataLoader(ds, batch_size=batch_size, num_workers=0), next(model.parameters()).device)


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32):
    """Cached-feature perturbations; fixed masks/target margin; not raw causal evidence.

    For M3ER this measures the complete model, including any changed reliability
    decision and proxy path. It is not an isolated original-modality contribution.
    """
    result = _explain(experiment, split, index, target_class, baseline, perturb_batch_size)
    result["model_name"] = experiment.config.model_name
    result["includes_proxy_rerouting"] = experiment.config.model_name == "m3er2020"
    return result

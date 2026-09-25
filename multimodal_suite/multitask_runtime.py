"""Joint classification/regression training for the user's ThisWork model.

The existing single-task runtime and dataset stay unchanged. Both targets refer
to the same sample; normalization fits train only; one validation-selected
checkpoint supplies both test predictions. Classification selection uses Macro-F1.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multi_fusion_model.weighted_sum_fusion import (
    FeatureDataset, audit_data, classification_metrics, regression_metrics,
    explain_feature_groups as _explain, fit_normalizers, resolve_masks, seed_everything,
)
from .runtime import Config as _BaseConfig, _batch, _json, _resolve_auxiliary_weights
from .common import MODALITIES
from .attention_models import ThisWork

FORMAT = "this_work_joint_classification_regression_v1"


@dataclass
class Config(_BaseConfig):
    model_name: str = "this_work"
    task: str = "multitask"
    pretrain_epochs: int = 3
    pretrain_auxiliary_weights: dict = field(default_factory=lambda: {"reconstruction": 1.0})
    auxiliary_weights: dict = field(default_factory=lambda: {"reconstruction": 0.0})
    classification_loss_weight: float = 1.0
    regression_loss_weight: float = 1.0
    selection_metric: str = "balanced"
    selection_classification_weight: float = 0.5
    selection_mae_scale: float | None = None
    output_root: str = "results/second_question/this_work"

    def validate(self):
        if self.model_name != "this_work" or self.task != "multitask":
            raise ValueError("This entry requires model_name='this_work', task='multitask'")
        # Validate inherited architecture/optimizer fields without altering the
        # single-task registry or modifying this Config object.
        values = {f.name: getattr(self, f.name) for f in fields(_BaseConfig)}
        values.update(model_name="zheng2022", task="classification")
        _BaseConfig(**values).validate()
        if not isinstance(self.pretrain_epochs, int) or self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be a nonnegative integer")
        for name in ("classification_loss_weight", "regression_loss_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive for joint training")
        if self.selection_metric not in ("balanced", "macro_f1", "mae"):
            raise ValueError("selection_metric must be balanced, macro_f1 or mae")
        if not np.isfinite(self.selection_classification_weight) or not 0 < self.selection_classification_weight < 1:
            raise ValueError("selection_classification_weight must lie strictly between 0 and 1")
        if self.selection_mae_scale is not None and (
            not np.isfinite(self.selection_mae_scale) or self.selection_mae_scale <= 0
        ):
            raise ValueError("selection_mae_scale must be finite and positive")


class MultiTaskFeatureDataset(FeatureDataset):
    """Use existing feature preprocessing once and return two aligned labels."""
    def __init__(self, split, masks, stats, class_values, require_labels=True):
        super().__init__(split, masks, stats, "classification", class_values, require_labels)
        self.classification_targets = self.targets
        # This object only validates/casts the regression labels; no feature
        # extraction or normalization is performed twice in __getitem__.
        regression = FeatureDataset(split, masks, stats, "regression", [], require_labels)
        self.regression_targets = regression.targets

    def __getitem__(self, index):
        item = super().__getitem__(index)
        target = item.pop("target", None)
        if target is not None:
            item["classification_target"] = target
        if self.regression_targets is not None:
            item["regression_target"] = torch.tensor(self.regression_targets[index], dtype=torch.float32)
        return item


def create_model(feature_dims, output_dim, max_length, config):
    config.validate()
    if output_dim < 2:
        raise ValueError("output_dim is the number of classification classes, not the regression width")
    return ThisWork(feature_dims, output_dim, max_length, config)


def compute_task_losses(output, batch, classification_criterion, regression_criterion, device):
    logits, prediction = output["classification_logits"], output["regression_prediction"]
    cls_target = batch["classification_target"].to(device)
    reg_target = batch["regression_target"].to(device)
    if logits.ndim != 2 or cls_target.shape != (logits.shape[0],):
        raise ValueError("Classification needs logits [B,C] and target [B]")
    if prediction.ndim != 1 or prediction.shape != reg_target.shape or prediction.shape != cls_target.shape:
        raise ValueError("Regression prediction and target must both be [B], without broadcasting")
    return {
        "classification": classification_criterion(logits, cls_target),
        "regression": regression_criterion(prediction, reg_target),
    }


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    parts = {key: [] for key in ("classification_logits", "regression_prediction",
                                "classification_targets", "regression_targets")}
    ids, indices = [], []
    for batch in loader:
        xx, mm = _batch(batch, device)
        output = model(xx, mm)  # Neither label is passed into prediction.
        batch_size = len(batch["index"])
        if output["classification_logits"].ndim != 2 or output["classification_logits"].shape[0] != batch_size:
            raise ValueError("Expected classification_logits [B,C]")
        if output["regression_prediction"].shape != (batch_size,):
            raise ValueError("Expected regression_prediction [B]")
        for key in ("classification_logits", "regression_prediction"):
            if not torch.isfinite(output[key]).all():
                raise FloatingPointError(f"Nonfinite {key}")
            parts[key].append(output[key].cpu().numpy())
        for singular, plural in (("classification_target", "classification_targets"),
                                 ("regression_target", "regression_targets")):
            if singular in batch:
                parts[plural].append(batch[singular].numpy())
        ids.extend(batch["id"])
        indices.extend(batch["index"].tolist())
    if not ids:
        raise ValueError("Cannot predict an empty split")
    result = {key: np.concatenate(value) for key, value in parts.items() if value}
    result.update(ids=np.asarray(ids, dtype=str), indices=np.asarray(indices, dtype=np.int64))
    result["predicted_class_indices"] = result["classification_logits"].argmax(1)
    return result


def selection_score(metrics, config, mae_scale):
    # Experimental selection convention, not a competition score. Regression
    # scale is fixed from TRAIN, so validation/test labels never fit this scale.
    weight = config.selection_classification_weight
    balanced_error = weight * (1 - metrics["macro_f1"]) + (1 - weight) * metrics["mae"] / mae_scale
    if config.selection_metric == "macro_f1":
        return metrics["macro_f1"], balanced_error
    if config.selection_metric == "mae":
        return -metrics["mae"], balanced_error
    return -balanced_error, balanced_error


def _metrics(prediction, config, output_dim, cls_criterion, reg_criterion, device, mae_scale):
    metrics = classification_metrics(
        prediction["classification_targets"], prediction["predicted_class_indices"], output_dim)
    metrics.update(regression_metrics(prediction["regression_targets"], prediction["regression_prediction"]))
    cls_loss = cls_criterion(torch.as_tensor(prediction["classification_logits"], device=device),
                             torch.as_tensor(prediction["classification_targets"], device=device))
    reg_loss = reg_criterion(torch.as_tensor(prediction["regression_prediction"], device=device),
                             torch.as_tensor(prediction["regression_targets"], device=device))
    metrics.update(classification_loss=float(cls_loss), regression_loss=float(reg_loss))
    metrics["loss"] = config.classification_loss_weight * float(cls_loss) + config.regression_loss_weight * float(reg_loss)
    metrics["selection_score"], metrics["balanced_error"] = selection_score(metrics, config, mae_scale)
    return metrics


def _export(prediction, run_dir, split_name, class_values):
    values = dict(prediction)
    values["predicted_labels"] = np.asarray(class_values)[values["predicted_class_indices"]]
    np.savez_compressed(run_dir / f"{split_name}_predictions.npz", **values,
                        class_values=np.asarray(class_values), modality_order=np.asarray(MODALITIES))
    rows = []
    for i, sample_id in enumerate(values["ids"]):
        row = {"id": sample_id, "index": int(values["indices"][i]),
               "predicted_class": float(values["predicted_labels"][i]),
               "predicted_score": float(values["regression_prediction"][i])}
        if "classification_targets" in values:
            row["true_class"] = class_values[int(values["classification_targets"][i])]
        if "regression_targets" in values:
            row["true_score"] = float(values["regression_targets"][i])
        row.update({f"logit_class_{k}": float(z) for k, z in enumerate(values["classification_logits"][i])})
        rows.append(row)
    with (run_dir / f"{split_name}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Experiment:
    model: nn.Module
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


def fit_experiment(data, config=None, mask_overrides=None):
    config = config or Config()
    config.validate()
    seed_everything(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mask_overrides = mask_overrides or {}
    if set(mask_overrides) - {"train", "valid", "test"}:
        raise ValueError("Unknown mask split")
    report = audit_data(data, mask_overrides)
    masks = {s: resolve_masks(data[s], mask_overrides.get(s)) for s in ("train", "valid", "test")}
    stats = fit_normalizers(data["train"], masks["train"], config.standardize_av)
    class_values = [float(v) for v in np.unique(data["train"]["classification_labels"])]
    if len(class_values) < 2 or not np.isfinite(class_values).all():
        raise ValueError("Need at least two finite training classes")
    datasets = {s: MultiTaskFeatureDataset(data[s], masks[s], stats, class_values)
                for s in ("train", "valid", "test")}
    if any(not len(ds) for ds in datasets.values()):
        raise ValueError("All splits must be nonempty")
    dims = {m: int(np.asarray(data["train"][m]).shape[-1]) for m in MODALITIES}
    max_length = int(np.asarray(data["train"]["text"]).shape[1])
    if any(np.asarray(data[s]["text"]).shape[1] > max_length for s in ("valid", "test")):
        raise ValueError("Validation/test length exceeds training maximum")
    train_y = datasets["train"].regression_targets.astype(np.float64)
    constant_mae = float(np.abs(train_y - np.median(train_y)).mean())
    mae_scale = (config.selection_mae_scale if config.selection_mae_scale is not None
                 else constant_mae if constant_mae > 1e-6 else 1.0)
    output_dim = len(class_values)
    model = create_model(dims, output_dim, max_length, config).to(device)
    auxiliary_weights, pretrain_weights = _resolve_auxiliary_weights(model, config)
    if config.pretrain_epochs and not any(v > 0 for v in pretrain_weights.values()):
        raise ValueError("Reconstruction warmup needs an active auxiliary loss")
    loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config.seed))
    eval_loaders = {s: DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0,
                                 generator=torch.Generator().manual_seed(config.seed + i + 1))
                    for i, (s, ds) in enumerate(datasets.items())}
    class_weights = None
    if config.class_weighted_loss:
        counts = np.bincount(datasets["train"].classification_targets, minlength=output_dim)
        class_weights = torch.tensor(len(datasets["train"]) / (output_dim * counts), device=device, dtype=torch.float32)

    # 这里计算分别计算 分类损失 和 回归损失        
    cls_criterion, reg_criterion = nn.CrossEntropyLoss(weight=class_weights), nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    run_dir = Path(config.output_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"format": FORMAT, "config": asdict(config), "feature_dims": dims, "output_dim": output_dim,
                "max_length": max_length, "class_values": class_values, "modality_order": list(MODALITIES),
                "selection_mae_scale": mae_scale, "selection_scale_source": "explicit" if config.selection_mae_scale is not None else "train_median_constant_mae",
                "effective_auxiliary_weights": auxiliary_weights, "effective_pretrain_auxiliary_weights": pretrain_weights,
                "normalizers": {m: {"mean": torch.from_numpy(stats[m]["mean"]), "std": torch.from_numpy(stats[m]["std"]),
                                   "observed_count": stats[m]["observed_count"]} for m in MODALITIES}}
    _json(run_dir / "config.json", {**asdict(config), **{k: v for k, v in metadata.items() if k not in ("config", "normalizers")},
                                   "actual_device": str(device)})
    _json(run_dir / "data_audit.json", report)
    print(f"ThisWork multitask | device={device} | classes={class_values} | output={run_dir}", flush=True)
    print(f"CE weight={config.classification_loss_weight}; MAE weight={config.regression_loss_weight}; "
          f"selection={config.selection_metric}; fixed MAE scale={mae_scale:.6f}", flush=True)
    history = []

    def train_epoch(pretraining=False):
        model.train()
        phase_weights = pretrain_weights if pretraining else auxiliary_weights
        totals, seen = {}, 0
        for batch in loader:
            xx, mm = _batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xx, mm)
            losses = {} if pretraining else compute_task_losses(output, batch, cls_criterion, reg_criterion, device)
            
            supervised = output["classification_logits"].new_zeros(())
            if not pretraining:
                supervised = config.classification_loss_weight * losses["classification"] + config.regression_loss_weight * losses["regression"]
            total = supervised
            aux = output.get("aux_losses", {})
            if set(aux) - set(phase_weights):
                raise ValueError("Unregistered auxiliary objective")
            if pretraining and not any(phase_weights[k] > 0 for k in aux):
                raise ValueError("No active warmup objective")
            for name, value in aux.items():
                if value.ndim != 0 or not torch.isfinite(value):
                    raise FloatingPointError(f"Invalid {name} loss")
                if phase_weights[name] > 0:
                    total = total + phase_weights[name] * value
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite joint loss")
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            n = len(batch["index"])
            terms = {**losses, **aux, "supervised": supervised, "total": total}
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + n * float(value.detach())
            seen += n
        return {name: value / seen for name, value in totals.items()}

    def evaluate(split):
        prediction = predict_loader(model, eval_loaders[split], device)
        return prediction, _metrics(prediction, config, output_dim, cls_criterion, reg_criterion, device, mae_scale)

    for epoch in range(1, config.pretrain_epochs + 1):
        training = train_epoch(True)
        history.append({"stage": "auxiliary_pretrain", "epoch": epoch, "train": training, "train_loss": training["total"]})
        print(f"warmup {epoch:02d} | reconstruction={training['reconstruction']:.4f}", flush=True)
    if config.pretrain_epochs:
        torch.save({**metadata, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}},
                   run_dir / "pretrained.pt")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_score, stale = -float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        training = train_epoch()
        _, valid = evaluate("valid")
        history.append({"stage": "supervised", "epoch": epoch, "train": training, "train_loss": training["total"], "valid": valid})
        score = valid["selection_score"]
        print(f"epoch {epoch:02d} | CE={training['classification']:.4f} | MAE={training['regression']:.4f} | "
              f"valid Macro-F1={valid['macro_f1']:.4f} | valid MAE={valid['mae']:.4f}", flush=True)
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            torch.save({**metadata, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "selected_epoch": epoch, "valid_metrics": valid}, run_dir / "best.pt")
        else:
            stale += 1
        _json(run_dir / "history.json", history)
        if stale >= config.patience:
            break
    selected = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["state_dict"])
    _, train_metrics = evaluate("train")
    valid_prediction, valid_metrics = evaluate("valid")
    test_prediction, test_metrics = evaluate("test")  # Exactly once after selection.
    _json(run_dir / "metrics.json", {"selected_epoch": selected["selected_epoch"], "selection_metric": config.selection_metric,
          "selection_mae_scale": mae_scale, "train": train_metrics, "valid": valid_metrics, "test": test_metrics})
    _export(valid_prediction, run_dir, "valid", class_values)
    _export(test_prediction, run_dir, "test", class_values)
    return Experiment(model, config, run_dir, datasets, stats, class_values, history,
                      train_metrics, valid_metrics, test_metrics, test_prediction, selected["selected_epoch"])


def load_experiment_model(checkpoint_path, device=None):
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if metadata.get("format") != FORMAT or "selected_epoch" not in metadata:
        raise ValueError("Expected a complete ThisWork multitask best.pt checkpoint")
    config = Config(**metadata["config"])
    model = create_model(metadata["feature_dims"], metadata["output_dim"], metadata["max_length"], config)
    model.load_state_dict(metadata["state_dict"])
    model.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))).eval()
    metadata["normalizers"] = {m: {"mean": v["mean"].numpy(), "std": v["std"].numpy(), "observed_count": v["observed_count"]}
                             for m, v in metadata["normalizers"].items()}
    return model, metadata


def predict_split(model, split, metadata, masks=None, batch_size=32):
    valid = resolve_masks(split, masks)
    for m in MODALITIES:
        x = np.asarray(split[m])
        if x.ndim != 3 or x.shape[:2] != valid[m].shape:
            raise ValueError(f"Invalid {m} shape")
        for start in range(0, len(x), 128):
            if not np.isfinite(x[start:start + 128][valid[m][start:start + 128]]).all():
                raise ValueError(f"Nonfinite observed {m}")
    ds = MultiTaskFeatureDataset(split, valid, metadata["normalizers"], metadata["class_values"], require_labels=False)
    result = predict_loader(model, DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False),
                            next(model.parameters()).device)
    result["predicted_labels"] = np.asarray(metadata["class_values"])[result["predicted_class_indices"]]
    return result


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32, task="classification"):
    """Explain either head using the existing fixed-slot perturbation helper."""
    if task not in ("classification", "regression"):
        raise ValueError("Choose task='classification' or task='regression' for explanation")

    class HeadView(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features, masks):
            output = self.model(features, masks)
            return {"logits": output["classification_logits"] if task == "classification"
                    else output["regression_prediction"].unsqueeze(-1)}

    view = HeadView(experiment.model)
    view.train(experiment.model.training)
    proxy = SimpleNamespace(model=view, datasets=experiment.datasets, class_values=experiment.class_values,
                            config=SimpleNamespace(task=task))
    result = _explain(proxy, split, index, target_class, baseline, perturb_batch_size)
    result["task"] = task
    return result

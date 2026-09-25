"""Frozen ThisWork evaluation on the four generated Attachment-2 test sets.

Run from the project root:
    .venv-align/bin/python -B evaluate_this_work_perturbations.py --device cuda:0

Uses the existing public inference API and checkpoint training normalizers.
No training, refitting, restoration, clipping, or test-based model selection.
Corruption masks are provenance only; retain the trained shared-attention-mask
protocol. Labels and raw_text are excluded from the model's input dictionary.
Each run writes summary.csv, report.json and per-level prediction CSV/NPZ files
into a new timestamped directory. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from multi_fusion_model.this_work import load_experiment_model, predict_split
from multi_fusion_model.weighted_sum_fusion import (
    classification_metrics, regression_metrics, seed_everything,
)

DEFAULT_RUN = ROOT / "results/second_question/this_work/20260925_150303_247629"
LEVELS = (0, 10, 20, 30)
INPUT_FIELDS = ("text", "audio", "vision", "text_bert", "id")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_historical(checkpoint, baseline):
    """A diagnostic comparison, never a checkpoint-selection criterion."""
    report = {}
    path = checkpoint.parent / "metrics.json"
    if path.is_file():
        old = json.loads(path.read_text())["test"]
        report["recorded_test_metrics"] = old
        report["p000_minus_recorded"] = {
            key: baseline["metrics"][key] - old[key] for key in ("macro_f1", "mae")}
    path = checkpoint.parent / "test_predictions.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"], baseline["prediction"]["ids"])
            for key in ("classification_logits", "regression_prediction"):
                current, previous = baseline["prediction"][key], saved[key]
                report[key + "_max_abs_difference"] = float(np.max(np.abs(current - previous)))
                report[key + "_matches_at_1e-4"] = bool(np.allclose(current, previous, atol=1e-4, rtol=1e-5))
            report["changed_class_predictions"] = int(np.count_nonzero(
                saved["predicted_class_indices"] != baseline["prediction"]["predicted_class_indices"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/附件2-同步扰动特征")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/second_question/this_work_robustness")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    paths = [args.data_dir.expanduser().resolve() / f"aligned_50_p{level:03d}.pkl" for level in LEVELS]
    for path in [checkpoint, *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Checkpoint loading also validates its format and strictly loads weights.
    checkpoint_hash = sha256(checkpoint)
    torch.set_num_threads(4)
    model, metadata = load_experiment_model(checkpoint, device=args.device)
    seed_everything(metadata["config"]["seed"])
    model.eval().requires_grad_(False)
    class_values = np.asarray(metadata["class_values"])
    mapping = {float(value): index for index, value in enumerate(class_values)}
    output_dir = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Checkpoint: {checkpoint} (selected epoch {metadata['selected_epoch']})", flush=True)
    print(f"Output: {output_dir}", flush=True)
    baseline, summaries, details = None, [], []
    protected = {checkpoint: checkpoint_hash}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).suffix == ".py" and Path(filename).resolve().is_relative_to(ROOT):
            path = Path(filename).resolve()
            if path.is_file() and not path.is_relative_to(ROOT / ".venv-align"):
                protected[path] = sha256(path)

    for level, path in zip(LEVELS, paths):
        dataset_hash = sha256(path)
        with path.open("rb") as stream:
            data = pickle.load(stream)  # Trusted, locally generated datasets.
        split, provenance = data["test"], data["_metadata"]
        del data  # The other two splits are not used for fitting or prediction.
        if provenance["format"] != "aligned_synchronous_corruption_v1" or not np.isclose(provenance["nominal_probability"], level / 100):
            raise ValueError(f"Unexpected dataset protocol: {path}")
        ids = np.asarray(split["id"], dtype=str)
        n = len(ids)
        labels = np.asarray(split["classification_labels"]).reshape(-1)
        targets = np.asarray([mapping[float(y)] for y in labels], dtype=np.int64)
        regression_targets = np.asarray(split["regression_labels"], dtype=np.float32).reshape(-1)
        if targets.shape != (n,) or regression_targets.shape != (n,) or not np.isfinite(regression_targets).all():
            raise ValueError("Invalid test targets")
        mask = np.asarray(split["corruption_mask"])
        eligible_count = np.asarray(split["eligible_count"])
        np.testing.assert_array_equal(mask.sum(1), split["corruption_count"])
        if level == 0 and mask.any():
            raise ValueError("The clean baseline contains corruption")
        if baseline is not None:
            for key, current in (("ids", ids), ("targets", targets), ("regression_targets", regression_targets)):
                np.testing.assert_array_equal(baseline[key], current)
            np.testing.assert_array_equal(baseline["attention_and_types"], split["text_bert"][:, 1:])
            np.testing.assert_array_equal(baseline["eligible_count"], eligible_count)
            for key in ("source_sha256", "checkpoint_sha256", "adapter_module_sha256", "seed"):
                if provenance[key] != baseline["provenance"][key]:
                    raise ValueError(f"Inconsistent dataset generation provenance: {key}")
        # Public API performs the original normalization using saved train stats.
        # Do not pass corruption_mask as a new mask override or model input.
        with torch.inference_mode():
            prediction = predict_split(model, {key: split[key] for key in INPUT_FIELDS}, metadata,
                                       masks=None, batch_size=args.batch_size)
        np.testing.assert_array_equal(prediction["ids"], ids)
        np.testing.assert_array_equal(prediction["indices"], np.arange(n))
        metrics = classification_metrics(targets, prediction["predicted_class_indices"], len(class_values))
        metrics.update(regression_metrics(regression_targets, prediction["regression_prediction"]))
        if baseline is None:
            baseline = dict(ids=ids, targets=targets, regression_targets=regression_targets,
                            attention_and_types=split["text_bert"][:, 1:].copy(), eligible_count=eligible_count,
                            provenance=provenance, prediction=prediction, metrics=metrics)
        summary = {"file": path.name, "nominal_percent": level, "samples": n,
                   "actual_percent": float(mask.sum() / max(int(eligible_count.sum()), 1) * 100),
                   **{key: metrics[key] for key in ("macro_f1", "mae", "accuracy", "rmse", "pearson")},
                   "macro_f1_change": metrics["macro_f1"] - baseline["metrics"]["macro_f1"],
                   "mae_change": metrics["mae"] - baseline["metrics"]["mae"],
                   "changed_class_predictions": int(np.count_nonzero(prediction["predicted_class_indices"] != baseline["prediction"]["predicted_class_indices"]))}
        tag = f"p{level:03d}"
        rows = []
        for i in range(n):
            rows.append({"id": ids[i], "index": i, "true_class": float(labels[i]),
                         "predicted_class": float(prediction["predicted_labels"][i]),
                         "true_score": float(regression_targets[i]),
                         "predicted_score": float(prediction["regression_prediction"][i]),
                         "absolute_error": float(abs(prediction["regression_prediction"][i] - regression_targets[i])),
                         "corrupted_slots": int(mask[i].sum()),
                         "corruption_rate": float(split["corruption_rate"][i])})
        write_csv(output_dir / f"{tag}_test_predictions.csv", rows)
        with (output_dir / f"{tag}_test_predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, **prediction, classification_targets=targets,
                                regression_targets=regression_targets, class_values=class_values,
                                corruption_mask=mask, corruption_rate=split["corruption_rate"])
        if sha256(path) != dataset_hash:
            raise RuntimeError(f"Dataset changed during evaluation: {path}")
        summaries.append(summary)
        details.append({"file": str(path), "sha256": dataset_hash, "metrics": metrics,
                        "generation_metadata": provenance})
        print(f"{tag} | n={n} | Macro-F1={metrics['macro_f1']:.6f} ({summary['macro_f1_change']:+.6f})"
              f" | MAE={metrics['mae']:.6f} ({summary['mae_change']:+.6f})", flush=True)
        del split, prediction, rows
        gc.collect()

    historical = compare_historical(checkpoint, baseline)
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Existing checkpoint or code changed: {path}")
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), metadata["state_dict"][key]):
            raise AssertionError(f"Model state changed during inference: {key}")
    write_csv(output_dir / "summary.csv", summaries)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
              "checkpoint_sha256": checkpoint_hash, "selected_epoch": metadata["selected_epoch"],
              "checkpoint_config": metadata["config"], "class_values": class_values.tolist(),
              "device": str(next(model.parameters()).device), "batch_size": args.batch_size,
              "protocol": "frozen checkpoint; saved train normalizers; masks=None/shared text attention; no refitting or restoration; test only",
              "scope": "one fixed corruption draw per level; changes relative to re-encoded p000, not a multi-seed robustness estimate",
              "historical_clean_comparison": historical, "summary": summaries, "datasets": details,
              "protected_code_and_checkpoint_sha256": {str(path): value for path, value in protected.items()},
              "software": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__)}}
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("Historical clean comparison: " + json.dumps(historical, ensure_ascii=False), flush=True)
    print(f"EVALUATION_COMPLETE: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

"""Compare frozen ThisWork predictions before/after frozen reconstruction.

Run: .venv-align/bin/python -B evaluate_reconstructed_this_work.py --device cuda:0
Only the four fixed test sets are evaluated. Saved synthetic corruption masks
are known inputs to the reconstructor, not new masks for the sentiment model.
Reconstruction returns raw-unit features: pass them directly to ThisWork's
saved preprocessing, without BERT re-encoding or refitting either normalizer.
Writes metrics and paired predictions only; never saves reconstructed features.
"""
from __future__ import annotations

import argparse
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
from question2.train_recap_reconstructor import load_reconstructor
from multi_fusion_model.this_work import load_experiment_model, predict_split
from multi_fusion_model.weighted_sum_fusion import (
    classification_metrics, regression_metrics, seed_everything,
)
from question2.evaluate_this_work_perturbations import sha256, write_csv

MODS = ("text", "audio", "vision")
LEVELS = (0, 10, 20, 30)
RESULTS = ROOT / "results/second_question"


def array_hashes(arrays):
    return {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in arrays.items()}


def metrics(prediction, classification, regression, classes):
    result = classification_metrics(classification, prediction["predicted_class_indices"], classes)
    result.update(regression_metrics(regression, prediction["regression_prediction"]))
    return result


def check_restoration(original, restored, valid, missing):
    intact = ~missing.any(1)
    for modality in MODS:
        before, after = original[modality], restored[modality]
        if after.shape != before.shape or after.dtype != before.dtype or not np.isfinite(after).all():
            raise AssertionError(f"Invalid reconstructed {modality}")
        change_allowed = valid & missing.any(1, keepdims=True) if modality == "text" else missing
        np.testing.assert_array_equal(after[~change_allowed], before[~change_allowed])
        np.testing.assert_array_equal(after[intact], before[intact])
        if np.any(after[~valid] != 0):
            raise AssertionError("Reconstructed padding must remain zero")
    return {"intact_samples_exactly_preserved": int(intact.sum()),
            "observed_av_exactly_preserved": True, "padding_zero": True,
            "shape_dtype_finite_checks_passed": True}


def historical_check(reference_dir, level, prediction, classification, regression):
    path = reference_dir / f"p{level:03d}_test_predictions.npz"
    with np.load(path, allow_pickle=False) as previous:
        for key in ("ids", "indices", "predicted_class_indices"):
            np.testing.assert_array_equal(prediction[key], previous[key])
        np.testing.assert_array_equal(classification, previous["classification_targets"])
        np.testing.assert_array_equal(regression, previous["regression_targets"])
        errors = {}
        for key in ("classification_logits", "regression_prediction"):
            np.testing.assert_allclose(prediction[key], previous[key], atol=1e-4, rtol=1e-5)
            errors[key + "_max_abs_difference"] = float(np.max(np.abs(prediction[key] - previous[key])))
    return errors


def assert_frozen(model, state):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Model must stay frozen and in eval mode")
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), state[name]):
            raise AssertionError(f"Model state changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructor", type=Path, default=RESULTS / "recap_reconstruction/20260925T103755_004716Z/best.pt")
    parser.add_argument("--checkpoint", type=Path, default=RESULTS / "this_work/20260925_150303_247629/best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/附件2-同步扰动特征")
    parser.add_argument("--reference-dir", type=Path, default=RESULTS / "this_work_robustness/run_20260925T090556_463095Z")
    parser.add_argument("--output-root", type=Path, default=RESULTS / "this_work_reconstruction_evaluation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    for name in ("reconstructor", "checkpoint", "data_dir", "reference_dir", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    paths = [args.data_dir / f"aligned_50_p{level:03d}.pkl" for level in LEVELS]
    reference_files = [args.reference_dir / "report.json", *[
        args.reference_dir / f"p{level:03d}_test_predictions.npz" for level in LEVELS]]
    protected = {path: sha256(path) for path in [args.reconstructor, args.checkpoint, *paths, *reference_files]}
    reference_report = json.loads(reference_files[0].read_text(encoding="utf-8"))
    if reference_report["checkpoint_sha256"] != protected[args.checkpoint]:
        raise ValueError("Historical baseline used a different sentiment checkpoint")
    recon_metadata = torch.load(args.reconstructor, map_location="cpu", weights_only=True)
    for path in paths:
        if recon_metadata["provenance_hashes"].get(str(path)) != protected[path]:
            raise ValueError(f"Reconstructor provenance does not match dataset: {path}")
    torch.set_num_threads(4)
    bundle = load_reconstructor(args.reconstructor, device=args.device)
    model, metadata = load_experiment_model(args.checkpoint, device=args.device)
    seed_everything(metadata["config"]["seed"])
    bundle.model.eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    classes = np.asarray(metadata["class_values"])
    mapping = {float(value): index for index, value in enumerate(classes)}
    # Ignore virtual module filenames; hash only real project source files.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / ".venv-align"):
                protected[path] = sha256(path)
    output_dir = args.output_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Reconstructor epoch: {recon_metadata['epoch']}; ThisWork epoch: {metadata['selected_epoch']}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    summary, details, clean_reference = [], [], None

    for level, path in zip(LEVELS, paths):
        with path.open("rb") as stream:
            dataset = pickle.load(stream)  # Trusted local generated dataset.
        split, provenance = dataset["test"], dataset["_metadata"]
        del dataset  # Train/valid are never used for fitting or prediction.
        if provenance["format"] != "aligned_synchronous_corruption_v1" or not np.isclose(provenance["nominal_probability"], level / 100):
            raise ValueError("Unexpected dataset protocol")
        ids = np.asarray(split["id"], dtype=str)
        n = len(ids)
        labels = np.asarray(split["classification_labels"]).reshape(-1)
        targets = np.asarray([mapping[float(label)] for label in labels], dtype=np.int64)
        regression = np.asarray(split["regression_labels"], dtype=np.float32).reshape(-1)
        if targets.shape != (n,) or regression.shape != (n,) or not np.isfinite(regression).all():
            raise ValueError("Invalid targets")
        tokens = np.asarray(split["text_bert"])
        if tokens.shape != (n, 3, 50) or not np.isin(tokens[:, 1], [0, 1]).all():
            raise ValueError("Invalid token input")
        valid = tokens[:, 1].astype(bool)
        missing = np.asarray(split["corruption_mask"])
        if missing.shape != valid.shape or missing.dtype != np.bool_:
            raise ValueError("Expected explicit boolean corruption mask")
        eligible = valid & ~np.isin(tokens[:, 0], [0, 101, 102])
        if (missing & ~eligible).any() or (tokens[:, 0][missing] != 100).any():
            raise ValueError("Invalid erased slots")
        np.testing.assert_array_equal(missing.sum(1), split["corruption_count"])
        np.testing.assert_array_equal(eligible.sum(1), split["eligible_count"])
        np.testing.assert_allclose(split["corruption_rate"], missing.sum(1) / np.maximum(eligible.sum(1), 1), atol=1e-7)
        if level == 0:
            if missing.any():
                raise ValueError("p000 must have no corruption")
            clean_reference = (ids.copy(), targets.copy(), regression.copy(), tokens.copy())
        else:
            for current, clean in zip((ids, targets, regression, tokens[:, 1:]), (*clean_reference[:3], clean_reference[3][:, 1:])):
                np.testing.assert_array_equal(current, clean)
            expected_ids = clean_reference[3][:, 0].copy()
            expected_ids[missing] = 100
            np.testing.assert_array_equal(tokens[:, 0], expected_ids)
        features = {modality: split[modality] for modality in MODS}
        inputs = {**features, "text_bert": tokens, "id": split["id"]}
        original_hashes = array_hashes({**features, "text_bert": tokens, "missing": missing})
        with torch.inference_mode():
            before = predict_split(model, inputs, metadata, masks=None, batch_size=args.batch_size)
            restored = bundle.reconstruct(features, valid, missing, batch_size=args.batch_size)
            invariants = check_restoration(features, restored, valid, missing)
            # Neither sentiment labels nor clean target features enter either model.
            after = predict_split(model, {**restored, "text_bert": tokens, "id": split["id"]},
                                  metadata, masks=None, batch_size=args.batch_size)
        if original_hashes != array_hashes({**features, "text_bert": tokens, "missing": missing}):
            raise AssertionError("Inference mutated its input arrays")
        for prediction in (before, after):
            np.testing.assert_array_equal(prediction["ids"], ids)
            np.testing.assert_array_equal(prediction["indices"], np.arange(n))
        historical = historical_check(args.reference_dir, level, before, targets, regression)
        if level == 0:
            for key in ("classification_logits", "regression_prediction"):
                np.testing.assert_allclose(after[key], before[key], atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(after["predicted_class_indices"], before["predicted_class_indices"])
        before_metrics = metrics(before, targets, regression, len(classes))
        after_metrics = metrics(after, targets, regression, len(classes))
        correct_before = before["predicted_class_indices"] == targets
        correct_after = after["predicted_class_indices"] == targets
        row = {"nominal_percent": level, "actual_percent": float(missing.sum() / max(int(eligible.sum()), 1) * 100),
               "samples": n, "before_macro_f1": before_metrics["macro_f1"], "after_macro_f1": after_metrics["macro_f1"],
               "macro_f1_gain": after_metrics["macro_f1"] - before_metrics["macro_f1"],
               "before_mae": before_metrics["mae"], "after_mae": after_metrics["mae"],
               "mae_reduction": before_metrics["mae"] - after_metrics["mae"],
               "corrected_classifications": int((~correct_before & correct_after).sum()),
               "degraded_classifications": int((correct_before & ~correct_after).sum())}
        tag = f"p{level:03d}"
        prediction_rows = [{"id": ids[i], "index": i, "true_class": float(labels[i]), "true_score": float(regression[i]),
                            "before_class": float(before["predicted_labels"][i]), "after_class": float(after["predicted_labels"][i]),
                            "before_score": float(before["regression_prediction"][i]), "after_score": float(after["regression_prediction"][i]),
                            "before_absolute_error": float(abs(before["regression_prediction"][i] - regression[i])),
                            "after_absolute_error": float(abs(after["regression_prediction"][i] - regression[i])),
                            "corrupted_slots": int(missing[i].sum()), "corruption_rate": float(split["corruption_rate"][i])}
                           for i in range(n)]
        write_csv(output_dir / f"{tag}_paired_predictions.csv", prediction_rows)
        arrays = {f"{prefix}_{key}": prediction[key] for prefix, prediction in (("before", before), ("after", after))
                  for key in ("classification_logits", "regression_prediction", "predicted_class_indices", "predicted_labels")}
        with (output_dir / f"{tag}_paired_predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays, ids=ids, indices=np.arange(n), class_values=classes,
                                classification_targets=targets, regression_targets=regression,
                                corruption_count=split["corruption_count"], corruption_rate=split["corruption_rate"])
        summary.append(row)
        details.append({"file": str(path), "sha256": protected[path], "before": before_metrics, "after": after_metrics,
                        "restoration_checks": invariants, "baseline_reproduction": historical,
                        "source_arrays_unchanged": True, "generation_metadata": provenance})
        print(f"{tag}: Macro-F1 {row['before_macro_f1']:.6f} -> {row['after_macro_f1']:.6f}; "
              f"MAE {row['before_mae']:.6f} -> {row['after_mae']:.6f}", flush=True)
        del split, features, inputs, restored, before, after, arrays, prediction, prediction_rows
        gc.collect()

    assert_frozen(bundle.model, recon_metadata["model"])
    assert_frozen(model, metadata["state_dict"])
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Existing input, weight or code changed: {path}")
    write_csv(output_dir / "summary.csv", summary)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "reconstructor_checkpoint": str(args.reconstructor), "reconstructor_epoch": recon_metadata["epoch"],
              "reconstructor_config": recon_metadata["config"], "this_work_checkpoint": str(args.checkpoint),
              "this_work_epoch": metadata["selected_epoch"], "this_work_config": metadata["config"],
              "class_values": classes.tolist(), "device": args.device, "batch_size": args.batch_size,
              "protocol": "frozen reconstruction with known synthetic missing slots; raw output -> saved ThisWork preprocessing; unchanged attention; no BERT re-encoding",
              "scope": "four fixed test sets, one corruption draw per level; no refitting, restoration tuning or test-based checkpoint selection",
              "reconstructed_features_saved": False, "model_states_unchanged": True,
              "existing_files_unchanged": True, "p000_identity_verified": True,
              "reference_run": str(args.reference_dir), "summary": summary, "datasets": details,
              "protected_sha256": {str(path): value for path, value in protected.items()},
              "software": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__)}}
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"EVALUATION_COMPLETE: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

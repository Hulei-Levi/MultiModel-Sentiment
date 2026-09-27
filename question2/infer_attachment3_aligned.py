"""Frozen Attachment-3 aligned inference: BERT adapter -> reconstructor -> ThisWork.

Run from any working directory:
  /path/to/MultiModel-Sentiment/.venv-align/bin/python -B \
      /path/to/MultiModel-Sentiment/question2/infer_attachment3_aligned.py

Only this script is added; existing code, datasets and weights are read-only.
Writes predictions, masks and a provenance/statistics report, not feature arrays.
No labels or clean reference are available: no task or reconstruction metrics.
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
for key, value in {
    "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ.setdefault(key, value)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from question2.bert_feature_adapter import load_adapter
from question2.train_recap_reconstructor import load_reconstructor, seed_all
from multi_fusion_model.this_work import load_experiment_model, predict_split

MODS = ("text", "audio", "vision")
RESULTS = ROOT / "results/second_question"
MASK_RULE = "attention=1 AND token_id=100 AND audio_all_zero AND vision_all_zero; exclude token IDs 0/101/102"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hashes(arrays):
    return {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in arrays.items()}


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(model):
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Every module must remain frozen and in evaluation mode")


def infer_masks(tokens, audio, vision):
    valid = tokens[:, 1].astype(bool)
    eligible = valid & ~np.isin(tokens[:, 0], [0, 101, 102])
    audio_zero = (audio == 0).all(-1)
    vision_zero = (vision == 0).all(-1)
    missing = eligible & (tokens[:, 0] == 100) & audio_zero & vision_zero
    return {
        "valid_mask": valid, "eligible_mask": eligible, "corruption_mask": missing,
        "extra_audio_zero_mask": eligible & audio_zero & ~missing,
        "extra_vision_zero_mask": eligible & vision_zero & ~missing,
        "unmatched_unk_mask": eligible & (tokens[:, 0] == 100) & ~missing,
    }


def self_test():
    tokens = np.zeros((2, 3, 8), dtype=np.int64)
    tokens[:, 0] = [101, 100, 100, 2082, 100, 102, 0, 0]
    tokens[:, 1, :6] = 1
    audio = np.zeros((2, 8, 74), dtype=np.float32)
    vision = np.zeros((2, 8, 35), dtype=np.float32)
    audio[:, 1:5] = 1
    vision[:, 1:5] = 1
    audio[0, 1] = vision[0, 1] = 0  # The only synchronous missing slot.
    vision[0, 3] = 0                # Natural/unknown vision-only zero: preserve.
    audio[0, 4] = 0                 # UNK + audio zero alone is insufficient.
    masks = infer_masks(tokens, audio, vision)
    assert np.argwhere(masks["corruption_mask"]).tolist() == [[0, 1]]
    assert np.argwhere(masks["extra_vision_zero_mask"]).tolist() == [[0, 3]]
    assert not masks["corruption_mask"][1].any()
    assert not (masks["corruption_mask"] & ~masks["eligible_mask"]).any()
    assert masks["unmatched_unk_mask"].sum() == 5
    print("SELF_TEST_PASSED: synchronous mask, isolated zeros, unmatched UNK, special tokens, padding", flush=True)


def load_inputs(directory):
    paths = [directory / f"附件3_{i:02d}.pkl" for i in range(1, 31)]
    if set(directory.glob("*.pkl")) != set(paths):
        raise ValueError("Expected exactly the 30 confirmed aligned pkl files")
    protected = {path: sha256(path) for path in paths}
    collected = {k: [] for k in ("text_bert", "audio", "vision")}
    for path in paths:
        with path.open("rb") as stream:
            data = pickle.load(stream)  # User-provided, trusted local dataset.
        if set(data) != {"test"} or set(data["test"]) != set(collected):
            raise ValueError(f"Unexpected input schema (confirmed unlabeled test only): {path}")
        split = data["test"]
        for key, dim in (("text_bert", None), ("audio", 74), ("vision", 35)):
            array = np.asarray(split[key])
            expected = (1, 3, 50) if dim is None else (1, 50, dim)
            if array.shape != expected or array.dtype.kind not in "ifu" or not np.isfinite(array).all():
                raise ValueError(f"Invalid shape/dtype/finite values: {path.name}/{key}")
            if dim is not None and array.dtype.kind != "f":
                raise ValueError(f"Expected floating point AV features: {path.name}/{key}")
            collected[key].append(array.copy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    tokens = arrays["text_bert"]
    if not np.array_equal(tokens, tokens.astype(np.int64)):
        raise ValueError("Token input must contain integral values")
    tokens = tokens.astype(np.int64)
    arrays["text_bert"] = tokens
    if not np.isin(tokens[:, 1], [0, 1]).all() or not np.isin(tokens[:, 2], [0, 1]).all():
        raise ValueError("Invalid attention/token-type channels")
    valid = tokens[:, 1].astype(bool)
    if not valid.any(1).all() or np.any(np.diff(valid.astype(int), axis=1) > 0):
        raise ValueError("Expected nonempty right-padded sequences")
    if np.any(tokens[:, 0][~valid] != 0) or np.any(tokens[:, 0][valid] == 0):
        raise ValueError("Unexpected token padding")
    lengths = valid.sum(1)
    if np.any(tokens[:, 0, 0] != 101) or np.any(tokens[np.arange(30), 0, lengths - 1] != 102):
        raise ValueError("Expected CLS/SEP boundaries")
    for modality in ("audio", "vision"):
        if np.any(arrays[modality][~valid] != 0):
            raise ValueError(f"Nonzero padding in {modality}")
    masks = infer_masks(tokens, arrays["audio"], arrays["vision"])
    if masks["unmatched_unk_mask"].any():
        raise ValueError("New unmatched UNK pattern: differs from the confirmed audit")
    return arrays, masks, paths, protected


def check_restoration(before, after, masks):
    valid, missing = masks["valid_mask"], masks["corruption_mask"]
    intact = ~missing.any(1)
    for modality in MODS:
        original, restored = before[modality], after[modality]
        if restored.shape != original.shape or restored.dtype != original.dtype or not np.isfinite(restored).all():
            raise AssertionError(f"Invalid reconstructed {modality}")
        allowed = valid & missing.any(1, keepdims=True) if modality == "text" else missing
        np.testing.assert_array_equal(restored[~allowed], original[~allowed])
        np.testing.assert_array_equal(restored[intact], original[intact])
        if np.any(restored[~valid] != 0):
            raise AssertionError("Padding changed")
    return {"shape_dtype_finite": True, "observed_av_exactly_preserved": True,
            "unflagged_zero_rows_preserved": True, "padding_zero": True,
            "samples_without_detected_missing_exactly_preserved": int(intact.sum())}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, [0, .25, .5, .75, 1])
    return dict(mean=float(values.mean()), std_population=float(values.std()),
                min=float(q[0]), q25=float(q[1]), median=float(q[2]), q75=float(q[3]), max=float(q[4]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/附件3-模态缺失特征样本/对齐版本")
    parser.add_argument("--adapter", type=Path, default=ROOT / "experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt")
    parser.add_argument("--reconstructor", type=Path, default=RESULTS / "recap_reconstruction/20260925T103755_004716Z/best.pt")
    parser.add_argument("--checkpoint", type=Path, default=RESULTS / "this_work/20260925_150303_247629/best.pt")
    parser.add_argument("--output-root", type=Path, default=RESULTS / "attachment3_aligned_inference")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for key in ("data_dir", "adapter", "reconstructor", "checkpoint", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    torch.set_num_threads(4)
    seed_all(42)
    torch.backends.cudnn.enabled = True
    arrays, masks, paths, protected = load_inputs(args.data_dir)
    ids = np.asarray([path.stem for path in paths], dtype=str)
    input_digest = array_hashes({**arrays, **masks})
    for path in (args.adapter, args.reconstructor, args.checkpoint):
        protected[path] = sha256(path)
    adapter_meta = torch.load(args.adapter, map_location="cpu", weights_only=True)
    recon_meta = torch.load(args.reconstructor, map_location="cpu", weights_only=True)
    expected_adapter = recon_meta["provenance_hashes"].get(recon_meta["config"]["adapter_checkpoint"])
    if expected_adapter != protected[args.adapter]:
        raise ValueError("Adapter checkpoint differs from the reconstructor's training encoder")
    if recon_meta["format"] != "recap_inspired_reconstructor_v1":
        raise ValueError("Expected the selected synchronous-missingness reconstructor")
    model_file = Path(adapter_meta["model_dir"]) / "model.safetensors"
    protected[model_file] = sha256(model_file)
    if protected[model_file] != adapter_meta["model_sha256"]:
        raise ValueError("BERT model hash mismatch")
    for path in model_file.parent.glob("*.json"):
        protected[path] = sha256(path)
    print(f"INPUT_AUDIT: samples={len(ids)}, content_slots={masks['eligible_mask'].sum()}, "
          f"detected_missing={masks['corruption_mask'].sum()}", flush=True)

    adapter = load_adapter(str(args.adapter), device=args.device)
    adapter.eval().requires_grad_(False)
    assert_frozen(adapter)
    adapter_state_before = state_hash(adapter)
    print("Encoding original corrupted tokens with frozen BERT + linear adapter...", flush=True)
    features = {m: arrays[m] for m in ("audio", "vision")}
    features["text"] = adapter.encode_text_bert(
        arrays["text_bert"], batch_size=args.batch_size, zero_padding=True, apply_adapter=True)
    if features["text"].shape != (len(ids), 50, 768) or features["text"].dtype != np.float32:
        raise AssertionError("Invalid encoded text shape/dtype")
    assert_frozen(adapter)
    if state_hash(adapter) != adapter_state_before:
        raise AssertionError("Text encoder state changed")
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bundle = load_reconstructor(args.reconstructor, device=args.device)
    model, metadata = load_experiment_model(args.checkpoint, device=args.device)
    bundle.model.eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    if metadata["config"]["task"] != "multitask" or metadata["feature_dims"] != {"text": 768, "audio": 74, "vision": 35}:
        raise ValueError("Unexpected ThisWork contract")
    classes = np.asarray(metadata["class_values"])
    before_states = {"reconstructor": state_hash(bundle.model), "this_work": state_hash(model)}
    feature_digest = array_hashes(features)
    # Include the actual relocated modules, not stale filenames in old checkpoints.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / ".venv-align"):
                protected[path] = sha256(path)
    protected[Path(__file__).resolve()] = sha256(__file__)
    print(f"Reconstructing: epoch {recon_meta['epoch']}; ThisWork: epoch {metadata['selected_epoch']}", flush=True)
    with torch.inference_mode():
        # Match reconstructor training arithmetic; restore all flags explicitly.
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
            restored = bundle.reconstruct(features, masks["valid_mask"], masks["corruption_mask"],
                                          batch_size=args.batch_size)
        restoration_checks = check_restoration(features, restored, masks)
        # Historical ThisWork predictions used cuDNN TF32 with FP32 matmul.
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True):
            prediction = predict_split(model, {**restored, "text_bert": arrays["text_bert"], "id": ids.tolist()},
                                       metadata, masks=None, batch_size=args.batch_size)

    if feature_digest != array_hashes(features) or input_digest != array_hashes({**arrays, **masks}):
        raise AssertionError("Input arrays were mutated")
    for key, module in (("reconstructor", bundle.model), ("this_work", model)):
        assert_frozen(module)
        if state_hash(module) != before_states[key]:
            raise AssertionError(f"{key} state changed")
    np.testing.assert_array_equal(prediction["ids"], ids)
    np.testing.assert_array_equal(prediction["indices"], np.arange(len(ids)))
    logits = np.asarray(prediction["classification_logits"])
    scores = np.asarray(prediction["regression_prediction"]).reshape(-1)
    predicted_indices = np.asarray(prediction["predicted_class_indices"])
    if logits.shape != (len(ids), len(classes)) or scores.shape != (len(ids),):
        raise AssertionError("Invalid prediction shapes")
    if not np.isfinite(logits).all() or not np.isfinite(scores).all():
        raise AssertionError("Nonfinite predictions")
    probability = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probability = np.exp(probability)
    probability /= probability.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probability.sum(1), 1, atol=1e-12)
    np.testing.assert_array_equal(predicted_indices, logits.argmax(1))
    np.testing.assert_array_equal(prediction["predicted_labels"], classes[predicted_indices])
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source data, weight or code changed: {path}")

    missing_count = masks["corruption_mask"].sum(1)
    eligible_count = masks["eligible_mask"].sum(1)
    rates = missing_count / np.maximum(eligible_count, 1)
    rows = []
    for i, path in enumerate(paths):
        row = {"sample_id": ids[i], "source_file": path.name, "source_row_index": 0,
               "valid_slots": int(masks["valid_mask"][i].sum()), "content_slots": int(eligible_count[i]),
               "detected_missing_slots": int(missing_count[i]), "detected_missing_rate": float(rates[i]),
               "detected_missing_positions_1based": json.dumps((np.flatnonzero(masks["corruption_mask"][i]) + 1).tolist()),
               "extra_audio_zero_positions_1based": json.dumps((np.flatnonzero(masks["extra_audio_zero_mask"][i]) + 1).tolist()),
               "extra_vision_zero_positions_1based": json.dumps((np.flatnonzero(masks["extra_vision_zero_mask"][i]) + 1).tolist()),
               "reconstruction_applied": bool(missing_count[i] > 0),
               "predicted_class_index": int(predicted_indices[i]), "predicted_class_value": float(classes[predicted_indices[i]]),
               "regression_prediction": float(scores[i])}
        for j, value in enumerate(classes):
            row[f"probability_class_{float(value):g}"] = float(probability[i, j])
        rows.append(row)
    class_rows = [{"class_index": j, "class_value": float(value),
                   "count": int((predicted_indices == j).sum()),
                   "proportion": float((predicted_indices == j).mean())} for j, value in enumerate(classes)]
    summary = {
        "sample_count": len(ids), "predicted_class_distribution": class_rows,
        "regression_score_distribution": describe(scores),
        "missingness": {
            "content_slots": int(eligible_count.sum()), "detected_missing_slots": int(missing_count.sum()),
            "detected_missing_rate_micro": float(missing_count.sum() / eligible_count.sum()),
            "per_sample_missing_rate_distribution": describe(rates),
            "samples_with_detected_missing": int((missing_count > 0).sum()),
            "samples_without_detected_missing": ids[missing_count == 0].tolist(),
            "extra_audio_zero_slots_preserved": int(masks["extra_audio_zero_mask"].sum()),
            "extra_vision_zero_slots_preserved": int(masks["extra_vision_zero_mask"].sum()),
        },
    }
    output = args.output_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "class_distribution.csv", class_rows)
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(
            stream, ids=ids, source_files=np.asarray([p.name for p in paths], dtype=str),
            indices=np.arange(len(ids)), class_values=classes,
            classification_logits=logits, classification_probabilities=probability,
            predicted_class_indices=predicted_indices, predicted_labels=classes[predicted_indices],
            regression_prediction=scores, **masks, missing_count=missing_count,
            eligible_count=eligible_count, detected_missing_rate=rates)
    report = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output), "input_dir": str(args.data_dir),
        "summary": summary, "class_semantics": "Numeric checkpoint class values only; no semantic label mapping assumed",
        "labels_available": False, "clean_reference_available": False,
        "unavailable_metrics": ["macro_f1", "accuracy", "regression_mae", "reconstruction_mae"],
        "adapter": {"checkpoint": str(args.adapter), "model_id": adapter_meta["model_id"],
                    "model_revision": adapter_meta["model_revision"], "model_dir": adapter_meta["model_dir"],
                    "selected_ridge_alpha": adapter_meta["selected_alpha"], "apply_adapter": True, "zero_padding": True,
                    "actual_module": str(ROOT / "question2/bert_feature_adapter.py")},
        "reconstructor": {"checkpoint": str(args.reconstructor), "epoch": recon_meta["epoch"],
                          "description": bundle.model.description, "config": recon_meta["config"],
                          "normalizer_source": "Selected checkpoint; no refit",
                          "training_adapter_checkpoint_hash_verified": True},
        "this_work": {"checkpoint": str(args.checkpoint), "epoch": metadata["selected_epoch"],
                      "config": metadata["config"], "class_values": classes.tolist(),
                      "normalizer_source": "Selected checkpoint; standardize once through predict_split"},
        "protocol": {
            "pipeline": "Original corrupted tokens -> frozen BERT+adapter -> reconstruction -> frozen ThisWork",
            "missing_mask_rule": MASK_RULE,
            "missing_mask_status": "Heuristic confirmed with user; not official ground truth",
            "missing_rate_denominator": "Valid content slots, excluding CLS/SEP/padding; not all 50 slots",
            "slot_indexing": "CSV positions are 1-based; NPZ array positions are 0-based",
            "text_repair": "All valid slots of samples with detected missingness",
            "av_repair": "Only detected synchronous missing slots; preserve other rows exactly",
            "padding": "Zero; original attention mask retained, including detected missing positions",
            "downstream_masks": "masks=None; existing shared text attention-mask protocol",
            "scope": "30 aligned files only; no training, refitting, tuning, raw-only comparison or feature export",
            "probabilities": "Uncalibrated softmax scores, not a guarantee of correctness",
            "regression": "Raw regression-head prediction; no clipping or class-based adjustment",
        },
        "verification": {**restoration_checks, "all_models_frozen": True, "model_states_unchanged": True,
                         "source_arrays_unchanged": True, "protected_files_unchanged": True,
                         "finite_predictions": True, "prediction_order_verified": True,
                         "probability_row_sums_verified": True, "reconstructed_features_saved": False},
        "numerics": {"device": args.device, "batch_size": args.batch_size, "seed": 42,
                     "dtype": "FP32 models", "matmul_tf32": False, "cudnn_enabled": True,
                     "cudnn_deterministic": True, "cudnn_benchmark": False,
                     "bert_and_reconstructor_cudnn_tf32": False, "this_work_cudnn_tf32": True},
        "protected_sha256": {str(path): value for path, value in sorted(protected.items())},
        "software": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__)},
    }
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    dist = summary["regression_score_distribution"]
    lines = [
        "# 附件 3 aligned：重建后 ThisWork 推理汇总", "",
        f"- 输入：{len(ids)} 个文件，每文件 1 个样本；全部处理完成。",
        f"- 重建权重：{args.reconstructor}（epoch {recon_meta['epoch']}）。",
        "- 此重建器是自定义多尺度卷积＋联合注意力残差模型，历史命名为 RECAP-inspired，并非原论文完整复现。",
        f"- ThisWork 权重：{args.checkpoint}（epoch {metadata['selected_epoch']}）。",
        f"- 文本编码器：{args.adapter}；冻结 BERT＋线性适配，保留原 [UNK]。",
        "- 使用各自权重保存的标准化参数，不重新拟合；先返回原单位重建特征，再进入 ThisWork 预处理。",
        "- 无真实标签及完整特征对照，不计算 Macro-F1、准确率、回归 MAE 或重建 MAE。",
        "- 以下是预测分布，不代表准确率或实际情感分布；概率未经校准。类别保留权重内编号，不推断其语义。", "",
        "## 预测类别分布", "",
        "| 类别编号 | 数量 | 占比 |", "|---|---:|---:|",
    ]
    lines += [f"| {x['class_value']:g} | {x['count']} | {x['proportion']:.2%} |" for x in class_rows]
    lines += [
        "", "## 回归分数分布", "",
        f"- 均值 {dist['mean']:.6f}，中位数 {dist['median']:.6f}，总体标准差 {dist['std_population']:.6f}。",
        f"- 最小值 {dist['min']:.6f}，Q1 {dist['q25']:.6f}，Q3 {dist['q75']:.6f}，最大值 {dist['max']:.6f}。",
        "", "## 缺失判定及重建范围", "",
        f"- 规则：{MASK_RULE}。该规则是数据观察后的约定，不是官方缺失真值。",
        f"- 共 {int(eligible_count.sum())} 个有效内容槽，识别 {int(missing_count.sum())} 个同步缺失槽，比例 {summary['missingness']['detected_missing_rate_micro']:.2%}。",
        f"- {int((missing_count > 0).sum())} 个样本应用重建；无该缺失标记的样本：{', '.join(ids[missing_count == 0])}，保持编码后的特征不变。",
        f"- 另有仅凭零值无法确定原因的视觉零槽 {int(masks['extra_vision_zero_mask'].sum())} 个，按约定保留。",
        "- 有缺失标记时，文本修复全部有效槽；音视频只修复已识别槽；特殊符号不算缺失，padding 保持零。",
        "- 文本恢复的是向量，不是原始词；本文比例以内容槽为分母，而非固定 50 步。",
        "", "## 逐样本结果", "",
        "| 文件 | 缺失槽/内容槽 | 缺失率 | 预测类别 | 回归分数 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [f"| {row['source_file']} | {row['detected_missing_slots']}/{row['content_slots']} | "
              f"{row['detected_missing_rate']:.2%} | {row['predicted_class_value']:g} | {row['regression_prediction']:.6f} |"
              for row in rows]
    lines += [
        "", "## 文件与校验", "",
        "- predictions.csv：逐样本结果、各类别概率、1-based 原槽位编号。",
        "- predictions.npz：预测数组及完整布尔 mask；数组索引为 0-based，不含特征或标签。",
        "- class_distribution.csv：预测类别数量与比例。",
        "- report.json：版本、路径、哈希、统计、数值设置及校验记录。",
        "- 已验证观测音视频、未识别为缺失的额外零行及无缺失标记样本保持原值；padding 为零；预测顺序正确。",
        "- 原文件、权重和模型状态未改变；未保存重建特征。", "",
    ]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ids"], ids)
        np.testing.assert_array_equal(saved["corruption_mask"], masks["corruption_mask"])
        np.testing.assert_array_equal(saved["regression_prediction"], scores)
        if any(k in saved for k in MODS):
            raise AssertionError("Feature arrays must not be exported")
    print("SUMMARY: " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"INFERENCE_COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()

"""Missing-position ablation with frozen reconstructors and frozen ThisWork.

One added script. Test-only head/center/tail contiguous, synchronous erasure at
10/20/30/40/50 percent. Existing datasets/code/checkpoints remain unchanged.
Run: .venv-align/bin/python -B question2/evaluate_missing_positions.py --devices cuda:0 cuda:1
Only predictions, masks, metrics and reports are persisted; no feature datasets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
import subprocess
import os
from pathlib import Path
import pickle
import sys
import time

sys.dont_write_bytecode = True
for key, value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "TORCHINDUCTOR_COMPILE_THREADS": "1"}.items():
    os.environ.setdefault(key, value)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from question2.bert_feature_adapter import load_adapter
from question2.train_recap_reconstructor import load_reconstructor, seed_all
from question2.generate_perturbed_dataset import content_mask
from question2.evaluate_reconstructed_this_work import check_restoration, metrics
from question2.infer_attachment3_aligned import sha256, array_hashes, state_hash, assert_frozen, write_csv
from multi_fusion_model.this_work import load_experiment_model, predict_split

RESULTS = ROOT / "results/second_question"
HISTORY = RESULTS / "frozen_high_missing_evaluation/run_20260926T094125_413222Z/report.json"
CLEAN = ROOT / "datasets/附件2-同步扰动特征/aligned_50_p000.pkl"
ANCHOR_DATA = ROOT / "datasets/附件2-同步扰动特征/aligned_50_p010.pkl"
ADAPTER = ROOT / "experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt"
BASE_REFERENCE = RESULTS / "this_work_robustness/run_20260925T090556_463095Z/p000_test_predictions.npz"
METHODS = ("MTSIT", "Centaur", "CSDI", "NAOMI", "BRITS", "RECAP")
DISPLAY = {"unreconstructed": "未重构", "RECAP": "本文重构器"}
ORDER = ("unreconstructed", *METHODS)
POSITIONS = ("head", "center", "tail")
POSITION_NAMES = {"head": "首部", "center": "中部", "tail": "尾部"}
RATES = (10, 20, 30, 40, 50)
MODS = ("text", "audio", "vision")
REFERENCES = {
    "MTSIT": "mtsit_this_work_evaluation/run_20260926T031219_565187Z",
    "Centaur": "centaur_this_work_evaluation/run_20260926T051546_307389Z",
    "CSDI": "csdi_this_work_evaluation/run_20260926T073855_150008Z",
    "NAOMI": "naomi_this_work_evaluation/run_20260926T054636_947171Z",
    "BRITS": "brits_this_work_evaluation/run_20260926T043408_925950Z",
    "RECAP": "this_work_reconstruction_evaluation/run_20260925T111339_061214Z",
}


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def position_mask(tokens, percent, position):
    if percent not in RATES or position not in POSITIONS:
        raise ValueError("Unknown position or severity")
    eligible = content_mask(tokens)
    missing = np.zeros_like(eligible)
    # Integer round-half-up, including zero when rounding a short sequence.
    counts = (eligible.sum(1) * percent + 50) // 100
    for i, count in enumerate(counts):
        slots = np.flatnonzero(eligible[i])
        if len(slots) > 1 and not np.all(np.diff(slots) == 1):
            raise ValueError("Content slots must be a contiguous sequence")
        count = int(count)
        start = 0 if position == "head" else len(slots) - count if position == "tail" else (len(slots) - count) // 2
        missing[i, slots[start:start + count]] = True
    np.testing.assert_array_equal(missing.sum(1), counts)
    assert not (missing & ~eligible).any()
    return missing


def self_test():
    lengths = np.arange(2, 49)
    tokens = np.zeros((len(lengths), 3, 50), np.int64)
    for i, length in enumerate(lengths):
        tokens[i, 0, 0] = 101
        tokens[i, 0, 1:length + 1] = 1234
        tokens[i, 0, length + 1] = 102
        tokens[i, 1, :length + 2] = 1
    for percent in RATES:
        masks = [position_mask(tokens, percent, p) for p in POSITIONS]
        for i, length in enumerate(lengths):
            count = int((length * percent + 50) // 100)
            for p, mask in zip(POSITIONS, masks):
                slots = np.flatnonzero(mask[i])
                assert len(slots) == count
                if count:
                    assert np.all(np.diff(slots) == 1)
                    if p == "head": assert slots[0] == 1
                    if p == "tail": assert slots[-1] == length
                    if p == "center": assert abs((slots[0] - 1) - (length - slots[-1])) <= 1
            assert len({int(m[i].sum()) for m in masks}) == 1
    assert position_mask(tokens[:1], 10, "head").sum() == 0
    log("SELF_TEST_PASSED: 47 lengths x 5 rates x 3 positions; counts/centering/boundaries/padding")


def load_test(path):
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)  # Trusted project datasets.
    split = data["test"]
    metadata = data["_metadata"]
    # Do not retain or use train/valid data.
    result = {key: split[key] for key in ("text", "audio", "vision", "text_bert", "id",
                                        "classification_labels", "regression_labels", "corruption_mask")}
    return result, metadata


def predict(model, meta, features, tokens, ids):
    with torch.inference_mode():
        return predict_split(model, {**features, "text_bert": tokens, "id": ids},
                             meta, masks=None, batch_size=32)


def check_prediction(pred, ids):
    np.testing.assert_array_equal(pred["ids"], np.asarray(ids, dtype=str))
    np.testing.assert_array_equal(pred["indices"], np.arange(len(ids)))
    for key in ("classification_logits", "regression_prediction"):
        if not np.isfinite(pred[key]).all():
            raise AssertionError("Nonfinite prediction")


def prediction_arrays(pred, classes, targets, scores):
    logits = np.asarray(pred["classification_logits"])
    probability = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    return {k: pred[k] for k in ("ids", "indices", "classification_logits",
                                 "predicted_class_indices", "predicted_labels", "regression_prediction")} | {
        "class_values": classes, "classification_probabilities": probability,
        "classification_targets": targets, "regression_targets": scores}


def verify_reference(pred, path, prefix="", count=None):
    with np.load(path, allow_pickle=False) as saved:
        n = len(pred["ids"]) if count is None else count
        np.testing.assert_array_equal(pred["ids"][:n], saved["ids"][:n])
        errors = {}
        for key in ("classification_logits", "regression_prediction"):
            reference = saved[prefix + key][:n]
            current = pred[key][:n]
            np.testing.assert_allclose(current, reference, atol=1e-4, rtol=1e-5)
            errors[key] = float(np.abs(current - reference).max())
        np.testing.assert_array_equal(pred["predicted_class_indices"][:n], saved[prefix + "predicted_class_indices"][:n])
        return errors


def worker(payload):
    device, conditions, output_name, registry, worker_key = payload
    label = f"{worker_key} (logical {device})"
    output = Path(output_name)
    torch.set_num_threads(4)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    seed_all(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Match the historical table's inference backend, not training-time defaults.
    clean, clean_meta = load_test(CLEAN)
    anchor, _ = load_test(ANCHOR_DATA)
    anchor = {key: value[:32] for key, value in anchor.items()}
    ids = clean["id"]
    valid = clean["text_bert"][:, 1].astype(bool)
    eligible = content_mask(clean["text_bert"])
    if clean["corruption_mask"].any():
        raise ValueError("The clean source has artificial corruption")
    original_arrays = {k: np.asarray(clean[k]) for k in (*MODS, "text_bert", "classification_labels", "regression_labels")}
    source_hashes = array_hashes(original_arrays)
    model, meta = load_experiment_model(registry["this_work"], device=device)
    model.eval().requires_grad_(False)
    classes = np.asarray(meta["class_values"])
    mapping = {float(value): i for i, value in enumerate(classes)}
    targets = np.asarray([mapping[float(x)] for x in clean["classification_labels"]], dtype=np.int64)
    scores = np.asarray(clean["regression_labels"], dtype=np.float32).reshape(-1)
    initial_model_state = state_hash(model)
    clean_pred = predict(model, meta, {m: clean[m] for m in MODS}, clean["text_bert"], ids)
    check_prediction(clean_pred, ids)
    clean_check = verify_reference(clean_pred, BASE_REFERENCE)
    clean_metrics = metrics(clean_pred, targets, scores, len(classes))
    log(f"{label}: clean reference reproduced; Macro-F1={clean_metrics['macro_f1']:.6f}, MAE={clean_metrics['mae']:.6f}")
    adapter = load_adapter(str(ADAPTER), device=device)
    adapter.eval().requires_grad_(False)
    adapter_state = state_hash(adapter)
    if clean_meta["checkpoint_sha256"] != sha256(ADAPTER):
        raise ValueError("Clean source encoder provenance mismatch")
    spot = np.asarray([0, len(ids)//2, len(ids)-1])
    encoded = adapter.encode_text_bert(clean["text_bert"][spot], batch_size=32, zero_padding=True, apply_adapter=True)
    np.testing.assert_allclose(encoded, clean["text"][spot], atol=1e-4, rtol=2e-4)
    bundles, before_states, anchor_checks = {}, {}, {}
    # Load CSDI last to allow fast-method anchors to fail early if needed.
    for method in ("MTSIT", "Centaur", "NAOMI", "BRITS", "RECAP", "CSDI"):
        bundle = load_reconstructor(registry["methods"][method], device=device)
        bundle.model.eval().requires_grad_(False)
        bundles[method] = bundle
        before_states[method] = state_hash(bundle.model)
        with torch.inference_mode():
            restored = bundle.reconstruct({m: anchor[m] for m in MODS},
                                          anchor["text_bert"][:, 1].astype(bool), anchor["corruption_mask"], batch_size=32)
        check_restoration({m: anchor[m] for m in MODS}, restored, anchor["text_bert"][:, 1].astype(bool), anchor["corruption_mask"])
        pred = predict(model, meta, restored, anchor["text_bert"], anchor["id"])
        reference = RESULTS / REFERENCES[method] / "p010_paired_predictions.npz"
        anchor_checks[method] = verify_reference(pred, reference, prefix="after_", count=32)
        log(f"{label}: {method} historical p010 anchor passed")

    records = []
    for position, percent in conditions:
        tag = f"{position}_p{percent:03d}"
        condition_dir = output / tag
        if (condition_dir / "complete.json").exists():
            cached = json.loads((condition_dir / "complete.json").read_text())["records"]
            records.extend(cached)
            log(f"{label}: reused completed {tag}")
            continue
        condition_dir.mkdir(exist_ok=True)
        missing = position_mask(clean["text_bert"], percent, position)
        tokens = clean["text_bert"].copy()
        tokens[:, 0][missing] = 100
        np.testing.assert_array_equal(tokens[:, 1:], clean["text_bert"][:, 1:])
        features = {"text": adapter.encode_text_bert(tokens, batch_size=32, zero_padding=True, apply_adapter=True)}
        for modality in ("audio", "vision"):
            features[modality] = clean[modality].copy()
            features[modality][missing] = 0
            np.testing.assert_array_equal(features[modality][~missing], clean[modality][~missing])
        for modality in MODS:
            if np.any(features[modality][~valid] != 0):
                raise AssertionError("Nonzero feature padding")
        corrupt_hashes = array_hashes({**features, "tokens": tokens, "missing": missing})
        mask_path = condition_dir / "masks.npz"
        if mask_path.exists():
            with np.load(mask_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ids"], np.asarray(ids, dtype=str))
                np.testing.assert_array_equal(saved["corruption_mask"], missing)
        else:
            with mask_path.open("xb") as stream:
                np.savez_compressed(stream, ids=np.asarray(ids, dtype=str), valid_mask=valid, eligible_mask=eligible,
                                    corruption_mask=missing, missing_count=missing.sum(1), eligible_count=eligible.sum(1))
        baseline_path = condition_dir / "unreconstructed_predictions.npz"
        baseline_begin = time.perf_counter()
        if baseline_path.exists():
            with np.load(baseline_path, allow_pickle=False) as saved:
                baseline = {key: saved[key] for key in saved.files}
        else:
            baseline = predict(model, meta, features, tokens, ids)
        baseline_seconds = time.perf_counter() - baseline_begin
        condition_records = []
        for method in ("unreconstructed", "MTSIT", "Centaur", "NAOMI", "BRITS", "RECAP", "CSDI"):
            result_path = condition_dir / f"{method.lower()}_metrics.json"
            prediction_path = condition_dir / f"{method.lower()}_predictions.npz"
            if result_path.exists() and prediction_path.exists():
                cached_row = json.loads(result_path.read_text())["summary"]
                with np.load(prediction_path, allow_pickle=False) as saved:
                    check_prediction(saved, ids)
                    np.testing.assert_array_equal(saved["classification_targets"], targets)
                    np.testing.assert_array_equal(saved["regression_targets"], scores)
                    check = metrics(saved, targets, scores, len(classes))
                    np.testing.assert_allclose([cached_row["macro_f1"], cached_row["mae"]],
                                               [check["macro_f1"], check["mae"]], atol=1e-12)
                condition_records.append(cached_row)
                records.append(cached_row)
                log(f"{label}: reused {tag}/{method}")
                continue
            if result_path.exists() or prediction_path.exists():
                raise ValueError(f"Incomplete result pair: {tag}/{method}")
            begin = time.perf_counter()
            if method == "unreconstructed":
                pred = baseline
                checks = None
            else:
                with torch.inference_mode():
                    restored = bundles[method].reconstruct(features, valid, missing, batch_size=32)
                checks = check_restoration(features, restored, valid, missing)
                pred = predict(model, meta, restored, tokens, ids)
                del restored
            check_prediction(pred, ids)
            measured = metrics(pred, targets, scores, len(classes))
            row = {"position": position, "position_name": POSITION_NAMES[position], "nominal_percent": percent,
                   "actual_percent": float(100 * missing.sum() / eligible.sum()), "method": method,
                   "method_name": DISPLAY.get(method, method), "samples": len(ids),
                   "missing_slots": int(missing.sum()), "content_slots": int(eligible.sum()),
                   "samples_without_corruption": int((~missing.any(1)).sum()),
                   "macro_f1": measured["macro_f1"], "mae": measured["mae"],
                   "accuracy": measured["accuracy"], "rmse": measured["rmse"],
                   "seconds": baseline_seconds if method == "unreconstructed" else time.perf_counter() - begin}
            with (condition_dir / f"{method.lower()}_predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, **prediction_arrays(pred, classes, targets, scores))
            write_json(condition_dir / f"{method.lower()}_metrics.json",
                       {"summary": row, "metrics": measured, "restoration_checks": checks})
            condition_records.append(row)
            records.append(row)
            log(f"{label} {tag} {method}: Macro-F1={row['macro_f1']:.4f}, MAE={row['mae']:.4f} ({row['seconds']:.1f}s)")
        if corrupt_hashes != array_hashes({**features, "tokens": tokens, "missing": missing}):
            raise AssertionError("Corrupted inputs changed across methods")
        write_json(condition_dir / "complete.json", {"records": condition_records, "same_inputs_for_all_methods": True})
        del features, tokens, baseline
        gc.collect()
    if array_hashes(original_arrays) != source_hashes:
        raise AssertionError("Original clean arrays changed")
    if state_hash(model) != initial_model_state or state_hash(adapter) != adapter_state:
        raise AssertionError("Frozen sentiment/text encoder state changed")
    assert_frozen(model)
    assert_frozen(adapter)
    for method, bundle in bundles.items():
        assert_frozen(bundle.model)
        if state_hash(bundle.model) != before_states[method]:
            raise AssertionError(f"{method} state changed")
    verification = {"device": device, "worker_key": worker_key, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "clean_metrics": clean_metrics, "clean_reference_check": clean_check,
                    "historical_p010_anchor_checks": anchor_checks, "source_arrays_unchanged": True,
                    "model_states_unchanged": True, "all_modules_frozen": True,
                    "condition_count": len(conditions), "normalizers_refitted": False,
                    "original_bert_attention_for_this_work": True}
    write_json(output / ("worker_" + worker_key + ".json"), {"records": records, **verification})
    return records, verification


def aggregate(output, records, verifications, manifest):
    if len(records) != 105 or len({(r["position"], r["nominal_percent"], r["method"]) for r in records}) != 105:
        raise AssertionError("Expected all 3 x 5 x 7 results")
    records.sort(key=lambda r: (POSITIONS.index(r["position"]), r["nominal_percent"], ORDER.index(r["method"])))
    lookup = {(r["position"], r["method"], r["nominal_percent"]): r for r in records}
    for r in records:
        baseline = lookup[r["position"], "unreconstructed", r["nominal_percent"]]
        r["macro_f1_gain_vs_unreconstructed"] = r["macro_f1"] - baseline["macro_f1"]
        r["mae_reduction_vs_unreconstructed"] = baseline["mae"] - r["mae"]
    # Recompute the requested metrics from saved predictions, independently of recorded summaries.
    for r in records:
        path = output / f"{r['position']}_p{r['nominal_percent']:03d}" / f"{r['method'].lower()}_predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            measured = metrics(saved, saved["classification_targets"], saved["regression_targets"], len(saved["class_values"]))
            np.testing.assert_allclose([r["macro_f1"], r["mae"]], [measured["macro_f1"], measured["mae"]], atol=1e-12)
    for percent in RATES:
        masks = []
        for position in POSITIONS:
            with np.load(output / f"{position}_p{percent:03d}" / "masks.npz", allow_pickle=False) as saved:
                masks.append(saved["missing_count"])
        for count in masks[1:]:
            np.testing.assert_array_equal(count, masks[0])
    write_csv(output / "summary.csv", records)
    tables = []
    for position in POSITIONS:
        rows = []
        for metric, label in (("macro_f1", "Macro-F1 ↑"), ("mae", "MAE ↓")):
            for method in ORDER:
                rows.append({"评价指标": label, "方法": DISPLAY.get(method, method),
                             **{f"{rate}%": lookup[position, method, rate][metric] for rate in RATES}})
        write_csv(output / f"table_{position}.csv", rows)
        tables.append((position, rows))
    sensitivity = []
    for metric in ("macro_f1", "mae"):
        for method in ORDER:
            for rate in RATES:
                values = {p: lookup[p, method, rate][metric] for p in POSITIONS}
                worst = min(values, key=values.get) if metric == "macro_f1" else max(values, key=values.get)
                sensitivity.append({"metric": metric, "method": method, "method_name": DISPLAY.get(method, method),
                                    "nominal_percent": rate, **values, "max_minus_min": max(values.values()) - min(values.values()),
                                    "worst_position": worst})
    write_csv(output / "position_sensitivity.csv", sensitivity)
    lines = [
        "# 缺失位置消融：首部、中部、尾部连续片段", "",
        "- 附件 2 干净测试集 727 个样本；冻结已有六份重建权重、文本编码器及 ThisWork，不重新训练或选模。",
        "- 三模态同槽同步扰动：文本 token 替换为 [UNK] 后整段重新编码，音视频对应整行置零。",
        "- 有效内容槽排除 padding、[CLS]、[SEP]；保持原 attention mask。每样本缺失数 k=floor(L×比例+0.5)。",
        "- 首部从第一个内容槽开始，尾部以最后内容槽结束，中部左右剩余长度之差至多为 1（多出的保留槽在右侧）。",
        "- 同一样本、同一比例下首/中/尾缺失数量严格相同；短样本可四舍五入为 0，此时保持未扰动。",
        "- MAE 是下游情感回归误差，不是重建误差；Macro-F1 对权重中的全部三个类别求平均。",
        "- 沿用各方法原有修复策略及训练标准化参数；CSDI 保留 50 步、3 轨迹、固定 seed=314159、均值聚合。",
        "- 此实验考察已有权重对连续片段缺失位置的敏感性。原表为随机散点缺失，因此两表差异不能只归因于位置。",
        "- 每个位置/比例只有一个确定性缺失方案，未估计训练随机性或统计显著性；不按测试表现重新选模型。", "",
        "## 实际缺失率", "", "| 标称比例 | 三个位置共同的实际比例 | 每位置未扰动样本数 |", "|---|---:|---:|",
    ]
    for rate in RATES:
        r = lookup["head", "unreconstructed", rate]
        lines.append(f"| {rate}% | {r['actual_percent']:.4f}% | {r['samples_without_corruption']} |")
    clean = verifications[0]["clean_metrics"]
    lines += ["", f"干净测试基线：Macro-F1={clean['macro_f1']:.6f}，MAE={clean['mae']:.6f}；已复现历史预测。"]
    for position, rows in tables:
        lines += ["", f"## {POSITION_NAMES[position]}缺失", "",
                  "| 评价指标 | 方法 | 10% | 20% | 30% | 40% | 50% |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| " + row["评价指标"] + " | " + row["方法"] + " | " +
                         " | ".join(f"{row[f'{rate}%']:.4f}" for rate in RATES) + " |")
    lines += ["", "## 位置敏感性汇总", "", "| 方法 | 五档平均 Macro-F1（首/中/尾） | 五档平均 MAE（首/中/尾） |",
              "|---|---|---|"]
    for method in ORDER:
        f1 = [np.mean([lookup[p, method, r]["macro_f1"] for r in RATES]) for p in POSITIONS]
        mae = [np.mean([lookup[p, method, r]["mae"] for r in RATES]) for p in POSITIONS]
        lines.append(f"| {DISPLAY.get(method,method)} | " + " / ".join(f"{v:.4f}" for v in f1) + " | " +
                     " / ".join(f"{v:.4f}" for v in mae) + " |")
    lines += ["", "说明：上述平均仅为五个缺失率的算术汇总，不是对模型重新选模的总分。",
              "", "## 文件与校验", "",
              "- table_head.csv / table_center.csv / table_tail.csv：三张论文表，原始精度。",
              "- comparison_tables.tex：三张 LaTeX 表；report.md 显示四位小数。",
              "- summary.csv：105 行完整结果及相对未重构的改变量。",
              "- position_sensitivity.csv：同一方法、同一比例下三个位置的跨度。",
              "- 各条件目录包含 masks.npz、七种方法的预测与真实标签 NPZ、指标 JSON。",
              "- 历史干净集预测及每种重建方法 p010 前 32 个样本均通过复现校验。",
              "- 已从所有保存的预测重算 Macro-F1/MAE；检查三位置缺失数量逐样本相等。",
              "- 观测音视频、无缺失样本、padding、模型状态及原文件均通过保持不变检查。",
              "- 运行记录中的 seconds 仅用于进度诊断，包含恢复前后不同计时范围，不用于方法速度比较。",
              "- 未落盘受扰特征、重建特征或新模型权重。本文重构器是项目自定义模型，不是 RECAP 原论文完整复现。", ""]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    tex = []
    for position, rows in tables:
        tex += [r"\begin{table}[htbp]", r"\centering",
                r"\caption{" + POSITION_NAMES[position] + "连续缺失下的情感预测性能}",
                r"\label{tab:missing-position-" + position + "}", r"\begin{tabular}{llccccc}", r"\toprule",
                r"评价指标 & 方法 & 10\% & 20\% & 30\% & 40\% & 50\% \\", r"\midrule"]
        for i, row in enumerate(rows):
            if i == 7: tex.append(r"\midrule")
            metric = ("Macro-F1 $\\uparrow$" if i == 0 else "MAE $\\downarrow$" if i == 7 else "")
            tex.append(metric + " & " + row["方法"] + " & " + " & ".join(f"{row[f'{rate}%']:.4f}" for rate in RATES) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with (output / "comparison_tables.tex").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(tex))
    for path, expected in manifest["protected_sha256"].items():
        if sha256(path) != expected:
            raise AssertionError(f"Protected file changed: {path}")
    write_json(output / "report.json", {
        **manifest, "status": "complete", "records": records, "worker_verifications": verifications,
        "checks": {"all_105_metric_rows_recomputed_from_saved_predictions": True,
                   "per_sample_missing_count_equal_across_positions": True,
                   "protected_files_unchanged": True, "no_features_saved": True},
    })
    log(f"ABLATION_COMPLETE: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--output-root", type=Path, default=RESULTS / "missing_position_ablation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        worker(json.loads(args.worker_spec))
        return
    if args.self_test:
        self_test()
        return
    if len(set(args.devices)) != len(args.devices):
        parser.error("Devices must be unique")
    history = json.loads(HISTORY.read_text())
    registry = {"this_work": history["this_work_checkpoint"],
                "methods": {name: history["methods"][name]["checkpoint"] for name in METHODS}}
    for name in METHODS:
        if sha256(registry["methods"][name]) != history["methods"][name]["checkpoint_sha256"]:
            raise ValueError(f"Historical checkpoint hash mismatch: {name}")
    paths = [CLEAN, ANCHOR_DATA, HISTORY, ADAPTER, BASE_REFERENCE, Path(registry["this_work"]),
             *[Path(p) for p in registry["methods"].values()],
             *[RESULTS / REFERENCES[m] / "p010_paired_predictions.npz" for m in METHODS]]
    for directory in ("question2", "multi_fusion_model", "multimodal_suite"):
        paths += list((ROOT / directory).glob("*.py"))
    adapter_meta = torch.load(ADAPTER, map_location="cpu", weights_only=True)
    paths += [Path(adapter_meta["model_dir"]) / "model.safetensors"]
    protected = {str(p.resolve()): sha256(p) for p in paths}
    protocol = {
        "test_only": True, "rates_percent": list(RATES), "positions": list(POSITIONS),
        "count_rule": "k=(eligible_count*nominal_percent+50)//100",
        "center_start": "(eligible_count-k)//2",
        "all_modalities_same_missing_mask": True, "batch_size": 32,
        "backend": {"matmul_precision": "highest", "matmul_tf32": False,
                    "cudnn_tf32": True, "cudnn_deterministic": True, "cudnn_benchmark": False},
        "metrics": {"macro_f1": "unweighted mean across all three checkpoint classes",
                    "mae": "downstream sentiment regression, original label units"},
        "no_retraining_or_test_selection": True, "fixed_existing_method_configs": True,
    }
    resumed_from = None
    if args.resume_run:
        output = args.resume_run.resolve()
        previous = json.loads((output / "manifest.json").read_text())
        if previous["registry"] != registry or previous["protocol"] != protocol:
            raise ValueError("Resume model registry/protocol mismatch")
        script_path = str(Path(__file__).resolve())
        for path, expected in previous["protected_sha256"].items():
            if path != script_path and sha256(path) != expected:
                raise ValueError(f"Protected dependency changed before resume: {path}")
        if (output / "report.json").exists():
            raise ValueError("This run is already complete")
        resumed_from = {"manifest": str(output / "manifest.json"),
                        "previous_script_sha256": previous["protected_sha256"][script_path],
                        "reason": "Explicit CUDA device binding and single-thread compiler recovery; metrics/protocol unchanged"}
    else:
        output = args.output_root.resolve() / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(output),
                "script": str(Path(__file__).resolve()), "devices": args.devices, "registry": registry,
                "source_dataset": str(CLEAN), "historical_reference": str(HISTORY), "protocol": protocol,
                "method_configs": {m: history["methods"][m]["config"] for m in METHODS},
                "method_epochs": {m: history["methods"][m]["epoch"] for m in METHODS},
                "this_work_epoch": history["this_work_epoch"], "protected_sha256": protected}
    manifest["resumed_from"] = resumed_from
    manifest["execution"] = "Independent subprocesses; one physical GPU per CUDA_VISIBLE_DEVICES, logical cuda:0; fail-fast child exit monitoring"
    manifest["compiler_threads"] = 1
    manifest_name = ("resume_manifest_" + datetime.now(timezone.utc).strftime("%H%M%S_%f") + ".json") if resumed_from else "manifest.json"
    write_json(output / manifest_name, manifest)
    conditions = [(p, r) for r in RATES for p in POSITIONS]
    pending, completed_records = [], []
    for position, rate in conditions:
        path = output / f"{position}_p{rate:03d}" / "complete.json"
        if path.exists():
            completed_records.extend(json.loads(path.read_text())["records"])
        else:
            pending.append((position, rate))
    log(f"OUTPUT: {output}; {len(pending)} pending conditions, {len(completed_records)} preserved rows, devices={args.devices}")
    processes, worker_keys = [], []
    launch_id = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    try:
        for i, device in enumerate(args.devices):
            assigned = pending[i::len(args.devices)]
            if not assigned:
                continue
            env = os.environ.copy()
            logical = device
            if device.startswith("cuda:"):
                env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
                logical = "cuda:0"
            env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
            worker_key = device.replace(":", "_") + "_" + launch_id
            payload = [logical, assigned, str(output), registry, worker_key]
            processes.append(subprocess.Popen(
                [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker-spec", json.dumps(payload)],
                env=env))
            worker_keys.append(worker_key)
        while any(process.poll() is None for process in processes):
            for process in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"Worker {process.pid} exited with {process.returncode}; completed outputs retained")
            time.sleep(1)
        for process in processes:
            if process.returncode:
                raise RuntimeError(f"Worker {process.pid} exited with {process.returncode}")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    results = [json.loads((output / ("worker_" + key + ".json")).read_text()) for key in worker_keys]
    aggregate(output, completed_records + [row for result in results for row in result["records"]],
              [{k: v for k, v in result.items() if k != "records"} for result in results], manifest)


if __name__ == "__main__":
    main()

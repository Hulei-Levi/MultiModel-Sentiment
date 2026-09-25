"""附件二文本向量适配：冻结 BERT + 单个 768 -> 768 仿射层。

输入 text_bert [N, 3, L] 的三个通道依次为 token IDs、attention mask、
token_type IDs；监督目标为附件二的 text [N, L, 768]。

训练只使用 train 的有效 token；valid 依据 MSE 选择 ridge 系数；test
只在选模完成后评估。既不使用情感标签，也不训练分类/回归融合模型。
线性层通过带 L2 正则的最小二乘精确拟合，无需设置 epoch 或学习率。
目标: mean_token(||X W + b - Y||_2^2) + alpha * ||W||_F^2。
截距不正则化，alpha=0 使用截断伪逆。报告的 MSE 是逐元素均值。

Jupyter 用法（在服务器项目目录运行）::

    from multi_fusion_model.bert_feature_adapter import Config, fit_experiment, load_adapter
    exp = fit_experiment(Config(model_dir="本地官方BERT目录"))
    adapter = load_adapter(exp.checkpoint_path, device="cuda:0")
    new_text = adapter.encode_text_bert(data["test"]["text_bert"])  # numpy [N,50,768]

encode_text_bert 默认将 padding 输出置零；有效位置保持原顺序，包括 CLS/SEP。
pad 输出不代表复原原始缓存的 pad 向量；下游仍应使用原 attention mask。
原始数据、现有模型代码及 notebook 不会被覆盖。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
MODEL_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / ".cache" / "bert_feature_adapter" / MODEL_REVISION


@dataclass
class Config:
    dataset_path: str = str(PROJECT_ROOT / "datasets" / "附件2-数据集特征文件" / "aligned_50.pkl")
    model_dir: str = str(DEFAULT_MODEL_DIR)
    output_root: str = str(PROJECT_ROOT / "experiments" / "bert_feature_adapter")
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    ridge_alphas: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    seed: int = 42
    cpu_threads: int = 4


@dataclass
class Experiment:
    run_dir: str
    checkpoint_path: str
    selected_alpha: float
    metrics: dict[str, Any]


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _configure(seed: int, threads: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 精确比较缓存向量，禁用 TF32/混合精度；BERT 始终 eval()。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _check_inputs(value: Any, *, vocab_size: int = 30522, type_vocab_size: int = 2) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[1] != 3 or not 1 <= value.shape[2] <= 512:
        raise ValueError(f"text_bert 必须是 [N,3,L]，1<=L<=512；实际 {value.shape}")
    if not np.issubdtype(value.dtype, np.integer):
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
            raise ValueError("text_bert 的三个通道必须为有限整数")
    value = value.astype(np.int64, copy=False)
    ids, mask, types = value[:, 0], value[:, 1], value[:, 2]
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("attention mask 只能包含 0/1")
    if ((ids < 0) | (ids >= vocab_size)).any():
        raise ValueError("token ID 超出 BERT 词表；不能直接套用这个 checkpoint")
    if ((types < 0) | (types >= type_vocab_size)).any():
        raise ValueError("token_type ID 超出 BERT 的 type_vocab_size")
    if len(value) and (mask.sum(axis=1) == 0).any():
        raise ValueError("发现没有有效 token 的样本，请先确认该样本的数据含义")
    return value


def audit_split(split: dict[str, Any]) -> dict[str, Any]:
    inputs = _check_inputs(split["text_bert"])
    targets = np.asarray(split["text"])
    if targets.shape != (len(inputs), inputs.shape[2], 768):
        raise ValueError(f"目标 text 与 BERT 槽位不一致: {targets.shape}")
    mask = inputs[:, 1].astype(bool)
    if not mask.any() or not np.isfinite(targets[mask]).all():
        raise ValueError("监督目标有效位置为空或含非有限值")
    if "id" not in split or len(split["id"]) != len(inputs):
        raise ValueError("需要与样本一一对应的 id，便于追溯")
    return {
        "samples": len(inputs), "slots": inputs.shape[2], "target_dim": targets.shape[2],
        "valid_tokens_including_special": int(mask.sum()),
        "lexical_tokens": int((mask & ~np.isin(inputs[:, 0], (101, 102))).sum()),
        "padding_tokens_excluded": int((~mask).sum()),
        "samples_at_full_length": int((mask.sum(axis=1) == inputs.shape[2]).sum()),
    }


def _load_bert(model_dir: str, device: str) -> nn.Module:
    from transformers import BertModel

    # local_files_only=True 禁止服务器训练时临时联网/更换版本。
    model, info = BertModel.from_pretrained(
        model_dir, local_files_only=True, add_pooling_layer=False,
        attn_implementation="eager", output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"BERT 主干权重加载不完整: {info}")
    # 官方 checkpoint 的 MLM/预训练头不属于所用 BertModel。
    unexpected = [key for key in info["unexpected_keys"] if not key.startswith(("cls.", "pooler.", "bert.pooler."))]
    if unexpected:
        raise RuntimeError(f"出现不符合预期的 checkpoint 权重: {unexpected}")
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    return model


@torch.inference_mode()
def _bert_batch(bert: nn.Module, inputs: np.ndarray, device: str) -> torch.Tensor:
    tokens = torch.as_tensor(np.ascontiguousarray(inputs), dtype=torch.long, device=device)
    return bert(input_ids=tokens[:, 0], attention_mask=tokens[:, 1], token_type_ids=tokens[:, 2]).last_hidden_state


def _encode_split(bert: nn.Module, split: dict[str, Any], name: str, config: Config, run_dir: Path) -> dict[str, Any]:
    inputs = _check_inputs(split["text_bert"])
    mask = inputs[:, 1].astype(bool)
    sample_index, token_index = np.nonzero(mask)
    cache = run_dir / f"{name}_bert_valid_tokens.npy"
    vectors = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(int(mask.sum()), 768))
    cursor = 0
    progress_at = time.monotonic()
    for start in range(0, len(inputs), config.batch_size):
        end = min(start + config.batch_size, len(inputs))
        hidden = _bert_batch(bert, inputs[start:end], config.device).cpu().numpy()
        selected = hidden[mask[start:end]]
        vectors[cursor:cursor + len(selected)] = selected
        cursor += len(selected)
        if time.monotonic() - progress_at > 15 or end == len(inputs):
            _log(f"BERT {name}: {end}/{len(inputs)} 样本，{cursor} 有效 token")
            progress_at = time.monotonic()
    vectors.flush()
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name}: BERT 产生非有限向量")
    np.savez_compressed(run_dir / f"{name}_token_index.npz", sample_index=sample_index, token_index=token_index)
    return {
        "x": vectors, "y": np.asarray(split["text"][mask], dtype=np.float32),
        "sample_index": sample_index, "token_index": token_index,
        "token_id": inputs[:, 0][mask], "lengths": mask.sum(axis=1),
        "ids": [str(value) for value in split["id"]],
    }


def ridge_statistics(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """仅传入 train 有效 token；float64 中心化后稳定积累协方差。"""
    if x.ndim != 2 or x.shape != y.shape or len(x) < 2:
        raise ValueError("训练 X/Y 必须是同形二维向量，且至少有两个有效 token")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("训练向量含非有限值")
    mx = x.mean(axis=0, dtype=np.float64)
    my = y.mean(axis=0, dtype=np.float64)
    cxx = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    cxy = np.zeros_like(cxx)
    for start in range(0, len(x), 4096):
        xc = np.asarray(x[start:start + 4096], dtype=np.float64) - mx
        yc = np.asarray(y[start:start + 4096], dtype=np.float64) - my
        cxx += xc.T @ xc
        cxy += xc.T @ yc
    cxx /= len(x)
    cxy /= len(x)
    eigenvalues, basis = np.linalg.eigh((cxx + cxx.T) / 2)
    return {"mx": mx, "my": my, "eigenvalues": eigenvalues, "basis": basis, "projected_cxy": basis.T @ cxy}


def solve_ridge(stats: dict[str, np.ndarray], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge alpha 必须为非负有限数")
    eigenvalues = np.maximum(stats["eigenvalues"], 0)
    if alpha == 0:
        cutoff = max(float(eigenvalues.max()), 1.0) * 1e-10
        inv = np.zeros_like(eigenvalues)
        np.divide(1.0, eigenvalues, out=inv, where=eigenvalues > cutoff)
    else:
        inv = 1.0 / (eigenvalues + alpha)
    # W 的列是目标维度；torch Linear 的权重需转置。
    weight = stats["basis"] @ (inv[:, None] * stats["projected_cxy"])
    bias = stats["my"] - stats["mx"] @ weight
    return weight.astype(np.float32), bias.astype(np.float32)


def _token_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    if pred.shape != target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("预测/目标形状不匹配或含非有限值")
    diff = np.asarray(pred, dtype=np.float64) - target
    pn = np.sqrt(np.einsum("ij,ij->i", pred, pred, dtype=np.float64))
    tn = np.sqrt(np.einsum("ij,ij->i", target, target, dtype=np.float64))
    denom = pn * tn
    cosine = np.zeros(len(pred), dtype=np.float64)
    defined = denom > 1e-12
    np.divide(np.einsum("ij,ij->i", pred, target, dtype=np.float64), denom, out=cosine, where=defined)
    return {"mse": np.mean(diff ** 2, axis=1), "mae": np.mean(np.abs(diff), axis=1),
            "cosine": np.clip(cosine, -1, 1), "cosine_defined": defined}


def _aggregate(values: dict[str, np.ndarray], selected: np.ndarray | None = None) -> dict[str, Any]:
    if selected is None:
        selected = np.ones(len(values["mse"]), dtype=bool)
    n = int(selected.sum())
    if n == 0:
        return {"tokens": 0, "mse": None, "mae": None, "cosine_mean": None}
    cos = values["cosine"][selected & values["cosine_defined"]]
    return {"tokens": n, "mse": float(values["mse"][selected].mean()),
            "mae": float(values["mae"][selected].mean()),
            "cosine_mean": float(cos.mean()) if len(cos) else None,
            "cosine_p05": float(np.quantile(cos, .05)) if len(cos) else None,
            "cosine_p50": float(np.quantile(cos, .5)) if len(cos) else None,
            "cosine_p95": float(np.quantile(cos, .95)) if len(cos) else None,
            "cosine_undefined_tokens": n - len(cos)}


@torch.inference_mode()
def _evaluate(encoded: dict[str, Any], weight: np.ndarray | None, bias: np.ndarray | None,
              device: str, csv_path: Path | None = None) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("mse", "mae", "cosine", "cosine_defined")}
    wt = torch.as_tensor(weight, device=device) if weight is not None else None
    bt = torch.as_tensor(bias, device=device) if bias is not None else None
    for start in range(0, len(encoded["x"]), 2048):
        x = np.asarray(encoded["x"][start:start + 2048])
        pred = x if wt is None else (torch.as_tensor(x, device=device) @ wt + bt).cpu().numpy()
        for key, value in _token_metrics(pred, encoded["y"][start:start + 2048]).items():
            arrays[key].append(value)
    values = {key: np.concatenate(value) for key, value in arrays.items()}
    result = _aggregate(values)
    result["lexical_only"] = _aggregate(values, ~np.isin(encoded["token_id"], (101, 102)))
    counts = np.bincount(encoded["sample_index"], minlength=len(encoded["ids"]))
    sample_values = {key: np.bincount(encoded["sample_index"], weights=values[key], minlength=len(counts)) / counts
                     for key in ("mse", "mae")}
    defined_counts = np.bincount(encoded["sample_index"], weights=values["cosine_defined"], minlength=len(counts))
    cos_sum = np.bincount(encoded["sample_index"], weights=values["cosine"], minlength=len(counts))
    sample_values["cosine"] = np.divide(cos_sum, defined_counts, out=np.zeros_like(cos_sum), where=defined_counts > 0)
    result["sample_macro_mse"] = float(sample_values["mse"].mean())
    result["sample_macro_mae"] = float(sample_values["mae"].mean())
    result["sample_macro_cosine"] = float(sample_values["cosine"][defined_counts > 0].mean()) if (defined_counts > 0).any() else None
    if csv_path is not None:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "id", "valid_tokens", "mse", "mae", "cosine_mean", "cosine_defined_tokens"])
            for i, sample_id in enumerate(encoded["ids"]):
                writer.writerow([i, sample_id, int(counts[i]), float(sample_values["mse"][i]),
                                 float(sample_values["mae"][i]), float(sample_values["cosine"][i]) if defined_counts[i] else "", int(defined_counts[i])])
    return result


class BertTextAdapter(nn.Module):
    """冻结 BERT 与拟合好的线性层；forward 输出保持 [B,L,768] 槽位。"""

    def __init__(self, bert: nn.Module, weight: np.ndarray, bias: np.ndarray, device: str):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(weight.shape[0], weight.shape[1])
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weight.T.copy()))
            self.linear.bias.copy_(torch.as_tensor(bias))
        self.to(device).requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, text_bert: torch.Tensor, zero_padding: bool = True, apply_adapter: bool = True) -> torch.Tensor:
        self.eval()
        if text_bert.ndim != 3 or text_bert.shape[1] != 3:
            raise ValueError("需要 [B,3,L] 输入")
        tokens = text_bert.to(device=next(self.parameters()).device, dtype=torch.long)
        hidden = self.bert(input_ids=tokens[:, 0], attention_mask=tokens[:, 1], token_type_ids=tokens[:, 2]).last_hidden_state
        output = self.linear(hidden) if apply_adapter else hidden
        if zero_padding:
            output = output.masked_fill(~tokens[:, 1].bool().unsqueeze(-1), 0)
        return output

    def encode_text_bert(self, inputs: Any, batch_size: int = 64, zero_padding: bool = True,
                         apply_adapter: bool = True) -> np.ndarray:
        """apply_adapter=False 直接返回冻结 BERT 的输出，用于基线及无需映射的情况。"""
        inputs = _check_inputs(inputs)
        if batch_size < 1:
            raise ValueError("batch_size 必须为正")
        output = np.empty((len(inputs), inputs.shape[2], self.linear.out_features), dtype=np.float32)
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            output[start:end] = self(torch.as_tensor(inputs[start:end]), zero_padding=zero_padding,
                                     apply_adapter=apply_adapter).cpu().numpy()
        return output


def load_adapter(checkpoint_path: str, device: str = "cpu", model_dir: str | None = None) -> BertTextAdapter:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("model_revision") != MODEL_REVISION:
        raise ValueError("不支持的适配器 checkpoint 或 BERT 版本")
    resolved = model_dir or checkpoint["model_dir"]
    if _sha256(Path(resolved) / "model.safetensors") != checkpoint["model_sha256"]:
        raise ValueError("推理 BERT 权重与训练时不一致")
    bert = _load_bert(resolved, device)
    return BertTextAdapter(bert, checkpoint["weight"].numpy(), checkpoint["bias"].numpy(), device)


def _write_report(run_dir: Path, result: dict[str, Any]) -> None:
    lines = ["# BERT 文本向量线性适配实验", "", "只复原附件二缓存 text 向量，不评估情感任务。",
             "", f"- BERT: `{MODEL_ID}` @ `{MODEL_REVISION}`，冻结、eval、FP32。",
             f"- 正则系数: {result['selected_alpha']}（仅依据 valid MSE 选择）。",
             "- 训练: train 有效 token 的正则化最小二乘；test 不参与拟合/选模。",
             "- mask=0 的 padding 排除；CLS/SEP 计入主指标，另报 lexical_only。",
             "- 余弦相似度衡量方向；MSE/MAE衡量数值误差，均不等于情感预测正确率。", "",
             "| Split | Method | Cosine mean ↑ | MSE ↓ | MAE ↓ |", "|---|---|---:|---:|---:|"]
    for split, methods in result["metrics"].items():
        for method, metrics in methods.items():
            cosine = metrics["cosine_mean"]
            cosine_text = f"{cosine:.8f}" if cosine is not None else "undefined"
            lines.append(f"| {split} | {method} | {cosine_text} | {metrics['mse']:.8g} | {metrics['mae']:.8g} |")
    raw_valid = result["metrics"]["valid"]["raw_bert"]
    if raw_valid["mse"] < 1e-10:
        lines += ["", "实测：未经适配的官方 BERT 已在数值误差范围内复现附件二有效位置的 text。",
                  "可以直接使用 BERT 输出；本次线性适配没有实际收益。可用 apply_adapter=False 跳过适配层。",
                  "这是对该数据的输出复现证据，不能据此确认原始生成脚本或所有预处理细节。"]
    lines += ["", "## 使用", "", "```python", "from multi_fusion_model.bert_feature_adapter import load_adapter",
              f"adapter = load_adapter({str(run_dir / 'adapter.pt')!r}, device='cuda:0')",
              "text = adapter.encode_text_bert(text_bert)  # [N,50,768], float32",
              "raw_text_vec = adapter.encode_text_bert(text_bert, apply_adapter=False)",
              "text_mask = text_bert[:, 1, :].astype(bool)", "```", "",
              "输出保留原槽位，padding 默认置零；不修改输入数据文件。词表/维度相同不证明原始提取器相同。",
              "该适配器在干净数据上的复原效果不代表对干扰样本的去噪能力；尚未评估下游任务。", "",
              "查看 metrics.json 获取完整指标与版本信息；validation_candidates.json 记录全部候选；",
              "每个 split 的 CSV 保存逐样本误差和 ID；*_token_index.npz 记录缓存向量对应的原始槽位。"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_experiment(config: Config | None = None) -> Experiment:
    config = config or Config()
    if config.batch_size < 1 or config.cpu_threads < 1 or not config.ridge_alphas:
        raise ValueError("batch_size/cpu_threads/候选列表无效")
    if any(not np.isfinite(alpha) or alpha < 0 for alpha in config.ridge_alphas):
        raise ValueError("ridge 候选必须为非负有限数")
    _configure(config.seed, config.cpu_threads)
    dataset_path = Path(config.dataset_path).resolve()
    model_dir = Path(config.model_dir).resolve()
    model_hash = _sha256(model_dir / "model.safetensors")
    if model_hash != MODEL_SHA256:
        raise ValueError("BERT 权重 SHA256 与固定官方版本不一致")
    _log("核对数据与官方权重；原始 pickle 仅只读加载")
    dataset_hash = _sha256(dataset_path)
    with dataset_path.open("rb") as handle:
        data = pickle.load(handle)  # 仅加载用户信任的本地附件二文件。
    audits = {name: audit_split(data[name]) for name in ("train", "valid", "test")}
    root = Path(config.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(exist_ok=False)
    _json(run_dir / "config.json", asdict(config))
    _json(run_dir / "audit.json", audits)
    _log(f"输出目录: {run_dir}")
    bert = _load_bert(str(model_dir), config.device)
    train = _encode_split(bert, data["train"], "train", config, run_dir)
    valid = _encode_split(bert, data["valid"], "valid", config, run_dir)
    _log("拟合 train 的 768×768 线性映射；valid 选择正则系数")
    stats = ridge_statistics(train["x"], train["y"])
    candidates = []
    best = None
    for alpha in config.ridge_alphas:
        weight, bias = solve_ridge(stats, float(alpha))
        metrics = _evaluate(valid, weight, bias, config.device)
        candidates.append({"alpha": float(alpha), "valid": metrics})
        _log(f"alpha={alpha:g}: valid MSE={metrics['mse']:.8g}, cosine={metrics['cosine_mean']:.8f}")
        if best is None or metrics["mse"] < best[0]:
            best = (metrics["mse"], float(alpha), weight, bias)
    _, alpha, weight, bias = best
    _json(run_dir / "validation_candidates.json", candidates)
    checkpoint_path = run_dir / "adapter.pt"
    torch.save({"format_version": 1, "weight": torch.from_numpy(weight), "bias": torch.from_numpy(bias),
                "selected_alpha": alpha, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "model_dir": str(model_dir), "model_sha256": model_hash, "dataset_sha256": dataset_hash,
                "input_layout": "[N,3,L]: input_ids,attention_mask,token_type_ids",
                "target_field": "text", "zero_padding_default": True}, checkpoint_path)
    _log(f"选模完成，alpha={alpha:g}；现在评估固定参数的 train/valid/test")
    all_metrics = {}
    for name in ("train", "valid", "test"):
        encoded = train if name == "train" else valid if name == "valid" else _encode_split(bert, data["test"], "test", config, run_dir)
        all_metrics[name] = {
            "raw_bert": _evaluate(encoded, None, None, config.device, run_dir / f"{name}_raw_bert.csv"),
            "linear_adapter": _evaluate(encoded, weight, bias, config.device, run_dir / f"{name}_linear_adapter.csv"),
        }
        for method, metrics in all_metrics[name].items():
            _log(f"{name}/{method}: MSE={metrics['mse']:.8g}, MAE={metrics['mae']:.8g}, cosine={metrics['cosine_mean']:.8f}")
    # 独立加载实际交付的 checkpoint，核验最终推理路径、槽位及 mask。
    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    exported = load_adapter(str(checkpoint_path), device=config.device)
    sample_inputs = np.asarray(data["test"]["text_bert"][:3])
    sample_mask = sample_inputs[:, 1].astype(bool)
    sample_raw = _bert_batch(exported.bert, sample_inputs, config.device).cpu().numpy()
    sample_output = exported.encode_text_bert(sample_inputs)
    expected = torch.as_tensor(sample_raw[sample_mask], device=config.device) @ torch.as_tensor(weight, device=config.device) + torch.as_tensor(bias, device=config.device)
    roundtrip_error = float(np.max(np.abs(sample_output[sample_mask] - expected.cpu().numpy())))
    if not np.allclose(sample_output[sample_mask], expected.cpu().numpy(), atol=3e-5, rtol=1e-5):
        raise AssertionError("导出推理路径与选模参数不一致")
    if (sample_output[~sample_mask] != 0).any():
        raise AssertionError("导出推理未正确屏蔽 padding")
    if _sha256(dataset_path) != dataset_hash:
        raise RuntimeError("实验期间原始数据文件发生变化，请核查后重新运行")
    import transformers
    result = {"selected_alpha": alpha, "metrics": all_metrics, "audit": audits,
              "dataset_path": str(dataset_path), "dataset_sha256": dataset_hash,
              "original_dataset_unchanged": True, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "model_sha256": model_hash, "adapter_parameters": int(weight.size + bias.size),
              "bert_parameters": sum(p.numel() for p in exported.bert.parameters()),
              "checkpoint_roundtrip_max_abs_error": roundtrip_error,
              "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "numpy": np.__version__, "transformers": transformers.__version__},
              "config": asdict(config)}
    _json(run_dir / "metrics.json", result)
    _write_report(run_dir, result)
    _log("完成：metrics.json / report.md / adapter.pt / 逐样本 CSV；原始数据哈希未变化")
    return Experiment(str(run_dir), str(checkpoint_path), alpha, all_metrics)


def self_test() -> None:
    """验证解析解、torch 权重方向、mask 及指标；不下载模型/改动数据。"""
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    truth = rng.normal(size=(8, 8)).astype(np.float32)
    offset = rng.normal(size=8).astype(np.float32)
    y = x @ truth + offset
    weight, bias = solve_ridge(ridge_statistics(x, y), 0)
    heldout = rng.normal(size=(100, 8)).astype(np.float32)
    np.testing.assert_allclose(heldout @ weight + bias, heldout @ truth + offset, atol=3e-6)
    layer = nn.Linear(8, 8)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight.T.copy()))
        layer.bias.copy_(torch.from_numpy(bias))
        np.testing.assert_allclose(layer(torch.from_numpy(heldout)).numpy(), heldout @ weight + bias, atol=3e-6)
    exact = _aggregate(_token_metrics(y, y))
    assert exact["mse"] == 0 and abs(exact["cosine_mean"] - 1) < 1e-12
    zeros = _aggregate(_token_metrics(np.zeros((2, 8)), np.zeros((2, 8))))
    assert zeros["cosine_mean"] is None and zeros["cosine_undefined_tokens"] == 2
    inputs = np.array([[[101, 100, 102, 0], [1, 1, 1, 0], [0, 0, 0, 0]]])
    _check_inputs(inputs)
    class FakeBert(nn.Module):
        def forward(self, input_ids, attention_mask, token_type_ids):
            from types import SimpleNamespace
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
            return SimpleNamespace(last_hidden_state=hidden)
    adapter = BertTextAdapter(FakeBert(), np.eye(8, dtype=np.float32), np.ones(8, dtype=np.float32), "cpu")
    pred = adapter.encode_text_bert(inputs)
    assert pred.shape == (1, 4, 8) and (pred[0, 3] == 0).all() and (pred[0, 0] == 102).all()
    raw = adapter.encode_text_bert(inputs, apply_adapter=False)
    assert (raw[0, 0] == 101).all() and (raw[0, 3] == 0).all()
    altered = inputs.copy()
    altered[0, 0, 3] = 1234
    np.testing.assert_array_equal(pred, adapter.encode_text_bert(altered))
    bad = inputs.copy()
    bad[:, 1] = 2
    try:
        _check_inputs(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("应拒绝非二值 mask")
    _log("SELF_TEST_PASS: affine recovery, Linear orientation, metrics, padding, input validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "self-test"))
    parser.add_argument("--dataset", default=Config.dataset_path)
    parser.add_argument("--model-dir", default=Config.model_dir)
    parser.add_argument("--output-root", default=Config.output_root)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        result = fit_experiment(Config(dataset_path=args.dataset, model_dir=args.model_dir,
                                      output_root=args.output_root, device=args.device,
                                      batch_size=args.batch_size, cpu_threads=args.cpu_threads))
        print(json.dumps({"run_dir": result.run_dir, "checkpoint_path": result.checkpoint_path,
                          "selected_alpha": result.selected_alpha}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

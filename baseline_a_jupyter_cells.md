# A：同一轻量骨干，分别训练分类和回归

这是后续双任务方法的对照基线，不是某篇论文的复现。两个实验使用相同结构、随机种子和训练设置，分别用三分类交叉熵和连续分数 MAE 训练；模型参数、优化器和最佳检查点各自独立。这里的“同一骨干”指结构与初始参数相同，训练结束后权重不会相同。

调用方式保持 `Config → fit_experiment → 中间结果 / 曲线 → load / predict → explain`。
分类默认按验证 **Macro-F1** 选模，回归按验证 **MAE** 选模；测试集只在各自最佳模型选定后评估一次。Accuracy 仅作为辅助指标报告。

本 Notebook 是新增文件，不需要改动原来的 `second_question_classfication_regression.ipynb`。


```python
import os
import sys
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# 在服务器项目目录打开本 Notebook；也支持从它的子目录启动。
PROJECT_DIR = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / "multi_fusion_model" / "lightweight_baseline.py").is_file()),
    None,
)
if PROJECT_DIR is None:
    raise RuntimeError("请在 MultiModel-Sentiment 项目目录启动 Notebook。")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from multi_fusion_model.lightweight_baseline import (
    Config as BaselineConfig,
    fit_experiment as fit_baseline_experiment,
    explain_feature_groups as explain_baseline_features,
    load_experiment_model as load_baseline_model,
    predict_split as predict_baseline_split,
    audit_data,
)

# 若接在现有 Notebook 后面，直接复用 data、MODULE_DIR、mask_overrides。
if "data" not in globals():
    candidates = list((PROJECT_DIR / "datasets").glob("*/aligned_50.pkl"))
    if len(candidates) != 1:
        raise RuntimeError("请先加载 data，或在这里指定你的 aligned_50.pkl 路径。")
    with candidates[0].open("rb") as stream:
        data = pickle.load(stream)
if "MODULE_DIR" not in globals():
    MODULE_DIR = str(PROJECT_DIR / "results" / "second_question")
if "mask_overrides" not in globals():
    mask_overrides = None

print("项目目录:", PROJECT_DIR)
print("训练集:", {m: data["train"][m].shape for m in ("text", "audio", "vision")})
```


## 1. 轻量结构

```text
text   [B,50,768] → Linear/LN/GELU → [B,50,64] → BiGRU(每方向32) ┐
audio  [B,50,74]  → Linear/LN/GELU → [B,50,64] → BiGRU(每方向32) ├→ 原槽位拼接 [B,50,192]
vision [B,50,35]  → Linear/LN/GELU → [B,50,64] → BiGRU(每方向32) ┘
  → Linear/GELU/Dropout [B,50,64] → masked mean [B,64]
  → Linear [B,3] 分类，或 Linear [B,1] 回归
```

三路编码器参数相互独立。没有跨模态注意力、动态门控、分阶段预训练或辅助损失。
回归头直接输出连续值，不把三分类概率映射成 -1/0/1，也不把回归标签离散化。

BiGRU 内部按 mask 编码有效位置，再放回原来的时间槽位；它保留观测顺序，但不会显式建模缺失位置之间相隔的真实秒数。拼接和解释仍对应原来的 50 个槽位。池化只统计至少一路有效的槽位。

数据工具沿用现有实现：音视频标准化只拟合训练集；默认三路 mask 使用 `text_bert[:,1,:]`。已有可靠的独立缺失标记时通过 `mask_overrides` 提供。音视频全零行不自动判定缺失。默认掩码要求特殊 token 和其他槽位的布局已正确对齐。


```python
baseline_common_config = BaselineConfig(
    d_model=64,
    hidden_dim=32,
    fusion_dim=64,
    bidirectional=True,
    dropout=0.2,
    epochs=40,
    patience=8,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    seed=42,
    standardize_av=True,
    class_weighted_loss=False,
    selection_metric="auto",  # classification→macro_f1；regression→mae
    device=None,
    output_root=str(Path(MODULE_DIR) / "baseline_a"),
)

baseline_cls_config = replace(baseline_common_config, task="classification")
baseline_reg_config = replace(baseline_common_config, task="regression")
```


## 2. 两个独立实验

两次 `fit_experiment` 会各自重设随机种子、创建新模型和优化器；不会从分类模型接着训练回归。骨干先于任务头初始化，训练随机状态也在构造模型后重设。

分类 `selection_metric="auto"` 解析为 `macro_f1`，与之前模型的选模目标一致；回归解析为 `mae`。也可分别显式指定这两个指标。早停和最佳模型选择均使用对应的验证指标。

若已有按 Accuracy 选模的 A 实验，应重新训练并按 Macro-F1 选择；历史检查点和单元输出仍代表原来的选模方式，不会因修改配置自动改变。已导入旧模块的内核请先重启。

下面是完整训练入口；完成实现验证不等于已经得到完整实验成绩。


```python
baseline_cls_experiment = fit_baseline_experiment(
    data, baseline_cls_config, mask_overrides=mask_overrides,
)
```


```python
baseline_reg_experiment = fit_baseline_experiment(
    data, baseline_reg_config, mask_overrides=mask_overrides,
)
baseline_experiments = {
    "classification": baseline_cls_experiment,
    "regression": baseline_reg_experiment,
}
```


## 3. 比较训练、验证、测试结果

分类和回归来自两个独立检查点。本表不能描述成“双任务模型同一个检查点的成绩”。
`train_metrics` 是最佳检查点下关闭 dropout 后重新计算的指标，可与验证指标比较；不能直接拿包含 dropout 的训练过程损失与验证 MAE 比大小。


```python
initial_hashes = {}
baseline_summary = []
for task, experiment in baseline_experiments.items():
    metadata = json.loads((experiment.run_dir / "config.json").read_text(encoding="utf-8"))
    initial_hashes[task] = metadata["initial_backbone_sha256"]
    print("\n任务:", task)
    print("类别映射:", experiment.class_values)
    print("最佳轮次:", experiment.selected_epoch)
    print("参数量:", sum(p.numel() for p in experiment.model.parameters()))
    print("输出目录:", experiment.run_dir)
    for split, metrics in (("train", experiment.train_metrics),
                           ("valid", experiment.valid_metrics),
                           ("test", experiment.test_metrics)):
        print(split, metrics)
        baseline_summary.append({"task": task, "split": split, **metrics})

assert initial_hashes["classification"] == initial_hashes["regression"], "两任务初始骨干不同"
print("\n相同初始骨干校验通过:", initial_hashes["classification"][:16])
try:
    import pandas as pd
except ImportError:
    pass
else:
    display(pd.DataFrame(baseline_summary))
```


```python
# 检查同一样本在两个实验中的中间形状。
for task, experiment in baseline_experiments.items():
    item = experiment.datasets["valid"][0]
    device = next(experiment.model.parameters()).device
    features = {m: x.unsqueeze(0).to(device) for m, x in item["features"].items()}
    masks = {m: x.unsqueeze(0).to(device) for m, x in item["masks"].items()}
    experiment.model.eval()
    with torch.no_grad():
        output = experiment.model(features, masks)
    print("\n", task)
    for m, sequence in output["unimodal_sequences"].items():
        print(m, tuple(sequence.shape))
    for name in ("concatenated", "fusion_sequence", "pooled_features", "logits"):
        print(name, tuple(output[name].shape))
```


```python
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("每次实验的曲线数据已保存在 history.json。")
else:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, (task, experiment) in enumerate(baseline_experiments.items()):
        history = experiment.history
        epochs = [record["epoch"] for record in history]
        metric = "macro_f1" if task == "classification" else "mae"
        axes[row, 0].plot(epochs, [record["train_loss"] for record in history])
        axes[row, 1].plot(epochs, [record["valid"][metric] for record in history])
        axes[row, 0].set(title=task, xlabel="Epoch", ylabel="Train CE" if row == 0 else "Train MAE")
        axes[row, 1].set(title=task, xlabel="Epoch", ylabel=f"Validation {metric}")
        for axis in axes[row]:
            axis.axvline(experiment.selected_epoch, color="gray", linestyle="--", alpha=.5)
    fig.tight_layout()
    plt.show()
```


## 4. 重载模型和无标签预测

重载时使用 `best.pt` 内的训练集标准化参数及类别映射。预测不需要标签。
如果训练时提供了自定义 mask，预测时也必须按同样约定提供。本示例在验证集上检查保存前后的输出一致性。


```python
unlabeled_valid = {key: value for key, value in data["valid"].items()
                   if key not in ("classification_labels", "regression_labels")}
for task, experiment in baseline_experiments.items():
    loaded_model, metadata = load_baseline_model(experiment.run_dir / "best.pt", device="cpu")
    prediction = predict_baseline_split(
        loaded_model, unlabeled_valid, metadata,
        masks=(mask_overrides or {}).get("valid"),
        batch_size=baseline_common_config.batch_size,
    )
    with np.load(experiment.run_dir / "valid_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(prediction["logits"], saved["logits"], rtol=2e-4, atol=2e-5)
        np.testing.assert_array_equal(prediction["ids"], saved["ids"])
    print(task, "重载一致；输出形状:", prediction["logits"].shape)
    if task == "classification":
        labels = np.asarray(metadata["class_values"])[prediction["logits"].argmax(axis=1)]
        print("预测类别:", labels[:5])
    else:
        print("预测分数:", prediction["logits"][:5, 0])
```


## 5. 按原槽位追溯影响

沿用已有的特征组遮挡接口，分别解释两个模型对同一个样本的判断。

- 分类 `score_drop`：遮挡前后固定目标类别相对固定参考类别的 logit 差值变化。
- 回归 `score_drop`：原预测分数减去遮挡后的预测分数；正值表示该组特征把预测往更高分方向推，不等于一定提高准确性。
- 无效槽位的解释分数为 NaN，不纳入排序。

`baseline=0` 对标准化音视频表示训练均值，对文本表示零特征向量。遮挡不改变 mask 和序列长度。
这是缓存特征层面的敏感性分析，不是可加和贡献分解，也不能直接当成原始词、语音、像素的因果结论；文本 BERT 特征已经包含上下文。使用返回的 `time_index` 和自己的词时间戳/帧号记录映射回原始材料。


```python
baseline_explanations = {}
for task, experiment in baseline_experiments.items():
    explanation = explain_baseline_features(
        experiment, split="valid", index=0, baseline=0.0, perturb_batch_size=32,
    )
    baseline_explanations[task] = explanation
    print("\n", task, "样本:", explanation["id"], "原分数:", explanation["original_score"])
    for row in explanation["rows"][:10]:
        print(row)
```


## 6. 后续对照规则

先完成这个 A 基线，再在相同编码器和数据协议下比较 B（联合训练）、C（任务专用门控）等改进。不要在 A 中提前加入这些模块。

候选宽度可以比较 `d_model=64/128`，但同一对照中的分类和回归必须保持相同骨干结构。入围方案再用 `seed=42/43/44` 重复；宽度与其他超参数只使用验证集选择。测试集用于最终报告。

每次运行独立保存 `config.json`、`data_audit.json`、`history.json`、`metrics.json`、`best.pt`、`valid_predictions.npz`、`test_predictions.npz/csv`。原来的模型代码和实验目录无需变更。

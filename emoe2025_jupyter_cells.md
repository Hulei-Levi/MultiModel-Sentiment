# EMOE：动态模态专家与在线单模态特征蒸馏

**本节是经过官方代码核查的缓存特征适配，保留三模态密集路由、加权求和、单模态监督及在线特征蒸馏。**

来源：[CVPR 2025 论文与补充材料](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)、[官方代码固定版本 c4759c748105e18354cd9082b1d8c6d310f1f9d8](https://github.com/fuyyyyy/EMOE/tree/c4759c748105e18354cd9082b1d8c6d310f1f9d8)。官方补充代码与该版本的核心模型、路由、训练和损失文件已对照；实验配置会保存来源及适配范围。

这里沿用你已有的 text/audio/vision 特征，BERT 本身不更新；默认三分类，也支持 `task="regression"`。保留原始特征展平后输入路由器，但把其隐藏宽度从原代码的 `输入维度//8` 调整为 128：当前 `50×(768+74+35)=43850`，原设置第一层约有 2.4 亿权重，本设置约 561 万权重。

为保留槽位索引，输入投影使用核为 1 的逐位置线性变换，采用显式观测 mask、最后有效位置读取及双向 Transformer；原代码使用时序卷积、固定最后位置和默认因果注意力。这些改动、缓存 BERT、三分类 CE 与 Brier 重要性误差、模型宽度及训练超参数都是本次适配，不是作者 checkpoint 的兼容复现。

若内核已导入旧版 `multimodal_suite`，先重启内核，再运行数据加载、原有 mask 设置及本节。


```python
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from multi_fusion_model.emoe2025_fusion import (
    Config as EMOEConfig,
    fit_experiment as fit_emoe_experiment,
    explain_feature_groups as explain_emoe_features,
    load_experiment_model as load_emoe_model,
    predict_split as predict_emoe_split,
    audit_data,
)

if "data" not in globals():
    raise RuntimeError("请先运行数据加载代码，得到 data。")
if "MODULE_DIR" not in globals():
    MODULE_DIR = str(Path.cwd() / "results" / "second_question")
if "mask_overrides" not in globals():
    mask_overrides = None
```

## 2. 先得到样本级模态权重，再融合专家表示

```text
text   [B,50,768] ─ 独立投影 ─┐                      ┌ text Transformer   → h_t [B,128]
audio  [B,50,74]  ─ 独立投影 ─┼ 共享 Linear(128,128) ┼ audio Transformer  → h_a [B,128]
vision [B,50,35]  ─ 独立投影 ─┘   + 正弦位置编码     └ vision Transformer → h_v [B,128]
                                                    每路取最后有效槽位

三路输入先按 mask 清零 → 每槽位拼接 [B,50,877] → 展平 [B,43850]
    → Linear(43850,128) → L2 normalize → ReLU → Linear(128,3)
    → 除以 temperature=0.1 → 可用模态上 softmax → w [B,3]

fused = w_t*h_t + w_a*h_a + w_v*h_v [B,128]
    → 融合残差 MLP → r_f [B,128] → Linear → logits [B,3]

每路 h_m → 各自残差 MLP → r_m [B,128] → Linear → 单模态 logits [B,3]
```

残差 MLP 为 `r = h + Linear(Dropout(ReLU(Linear(h))))`。三个专家分别是文本、音频和视觉分支；权重是每条样本的三个数，使用密集 softmax，没有 top-k 或路由噪声。权重只在序列表征完成后参与融合，三路 Transformer 参数独立，前面的共享 Linear 使用同一组参数。

本实现的模态顺序固定为 **text、audio、vision（T/A/V）**，与官方代码的 T/V/A 顺序不同。整路缺失时其表示、单模态输出及路由权重均为零，其他可用模态权重重新归一化；三路全缺失则拒绝输入。短于最大长度的输入仅为固定宽度路由右侧补零，不改变已有槽位索引。

默认 mask 来自 `text_bert[:,1,:]`，要求三路槽位布局及特殊 token 占位一致；独立缺失信息通过 `mask_overrides` 提供。全零行不自动判缺失。音视频标准化参数仅拟合训练集；图中的输入指进入模型的特征，音视频按配置已标准化。


## 3. 四项辅助目标与联合训练

对每条样本，设可用模态集合为 `A`，数量为 `n_available`。默认总目标为：

```text
L_total = L_fused_task
        + 1.00 * L_unimodal
        + 0.01 * L_router_fit
        + 0.10 * L_router_entropy
        + 0.10 * L_distillation
```

| 辅助目标 | 当前计算方式 |
| --- | --- |
| `emoe_unimodal` | 每个样本的可用单模态任务损失取均值，再对 batch 取均值；分类用 CE，回归用 MAE |
| `emoe_router_fit` | 每路可靠性 `q_m=1/(error_m+0.1)`；在可用模态中归一化得到 `I=stop_gradient(q/sum(q))`，计算 `w` 与 `I` 的可用模态均方差，再取 batch 均值 |
| `emoe_router_entropy` | `mean_batch[n_available * sum_m(w_m * log(clamp_min(w_m,1e-9)))]`，缺失模态权重为零 |
| `emoe_distillation` | 融合残差特征 `r_f` 与目标 `stop_gradient(sum_m w_m*r_m)`，分别沿特征维 softmax，再对 batch 和特征维做 MSE |

三分类重要性误差是 Brier 形式：`error_m=mean_class((softmax(unimodal_logits_m)-one_hot(y))^2)`；回归误差是 `(prediction_m-y)^2`。三分类可靠性规则是本次扩展，不把类别编号当作连续情感分数。类别 0/1/2 的情感含义仍由你的数据定义。

负熵项鼓励路由保留多模态的学习机会，防止一开始完全集中于单一路；不是要求每个样本的最终权重均匀。**它可以为负，总训练损失也可以为负**。有三路观测时 `n_available=3`，恢复作者代码中的专家数系数；缺失时按实际可用数量调整。

蒸馏目标来自同一模型当前的单模态残差特征，整个加权目标（包括其中的路由权重）停止梯度。单模态分支通过各自任务监督继续训练；蒸馏的 student 是融合残差特征。共享上游参数仍可通过融合路径得到梯度。这里没有另一个冻结的预训练教师，也不设预训练阶段；四项辅助目标从第一轮与融合任务共同训练。`temperature=0.1` 仅用于路由，蒸馏没有额外温度，也不是类别概率 KL。

论文公式将重要性写成倒数误差后 softmax，发布代码实际使用倒数误差除和；本节采用发布代码。论文蒸馏部分的 logits 与方向措辞也存在歧义，本节按实际梯度路径执行隐藏特征 softmax-MSE：融合分支学习停止梯度的单模态组合目标。

标签只用于训练任务和可靠性目标，不进入路由或预测。普通 eval 的辅助损失为空。默认三分类用验证 Macro-F1 选模；回归用验证 MAE。最多 40 轮、patience=8，最后加载验证最佳模型评估测试集。


```python
emoe_config = EMOEConfig(
    task="classification",
    d_model=128,
    nhead=4,
    num_layers=2,
    dropout=0.2,
    pretrain_epochs=0,
    model_options={
        "emoe_router_hidden": 128,
        "emoe_temperature": 0.1,
        "emoe_reliability_epsilon": 0.1,
    },
    auxiliary_weights={
        "emoe_unimodal": 1.0,
        "emoe_router_fit": 0.01,
        "emoe_router_entropy": 0.1,
        "emoe_distillation": 0.1,
    },
    epochs=40,
    patience=8,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    seed=42,
    standardize_av=True,
    class_weighted_loss=False,
    device=None,
    output_root=str(os.path.join(MODULE_DIR, "emoe2025_fusion")),
)

emoe_experiment = fit_emoe_experiment(data, emoe_config, mask_overrides=mask_overrides)
```

## 4. 查看形状、最后有效位置与路由表

在 `eval()` 与 `torch.no_grad()` 下检查第一个测试样本，不传标签、不更新模型。`last_valid_indices=-1` 表示整路缺失。`expert_representations`、`modality_logits` 和 `routing_weights` 的模态轴顺序均为 T/A/V。

`fused_features` 是加权求和后的向量；`fusion_residual_features` 是融合预测头中的残差向量。兼容字段 `pooled_features` 等于后者，不是时间均值池化。


```python
print("类别映射:", emoe_experiment.class_values)
metrics_record = json.loads((emoe_experiment.run_dir / "metrics.json").read_text(encoding="utf-8"))
print("最佳轮次:", metrics_record["selected_epoch"])
print("验证集:", emoe_experiment.valid_metrics)
print("测试集:", emoe_experiment.test_metrics)
print("输出目录:", emoe_experiment.run_dir)

item = emoe_experiment.datasets["test"][0]
device = next(emoe_experiment.model.parameters()).device
features = {m: value.unsqueeze(0).to(device) for m, value in item["features"].items()}
masks = {m: value.unsqueeze(0).to(device) for m, value in item["masks"].items()}
emoe_experiment.model.eval()
with torch.no_grad():
    emoe_output = emoe_experiment.model(features, masks, return_intermediates=True)

for m, value in emoe_output["unimodal_sequences"].items():
    print(m, tuple(value.shape), "读取槽位:", int(emoe_output["last_valid_indices"][m][0]))
for key in ("router_features", "router_hidden", "routing_weights", "expert_representations",
            "unimodal_residual_features", "modality_logits", "modality_availability",
            "fused_features", "fusion_residual_features", "pooled_features", "logits"):
    print(key, tuple(emoe_output[key].shape))
print("加权求和最大误差:", float((emoe_output["fused_features"] -
      (emoe_output["routing_weights"][..., None] * emoe_output["expert_representations"]).sum(1)).abs().max()))
print("普通 eval 辅助损失:", emoe_output["aux_losses"])

emoe_routing_rows = []
for j, m in enumerate(("text", "audio", "vision")):
    observed = bool(emoe_output["modality_availability"][0, j])
    raw_prediction = emoe_output["modality_logits"][0, j].cpu()
    if not observed:
        prediction = None
    elif emoe_config.task == "classification":
        prediction = emoe_experiment.class_values[int(raw_prediction.argmax())]
    else:
        prediction = float(raw_prediction[0])
    emoe_routing_rows.append({
        "modality": m, "observed": observed,
        "routing_weight": float(emoe_output["routing_weights"][0, j]),
        "unimodal_prediction": prediction,
    })
try:
    import pandas as pd
except ImportError:
    for row in emoe_routing_rows:
        print(row)
else:
    print(pd.DataFrame(emoe_routing_rows).to_string(index=False))
print("路由权重之和:", float(emoe_output["routing_weights"][0].sum()))
```

路由表说明模型如何混合当前样本的三路表示。权重不是准确率、解释置信度或最终分数的贡献比例；即使视觉权重较大，也不能直接认定某一帧决定了预测。下一步可以用完整模型的槽位替换检查预测敏感性。

## 5. 在训练小批次上检查四项辅助损失

从 **train** 取最多 12 个样本，在 eval 模式显式启用 `compute_auxiliary_losses=True`。这一检查关闭 dropout、不反向传播，标签仅用于展示训练可靠性目标和单模态任务损失；不会拿测试标签调整路由。辅助目标开关不改变预测路径。


```python
train_dataset = emoe_experiment.datasets["train"]
diagnostic_batch = next(iter(DataLoader(train_dataset, batch_size=min(12, len(train_dataset)), shuffle=False)))
diagnostic_features = {m: value.to(device) for m, value in diagnostic_batch["features"].items()}
diagnostic_masks = {m: value.to(device) for m, value in diagnostic_batch["masks"].items()}
diagnostic_targets = diagnostic_batch["target"].to(device)
with torch.no_grad():
    emoe_plain = emoe_experiment.model(diagnostic_features, diagnostic_masks)
    emoe_diagnostics = emoe_experiment.model(
        diagnostic_features, diagnostic_masks, targets=diagnostic_targets,
        return_intermediates=True, compute_auxiliary_losses=True,
    )
print("四项原始辅助损失:", {name: float(value) for name, value in emoe_diagnostics["aux_losses"].items()})
print("辅助诊断前后 logits 最大误差:", float((emoe_plain["logits"] - emoe_diagnostics["logits"]).abs().max()))
print("训练样本 0 的可用模态 T/A/V:", emoe_diagnostics["modality_availability"][0].cpu().tolist())
print("训练样本 0 的可靠性误差 T/A/V:", emoe_diagnostics["routing_errors"][0].cpu().tolist())
print("训练样本 0 的路由权重 T/A/V:", emoe_diagnostics["routing_weights"][0].cpu().tolist())
print("训练样本 0 的可靠性目标 T/A/V:", emoe_diagnostics["importance_targets"][0].cpu().tolist())
print("在线蒸馏目标形状:", tuple(emoe_diagnostics["distillation_target"].shape))
```

## 6. 分开查看任务损失与辅助目标

训练总损失包含单模态、路由和蒸馏目标，验证损失仅为最终融合任务损失。下面分别画融合任务损失、四项未经系数缩放的辅助目标，以及验证指标；负熵使总损失可能为负，因此不把训练总损失和验证任务损失直接作差。完整总损失保存在 `history.json`。


```python
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("曲线数据已保存在 history.json")
else:
    history = emoe_experiment.history
    epochs = [r["epoch"] for r in history]
    metric = "macro_f1" if emoe_config.task == "classification" else "mae"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()
    axes[0].plot(epochs, [r["train_task_loss"] for r in history], label="train fused task")
    axes[0].plot(epochs, [r["valid"]["loss"] for r in history], label="valid fused task")
    axes[0].set(xlabel="Epoch", ylabel="Task loss")
    axes[0].legend()
    for axis, name in zip(axes[1:5], emoe_config.auxiliary_weights):
        axis.plot(epochs, [r["train"][name] for r in history])
        axis.set(xlabel="Epoch", ylabel="Raw auxiliary objective", title=name)
    axes[5].plot(epochs, [r["valid"][metric] for r in history])
    axes[5].set(xlabel="Epoch", ylabel=f"Validation {metric}")
    fig.tight_layout()
    plt.show()
```

## 7. 槽位追溯、保存的路由与模型重载

输出目录保存来源与配置、训练集标准化参数、mask 审计、训练历史、最佳模型和验证/测试指标。预测 NPZ 同时保存样本 ID、最终 logits、`routing_weights [N,3]`、`modality_logits [N,3,C]` 与 `modality_availability [N,3]`，顺序都是 T/A/V；回归时 `C=1`。`predict_split` 重载预测也返回这些字段。

解释接口固定 mask 和目标/参照类别，替换某个模态某个槽位的缓存特征，再运行整个模型。因此它同时包含专家表示变化与路由重新加权，而非固定路由下的孤立影响。返回结果是特征替换后的预测敏感性，不是因果贡献；可通过 `id + modality + time_index` 关联原始词时间戳、音频和帧号。缓存 BERT 会把词的信息分散到其他上下文槽位中，原始词删除需要重新提取特征才能检验。


```python
emoe_explanation = explain_emoe_features(emoe_experiment, split="test", index=0)
print("解释是否包含路由重新加权:", emoe_explanation["includes_router_reweighting"])
for row in emoe_explanation["rows"][:15]:
    print(row)

emoe_model, emoe_metadata = load_emoe_model(emoe_experiment.run_dir / "best.pt")
one_sample = {key: value[:1] for key, value in data["test"].items()}
one_sample_masks = {
    m: value[:1] for m, value in (mask_overrides or {}).get("test", {}).items()
} or None
emoe_prediction = predict_emoe_split(emoe_model, one_sample, emoe_metadata, masks=one_sample_masks)
print("重载后的 logits:", emoe_prediction["logits"])
print("重载后的路由 T/A/V:", emoe_prediction["routing_weights"])
print("重载后的可用模态 T/A/V:", emoe_prediction["modality_availability"])
```
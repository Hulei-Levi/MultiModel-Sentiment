# MultiModel-Sentiment

## 基于跨模态级联融合与残差重构的可解释多模态情感分析

本项目面向文本、语音和视觉三种模态，完成从词级时间对齐、特征提取，到情感预测、缺失特征修复和预测依据解释的建模与实验。预测任务同时包含**情感极性三分类**与**连续情感强度回归**；解释结果进一步细化到模态、输入位置、特征维度和词语。

项目围绕三问展开：

| 任务 | 主要方法 | 已完成的效果 |
| --- | --- | --- |
| 第一问：特征提取与时间对齐 | Wav2Vec2 CTC 强制对齐、BERT、双分支语音特征、DINOv2、自适应视频采帧 | 完成 100 条视频、1,931 个原词的处理，其中 1,928 个词获得时间区间 |
| 第二问：预测与抗缺失建模 | ThisWork 跨模态级联融合、分类／回归联合训练、多尺度时序残差重构 | 完整测试集 Macro-F1 为 0.6193、回归 MAE 为 0.6242；50% 同步缺失时，重构使 Macro-F1 从 0.4451 提升至 0.5106 |
| 第三问：可解释性分析 | 精确模态 Shapley、条件积分梯度、参考替换与整词干预 | 完成 728 条验证样本的总体分析、60 条样本的详细诊断，以及附件四全部 20 条样本的预测与解释 |

以下数值来自当前服务器保存的实验报告。完整输入、受扰输入、验证集诊断和无标签附件的统计分别呈现。

## 核心代码快速导航

如果第一次阅读项目，先按下面的入口查看，再进入各问的脚本职责表。表中的结果目录指**当前已保存的实验批次**，新运行的输出位置由实际配置决定。

| 想了解的问题 | 建议首先打开 | 接着看什么 |
| --- | --- | --- |
| 第一问：如何把原始媒体变成对齐特征？ | [extract_this_work.py](question1/extract_this_work.py) 的 `extract_sample()` | `align_words()` → 各模态特征函数 → `pool_intervals()`；[第一问脚本导航](#q1-code) |
| 第二问：本文模型如何训练和预测？ | [second_question_ourwork.ipynb](question2/second_question_ourwork.ipynb) | [attention_models.py](multimodal_suite/attention_models.py) 的 `ThisWork` → [multitask_runtime.py](multimodal_suite/multitask_runtime.py)；[第二问脚本导航](#q2-code) |
| 第二问：缺失数据如何修复、怎样比较前后效果？ | [train_recap_reconstructor.py](question2/train_recap_reconstructor.py) 的 `RECAPReconstructor` | [evaluate_reconstructed_this_work.py](question2/evaluate_reconstructed_this_work.py)，再看单模态与位置消融入口 |
| 第三问：贡献值与关键证据怎样计算？ | [explain_this_work.py](question3/explain_this_work.py) 的 `run_analysis()` | `shapley_from_coalitions()` → `conditional_ig()` → 干预与媒体定位；[第三问脚本导航](#q3-code) |
| 只想查看／重画一个样本的解释图？ | [plot_sample_waterfall.py](question3/plot_sample_waterfall.py) | 读取已有归因与媒体，生成分类、回归综合图，不重新推理或训练 |

**区分三个层次：**Notebook 负责组织实验；`multi_fusion_model/this_work.py` 提供导入接口；真正的 ThisWork 网络与训练实现分别位于 `multimodal_suite/attention_models.py` 和 `multimodal_suite/multitask_runtime.py`。已有结果放在 `results/` 和第三问的输出目录中，不能将这些目录当作模型源码入口。

## 数据与任务设置

| 数据 | 用途 | 当前规模与形式 |
| --- | --- | --- |
| 附件一 | 原始媒体的词级对齐与特征提取 | 100 条视频及对应文本 |
| 附件二 | 模型训练、验证与有标签测试 | train 3,395 条、valid 728 条、test 727 条 |
| 附件三 aligned | 模态缺失条件下的实际推理 | 30 个文件、30 条样本，无真实标签和完整特征对照 |
| 附件四 | 全量预测、模态贡献与关键证据分析 | 20 条样本，无真实标签 |

第二、三问使用附件中已有的对齐特征。每条样本最多 50 个位置：

```text
text       [50, 768]   文本特征
audio      [50,  74]   语音特征
vision     [50,  35]   视觉特征
text_bert  [ 3,  50]   token IDs、attention mask、token type IDs
```

有效位置由 attention mask 确定。人工缺失与 padding 分开记录；音视频的自然零值不直接等同于人工缺失。标准化统计量由训练集拟合，并随模型权重保存。

第一问重新提取的特征为词级变长序列，其音视频维度与上述预提取特征不同。两条数据流程分别保存；第一问的新特征不能直接送入当前按 `768/74/35` 维训练的最优权重。

## 第一问：词级对齐与特征提取

<a id="q1-code"></a>

### 核心脚本与结果对应

**推荐顺序：**先读完整提取器的 `extract_sample()`，理解输入到输出的流程；需要单独查看强制对齐和异常复核规则时，再读 `first_question.py`。

| 脚本 | 负责的工作与核心函数 | 对应的主要结果 |
| --- | --- | --- |
| [question1/extract_this_work.py](question1/extract_this_work.py) | 完整特征提取主入口。`extract_sample()` 串起流程；`align_words()` 生成词级时间区间；`text_features()`、`audio_features()`、`visual_features()` 提取三模态特征；`pool_intervals()` 按时间重叠汇聚；`video_frame_budget()` 控制采帧预算 | 每样本 `samples/*.npz` 与 `samples/*.json`、批次 `summary.csv` 和 `run_config.json`；对应本节 100 条样本与 6,553 帧的完整动态提取结果 |
| [question1/first_question.py](question1/first_question.py) | 独立词级对齐工具。`process()` 组织单样本处理；`align_characters()` 完成 CTC 对齐；`media_timeline()`、`frame_ids()` 关联原始视频时间轴与帧索引 | [results/word_alignment/](results/word_alignment/) 中的 `summary.csv`、`words.csv` 与逐样本 JSON；用于检查对齐覆盖与待复核词项 |

两个脚本都可以直接读取原始媒体和文本。完整提取器内部会执行对齐，不要求先运行 `first_question.py`，也不读取后者的输出作为中间输入。

### 方法

1. 将视频音轨解码为 16 kHz 单声道音频，并读取视频帧的实际呈现时间戳（PTS）。
2. 使用 WhisperX 加载英文 Wav2Vec2 对齐模型，通过 CTC 字符强制对齐获得时间位置，再汇聚到原始英文词。这里使用给定文本进行对齐。
3. 使用 BERT 提取子词向量，根据原文字符偏移归并到原词。
4. 分别提取 openSMILE eGeMAPSv02 LLD 和 Wav2Vec2 第 6 层语音表示，按原生时间窗与词区间的重叠时长加权汇聚。
5. 根据词级声学对齐分数设置视觉采样预算：低于 0.5 时最多 9 帧、0.5～0.6 时最多 6 帧、不低于 0.6 时最多 3 帧。对选中帧提取 DINOv2 CLS 表示，再汇聚为词级视觉特征。

| 特征分支 | 编码器／描述符 | 每词维度 |
| --- | --- | ---: |
| 文本 | BERT base uncased | 768 |
| 语音声学描述 | eGeMAPSv02 LLD | 25 |
| 语音上下文表示 | Wav2Vec2 第 6 层 | 768 |
| 视觉 | DINOv2 ViT-S/14 CLS | 384 |

### 已有结果

完整动态采帧批次处理 **100 条样本，失败 0 条**。共保留 1,931 个原词：文本特征有效词数为 1,931，语音与视觉特征有效词数均为 1,928。实际编码的视频帧按样本内去重后合计 **6,553 帧**。

每条样本保存一份 NPZ 特征和一份 JSON 对齐记录，包含原词、字符范围、词级时间区间、所选视频帧及有效性信息。没有可靠时间位置的词保留原始记录和状态标记。

- 对齐工具：[question1/first_question.py](question1/first_question.py)
- 完整提取入口：[question1/extract_this_work.py](question1/extract_this_work.py)
- 仓库内对齐结果：[results/word_alignment/](results/word_alignment/)
- 当前完整动态提取批次保存在仓库外：`/data_disk/disk_1/lhx/huawei_cup/q1/20260926_dynamic/outputs/runs/q1_dynamic_all100_20260926_01/`，统计来源为其中的 `run_config.json`、`summary.csv` 和 `samples/`。

## 第二问：双任务情感预测与缺失重构

<a id="q2-code"></a>

### 核心脚本与结果对应

**推荐顺序：**先看 ThisWork Notebook 中的数据与配置，再看网络和训练代码；需要理解抗缺失结果时，依次阅读扰动生成、重构模型和下游评估。

#### 预测模型与训练

| 文件 | 负责的工作与定位点 | 对应的主要结果 |
| --- | --- | --- |
| [second_question_ourwork.ipynb](question2/second_question_ourwork.ipynb) | 本文模型的实验组织入口：读取附件二，设置 `ThisWorkConfig`，调用 `fit_this_work_experiment()` | [results/second_question/this_work/](results/second_question/this_work/) 下的训练记录、最佳权重、分类／回归指标与预测 |
| [multi_fusion_model/this_work.py](multi_fusion_model/this_work.py) | 面向 Notebook 和推理脚本的公共接口，导出 `Config`、`Model`、`fit_experiment()`、`load_experiment_model()`、`predict_split()` | 本文件主要转导出实现；调用现有模型时从这里导入 |
| [attention_models.py](multimodal_suite/attention_models.py) | 网络主体。`ThisWork` 定义双任务模型；`_ResidualSequenceEncoder` 实现模态内编码；`_BidirectionalPairFusion` 与 `_CascadeFusion` 实现跨模态融合 | 决定 ThisWork 的网络结构、前向输出和辅助重建分支 |
| [multitask_runtime.py](multimodal_suite/multitask_runtime.py) | 联合训练与推理。`fit_experiment()` 组织预训练和监督训练；`compute_task_losses()` 计算分类／回归损失；`selection_score()` 选优；`predict_split()` 执行预测 | `best.pt`、`history.json`、`metrics.json`、`valid/test_predictions.*` |
| [common.py](multimodal_suite/common.py) | 公共网络工具：`ProjectedInputs`、`SafeAttention`、`masked_mean()`，用于投影、掩码注意力和池化 | 无独立实验输出；排查输入维度或全缺失分支时查看 |
| [weighted_sum_fusion.py](multi_fusion_model/weighted_sum_fusion.py) | 同时包含被其他模型复用的数据工具：`resolve_masks()`、`fit_normalizers()`、`FeatureDataset` | 查有效掩码、训练集标准化和数据装载逻辑时查看 |

[分类对比 Notebook](question2/second_question_classfication.ipynb) 和 [分类／回归对比 Notebook](question2/second_question_classfication_regression.ipynb) 负责比较模型的配置与调用；后者也包含 ThisWork 的实验单元。历史基线结果分别保存在 [calssification/](results/second_question/calssification/) 和 [regression/](results/second_question/regression/)（前者保留已有目录拼写）。

<details>
<summary>其他比较模型：网络实现位置</summary>

| 模型 | 主要网络文件／类 |
| --- | --- |
| Weighted Sum | [weighted_sum_fusion.py](multi_fusion_model/weighted_sum_fusion.py)：`WeightedSumFusion` |
| Poria / bc-LSTM | [context_lstm_fusion.py](multi_fusion_model/context_lstm_fusion.py)：`ContextLSTMFusion` |
| Ren、Zheng、Pan | [attention_models.py](multimodal_suite/attention_models.py)：`Ren2021Model`、`Zheng2022Model`、`Pan2020Model` |
| MFRM、TransModality | [memory_translation.py](multimodal_suite/memory_translation.py)：`MFRM2022Model`、`TransModality2020Model` |
| M3ER、MEmoBERT、HyCon | [robust_pretraining.py](multimodal_suite/robust_pretraining.py)：`M3ER2020Model`、`MEmoBERT2022Model`、`HyCon2022Model` |
| EMOE | [emotion_experts.py](multimodal_suite/emotion_experts.py)：`EMOE2025Model` |
| CaReFlow | [rectified_flow.py](multimodal_suite/rectified_flow.py)：`CaReFlow2026Model` |

各模型的调用入口位于 [multi_fusion_model/](multi_fusion_model/) 对应的 `*_fusion.py` 文件；公共单任务运行逻辑位于 [runtime.py](multimodal_suite/runtime.py)。Poria、Pan 等带专门分阶段训练的实现还在各自入口文件中定义训练流程。

</details>

#### 扰动、重构与鲁棒性评估

| 脚本 | 负责的工作与定位点 | 对应的主要结果 |
| --- | --- | --- |
| [bert_feature_adapter.py](question2/bert_feature_adapter.py) | `fit_experiment()` 拟合 BERT 输出到既有文本特征空间的线性映射；`BertTextAdapter.encode_text_bert()`、`load_adapter()` 提供编码接口 | [experiments/bert_feature_adapter/](experiments/bert_feature_adapter/) 下的 `adapter.pt` 和适配评估 |
| [generate_perturbed_dataset.py](question2/generate_perturbed_dataset.py) | `generate_datasets()` 组织生成；`make_corruption_mask()`、`perturb_split()` 添加同步扰动；`encode_split()` 重新编码受扰文本 | [datasets/附件2-同步扰动特征/](datasets/附件2-同步扰动特征/) 中的特征 PKL、缺失掩码和生成记录 |
| [train_recap_reconstructor.py](question2/train_recap_reconstructor.py) | 六种重构方法的统一实现与训练。`RECAPReconstructor` 是本文模型；`BaseReconstructor` 规定共同输入输出；`train()`、`evaluate()` 训练并评估重建质量；`ReconstructionBundle.reconstruct()` 提供修复接口 | `results/second_question/*_reconstruction/` 下的权重、重建误差和报告；本脚本不负责下游情感任务评估 |
| [evaluate_this_work_perturbations.py](question2/evaluate_this_work_perturbations.py) | `main()` 加载冻结 ThisWork，直接预测受扰测试集；`compare_historical()` 核对完整输入基准 | [this_work_robustness/](results/second_question/this_work_robustness/)：未重构条件的分类／回归指标和预测 |
| [evaluate_reconstructed_this_work.py](question2/evaluate_reconstructed_this_work.py) | `main()` 串联“冻结重构器 → 冻结 ThisWork”；`check_restoration()` 检查修复范围；`metrics()` 计算下游指标 | [this_work_reconstruction_evaluation/](results/second_question/this_work_reconstruction_evaluation/) 及指定的其他方法评估目录：修复后的下游预测与效果比较 |
| [single_modality_robustness.py](question2/single_modality_robustness.py) | `run_experiment()` 训练单模态受扰版本；`SingleModalityRECAP`、`apply_corruption()` 保持未受扰模态；`evaluate_test()` 评估；`reconstruction_metrics()` 统计误差；`recompute_reconstruction_mae()` 对应补算 MAE 的 `reconstruction-mae` 入口 | [single_modality_robustness/](results/second_question/single_modality_robustness/)：文本／音频／视觉分别受扰的重建与预测结果 |
| [evaluate_missing_positions.py](question2/evaluate_missing_positions.py) | `position_mask()` 生成首／中／尾连续缺失；`worker()` 执行冻结模型比较；`aggregate()` 汇总 | [missing_position_ablation/](results/second_question/missing_position_ablation/)：`summary.csv`、三种位置对照表、`comparison_tables.tex` 和报告 |
| [infer_attachment3_aligned.py](question2/infer_attachment3_aligned.py) | 附件三实际推理入口。`load_inputs()` 读取样本；`infer_masks()` 识别缺失标记；`main()` 串联编码、重构与 ThisWork | [attachment3_aligned_inference/](results/second_question/attachment3_aligned_inference/)：30 条样本的预测 CSV、掩码与统计报告 |

**复现范围说明：**两个 `evaluate_*this_work*` 脚本中的 `LEVELS` 当前固定为 `(0, 10, 20, 30)`。40%／50% 的历史追加测试及 `comparison_all_rates.csv` 已保存在 [frozen_high_missing_evaluation/](results/second_question/frozen_high_missing_evaluation/)，当前仓库未保留单独的一键生成该整合表的脚本。位置实验中的 `position_before_after.csv` 是主实验之后补充整理的结果，也不是 `aggregate()` 直接生成的文件。

`generate_perturbed_dataset.py` 和上述两个评估脚本迁移到 `question2/` 后，部分默认数据／结果路径仍按脚本目录拼接；复现时应显式传入项目根目录下的数据、权重和输出路径。这里的导航用于定位现有实现，不意味着直接使用所有默认参数就能复现每份历史结果。

### ThisWork：共享融合表示，分别预测类别与强度

ThisWork 将三个模态投影到 128 维，使用残差时序卷积与自注意力提取模态内上下文，再通过三条级联路径建模跨模态交互：

```text
文本 [B,50,768] ── 投影 + 残差卷积/自注意力 ──┐
语音 [B,50, 74] ── 投影 + 残差卷积/自注意力 ──┼─→ 三条级联融合路径
视觉 [B,50, 35] ── 投影 + 残差卷积/自注意力 ──┘

路径一：(语音 + 文本) → 视觉
路径二：(语音 + 视觉) → 文本
路径三：(文本 + 视觉) → 语音

每次两路融合：双向跨模态注意力 → 拼接 → 1×1 卷积
三条路径整合 → masked mean pooling
             ├─ 分类头 → 三类 logits
             └─ 回归头 → 连续情感强度
```

训练采用两个阶段：先进行 3 轮辅助特征重建预训练，再以 `交叉熵 + MAE` 等权联合训练分类与回归。所选配置在联合监督阶段关闭辅助重建损失。验证集选优同时考虑 Macro-F1 和归一化回归 MAE，归一化尺度固定来自训练集的中位数常数预测基线。

模型实现位于 [attention_models.py](multimodal_suite/attention_models.py)，联合训练位于 [multitask_runtime.py](multimodal_suite/multitask_runtime.py)，对外接口为 [multi_fusion_model/this_work.py](multi_fusion_model/this_work.py)。

### 完整输入下的预测效果

当前下游推理与解释使用 `20260925_150303_247629/best.pt`，由验证集选中第 4 个联合训练 epoch。以下为附件二原始测试集 **727 条样本**的结果：

| 模型 | Accuracy ↑ | Macro-F1 ↑ | 回归 MAE ↓ | Pearson ↑ |
| --- | ---: | ---: | ---: | ---: |
| ThisWork | 0.6768 | 0.6193 | 0.6242 | 0.6694 |

来源：[完整指标](results/second_question/this_work/20260925_150303_247629/metrics.json)、[数据划分审计](results/second_question/this_work/20260925_150303_247629/data_audit.json)、[实验配置](results/second_question/this_work/20260925_150303_247629/config.json)。

项目还实现了 Weighted Sum、Poria/bc-LSTM、Ren/IMAN、Zheng、Pan、M3ER、MFRM、TransModality、MEmoBERT、HyCon、EMOE 和 CaReFlow 等比较方法。它们基于本项目的预提取特征与任务接口进行适配，相关结果为本项目实验结果，不等同于原论文的复现分数。

### 本文重构器：多尺度时序卷积与联合注意力

独立缺失重构器保留历史名称 `RECAPReconstructor`，实际为 **RECAP-inspired 自定义特征重构模型**，没有实现原论文完整的对抗生成、FID 等模块。它与 ThisWork 内部用于预训练的辅助重建头是两个独立模块。

重构流程：

```text
受扰文本 tokens → 冻结 BERT + 已训练 adapter → 受扰文本向量
受扰音频、视觉 → 人工缺失位置置零
                         ↓
训练集标准化 + 64维投影 + 位置/模态/缺失状态嵌入
                         ↓
三路独立多尺度卷积（卷积核 3、5、9）
                         ↓
沿序列维拼接为 [B,150,64]
                         ↓
2层、4头 Transformer：样本内跨模态交互
                         ↓
三路残差解码 → 受控回加 → 恢复原始特征单位
                         ↓
ThisWork 自身的预处理 → 分类与回归
```

修复对象是特征向量，不会将 `[UNK]` 还原成具体词语。同步缺失版本允许受扰样本的文本在全部有效位置更新，以处理整句 BERT 重新编码带来的上下文变化；音视频仅更新明确标记的人工缺失位置。无扰动样本原样返回，padding 保持为零。

在同一文件中还实现了 MTSIT、BRITS、Centaur、NAOMI 和 CSDI 的重构适配版本，统一输入、输出与评估接口。主体代码为 [train_recap_reconstructor.py](question2/train_recap_reconstructor.py)。

### 同步随机缺失下的效果

在有效内容位置同步扰动三模态：文本替换为 `[UNK]` 后重新编码，音视频对应特征置零；排除 padding 与特殊起止标记。百分比表示 Bernoulli 名义缺失率，单条样本的实际缺失比例可以不同。

以下全部使用附件二原 test 的 727 条样本，并固定 ThisWork 与已有同步缺失重构器权重。10%～30% 整合历史结果；40%～50% 是冻结权重后的追加测试，超出了该重构器原有的 10%～30% 训练扰动范围。

| 名义缺失率 | 未重构 Macro-F1 ↑ | 重构后 Macro-F1 ↑ | 未重构回归 MAE ↓ | 重构后回归 MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| 10% | 0.5890 | 0.6030 | 0.6636 | 0.6547 |
| 20% | 0.5757 | 0.5906 | 0.7031 | 0.6861 |
| 30% | 0.5378 | 0.5668 | 0.7391 | 0.7177 |
| 40% | 0.4852 | 0.5268 | 0.7664 | 0.7391 |
| 50% | 0.4451 | 0.5106 | 0.8004 | 0.7792 |

五档扰动中，重构均改善了两个下游指标。50% 缺失时，Macro-F1 提升约 **6.55 个百分点**，回归 MAE 降低 **0.0212**。

来源：[全部方法与缺失率对照](results/second_question/frozen_high_missing_evaluation/run_20260926T094125_413222Z/comparison_all_rates.csv)、[完整报告](results/second_question/frozen_high_missing_evaluation/run_20260926T094125_413222Z/report.md)。报告还包含未重构与六种重构方法的完整比较。

**误差口径：**上表 MAE 是情感强度预测误差。历史重构报告中的 `primary_mse`（例如 10% 条件下的 0.32031537）是标准化特征空间的重建 MSE，不能当作重建 MAE。重建质量和下游预测质量分别记录，特征误差下降也不保证每种方法的下游表现都改善。

### 位置与受扰模态消融

项目分别测试了首部、中部、尾部连续缺失，以及仅文本、仅音频、仅视觉受扰。位置实验固定既有权重，同一样本、同一缺失率下，各位置方案的缺失数量一致。

50% 连续缺失时，本文重构器的下游结果为：

| 缺失位置 | Macro-F1：重构前 → 后 | 回归 MAE：重构前 → 后 |
| --- | --- | --- |
| 首部 | 0.4626 → 0.5426 | 0.7664 → 0.7346 |
| 中部 | 0.4439 → 0.5201 | 0.7690 → 0.7281 |
| 尾部 | 0.4203 → 0.5279 | 0.7693 → 0.7225 |

来源：[位置实验前后对照](results/second_question/missing_position_ablation/run_20260926T212817_809893Z/position_before_after.csv)。

单模态实验另行训练了一个覆盖三种受扰模态的重构器版本，保留两路未受扰输入；其权重与上述同步缺失重构器不同。结果中，文本修复能改善预测，而部分音视频条件下虽然重建误差下降，下游指标仍可能变差。详见 [单模态实验报告](results/second_question/single_modality_robustness/run_20260926T110006_216575Z/report.md)。

### 附件三 aligned 全量推理

已完成 **30 条样本**的“BERT adapter → 本文重构器 → ThisWork”推理，其中 27 条识别到缺失标记并应用重构，3 条未识别到缺失标记。输出类别、类别概率、连续情感分数、缺失统计与模型来源。

附件三没有真实标签或完整特征对照，因此这里报告预测与缺失统计，不计算分类、回归或重构性能指标。

成果：[预测 CSV](results/second_question/attachment3_aligned_inference/run_20260926T172935_071166Z/predictions.csv)、[推理报告](results/second_question/attachment3_aligned_inference/run_20260926T172935_071166Z/report.md)。

## 第三问：分层归因与关键证据追踪

<a id="q3-code"></a>

### 核心脚本与结果对应

**推荐顺序：**先看 `run_analysis()` 理解单样本解释如何产生，再看 Shapley 与积分梯度；需要完整验证分析与媒体证据时，继续看 `q3_completion_main()`。只需重画已有样本图时，可直接使用独立绘图脚本。

主流程和算法集中在 [question3/explain_this_work.py](question3/explain_this_work.py)，可按下面的函数名直接定位：

| 想查看的内容 | 核心函数／入口 | 对应的主要结果 |
| --- | --- | --- |
| 附件四全量预测与归因 | `main()` → `run_analysis()` | [attachment4_all_20260925T152718_260737Z/](question3/attachment4_all_20260925T152718_260737Z/) 中的 `analysis_report.json` 和 `attributions.npz` |
| 模态贡献如何计算 | `scalar_outputs()` 固定解释目标；`coalition_predictions()` 枚举组合；`shapley_from_coalitions()` 计算 Shapley | 分类／回归模态贡献、组合预测与总体作用度 |
| 时间步和特征维度如何归因 | `integrate_edge()`、`conditional_ig()` | `attributions.npz` 中的局部归因及数值收敛记录 |
| 解释是否符合模型响应 | `deletion_check()`、`word_validation()` | 位置替换、整词替换记录，以及验证集的 `diagnostic_occlusion.csv` |
| 728 条验证样本与 60 条详细诊断 | `q3_completion_main()` 调度；`q3_validation_prepare()`、`q3_select_diagnostic_cases()`、`q3_validation_details()`、`q3_validation_summary()` 计算与汇总 | [completion_20260926/validation/](question3/completion_20260926/validation/) 中的全量预测、诊断样本、逐例归因和分析汇总 |
| 归因怎样对应词语、音频与视频 | `q3_build_attachment4_evidence()`、`q3_media_word_mapping()`、`q3_media_export_asset()` | [completion_20260926/attachment4/](question3/completion_20260926/attachment4/) 下的对齐 JSON、音频片段、视频帧与关键证据 CSV |
| 完整图表与结果汇编 | `q3_make_completion_figures()`、`q3_plot_validation()`、`q3_plot_attachment4()`、`q3_write_result_index()` | 综合结果索引、中文表格、验证和附件四解释卡 PDF |

`--complete-q3` 流程以**已有附件四归因报告**为前置输入，通过 `--completion-stage prepare/details/media/figures` 分阶段组织验证分析、媒体证据和图表；它不会自动重算前面的附件四归因。当前对应成果为 [completion_20260926/](question3/completion_20260926/)。

独立绘图入口为 [question3/plot_sample_waterfall.py](question3/plot_sample_waterfall.py)：`main()` → `q3_overview_source()` → `q3_plot_sample_overview()`，默认输出至 [question3/waterfall_results/](question3/waterfall_results/) 的 `sampleXX/`，每个样本分别生成分类、回归综合图及合并 PDF。它只读取已有归因、对齐记录与媒体，不加载 ThisWork，不重新计算 IG，也不重新执行对齐。

```bash
# 从项目根目录重画附件四第 1 个样本，要求已有对应归因和媒体结果
.venv-align/bin/python -B question3/plot_sample_waterfall.py 1
```

### 方法

固定已训练的 ThisWork，使用三层分析解释预测：

1. **模态贡献：**枚举三模态全部 8 个组合，计算精确模态 Shapley。未进入组合的模态替换为参考特征，保持原掩码与位置不变。
2. **局部贡献：**对 12 条条件路径计算积分梯度，再按 Shapley 权重汇总到特征维度、内容槽位与完整词语。分别保留有符号贡献和绝对重要性。
3. **解释复核：**比较高重要性、低重要性和 20 组随机位置替换带来的输出变化；补充整词 `[UNK]` 替换与重新编码、训练均值／零参考对照及数值收敛检查。

分类解释目标为完整输入下原预测类别相对原次高类别的**固定 logit 差**；回归解释目标为连续情感强度。参考均值仅由训练集计算。局部解释不把注意力权重直接当作贡献，也不把模型归因直接解释为因果关系。

### 总体模态作用

附件四 **20 条样本**在训练均值参考下，各模态的平均绝对 Shapley 份额为：

| 解释任务 | 文本 | 语音 | 视觉 |
| --- | ---: | ---: | ---: |
| 情感极性分类 | 66.11% | 13.53% | 20.36% |
| 情感强度回归 | 64.25% | 14.15% | 21.60% |

在该批样本中，文本平均作用度最高，视觉与语音提供补充信息；不同样本仍有不同的模态依赖。这些百分比表示输出归因份额，不是准确率贡献。附件四无真实标签，因此不报告其 Accuracy、Macro-F1 或回归 MAE。

来源：[附件四分析报告](question3/attachment4_all_20260925T152718_260737Z/analysis_report.json)。

### 解释是否对应模型实际响应

从 728 条有标签验证样本中，按类别、分类对错与回归误差分层选取 60 条详细诊断样本。在相同预算下，三模态同步替换最多 4 个内容位置（不足 4 个时使用全部内容位置）。将高归因位置替换为参考特征，模型输出平均变化高于随机与低归因替换：

| 解释目标 | 高归因位置 | 随机位置 | 低归因位置 |
| --- | ---: | ---: | ---: |
| 分类 margin 的平均绝对变化 | 0.7448 | 0.3083 | 0.2083 |
| 回归分数的平均绝对变化 | 0.2533 | 0.1033 | 0.0685 |

分类中 43/60 条、回归中 44/60 条的高归因干预响应超过随机对照均值。这支持归因排序与模型输出敏感性具有一定一致性，统计范围仅为该诊断子集。60 条样本、两种参考对应的 **1,440 条条件 IG 积分边均通过数值收敛检查**。

来源：[验证分析汇总](question3/completion_20260926/validation/analysis_summary.json)、[干预明细](question3/completion_20260926/validation/diagnostic_occlusion.csv)、[完成记录](question3/completion_20260926/completion_manifest.json)。

### 原始媒体证据与成果

附件四的解释卡同时展示预测结果、模态贡献、局部特征与词语重要性，以及词级时间对应的音频片段和视频画面。媒体定位采用事后词级时间锚点；当前未恢复缓存特征原始提取时间窗，因此待复核的画面保留对应标记，不能直接把模型槽位编号当成秒数或帧号。

- [结果总索引](question3/completion_20260926/结果索引.html)
- [中文结果表格](question3/completion_20260926/中文表格/)
- [附件四全量预测与解释 CSV](question3/completion_20260926/attachment4/attachment4_predictions_and_explanations.csv)
- [附件四关键证据 CSV](question3/completion_20260926/attachment4/attachment4_key_evidence.csv)
- [附件四解释卡 PDF](question3/completion_20260926/attachment4/attachment4_explanation_cards.pdf)
- [验证诊断解释卡 PDF](question3/completion_20260926/validation/validation_explanation_cards.pdf)

## 代码组织

```text
MultiModel-Sentiment/
├── question1/                 原始媒体对齐与词级特征提取
├── question2/                 预测实验、扰动生成、重构、鲁棒性与附件三推理
│   ├── second_question_ourwork.ipynb
│   ├── bert_feature_adapter.py
│   ├── generate_perturbed_dataset.py
│   ├── train_recap_reconstructor.py
│   ├── single_modality_robustness.py
│   ├── evaluate_missing_positions.py
│   └── infer_attachment3_aligned.py
├── question3/                 分层解释、关键证据、解释卡与全量汇总
│   ├── explain_this_work.py
│   └── completion_20260926/
├── multi_fusion_model/        各模型对外统一接口
├── multimodal_suite/          网络结构、掩码处理与训练运行逻辑
├── datasets/                  原始预提取特征与人工扰动数据
├── experiments/               BERT 特征适配器等实验权重
└── results/                   预测、重构及消融实验报告
```

## 使用入口与当前权重

环境主要依赖 PyTorch、NumPy、Transformers；原始媒体处理还需要 FFmpeg/FFprobe、torchaudio、WhisperX、openSMILE 等。当前服务器使用 `.venv-align`。数据、模型缓存和大部分权重被 `.gitignore` 排除，仅克隆源码时需要另外准备。

从项目根目录打开 [ThisWork 实验 Notebook](question2/second_question_ourwork.ipynb) 可训练联合预测模型；完整对比实验分别位于 [分类 Notebook](question2/second_question_classfication.ipynb) 和 [分类／回归 Notebook](question2/second_question_classfication_regression.ipynb)。Notebook 的相对数据路径以项目根目录为基准。

当前附件推理与解释使用的固定权重：

| 组件 | 权重位置 |
| --- | --- |
| BERT 特征适配器 | `experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt` |
| ThisWork 双任务模型 | `results/second_question/this_work/20260925_150303_247629/best.pt` |
| 同步缺失重构器 | `results/second_question/recap_reconstruction/20260925T103755_004716Z/best.pt` |

在已有数据、缓存及上述权重齐备的当前服务器上，可执行附件三推理：

```bash
cd /data_disk/disk_1/hl/workspace/huawei_cup/MultiModel-Sentiment
.venv-align/bin/python -B question2/infer_attachment3_aligned.py --device cuda:0
```

其他入口可先查看参数，再明确指定输入、输出和设备：

```bash
.venv-align/bin/python -B question1/extract_this_work.py --help
.venv-align/bin/python -B question2/train_recap_reconstructor.py --help
.venv-align/bin/python -B question2/evaluate_missing_positions.py --help
.venv-align/bin/python -B question3/explain_this_work.py --help
```

第一问提取脚本的默认 `/outputs` 路径来自原容器环境；迁移运行时请显式设置媒体目录、输出目录和模型缓存目录。实验目录中的 `config.json`、`metrics.json`、预测文件与报告保存了各批次的实际配置和统计口径。

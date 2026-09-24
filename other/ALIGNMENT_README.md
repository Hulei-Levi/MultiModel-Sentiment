# 附件1单词级音视频对齐

入口为 `align_words.py`。读取原始 `label-100.xlsx` 和100个 MP4，用 WhisperX 加载英文 wav2vec2 模型，再用 TorchAudio 的 CTC 强制对齐接口将给定转写对齐到音轨，映射到实际解码的视频帧。原始视频、Excel 和 Notebook 均不修改。

使用原生 CTC 接口是为避开 WhisperX 3.3.4 内置回溯的首尾边界问题：它会把部分首尾空白计入单词，并可能使末词结束时间超出音频约一个声学时间步。没有修改第三方包；实际后端和版本写在运行配置中。

## 运行

在 apple 服务器执行：

```bash
cd /data_disk/disk_1/hl/workspace/huawei_cup/MultiModel-Sentiment
export PATH="$PWD/.venv-align/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1
.venv-align/bin/python align_words.py
```

少量样本验证：

```bash
.venv-align/bin/python align_words.py --output work/alignment_pilot \
  --sample=-mJ2ud6oKI8/6 --sample=-aqamKhZ1Ec/0 --sample=-HwX2H8Z4hY/2
```

默认使用GPU。没有GPU时加 `--device cpu`。默认结果写入 `results/word_alignment/`。重新运行会重新生成该目录内同名结果文件；可用 `--output` 指定另一个结果目录。

## 输出

- `samples/<video_id>__<clip_id>.json`：逐样本原文、源文件SHA256、规范化文本、解码帧时间戳和逐词记录。
- `words.csv`：全量逐词结果，UTF-8 BOM，可用Excel打开；列表列使用JSON字符串。
- `summary.csv`：每个样本的词数、状态计数、音频时长、解码视频帧数及错误信息。
- `run_config.json`：参数、依赖版本、脚本/对齐模块/模型校验值和处理规则。

词字段包括 `word_index`、`word`、`raw_char_span`、`alignment_text`、`alignment_char_span`、`start_s`、`end_s`、`score`、`char_coverage`、`frame_indices`、`frame_start`、`frame_end_exclusive`、`status`、`flags`。

所有索引从0开始，字符区间和时间区间为左闭右开。`frame_start` 包含首帧，`frame_end_exclusive` 不包含末端索引。例如 `[12, 15)` 对应12、13、14帧。`frame_indices` 明确列出对应帧；边界处同一帧可以同时与两个相邻单词有重叠。

## 时间和文本规则

1. 使用默认MP4 edit list语义进行真实解码，以展示PTS定义视频时间。不能使用容器 `nb_frames / fps` 推算。本批数据存在裁剪前的编码样本。
2. 音轨重采样为16 kHz单声道；保留源播放时间轴，用 `aresample=16000:async=1:first_pts=0` 处理开头空白或时间戳间隙。视频PTS不另行归零。
3. 根据 `torchaudio.functional.forced_align` 和 `merge_tokens` 返回的非blank字符声学区间合并单词，不插值。无法对齐的词保留，时间为 `null`、帧列表为空。JSON记录声学时间步；小数位数不代表时间估计具有同等精度。
4. 英文缩约词保留为一个词，连字符或破折号分隔为多个词。原始大小写和字符位置保留，模型输入统一小写和撇号。标点不单独作为单词。
5. 数字在对齐副本中展开为英文读法，1900–1999采用年份读法；`US`、`Ph` 按字母分开。数字及 `US`、`Ph`、`BIO` 标记复核，原文不改。
6. 以冒号结束的方括号说话人标签作为非发音候选排除，例如 `[President Ronald Reagan:]`；其中的原词保留为 `annotation_candidate`，附 `speaker_label_candidate` 标记，仍需回听确认。

## 质量标记

- `aligned`：字符覆盖完整，满足自动检查；不等于人工核验准确。
- `needs_review`：低分、部分字符未对齐、单词时长超过1.5秒、没有对应视频帧、数字/缩写读法待核查，或音频的贪心解码文本与原转写差异较大。
- `unaligned`：没有可用词区间。
- `annotation_candidate`：按已记录规则暂时排除的说话人说明，未人工确认，时间为空。

默认 `--min-score 0.5` 仅为复核筛选规则，不是经过标定的正确概率，也不用于删除样本。置信分数是非blank字符区间分数的均值。另用同一声学模型的CTC贪心解码作自动核查，`text_match_ratio` 为两份单词序列的 SequenceMatcher 相似率；低于0.5标记整条样本的词待复核。它不是独立人工真值，不会替换原始转写。强制对齐无法证明原文确实被说出，低分和特殊转写应回听检查。

本工作仅完成单词时间及视频帧对应关系；不包含三模态情感特征提取或模型训练。

## 本次100条结果

2026-09-23完成全部100条，处理异常为0。共保留1931个原文词记录：1533个 `aligned`、395个 `needs_review`、3个 `annotation_candidate`；其中1928个具有时间及帧对应关系。自动检查通过：原文/规范化字符区间、时间边界及顺序、逐词全部帧映射、JSON与CSV计数一致。

21条样本的 `text_match_ratio < 0.5`，这些样本的有时间戳词均标记待复核。该现象可能来自转写与音轨不一致，也可能来自模型识别错误，尚未人工回听定论。可在 `summary.csv` 按此列筛选优先检查；395个待复核词还包括其他低分及特殊读法情况。自动检查通过不代表人工评估的对齐精度。

真实输出示例：`-3g5yACwYnA/2` 中的 `technical`：

```json
{
  "word_index": 4,
  "word": "technical",
  "start_s": 3.764613,
  "end_s": 4.14508,
  "frame_start": 112,
  "frame_end_exclusive": 125,
  "frame_indices": [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124],
  "score": 0.9952,
  "status": "aligned"
}
```

即该词估计在3.764613–4.145080秒发音，与从0编号的第112至124帧有时间重叠。这是自动结果示例，未经过人工边界标注评估。

## 环境复现

服务器的 `.venv-align`、FFmpeg和模型已经准备好，直接使用前面的运行命令。交付包不包含原始数据、环境或模型；其中脚本应放回项目根目录，或显式设置 `--input` 和 `--output`。

只有在重建环境时才执行以下命令，要求 Linux x86_64、Python 3.12及兼容CUDA 12.4的NVIDIA驱动：

```bash
python3.12 -m venv .venv-align
.venv-align/bin/python -m pip install -r requirements-align.txt
.venv-align/bin/python -m pip install --no-deps whisperx==3.3.4
```

这是仅支持对齐的精简环境，未安装WhisperX的转写/说话人分离可选依赖。另需在PATH中提供FFmpeg/ffprobe；本次版本为7.0.2 static。模型为TorchAudio的 `WAV2VEC2_ASR_BASE_960H`，缓存于 `.cache/alignment_models/`，首次加载会下载，后续运行复用。具体版本、模型下载地址及SHA256见 `alignment-environment-versions.txt` 和 `run_config.json`。

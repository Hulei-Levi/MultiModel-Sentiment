#!/usr/bin/env python3
"""Align attachment 1's supplied English words to audio and decoded video frames."""
import argparse
import bisect
import csv
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
RATE = 16000
WORD = re.compile(r"\d+(?:,\d{3})*(?:st|nd|rd|th)?|[A-Za-z]+(?:['’‘][A-Za-z]+)*")


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def normalize(text):
    """Keep every original lexical item; map normalized characters back to it."""
    from num2words import num2words
    # Bracketed speaker labels ending with ':' are excluded candidates, retained for review.
    annotations = [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*:\]", text)]
    words, parts, position = [], [], 0
    for match in WORD.finditer(text):
        raw = match.group()
        spoken = raw.lower().replace("’", "'").replace("‘", "'")
        flags = []
        if raw[0].isdigit():
            number = int(re.sub(r"\D", "", raw))
            mode = "ordinal" if re.search(r"(?:st|nd|rd|th)$", raw) else "cardinal"
            if mode == "cardinal" and 1900 <= number <= 1999:
                mode = "year"
            spoken = num2words(number, lang="en", to=mode)
            spoken = " ".join(WORD.findall(spoken.lower())).replace(" and ", " ")
            flags.append("number_normalized")
        if raw in ("US", "Ph"):
            spoken = " ".join(raw.lower())
        if raw in ("US", "Ph", "BIO"):
            flags.append("acronym_reading_review")
        non_speech = any(a <= match.start() < b for a, b in annotations)
        start = position
        if not non_speech:
            parts.append(spoken)
            position += len(spoken) + 1
        words.append({
            "word_index": len(words), "word": raw, "raw_char_span": list(match.span()),
            "alignment_text": "" if non_speech else spoken,
            "alignment_char_span": None if non_speech else [start, position - 1],
            "start_s": None, "end_s": None, "score": None, "char_coverage": 0.0,
            "frame_start": None, "frame_end_exclusive": None, "frame_indices": [],
            "status": "annotation_candidate" if non_speech else "unaligned",
            "time_source": None, "flags": flags + (["speaker_label_candidate"] if non_speech else []),
        })
    return " ".join(parts), words


def read_samples(folder):
    from openpyxl import load_workbook
    book = load_workbook(folder / "label-100.xlsx", read_only=True, data_only=True)
    rows = iter(book["label"].values)
    header = next(rows)
    samples = [dict(zip(header, row)) for row in rows if any(v is not None for v in row)]
    book.close()
    for row in samples:
        row["video_id"], row["clip_id"] = str(row["video_id"]), str(row["clip_id"])
        row["sample_id"] = row["video_id"] + "/" + row["clip_id"]
    expected = {row["sample_id"] for row in samples}
    actual = {p.parent.name + "/" + p.stem for p in folder.glob("*/*.mp4")}
    if len(expected) != len(samples) or expected != actual:
        raise ValueError(f"Video/label mismatch: missing={expected-actual}, extra={actual-expected}")
    return samples


def media_timeline(path, ffprobe):
    """Decode frames: MP4 sample counts include edit-list preroll and are unsuitable."""
    data = json.loads(run([ffprobe, "-v", "error", "-show_frames", "-show_streams",
                          "-show_entries", "stream=index,codec_type,time_base,sample_rate,start_time,duration:"
                          "frame=media_type,stream_index,pts,best_effort_timestamp_time,duration_time,pkt_duration_time,nb_samples",
                          "-of", "json", str(path)]))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    vf = [f for f in data["frames"] if f["stream_index"] == video["index"]]
    af = [f for f in data["frames"] if f["stream_index"] == audio["index"]]
    pts = [float(f["best_effort_timestamp_time"]) for f in vf]
    if not pts or not af or pts[0] < -1e-6 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Missing/non-monotonic decoded presentation timestamps")
    last_duration = float(vf[-1].get("duration_time", vf[-1].get("pkt_duration_time", 0)))
    if last_duration <= 0:
        raise ValueError("Last video frame has no presentation duration")
    return {"video_pts_s": pts, "video_frame_end_s": pts[1:] + [pts[-1] + last_duration],
            "video_pts": [f.get("pts") for f in vf], "video_time_base": video["time_base"],
            "audio_first_pts_s": float(af[0]["best_effort_timestamp_time"]),
            "audio_last_end_s": float(af[-1]["best_effort_timestamp_time"]) +
                                int(af[-1]["nb_samples"]) / int(audio["sample_rate"]),
            "source_audio_sample_rate": int(audio["sample_rate"])}


def frame_ids(start, end, pts, ends):
    # Half-open intervals overlap iff frame_end > word_start and frame_start < word_end.
    return list(range(bisect.bisect_right(ends, start), bisect.bisect_left(pts, end)))


def align_characters(audio, text, model, metadata, device):
    """Native CTC keeps edge blanks outside words (WhisperX 3.3.4 stretches edges)."""
    import torch
    import torchaudio.functional as F
    dictionary = metadata["dictionary"]
    tokens = [dictionary[c if c != " " else "|"] for c in text]
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy(audio).unsqueeze(0).to(device))
        log_probs = logits.log_softmax(-1).cpu()
    path, scores = F.forced_align(log_probs, torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], scores[0].exp(), blank=0)
    if [s.token for s in spans] != tokens:
        raise ValueError("CTC path differs from supplied transcript")
    step = len(audio) / RATE / log_probs.shape[1]
    chars = [{"char": c, "start": s.start*step, "end": s.end*step, "score": s.score}
             for c,s in zip(text, spans)]
    labels = {v:k for k,v in dictionary.items()}
    greedy = "".join(labels[int(i)] for i in torch.unique_consecutive(log_probs[0].argmax(-1)) if int(i) != 0)
    return chars, greedy.replace("|", " ").strip(), step


def process(row, folder, model, metadata, args):
    import numpy as np
    path = folder / row["video_id"] / (row["clip_id"] + ".mp4")
    text, words = normalize(row["text"])
    result = {"schema_version": 1, "sample_id": row["sample_id"],
              "source_video": os.path.relpath(path, ROOT), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "raw_text": row["text"], "label": row["label"], "annotation": row["annotation"],
              "alignment_text": text, "time_origin": "source MP4 presentation timeline after edit lists, seconds",
              "frame_index_convention": "0-based decoded presentation order; end exclusive",
              "words": words, "error": None}
    try:
        """============================== 这里开始提取视频的时间轴 =============================="""
        timeline = media_timeline(path, args.ffprobe)
        # Preserve the common media timeline, including leading silence or timestamp gaps.
        # 下面的这一段是用于说明获取了音轨的相关特征
        audio = np.frombuffer(run([args.ffmpeg, "-v", "error", "-threads", "2", "-copyts", "-i", str(path),
                                  "-map", "0:a:0", "-vn", "-ac", "1", "-af",
                                  f"aresample={RATE}:async=1:first_pts=0", "-f", "f32le", "-ar", str(RATE), "-" ]),
                              dtype=np.float32).copy()
        # duration是音频时长，等于“采样点数量 ÷ 每秒采样点数量”。
        duration = len(audio) / RATE
        # 确保视频时长有效
        if abs(duration - timeline["audio_last_end_s"]) > 0.05:
            raise ValueError("Decoded audio duration disagrees with presentation timestamps")

        result.update(timeline, audio_duration_s=duration, alignment_sample_rate=RATE)

        # 得到字符时间，并粗查原文与音频是否匹配
        # chars：给定文本中各字符的开始时间、结束时间和分数。
        # greedy：同一声学模型直接解码得到的文本，不强制使用给定转写。
        # step：模型一个声学时间步对应多少秒。
        chars, greedy, step = align_characters(audio, text, model, metadata, args.device)

        # 接着用于对比声音与文本是否直接匹配，用于发现疑似“转写和音频内容不匹配”的样本
        match_ratio = SequenceMatcher(None, text.split(), greedy.split(), autojunk=False).ratio()

        result.update(acoustic_text_check=greedy, acoustic_time_step_s=step, text_match_ratio=round(match_ratio, 4))
        for word in words:
            span = word["alignment_char_span"]
            if span is None:
                continue
            relevant = [c for c in chars[span[0]:span[1]] if not c["char"].isspace()]
            observed = [c for c in relevant if all(k in c for k in ("start", "end", "score"))]
            expected = sum(not c.isspace() for c in word["alignment_text"])

            # word char_coverage 用于说明覆盖率
            word["char_coverage"] = round(len(observed) / expected, 4) if expected else 0.0

            if not observed:
                continue

            # 然后获取 开始时间 和 结束时间
            start, end = min(c["start"] for c in observed), max(c["end"] for c in observed)
            score = float(np.mean([c["score"] for c in observed]))
            if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= duration + 0.002):
                word["flags"].append("invalid_interval")
                continue
            start, end = round(start, 6), round(min(end, duration), 6)
            indices = frame_ids(start, end, timeline["video_pts_s"], timeline["video_frame_end_s"])

            # 根据字符长度是否一致、以及分数是否大于阈值来判定是否需要检查
            word.update(start_s=round(start, 6), end_s=round(end, 6), score=round(score, 4),
                        frame_indices=indices, frame_start=indices[0] if indices else None,
                        frame_end_exclusive=indices[-1]+1 if indices else None, time_source="ctc_characters",
                        status="aligned" if len(observed) == expected and score >= args.min_score else "needs_review")

            ## 这里是对更对情况进行了阐述，也就是针对需要复核的样本，进一步进行细分
            if len(observed) != expected:
                word["flags"].append("partial_char_alignment")
            if score < args.min_score:
                word["flags"].append("low_alignment_score")
            if any(f in word["flags"] for f in ("number_normalized", "acronym_reading_review")):
                word["status"] = "needs_review"
            if match_ratio < 0.5:
                word["flags"].append("transcript_mismatch_candidate")
                word["status"] = "needs_review"
            if end-start > 1.5:
                word["flags"].append("long_word_interval")
                word["status"] = "needs_review"
            if not indices:
                word["flags"].append("no_video_frame")
                word["status"] = "needs_review"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError):
            result["error"] += " " + error.stderr.decode(errors="replace")[-1500:]
    # 最后更新结果
    return result

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条")
    parser.add_argument("--output", type=Path, default=ROOT / "results/word_alignment")
    parser.add_argument("--sample", action="append", help="video_id/clip_id; repeat to select pilot samples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="WAV2VEC2_ASR_BASE_960H")
    parser.add_argument("--min-score", type=float, default=0.5, help="Review flag only; not a calibrated probability")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    cache = ROOT / ".cache"
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("NLTK_DATA", str(cache / "nltk_data"))
    import torch
    import whisperx.alignment as alignment_module
    from whisperx.alignment import load_align_model
    torch.set_num_threads(4)
    samples = read_samples(args.input)          # 这个部分主要是读取数据
    if args.sample:
        samples = [r for r in samples if r["sample_id"] in args.sample]
        if len(samples) != len(set(args.sample)):
            parser.error("Unknown --sample ID")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "samples").mkdir(exist_ok=True)
    model, metadata = load_align_model("en", args.device, model_name=args.model,
                                      model_dir=str(cache / "alignment_models"))
    model.eval()
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "alignment_module_sha256": hashlib.sha256(Path(alignment_module.__file__).read_bytes()).hexdigest(),
              "model_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (cache / "alignment_models").glob("*.pth")},
              "versions": {p: importlib.metadata.version(p) for p in ["whisperx", "torch", "torchaudio", "numpy", "openpyxl", "num2words"]},
              "ffmpeg": run([args.ffmpeg, "-version"]).decode().splitlines()[0],
              "word_definition": "English words with contractions retained; hyphens split; numbers retain source identity",
              "normalization": "lowercase, smart apostrophes, number readings; bracketed speaker labels ending : excluded but retained",
              "interpolation": "none: timestamps derived only from returned aligned characters",
              "alignment_backend": "WhisperX model loader + torchaudio.functional.forced_align + merge_tokens",
              "min_text_match_ratio_review": 0.5,
              "padding": "none", "min_score_is_review_heuristic": True}
    save_json(args.output / "run_config.json", config)
    all_words, summary = [], []
    for i, row in enumerate(samples, 1):
        result = process(row, args.input, model, metadata, args)
        save_json(args.output / "samples" / (row["sample_id"].replace("/", "__") + ".json"), result)
        counts = {status: sum(w["status"] == status for w in result["words"])
                  for status in ("aligned", "needs_review", "unaligned", "annotation_candidate")}
        summary.append({"sample_id": row["sample_id"], "word_count": len(result["words"]), **counts,
                        "audio_duration_s": result.get("audio_duration_s"),
                        "text_match_ratio": result.get("text_match_ratio"),
                        "decoded_video_frames": len(result.get("video_pts_s", [])), "error": result["error"]})
        all_words.extend({"sample_id": row["sample_id"], **w} for w in result["words"])
        print(f"[{i}/{len(samples)}] {row['sample_id']} {counts} {result['error'] or ''}", flush=True)
    write_csv(args.output / "words.csv", all_words)
    write_csv(args.output / "summary.csv", summary)
    print(f"Saved {len(samples)} samples, {len(all_words)} words to {args.output}")
    return int(any(r["error"] for r in summary))


if __name__ == "__main__":
    sys.exit(main())

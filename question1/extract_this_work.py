#!/usr/bin/env python3
"""Q1: extract word-aligned text, audio and visual features from raw attachment 1.

Input is only label-100.xlsx and its 100 MP4 files. No previous alignment output
is read. Each run writes one JSON + NPZ per video, a summary CSV and a manifest.
Video sampling uses up to 9/6/3 distinct frames for scores below 0.5, from
0.5 to below 0.6, and at least 0.6. Audio uses all native steps as before.

Example (from a container with read-only input and writable /outputs):
  python /work/q1_extract_features.py --media-root /data/attachment1 --output-root /outputs/runs --device cuda:0

Dependencies for full extraction: FFmpeg/FFprobe, openpyxl, num2words, numpy,
torch, torchaudio, whisperx, transformers, opensmile and Pillow. --plan-only
needs only openpyxl and reads the raw input inventory without loading models.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any


RATE = 16_000
WORD = re.compile(r"\d+(?:,\d{3})*(?:st|nd|rd|th)?|[A-Za-z]+(?:['’‘][A-Za-z]+)*")
DEFAULT_MEDIA_ROOT = Path(
    "datasets/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"
)
TIME_ORIGIN = "source MP4 presentation timeline after edit lists, seconds"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> bytes:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.decode(errors='replace')[-800:]}")
    return result.stdout


def read_samples(media_root: Path) -> list[dict[str, Any]]:
    """Match every Excel row to exactly one original MP4; never use old outputs."""
    from openpyxl import load_workbook

    book = load_workbook(media_root / "label-100.xlsx", read_only=True, data_only=True)
    try:
        rows = iter(book["label"].values)
        header = next(rows)
        required = {"video_id", "clip_id", "text", "label", "annotation"}
        if not required.issubset(header):
            raise ValueError(f"label worksheet is missing: {sorted(required - set(header))}")
        samples = [dict(zip(header, row)) for row in rows if any(v is not None for v in row)]
    finally:
        book.close()
    seen: set[str] = set()
    for row in samples:
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in (video_id, clip_id)):
            raise ValueError(f"unsafe video/clip ID: {video_id}/{clip_id}")
        row["video_id"], row["clip_id"] = video_id, clip_id
        row["sample_id"] = f"{video_id}/{clip_id}"
        if row["sample_id"] in seen or not isinstance(row["text"], str):
            raise ValueError(f"duplicate sample or missing text: {row['sample_id']}")
        seen.add(row["sample_id"])
    actual = {f"{path.parent.name}/{path.stem}" for path in media_root.glob("*/*.mp4")}
    if seen != actual:
        raise ValueError(f"Excel/MP4 mismatch: missing={sorted(seen-actual)}, extra={sorted(actual-seen)}")
    return samples


def normalize(raw_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Retain source words and their spans while preparing English CTC characters."""
    from num2words import num2words

    labels = [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*:\]", raw_text)]
    words: list[dict[str, Any]] = []
    parts: list[str] = []
    cursor = 0
    for match in WORD.finditer(raw_text):
        raw = match.group()
        spoken = raw.lower().replace("’", "'").replace("‘", "'")
        flags: list[str] = []
        if raw[0].isdigit():
            number = int(re.sub(r"\D", "", raw))
            mode = "ordinal" if re.search(r"(?:st|nd|rd|th)$", raw) else "cardinal"
            if mode == "cardinal" and 1900 <= number <= 1999:
                mode = "year"
            spoken = " ".join(WORD.findall(num2words(number, lang="en", to=mode).lower()))
            spoken = spoken.replace(" and ", " ")
            flags.append("number_normalized")
        if raw in ("US", "Ph"):
            spoken = " ".join(raw.lower())
        if raw in ("US", "Ph", "BIO"):
            flags.append("acronym_reading_review")
        annotation = any(left <= match.start() < right for left, right in labels)
        span = None if annotation else [cursor, cursor + len(spoken)]
        if annotation:
            flags.append("speaker_label_candidate")
        else:
            parts.append(spoken)
            cursor += len(spoken) + 1
        words.append({
            "word_index": len(words), "word": raw, "raw_char_span": list(match.span()),
            "alignment_text": "" if annotation else spoken, "alignment_char_span": span,
            "start_s": None, "end_s": None, "score": None,
            "time_source": None,
            "status": "annotation_candidate" if annotation else "unaligned", "flags": flags,
            "candidate_frame_indices": [], "selected_frame_indices": [],
        })
    return " ".join(parts), words


def media_timeline(video: Path, ffprobe: str) -> dict[str, Any]:
    """Use decoded presentation timestamps, including MP4 edit lists."""
    spec = (
        "stream=index,codec_type,time_base,sample_rate:"
        "frame=media_type,stream_index,best_effort_timestamp_time,duration_time,pkt_duration_time,nb_samples"
    )
    info = json.loads(run([ffprobe, "-v", "error", "-show_frames", "-show_streams",
                           "-show_entries", spec, "-of", "json", str(video)]))
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    frames = [f for f in info["frames"] if f.get("stream_index") == video_stream["index"]]
    audio_frames = [f for f in info["frames"] if f.get("stream_index") == audio_stream["index"]]
    pts = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if not pts or not audio_frames or pts[0] < -1e-6 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("empty or non-monotonic decoded presentation timestamps")
    last_duration = float(frames[-1].get("duration_time") or frames[-1].get("pkt_duration_time") or 0)
    if last_duration <= 0:
        raise ValueError("last video frame has no presentation duration")
    first_audio = float(audio_frames[0]["best_effort_timestamp_time"])
    last_audio = float(audio_frames[-1]["best_effort_timestamp_time"])
    last_audio += int(audio_frames[-1]["nb_samples"]) / int(audio_stream["sample_rate"])
    return {"video_pts_s": pts, "video_frame_end_s": pts[1:] + [pts[-1] + last_duration],
            "video_time_base": video_stream["time_base"], "audio_first_pts_s": first_audio,
            "audio_last_end_s": last_audio}


def decode_audio(video: Path, ffmpeg: str):
    import numpy as np

    raw = run([ffmpeg, "-v", "error", "-threads", "2", "-nostdin", "-copyts", "-i", str(video),
               "-map", "0:a:0", "-vn", "-ac", "1", "-af",
               f"aresample={RATE}:async=1:first_pts=0", "-f", "f32le", "-ar", str(RATE), "-"])
    if len(raw) % 4:
        raise ValueError("incomplete float32 audio sample")
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("empty or non-finite decoded audio")
    return audio


def candidate_frames(start: float, end: float, pts: list[float], ends: list[float]) -> list[int]:
    """Half-open word/frame intersection on the original MP4 presentation axis."""
    return list(range(bisect.bisect_right(ends, start), bisect.bisect_left(pts, end)))


def video_frame_budget(score: float, thresholds: tuple[float, float],
                       budgets: tuple[int, int, int]) -> int:
    """Select the low/middle/high-confidence budget using the unrounded score."""
    return budgets[0 if score < thresholds[0] else 1 if score < thresholds[1] else 2]


def select_word_frames(candidates: list[int], pts: list[float], ends: list[float],
                       start: float, end: float, budget: int = 3) -> list[int]:
    """Keep first/middle/last, then fill the largest gaps in presentation time."""
    if budget < 3:
        raise ValueError("video frame budget must be at least three")
    ids = sorted(set(candidates))
    if len(ids) <= budget:
        return ids
    centers = {k: (pts[k] + ends[k]) / 2 for k in ids}
    selected = {ids[0], ids[-1]}
    selected.add(min(ids[1:-1], key=lambda k: (abs(centers[k] - (start + end) / 2), k)))
    while len(selected) < budget:
        remaining = (k for k in ids if k not in selected)
        selected.add(min(remaining, key=lambda k: (
            -min(abs(centers[k] - centers[j]) for j in selected), k)))
    return sorted(selected)


def align_words(audio: Any, text: str, words: list[dict[str, Any]], timeline: dict[str, Any],
                model: Any, dictionary: dict[str, int], device: str, min_score: float,
                wav2vec_layer: int, video_thresholds: tuple[float, float] = (0.5, 0.6),
                video_budgets: tuple[int, int, int] = (9, 6, 3)) -> tuple[Any, dict[str, Any]]:
    """Use one loaded Wav2Vec2 model for both CTC timing and middle-layer features."""
    import numpy as np
    import torch
    import torchaudio.functional as F

    if not text:
        raise ValueError("no alignable words in transcript")
    tokens = [dictionary[c if c != " " else "|"] for c in text]
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits, _ = model(waveform)
        log_probs = logits.log_softmax(-1).cpu()
        layers, lengths = model.extract_features(waveform, num_layers=wav2vec_layer)
    if len(layers) != wav2vec_layer:
        raise ValueError("Wav2Vec2 did not return the requested middle layer")
    hidden = layers[-1][0].float().cpu().numpy()
    if lengths is not None:
        hidden = hidden[:int(lengths[0])]
    if len(hidden) != log_probs.shape[1]:
        raise ValueError("CTC logits and Wav2Vec2 hidden features have different time grids")
    path, scores = F.forced_align(log_probs, torch.tensor([tokens], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], scores[0].exp(), blank=0)
    if [span.token for span in spans] != tokens:
        raise ValueError("forced CTC path differs from supplied transcript")
    duration = len(audio) / RATE
    step = duration / len(hidden)
    chars = [{"char": char, "start": span.start * step, "end": span.end * step,
              "score": span.score} for char, span in zip(text, spans)]
    pts, ends = timeline["video_pts_s"], timeline["video_frame_end_s"]
    for word in words:
        span = word["alignment_char_span"]
        if span is None:
            continue
        observed = [char for char in chars[span[0]:span[1]] if not char["char"].isspace()]
        if not observed:
            continue
        start = min(char["start"] for char in observed)
        end = max(char["end"] for char in observed)
        score = float(np.mean([char["score"] for char in observed]))
        if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= duration + 0.002):
            word["flags"].append("invalid_interval")
            continue
        start, end = round(start, 6), round(min(end, duration), 6)
        candidates = candidate_frames(start, end, pts, ends)
        budget = video_frame_budget(score, video_thresholds, video_budgets)
        selected = select_word_frames(candidates, pts, ends, start, end, budget)
        word.update(start_s=start, end_s=end, score=round(score, 4), time_source="ctc_characters",
                    candidate_frame_indices=candidates, selected_frame_indices=selected,
                    video_sampling_score=score, video_frame_budget=budget)
        if score < min_score:
            word["flags"].append("low_alignment_score")
        if end - start > 1.5:
            word["flags"].append("long_word_interval")
        if not candidates:
            word["flags"].append("no_video_frame")
        word["status"] = "needs_review" if word["flags"] else "aligned"
    return hidden, {"acoustic_time_step_s": step}


def intervals_of(words: list[dict[str, Any]]) -> list[tuple[float, float] | None]:
    return [(word["start_s"], word["end_s"]) if word["start_s"] is not None else None
            for word in words]


def pool_intervals(values: Any, starts: Any, ends: Any,
                   intervals: list[tuple[float, float] | None]) -> tuple[Any, Any, list[list[int]]]:
    """Pool native audio windows by duration of overlap with each word."""
    import numpy as np

    matrix = np.asarray(values, dtype=np.float32)
    left, right = np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(left) or len(matrix) != len(right):
        raise ValueError("feature/time grid lengths differ")
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(left) & np.isfinite(right) & (right > left)
    pooled = np.zeros((len(intervals), matrix.shape[1]), dtype=np.float32)
    mask = np.zeros(len(intervals), dtype=np.uint8)
    contributors = []
    for i, interval in enumerate(intervals):
        if interval is None:
            contributors.append([])
            continue
        start, end = interval
        overlap = np.minimum(right, end) - np.maximum(left, start)
        ids = np.flatnonzero(valid & (overlap > 0))
        contributors.append(ids.tolist())
        if ids.size:
            pooled[i] = np.average(matrix[ids], axis=0, weights=overlap[ids])
            mask[i] = 1
    return pooled, mask, contributors


def text_features(raw_text: str, words: list[dict[str, Any]], tokenizer: Any,
                  model: Any, device: str) -> tuple[Any, Any]:
    import numpy as np
    import torch

    encoded = tokenizer(raw_text, truncation=True, max_length=model.config.max_position_embeddings,
                        return_overflowing_tokens=True, return_offsets_mapping=True)
    dimension = int(model.config.hidden_size)
    sums = np.zeros((len(words), dimension), dtype=np.float64)
    counts = np.zeros(len(words), dtype=np.int32)
    for token_ids, attention, offsets in zip(encoded["input_ids"], encoded["attention_mask"],
                                              encoded["offset_mapping"]):
        with torch.inference_mode():
            hidden = model(input_ids=torch.tensor([token_ids], device=device),
                           attention_mask=torch.tensor([attention], device=device)).last_hidden_state[0]
        vectors = hidden.float().cpu().numpy()
        for k, (start, end) in enumerate(offsets):
            if start == end:
                continue
            for i, word in enumerate(words):
                left, right = word["raw_char_span"]
                if start < right and end > left:
                    sums[i] += vectors[k]
                    counts[i] += 1
    output = np.zeros_like(sums, dtype=np.float32)
    mask = (counts > 0).astype(np.uint8)
    valid = counts > 0
    output[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    return output, mask


def audio_features(audio: Any, hidden: Any, intervals: list[tuple[float, float] | None],
                   smile: Any) -> tuple[dict[str, Any], dict[str, list[list[int]]], list[list[float]]]:
    import numpy as np

    table = smile.process_signal(audio, RATE)
    egemaps = table.to_numpy(dtype=np.float32)
    if egemaps.ndim != 2 or egemaps.shape[1] != 25:
        raise ValueError(f"eGeMAPSv02 LLD expected 25 values, got {egemaps.shape}")
    starts = np.asarray(table.index.get_level_values(0).total_seconds())
    ends = np.asarray(table.index.get_level_values(1).total_seconds())
    eg_word, eg_mask, eg_steps = pool_intervals(egemaps, starts, ends, intervals)
    step = len(audio) / RATE / len(hidden)
    wav_starts = np.arange(len(hidden), dtype=np.float64) * step
    wav_word, wav_mask, wav_steps = pool_intervals(hidden, wav_starts, wav_starts + step, intervals)
    return {"audio_egemaps": eg_word, "audio_wav2vec": wav_word,
            "audio_egemaps_mask": eg_mask, "audio_wav2vec_mask": wav_mask,
            "audio_mask": eg_mask & wav_mask}, {"egemaps": eg_steps, "wav2vec": wav_steps}, \
           np.column_stack((starts, ends)).round(9).tolist()


def video_shape(video: Path, ffprobe: str) -> tuple[int, int]:
    info = json.loads(run([ffprobe, "-v", "error", "-select_streams", "v:0",
                           "-show_streams", "-of", "json", str(video)]))
    stream = info["streams"][0]
    if any(abs(float(item.get("rotation", 0))) > 1e-6
           for item in stream.get("side_data_list", [])):
        raise ValueError("rotated video needs an explicit frame orientation rule")
    return int(stream["width"]), int(stream["height"])


def selected_video_frames(video: Path, wanted: set[int], frame_count: int,
                          width: int, height: int, ffmpeg: str):
    """Decode in presentation order; keep each requested frame just once."""
    import numpy as np

    command = [ffmpeg, "-v", "error", "-threads", "2", "-nostdin", "-noautorotate", "-i", str(video),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0", "-pix_fmt", "rgb24",
               "-f", "rawvideo", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    count, size = 0, width * height * 3
    try:
        assert process.stdout is not None and process.stderr is not None
        while True:
            raw = bytearray()
            while len(raw) < size:
                chunk = process.stdout.read(size - len(raw))
                if not chunk:
                    break
                raw.extend(chunk)
            if not raw:
                break
            if len(raw) != size:
                raise ValueError(f"partial decoded video frame {count}")
            if count in wanted:
                yield count, np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            count += 1
        error = process.stderr.read().decode(errors="replace")
        if process.wait():
            raise RuntimeError(f"FFmpeg video decode failed: {error[-500:]}")
        if count != frame_count:
            raise ValueError(f"decoded {count} frames, FFprobe reported {frame_count}")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def visual_features(video: Path, words: list[dict[str, Any]], timeline: dict[str, Any],
                    processor: Any, model: Any, device: str, batch_size: int,
                    ffmpeg: str, ffprobe: str) -> tuple[Any, Any, int]:
    import numpy as np
    import torch
    from PIL import Image

    wanted = {k for word in words for k in word["selected_frame_indices"]}
    vectors: dict[int, Any] = {}
    if wanted:
        width, height = video_shape(video, ffprobe)
        ids, images = [], []

        def flush() -> None:
            if not images:
                return
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode():
                encoded = model(pixel_values=pixels).last_hidden_state[:, 0, :].float().cpu().numpy()
            vectors.update(zip(ids, encoded))
            ids.clear()
            images.clear()

        for k, frame in selected_video_frames(video, wanted, len(timeline["video_pts_s"]),
                                               width, height, ffmpeg):
            ids.append(k)
            images.append(Image.fromarray(frame))
            if len(images) >= batch_size:
                flush()
        flush()
        if set(vectors) != wanted:
            raise ValueError(f"missing selected frames: {sorted(wanted-set(vectors))[:8]}")
    dimension = int(model.config.hidden_size)
    output = np.zeros((len(words), dimension), dtype=np.float32)
    mask = np.zeros(len(words), dtype=np.uint8)
    pts, ends = timeline["video_pts_s"], timeline["video_frame_end_s"]
    for i, word in enumerate(words):
        if word["start_s"] is None:
            continue
        weights = [max(0.0, min(word["end_s"], ends[k]) - max(word["start_s"], pts[k]))
                   for k in word["selected_frame_indices"]]
        if sum(weights) > 0:
            output[i] = np.average([vectors[k] for k in word["selected_frame_indices"]],
                                   axis=0, weights=weights)
            mask[i] = 1
    return output, mask, len(wanted)


def model_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("whisperx", "torch", "torchaudio", "transformers", "opensmile", "numpy", "openpyxl", "num2words"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def local_model_hashes(identifier: str) -> dict[str, str] | None:
    root = Path(identifier).expanduser()
    if not root.is_dir():
        return None
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def load_models(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import opensmile
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer
    from whisperx.alignment import load_align_model

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: {args.device}")
    align_model, align_meta = load_align_model(
        "en", args.device, model_name=args.alignment_model, model_dir=str(args.model_cache))
    if not hasattr(align_model, "extract_features"):
        raise TypeError("loaded WhisperX alignment model lacks extract_features; use the pinned compatible version")
    align_model.eval()
    align_weights = {p.relative_to(args.model_cache).as_posix(): sha256_file(p)
                     for p in sorted(args.model_cache.rglob("*.pth")) if p.is_file()}
    if not align_weights:
        raise FileNotFoundError(f"no alignment weights found under {args.model_cache}")

    def hf_model(identifier: str, revision: str | None, image: bool = False):
        options: dict[str, Any] = {"cache_dir": str(args.hf_cache),
                                   "local_files_only": args.local_files_only}
        if revision:
            options["revision"] = revision
        config = AutoConfig.from_pretrained(identifier, **options)
        commit = getattr(config, "_commit_hash", None)
        files = local_model_hashes(identifier)
        if files is None and commit:
            options["revision"] = commit
        elif files is None and not commit:
            raise ValueError(f"could not resolve immutable model revision: {identifier}")
        if image:
            processor = AutoImageProcessor.from_pretrained(identifier, **options)
        else:
            processor = AutoTokenizer.from_pretrained(identifier, use_fast=True, **options)
            if not processor.is_fast:
                raise ValueError("BERT tokenizer must provide raw character offsets")
        model = AutoModel.from_pretrained(identifier, config=config, **options).to(args.device).eval()
        return processor, model, {"id": identifier, "requested_revision": revision,
                                  "resolved_revision": commit, "local_files_sha256": files}

    tokenizer, bert, bert_info = hf_model(args.bert_model, args.bert_revision)
    image_processor, dino, dino_info = hf_model(args.dino_model, args.dino_revision, image=True)
    if int(dino.config.hidden_size) != 384:
        raise ValueError("DINOv2 ViT-S/14 must output 384-dimensional frame vectors")
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    models = {"align": align_model, "dictionary": align_meta["dictionary"], "bert": bert,
              "tokenizer": tokenizer, "smile": smile, "dino": dino, "processor": image_processor}
    info = {"alignment_model": args.alignment_model, "alignment_weights_sha256": align_weights,
            "wav2vec_layer_1_based": args.wav2vec_layer, "bert": bert_info, "dino": dino_info,
            "dino_processor": json.loads(image_processor.to_json_string()),
            "egemaps": "eGeMAPSv02/LowLevelDescriptors (25)", "versions": model_versions()}
    return models, info


def extract_sample(row: dict[str, Any], args: argparse.Namespace, models: dict[str, Any],
                   run_dir: Path) -> dict[str, Any]:
    import numpy as np

    sample_id = row["sample_id"]
    video = args.media_root / row["video_id"] / f"{row['clip_id']}.mp4"
    timeline = media_timeline(video, args.ffprobe)
    audio = decode_audio(video, args.ffmpeg)
    duration = len(audio) / RATE
    if abs(duration - timeline["audio_last_end_s"]) > 0.05:
        raise ValueError("decoded audio duration disagrees with MP4 presentation timestamps")
    alignment_text, words = normalize(row["text"])
    hidden, alignment = align_words(audio, alignment_text, words, timeline, models["align"],
                                    models["dictionary"], args.device, args.min_score,
                                    args.wav2vec_layer, tuple(args.video_sampling_thresholds),
                                    tuple(args.video_sampling_frames))
    text, text_mask = text_features(row["text"], words, models["tokenizer"],
                                    models["bert"], args.device)
    audio_arrays, audio_steps, egemaps_windows = audio_features(
        audio, hidden, intervals_of(words), models["smile"])
    visual, visual_mask, unique_frames = visual_features(
        video, words, timeline, models["processor"], models["dino"], args.device,
        args.dino_batch_size, args.ffmpeg, args.ffprobe)
    arrays = {"word_index": np.arange(len(words), dtype=np.int32), "text": text,
              "text_mask": text_mask, **audio_arrays, "visual": visual,
              "visual_mask": visual_mask}
    for i, word in enumerate(words):
        word["egemaps_step_indices"] = audio_steps["egemaps"][i]
        word["wav2vec_step_indices"] = audio_steps["wav2vec"][i]
        word["selected_frame_pts_s"] = [[timeline["video_pts_s"][k], timeline["video_frame_end_s"][k]]
                                        for k in word["selected_frame_indices"]]
    source_hash = sha256_file(video)
    record = {"schema_version": 1, "sample_id": sample_id, "modalities": ["text", "audio", "visual"],
              "source_video": str(video.relative_to(args.media_root)),
              "source_sha256": source_hash, "raw_text": row["text"], "label": row["label"],
              "annotation": row["annotation"], "alignment_text": alignment_text,
              "time_origin": TIME_ORIGIN, "alignment_granularity": "word",
              "frame_index_convention": "0-based decoded presentation order; half-open intervals",
              "audio_duration_s": duration, "audio_first_pts_s": timeline["audio_first_pts_s"],
              "audio_last_end_s": timeline["audio_last_end_s"],
              "video_first_pts_s": timeline["video_pts_s"][0],
              "video_last_end_s": timeline["video_frame_end_s"][-1],
              "video_time_base": timeline["video_time_base"],
              "egemaps_native_windows_s": egemaps_windows, **alignment,
              "dimensions": {"text": text.shape[1], "egemaps": 25,
                             "wav2vec": audio_arrays["audio_wav2vec"].shape[1], "visual": visual.shape[1]},
              "words": words}
    stem = sample_id.replace("/", "__")
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    numeric_path, record_path = samples_dir / f"{stem}.npz", samples_dir / f"{stem}.json"
    stored = {}
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite feature: {key}")
        cast = value.astype(args.storage_dtype) if value.dtype.kind == "f" else value
        if not np.isfinite(cast).all():
            raise ValueError(f"feature overflow in {args.storage_dtype}: {key}")
        stored[key] = cast
    with numeric_path.open("wb") as stream:
        np.savez_compressed(stream, **stored)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return {"sample_id": sample_id, "modalities": "text,audio,visual",
            "alignment_granularity": "word", "source_sha256": source_hash, "word_count": len(words),
            "timed_words": sum(word["start_s"] is not None for word in words),
            "text_valid": int(text_mask.sum()), "audio_valid": int(audio_arrays["audio_mask"].sum()),
            "visual_valid": int(visual_mask.sum()), "unique_encoded_frames": unique_frames,
            "audio_duration_s": round(duration, 6),
            "audio_effective_duration_s": round(
                timeline["audio_last_end_s"] - timeline["audio_first_pts_s"], 6),
            "video_effective_duration_s": round(timeline["video_frame_end_s"][-1] - timeline["video_pts_s"][0], 6),
            "dimensions": json.dumps(record["dimensions"]),
            "feature_file": numeric_path.relative_to(run_dir).as_posix(),
            "feature_sha256": sha256_file(numeric_path),
            "metadata_file": record_path.relative_to(run_dir).as_posix(), "error": ""}


def load_feature_batch(paths: list[Path]) -> dict[str, Any]:
    """Pad variable word sequences only when loading a batch for Q2/Q3."""
    import numpy as np

    if not paths:
        raise ValueError("empty feature batch")
    with np.load(paths[0], allow_pickle=False) as file:
        keys = file.files
    samples = []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            if file.files != keys:
                raise ValueError("feature fields differ between samples")
            samples.append({key: file[key] for key in keys})
    lengths = np.asarray([len(sample["word_index"]) for sample in samples], dtype=np.int32)
    maximum = int(lengths.max())
    batch: dict[str, Any] = {"lengths": lengths,
                             "padding_mask": (np.arange(maximum)[None, :] < lengths[:, None]).astype(np.uint8)}
    for key in keys:
        shape = samples[0][key].shape[1:]
        output = np.zeros((len(samples), maximum, *shape), dtype=samples[0][key].dtype)
        if key == "word_index":
            output.fill(-1)
        for i, sample in enumerate(samples):
            if sample[key].shape != (lengths[i], *shape):
                raise ValueError(f"inconsistent {key} shape")
            output[i, :lengths[i]] = sample[key]
        batch[key] = output
    return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("/outputs/runs"))
    parser.add_argument("--sample", action="append", help="video_id/clip_id; repeat for a pilot")
    parser.add_argument("--plan-only", action="store_true", help="check raw Excel/MP4 inventory only")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--model-cache", type=Path, default=Path("/outputs/model_cache/alignment"))
    parser.add_argument("--hf-cache", type=Path, default=Path("/outputs/model_cache/huggingface"))
    parser.add_argument("--alignment-model", default="WAV2VEC2_ASR_BASE_960H")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--bert-revision")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--dino-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--wav2vec-layer", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="alignment review heuristic; not a calibrated probability")
    parser.add_argument("--video-sampling-thresholds", type=float, nargs=2, default=(0.5, 0.6),
                        metavar=("LOW", "HIGH"), help="video sampling score thresholds")
    parser.add_argument("--video-sampling-frames", type=int, nargs=3, default=(9, 6, 3),
                        metavar=("LOW", "MID", "HIGH"), help="frame budgets by confidence; 3 3 3 restores fixed sampling")
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args(argv)
    low, high = args.video_sampling_thresholds
    low_frames, mid_frames, high_frames = args.video_sampling_frames
    if not 0 <= low < high <= 1:
        parser.error("video sampling thresholds must satisfy 0 <= LOW < HIGH <= 1")
    if not low_frames >= mid_frames >= high_frames >= 3:
        parser.error("video sampling budgets must satisfy LOW >= MID >= HIGH >= 3")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.media_root = args.media_root.expanduser().resolve(strict=True)
    samples = read_samples(args.media_root)
    if len(samples) != 100:
        raise ValueError(f"attachment 1 should contain 100 samples, found {len(samples)}")
    if args.sample:
        selected = set(args.sample)
        samples = [row for row in samples if row["sample_id"] in selected]
        if len(samples) != len(selected):
            raise ValueError(f"unknown --sample ID: {sorted(selected-{r['sample_id'] for r in samples})}")
    if args.plan_only:
        print(json.dumps({"source": str(args.media_root), "samples": len(samples),
                          "total_video_bytes": sum((args.media_root / r["video_id"] /
                                                    f"{r['clip_id']}.mp4").stat().st_size for r in samples)},
                         ensure_ascii=False, indent=2))
        return 0
    if args.wav2vec_layer < 1 or args.dino_batch_size < 1 or not 0 <= args.min_score <= 1:
        raise ValueError("invalid layer, batch size or score threshold")
    for executable in (args.ffmpeg, args.ffprobe):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"required executable missing: {executable}")
    args.model_cache = args.model_cache.expanduser().resolve()
    args.hf_cache = args.hf_cache.expanduser().resolve()
    for cache in (args.model_cache, args.hf_cache):
        cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_cache)
    models, model_info = load_models(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("q1_features_%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("unsafe run ID")
    run_dir = args.output_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "source_root": str(args.media_root),
                "source_labels_sha256": sha256_file(args.media_root / "label-100.xlsx"),
                "script_sha256": sha256_file(Path(__file__)), "models": model_info,
                "settings": {"device": args.device, "min_score": args.min_score,
                              "video_sampling_strategy": "confidence_tiers_pts_farthest_v1",
                              "video_sampling_thresholds": args.video_sampling_thresholds,
                              "video_sampling_frames": args.video_sampling_frames,
                             "storage_dtype": args.storage_dtype, "dino_batch_size": args.dino_batch_size,
                             "model_cache": str(args.model_cache), "hf_cache": str(args.hf_cache)},
                "schema": "word rows, no on-disk padding; masks 1=valid; load_feature_batch pads and returns true lengths"}
    manifest_path = run_dir / "run_config.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary: list[dict[str, Any]] = []
    try:
        for row in samples:
            try:
                result = extract_sample(row, args, models, run_dir)
                print(f"OK {row['sample_id']}: {result['word_count']} words", flush=True)
            except Exception as exc:
                result = {"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}"}
                print(f"ERROR {row['sample_id']}: {result['error']}", file=sys.stderr, flush=True)
            summary.append(result)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    else:
        manifest["status"] = "complete" if all(not row["error"] for row in summary) else "failed"
    finally:
        if summary:
            fields = ["sample_id", "modalities", "alignment_granularity", "source_sha256",
                      "word_count", "timed_words", "text_valid",
                      "audio_valid", "visual_valid", "unique_encoded_frames", "audio_duration_s",
                      "audio_effective_duration_s", "video_effective_duration_s", "dimensions",
                      "feature_file", "feature_sha256",
                      "metadata_file", "error"]
            with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in fields} for row in summary)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_samples"] = sum(not row["error"] for row in summary)
        manifest["failed_samples"] = sum(bool(row["error"]) for row in summary)
        manifest["payload_bytes"] = sum(p.stat().st_size for p in run_dir.rglob("*")
                                        if p.is_file() and p != manifest_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "status": manifest["status"]}, ensure_ascii=False))
    return int(manifest["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())

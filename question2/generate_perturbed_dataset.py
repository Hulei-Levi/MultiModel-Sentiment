"""Generate Attachment-2 synchronous corruption and adapted BERT text vectors.

这个文件用于生成扰动样本

One script owns generation and verification; no training or new model downloads.
Run from MultiModel-Sentiment:
    .venv-align/bin/python -B generate_perturbed_dataset.py self-test
    .venv-align/bin/python -B generate_perturbed_dataset.py generate --device cuda:0

Outputs: datasets/附件2-同步扰动特征/aligned_50_p000.pkl, p010, p020, p030.
Each pickle contains train/valid/test plus _metadata. Original split fields are
preserved except text_bert, audio, vision and regenerated text. Each split also
contains corruption_mask [N,L], corruption_rate [N], corruption_count [N], and
eligible_count [N]. Existing fusion fit_experiment(data, ...) APIs read the
three named splits and ignore top-level _metadata.

The saved audio/vision arrays remain in their original feature units. Detect
missingness using corruption_mask, not zeros after normalization. Keep the
original attention mask: a corrupted token is not padding. raw_text remains
clean provenance only and MUST NOT be re-tokenized as the corrupted input.

The 10/20/30 percent Bernoulli probabilities approximate the observed pattern;
they are not a claim to reproduce the attachment author's unknown RNG/algorithm.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Reusing the existing encoder must not create additional Python cache files.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
MODALITIES = ("text", "audio", "vision")
EXTRA_FIELDS = ("corruption_mask", "corruption_rate", "corruption_count", "eligible_count")
FORMAT = "aligned_synchronous_corruption_v1"


@dataclass
class Config:
    dataset_path: str = str(ROOT / "datasets/附件2-数据集特征文件/aligned_50.pkl")
    adapter_module: str = str(ROOT / "bert_feature_adapter.py")
    checkpoint_path: str = str(ROOT / "experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt")
    output_dir: str = str(ROOT / "datasets/附件2-同步扰动特征")
    probabilities: tuple = (0.0, 0.1, 0.2, 0.3)
    seed: int = 42
    batch_size: int = 64
    device: str = "cuda:0"
    cpu_threads: int = 4

    def validate(self):
        if not self.probabilities or any(not math.isfinite(p) or not 0 <= p <= 1
                                         or not math.isclose(p * 100, round(p * 100), abs_tol=1e-8)
                                         for p in self.probabilities):
            raise ValueError("Probabilities must be finite multiples of 0.01 within [0,1]")
        if len(set(self.probabilities)) != len(self.probabilities):
            raise ValueError("Duplicate probabilities would create duplicate output names")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        if min(self.batch_size, self.cpu_threads) < 1:
            raise ValueError("batch_size and cpu_threads must be positive")


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def content_mask(text_bert):
    value = np.asarray(text_bert)
    if value.ndim != 3 or value.shape[1:] != (3, 50):
        raise ValueError(f"Expected text_bert [N,3,50], found {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("text_bert must contain finite numeric values")
    if not np.equal(value, np.floor(value)).all() or not np.isin(value[:, 1], (0, 1)).all():
        raise ValueError("Token/type IDs must be integral and the attention mask binary")
    if ((value[:, 0] < 0) | (value[:, 0] >= 30522)).any():
        raise ValueError("Token IDs do not match the fixed bert-base-uncased vocabulary")
    if not np.isin(value[:, 2], (0, 1)).all() or not value[:, 1].any(1).all():
        raise ValueError("Invalid token types or empty attention-mask sample")
    return value[:, 1].astype(bool) & ~np.isin(value[:, 0], (0, 101, 102))


def make_corruption_mask(text_bert, probability, seed, split_name):
    """Independent deterministic streams per split AND severity; no label inputs."""
    eligible = content_mask(text_bert)
    if split_name not in SPLITS or not 0 <= probability <= 1:
        raise ValueError("Invalid split/probability")
    # Each severity is independently drawn; masks are not required to be nested.
    entropy = [int(seed), SPLITS.index(split_name), int(round(probability * 100))]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    return (rng.random(eligible.shape) < probability) & eligible


def perturb_split(source, mask):
    """Copy only changed arrays; originals and labels are never modified."""
    if set(EXTRA_FIELDS) & set(source):
        raise ValueError("Source already contains corruption metadata; refusing double corruption")
    eligible = content_mask(source["text_bert"])
    if mask.dtype != np.bool_ or mask.shape != eligible.shape or (mask & ~eligible).any():
        raise ValueError("Corruption mask must select only valid content slots")
    result = dict(source)
    result["text_bert"] = np.array(source["text_bert"], copy=True)
    result["text_bert"][:, 0, :][mask] = 100
    for modality in ("audio", "vision"):
        result[modality] = np.array(source[modality], copy=True)
        result[modality][mask] = 0
    # Do not allow a caller to accidentally retain clean contextual embeddings.
    result.pop("text", None)
    counts = mask.sum(1).astype(np.int32)
    eligible_counts = eligible.sum(1).astype(np.int32)
    rates = np.zeros(len(mask), dtype=np.float32)
    np.divide(counts, eligible_counts, out=rates, where=eligible_counts > 0)
    result.update(corruption_mask=mask.copy(), corruption_count=counts,
                  eligible_count=eligible_counts, corruption_rate=rates)
    return result


def same_value(left, right):
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
            return False
        if np.issubdtype(left.dtype, np.number):
            return np.array_equal(left, right, equal_nan=True)
        return np.array_equal(left, right)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_value(left[k], right[k]) for k in left)
    return left == right


def verify_split(source, generated, expected_mask):
    if set(generated) != set(source) | set(EXTRA_FIELDS):
        raise AssertionError("Unexpected split schema changes")
    mask = generated["corruption_mask"]
    np.testing.assert_array_equal(mask, expected_mask)
    expected_ids = source["text_bert"][:, 0].copy()
    expected_ids[mask] = 100
    np.testing.assert_array_equal(generated["text_bert"][:, 0], expected_ids)
    np.testing.assert_array_equal(generated["text_bert"][:, 1:], source["text_bert"][:, 1:])
    if generated["text_bert"].dtype != source["text_bert"].dtype:
        raise AssertionError("text_bert dtype changed")
    for modality in ("audio", "vision"):
        if generated[modality].dtype != source[modality].dtype or generated[modality].shape != source[modality].shape:
            raise AssertionError("A/V shape or dtype changed")
        if (generated[modality][mask] != 0).any():
            raise AssertionError("Selected A/V rows were not fully erased")
        np.testing.assert_array_equal(generated[modality][~mask], source[modality][~mask])
    for key in set(source) - {"text_bert", "audio", "vision", "text"}:
        if not same_value(source[key], generated[key]):
            raise AssertionError(f"Unrelated field changed: {key}")
    x = generated["text"]
    if x.shape != source["text"].shape or x.dtype != np.float32 or not np.isfinite(x).all():
        raise AssertionError("Invalid regenerated text shape/dtype/values")
    attention = generated["text_bert"][:, 1].astype(bool)
    if (x[~attention] != 0).any():
        raise AssertionError("Encoded padding must be zero")
    eligible = content_mask(source["text_bert"])
    np.testing.assert_array_equal(generated["corruption_count"], mask.sum(1))
    np.testing.assert_array_equal(generated["eligible_count"], eligible.sum(1))
    np.testing.assert_allclose(generated["corruption_rate"], mask.sum(1) / np.maximum(eligible.sum(1), 1), atol=1e-7)
    return {
        "samples": len(mask), "eligible_slots": int(eligible.sum()), "corrupted_slots": int(mask.sum()),
        "realized_fraction": float(mask.sum() / max(int(eligible.sum()), 1)),
        "samples_without_corruption": int((~mask.any(1)).sum()),
        "samples_with_all_content_corrupted": int(((mask.sum(1) == eligible.sum(1)) & eligible.any(1)).sum()),
        "min_sample_fraction": float(generated["corruption_rate"].min()),
        "max_sample_fraction": float(generated["corruption_rate"].max()),
    }


def encode_split(adapter, split, batch_size, split_name):
    tokens = split["text_bert"]
    output = np.empty((len(tokens), 50, 768), dtype=np.float32)
    since = time.monotonic()
    for start in range(0, len(tokens), batch_size):
        end = min(start + batch_size, len(tokens))
        output[start:end] = adapter.encode_text_bert(
            tokens[start:end], batch_size=batch_size, zero_padding=True, apply_adapter=True)
        if time.monotonic() - since >= 10 or end == len(tokens):
            log(f"encode {split_name}: {end}/{len(tokens)}")
            since = time.monotonic()
    return output


def verify_encoding(adapter, split):
    n = len(split["text"])
    changed = np.flatnonzero(split["corruption_mask"].any(1))
    indices = sorted(set([0, n // 2, n - 1] + ([int(changed[0])] if len(changed) else [])))
    encoded = adapter.encode_text_bert(split["text_bert"][indices], batch_size=len(indices),
                                       zero_padding=True, apply_adapter=True)
    saved = split["text"][indices]
    np.testing.assert_allclose(saved, encoded, rtol=2e-5, atol=3e-5)
    return {"indices": indices, "max_abs_error": float(np.max(np.abs(saved - encoded)))}


def import_adapter(module_path):
    spec = importlib.util.spec_from_file_location("_existing_bert_feature_adapter", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_source(data):
    if set(data) != set(SPLITS):
        raise ValueError("Source must contain exactly train/valid/test; use the original aligned_50.pkl")
    for name in SPLITS:
        split = data[name]
        eligible = content_mask(split["text_bert"])
        n = len(eligible)
        if not n or len(split["id"]) != n:
            raise ValueError("Empty split or incorrect sample-ID count")
        for modality, dim in (("text", 768), ("audio", 74), ("vision", 35)):
            value = np.asarray(split[modality])
            if value.shape != (n, 50, dim) or not np.isfinite(value).all():
                raise ValueError(f"Unexpected shape/nonfinite input: {name}/{modality}")
        for label in ("classification_labels", "regression_labels"):
            y = np.asarray(split[label])
            if y.shape not in ((n,), (n, 1)) or not np.isfinite(y).all():
                raise ValueError(f"Invalid labels in {name}/{label}")
        if set(EXTRA_FIELDS) & set(split):
            raise ValueError("Input has already been augmented")


def atomic_save_verified(path, payload, original, adapter, probability, seed):
    """Publish only after reloading and verifying, without overwriting a target."""
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(str(path))
    with partial.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    # This independent disk read catches serialization / saved-array issues.
    with partial.open("rb") as stream:
        saved = pickle.load(stream)
        if stream.read(1):
            raise AssertionError("Unexpected trailing pickle data")
    if saved.keys() != payload.keys() or saved["_metadata"] != payload["_metadata"]:
        raise AssertionError("Saved dataset metadata mismatch")
    for name in SPLITS:
        mask = make_corruption_mask(original[name]["text_bert"], probability, seed, name)
        verify_split(original[name], saved[name], mask)
        np.testing.assert_array_equal(saved[name]["text"], payload[name]["text"])
        verify_encoding(adapter, saved[name])
    del saved
    # Hard-link creation is atomic and fails if a target already exists.
    # Both names are on the same filesystem; unlink only our own partial file.
    os.link(partial, path)
    partial.unlink()
    return sha256(path)


def generate_datasets(config=None):
    config = config or Config()
    config.validate()
    import torch
    source = Path(config.dataset_path).expanduser().resolve()
    module_path = Path(config.adapter_module).expanduser().resolve()
    checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
    destination = Path(config.output_dir).expanduser().resolve()
    for path in (source, module_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if destination == source.parent:
        raise ValueError("Use a new directory for derived datasets")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = int(source.stat().st_size * len(config.probabilities) * 1.4)
    if shutil.disk_usage(destination.parent).free < required:
        raise OSError(f"Insufficient disk space; allow approximately {required / 1024**3:.2f} GiB")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    log("Read-only hash checks for source data and existing adapter")
    source_hash, module_hash, checkpoint_hash = map(sha256, (source, module_path, checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("dataset_sha256") != source_hash:
        raise ValueError("Adapter was fitted to a different original dataset hash")
    
    module = import_adapter(module_path)
    adapter = module.load_adapter(str(checkpoint_path), device=config.device)

    if adapter.training or any(p.requires_grad for p in adapter.parameters()):
        raise AssertionError("Adapter must be frozen and in eval mode")
    with source.open("rb") as stream:
        data = pickle.load(stream)  # User-provided trusted local attachment.
    validate_source(data)
    destination.mkdir(exist_ok=False)
    script_hash = sha256(Path(__file__).resolve())
    import transformers
    
    log(f"Output directory: {destination}")
    summaries = []
    for probability in config.probabilities:
        log(f"Generating p={probability:.2f}; token corruption precedes BERT encoding")
        result, statistics, encoding_checks = {}, {}, {}
        for name in SPLITS:
            mask = make_corruption_mask(data[name]["text_bert"], probability, config.seed, name)
            split = perturb_split(data[name], mask)
            split["text"] = encode_split(adapter, split, config.batch_size, name)
            statistics[name] = verify_split(data[name], split, mask)
            encoding_checks[name] = verify_encoding(adapter, split)
            result[name] = split
            log(f"{name}: erased {statistics[name]['corrupted_slots']}/{statistics[name]['eligible_slots']} content slots")
        # Ensure provenance is still current BEFORE publishing a finished file.
        for path, expected in ((source, source_hash), (module_path, module_hash), (checkpoint_path, checkpoint_hash)):
            if sha256(path) != expected:
                raise RuntimeError(f"Original input changed during generation: {path}")
        result["_metadata"] = {
            "format": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": {**asdict(config), "probabilities": list(config.probabilities)},
            "nominal_probability": probability, "seed": config.seed,
            "sampler": "NumPy PCG64 Bernoulli; independent streams per split/probability",
            "seed_sequence_entropy": {name: [config.seed, SPLITS.index(name), int(round(probability * 100))] for name in SPLITS},
            "scope": "approximation of aligned Attachment-3 synchronous slot erasure; official sampler unknown",
            "eligible_rule": "attention_mask == 1 and token_id not in [0,101,102]",
            "replacement_token_id": 100, "same_mask_for_text_audio_vision": True,
            "attention_mask_unchanged": True, "token_type_ids_unchanged": True,
            "labels_ids_splits_unchanged": True, "raw_text_usage": "clean provenance only; never re-tokenize as corrupted input",
            "source_path": str(source), "source_sha256": source_hash,
            "adapter_module": str(module_path), "adapter_module_sha256": module_hash,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
            "bert_model_dir": checkpoint["model_dir"], "bert_model_revision": checkpoint["model_revision"],
            "bert_model_sha256": checkpoint["model_sha256"], "selected_adapter_alpha": checkpoint["selected_alpha"],
            "apply_adapter": True, "zero_padding": True, "text_dtype": "float32",
            "text_encoding": "whole corrupted sequence -> frozen BERT -> trained affine adapter",
            "av_storage": "original unnormalized units and original dtype; selected rows zeroed",
            "normalization_note": "use train-only statistics; carry corruption_mask instead of detecting zeros after scaling",
            "script_sha256": script_hash, "split_statistics": statistics,
            "encoding_spot_checks": encoding_checks,
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "torch": str(torch.__version__), "transformers": transformers.__version__},
        }
        target = destination / f"aligned_50_p{int(round(probability * 100)):03d}.pkl"
        output_hash = atomic_save_verified(target, result, data, adapter, probability, config.seed)
        summary = {"path": str(target), "sha256": output_hash, "bytes": target.stat().st_size,
                   "probability": probability, "splits": statistics}
        summaries.append(summary)
        log("Verified saved dataset: " + json.dumps(summary, ensure_ascii=False))
        del result, split
        gc.collect()
    for path, expected in ((source, source_hash), (module_path, module_hash), (checkpoint_path, checkpoint_hash)):
        if sha256(path) != expected:
            raise RuntimeError(f"Original input changed during generation: {path}")
    log("GENERATION_COMPLETE: all files reloaded and verified; source/adapter/checkpoint unchanged")
    return summaries


def self_test():
    """In-memory behavioral tests; no output files or extra Python scripts."""
    import copy
    rng = np.random.default_rng(9)
    n, length = 12, 50
    tokens = np.zeros((n, 3, length), dtype=np.int64)
    tokens[:, 0, :8] = [101, 201, 202, 203, 204, 205, 206, 102]
    tokens[:, 1, :8] = 1
    # An interior invalid slot and a sample containing only special tokens.
    tokens[0, 1, 3] = 0
    tokens[1, 0] = 0; tokens[1, 1] = 0
    tokens[1, 0, :2] = [101, 102]; tokens[1, 1, :2] = 1
    original = {"text_bert": tokens, "text": rng.normal(size=(n, length, 768)).astype(np.float32),
                "audio": rng.normal(size=(n, length, 74)), "vision": rng.normal(size=(n, length, 35)),
                "classification_labels": np.arange(n) % 3, "regression_labels": np.linspace(-3, 3, n),
                "id": [f"sample_{i}" for i in range(n)], "raw_text": np.array(["clean provenance"] * n)}
    original["vision"][2, 2] = 0  # Legitimate pre-existing zero must remain untouched.
    before = copy.deepcopy(original)

    class ContextEncoder:
        def encode_text_bert(self, inputs, batch_size=64, zero_padding=True, apply_adapter=True):
            if not (zero_padding and apply_adapter):
                raise AssertionError("Both agreed encoder options are required")
            ids, mask = inputs[:, 0].astype(np.float32), inputs[:, 1].astype(bool)
            context = (ids * mask).sum(1, keepdims=True)
            x = np.broadcast_to((ids + context)[..., None], (len(ids), 50, 768)).copy()
            x[~mask] = 0
            return x

    encoder = ContextEncoder()
    outputs = {}
    for probability in (0., .1, .2, .3, 1.):
        mask = make_corruption_mask(tokens, probability, 42, "train")
        np.testing.assert_array_equal(mask, make_corruption_mask(tokens, probability, 42, "train"))
        generated = perturb_split(original, mask)
        assert "text" not in generated, "Clean cached embeddings must be invalidated"
        generated["text"] = encode_split(encoder, generated, 5, "self-test")
        verify_split(original, generated, mask)
        verify_encoding(encoder, generated)
        outputs[probability] = generated
    assert same_value(original, before), "Input arrays were mutated"
    assert not outputs[0.]["corruption_mask"].any()
    np.testing.assert_array_equal(outputs[1.]["corruption_mask"], content_mask(tokens))
    assert not np.array_equal(outputs[0.]["text"][2, 0], outputs[1.]["text"][2, 0]), "Context at uncorrupted CLS must also update"
    large = np.repeat(tokens[2:3], 4000, axis=0)
    mask = make_corruption_mask(large, .2, 42, "train")
    assert abs(mask.sum() / content_mask(large).sum() - .2) < .015
    assert not np.array_equal(mask, make_corruption_mask(large, .2, 42, "valid"))
    assert not np.array_equal(mask, make_corruption_mask(large, .2, 43, "train"))
    invalid = tokens.copy(); invalid[0, 1, 0] = 2
    try:
        content_mask(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonbinary attention masks must be rejected")
    log("SELF_TEST_PASS: synchronized masks, special/padding protection, seed streams, context re-encoding, no source mutations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("self-test", "generate"))
    parser.add_argument("--dataset", default=Config.dataset_path)
    parser.add_argument("--adapter-module", default=Config.adapter_module)
    parser.add_argument("--checkpoint", default=Config.checkpoint_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0., .1, .2, .3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        generate_datasets(Config(dataset_path=args.dataset, adapter_module=args.adapter_module,
                                 checkpoint_path=args.checkpoint, output_dir=args.output_dir,
                                 probabilities=tuple(args.probabilities), seed=args.seed,
                                 batch_size=args.batch_size, device=args.device, cpu_threads=args.cpu_threads))


if __name__ == "__main__":
    main()

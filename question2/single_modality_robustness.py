"""Single-modality partial-missing robustness with one freshly trained RECAP.

Only this script is new. Existing model/data sources are imported read-only.
Run from the project root:
 .venv-align/bin/python -B question2/single_modality_robustness.py self-test
 .venv-align/bin/python -B question2/single_modality_robustness.py run --device cuda:0

One uniform RECAP, random initialization, frozen BERT/adapter and ThisWork.
Same masks across the three modality conditions at a given split/severity.
Text restoration covers all valid slots of text-corrupted samples; AV restores
only erased slots. The two unaffected modalities are copied exactly.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, io, json, math, os, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from question2 import train_recap_reconstructor as rec
from question2 import generate_perturbed_dataset as gen
from question2 import evaluate_reconstructed_this_work as ev

MODS = rec.MODS
RATES = (10, 20, 30, 40, 50)
FORMAT = "single_modality_recap_v1"


@dataclass
class Config:
    data_dir: str = str(ROOT / "datasets/附件2-同步扰动特征")
    adapter_module: str = str(ROOT / "question2/bert_feature_adapter.py")
    adapter_checkpoint: str = str(ROOT / "experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt")
    this_work_checkpoint: str = str(ROOT / "results/second_question/this_work/20260925_150303_247629/best.pt")
    output_root: str = str(ROOT / "results/second_question/single_modality_robustness")
    device: str = "cuda:0"
    seed: int = 42
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    max_length: int = 50
    kernels: tuple = (3, 5, 9)
    dropout: float = 0.1
    batch_size: int = 32
    encoder_batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.
    std_floor: float = 1e-4
    cpu_threads: int = 4


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_masks(valid, missing):
    if valid.dtype != torch.bool or valid.ndim != 2 or set(missing) != set(MODS):
        raise ValueError("Boolean valid [B,L] and three modality masks required")
    flags = []
    for m in MODS:
        k = missing[m]
        if k.dtype != torch.bool or k.shape != valid.shape or (k & ~valid).any():
            raise ValueError("Invalid modality-specific missing mask: " + m)
        flags.append(k.any(1))
    if torch.stack(flags).sum(0).gt(1).any():
        raise ValueError("At most one modality may be artificially missing per sample")


def change_regions(valid, missing):
    return {m: valid & missing[m].any(1, keepdim=True) if m == "text" else missing[m]
            for m in MODS}


class SingleModalityRECAP(rec.RECAPReconstructor):
    """Original RECAP modules/parameter shapes with independent mask semantics."""
    description = "RECAP-inspired reconstruction adapted to single-modality missingness"

    def predict_residuals(self, features, valid_mask, corruption_masks):
        validate_masks(valid_mask, corruption_masks)
        length = valid_mask.shape[1]
        h = []
        for mi, m in enumerate(MODS):
            z = self.project[m](features[m]) + self.position[:, :length] + self.modality[mi]
            z = z + self.erasure(corruption_masks[m].long())
            z = z.masked_fill(~valid_mask[..., None], 0)
            h.append(self.temporal[m](z, valid_mask))
        z = self.interaction(torch.cat(h, dim=1),
                             src_key_padding_mask=~valid_mask.repeat(1, 3))
        return {m: self.decode[m](z[:, mi * length:(mi + 1) * length])
                for mi, m in enumerate(MODS)}

    def apply_residuals(self, features, valid_mask, corruption_masks, residuals):
        regions = change_regions(valid_mask, corruption_masks)
        return {m: torch.where(regions[m][..., None], features[m] + residuals[m],
                               features[m]).masked_fill(~valid_mask[..., None], 0)
                for m in MODS}

    def training_objective(self, features, targets, valid, missing, modes, rates):
        prediction = self(features, valid, missing)
        terms = {}
        for mi, m in enumerate(MODS):
            per_slot = (prediction[m] - targets[m]).square().mean(-1)
            for rate in RATES:
                selected = (modes == mi) & (rates == rate)
                if not selected.any():
                    continue
                region = (valid if m == "text" else missing[m]) & selected[:, None]
                # A present but all-intact condition contributes zero.
                terms[f"{m}_{rate}"] = (per_slot * region).sum() / region.sum().clamp_min(1)
        if not terms:
            raise ValueError("No training conditions")
        return sum(terms.values()) / len(terms), terms


class SingleModalityBundle:
    def __init__(self, model, normalizer, device):
        self.model, self.normalizer, self.device = model, normalizer, device

    @torch.no_grad()
    def reconstruct(self, features, valid, missing, batch_size=32):
        valid = np.asarray(valid, dtype=bool)
        missing = {m: np.asarray(missing[m], dtype=bool) for m in MODS}
        validate_masks(torch.from_numpy(valid), {m: torch.from_numpy(k) for m, k in missing.items()})
        for m in MODS:
            x = np.asarray(features[m])
            if x.shape != (*valid.shape, rec.DIMS[m]) or x.dtype.kind != "f" or not np.isfinite(x).all():
                raise ValueError("Invalid input " + m)
            if np.any(x[~valid] != 0):
                raise ValueError("Nonzero padding")
        result = {m: np.asarray(features[m]).copy() for m in MODS}
        self.model.eval()
        for start in range(0, len(valid), batch_size):
            end = start + batch_size
            v = valid[start:end]
            k = {m: missing[m][start:end] for m in MODS}
            if not any(x.any() for x in k.values()):
                continue
            raw = {m: features[m][start:end] for m in MODS}
            x = self.normalizer.normalize(raw, v, self.device)
            vt = torch.as_tensor(v, device=self.device)
            kt = {m: torch.as_tensor(k[m], device=self.device) for m in MODS}
            prediction = self.model(x, vt, kt)
            for m in MODS:
                region = v & k[m].any(1, keepdims=True) if m == "text" else k[m]
                delta = (prediction[m] - x[m]).cpu().numpy() * self.normalizer.stats[m]["std"]
                result[m][start:end][region] = (raw[m] + delta)[region].astype(result[m].dtype)
        return result


def load_bundle(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError("Expected single-modality checkpoint; old synchronous checkpoints are incompatible")
    cfg = Config(**saved["config"])
    model = SingleModalityRECAP(cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    return SingleModalityBundle(model, rec.Normalizer.from_checkpoint(saved["normalizer"]), device), saved


def apply_corruption(clean, modes, mask, adapter, encoder_batch_size):
    """Clean inputs are used solely to construct the corrupted observation."""
    features = {m: clean[m].copy() for m in MODS}
    tokens = clean["text_bert"].copy()
    missing = {m: mask & (modes == mi)[:, None] for mi, m in enumerate(MODS)}
    text_rows = np.flatnonzero(missing["text"].any(1))
    tokens[:, 0][missing["text"]] = 100
    if len(text_rows):
        features["text"][text_rows] = adapter.encode_text_bert(
            tokens[text_rows], batch_size=encoder_batch_size, zero_padding=True, apply_adapter=True)
    for m in ("audio", "vision"):
        features[m][missing[m]] = 0
    return features, tokens, missing


def training_masks(clean, epoch, seed):
    n = len(clean["id"])
    # Each condition count differs by at most one; rotate remainder assignments.
    assignment = (np.arange(n) + epoch) % 15
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2609]))
    assignment = assignment[rng.permutation(n)]
    modes = assignment // len(RATES)
    rates = np.asarray(RATES, dtype=np.int64)[assignment % len(RATES)]
    eligible = rec.eligible_slots(clean["text_bert"])
    mask = np.zeros_like(eligible)
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, i, 3817]))
        mask[i] = (rng.random(eligible.shape[1]) < rates[i] / 100.) & eligible[i]
    return modes, rates, mask


def build_cases(clean, split_name, cfg, adapter, metadata):
    """Fixed paired masks across modalities; cached text remains encoding-identical."""
    cases, mask_archive = [], {}
    for rate in RATES:
        mask = gen.make_corruption_mask(clean["text_bert"], rate / 100., cfg.seed, split_name)
        tokens = clean["text_bert"].copy()
        tokens[:, 0][mask] = 100
        text = None
        path = Path(cfg.data_dir) / f"aligned_50_p{rate:03d}.pkl"
        if path.exists():
            data = rec.load_pickle(path)
            meta = data["_metadata"]
            for key in ("source_sha256", "adapter_module_sha256", "checkpoint_sha256",
                        "apply_adapter", "zero_padding", "bert_model_sha256"):
                if meta[key] != metadata[key]:
                    raise ValueError("Encoding provenance differs: " + key)
            if split_name in data:
                cached = data[split_name]
                np.testing.assert_array_equal(mask, cached["corruption_mask"])
                np.testing.assert_array_equal(tokens, cached["text_bert"])
                np.testing.assert_array_equal(clean["id"], cached["id"])
                text = cached["text"].copy()
                # Intact samples must be exactly the cached clean sequence.
                np.testing.assert_array_equal(text[~mask.any(1)], clean["text"][~mask.any(1)])
            del data
            gc.collect()
        if text is None:
            modes = np.zeros(len(mask), dtype=np.int64)
            features, _, _ = apply_corruption(clean, modes, mask, adapter, cfg.encoder_batch_size)
            text = features["text"]
        check = sorted(set([0, len(mask)//2, len(mask)-1]))
        encoded = adapter.encode_text_bert(tokens[check], batch_size=len(check),
                                           zero_padding=True, apply_adapter=True)
        np.testing.assert_allclose(text[check], encoded, rtol=2e-4, atol=1e-4)
        mask_archive[f"p{rate:03d}"] = mask.copy()
        for mi, m in enumerate(MODS):
            features = {name: clean[name] for name in MODS}
            if m == "text":
                features["text"] = text
            else:
                features[m] = clean[m].copy()
                features[m][mask] = 0
            masks = {name: mask if name == m else np.zeros_like(mask) for name in MODS}
            cases.append({"modality": m, "percent": rate, "features": features,
                          "text_bert": tokens if m == "text" else clean["text_bert"],
                          "missing": masks, "valid": clean["text_bert"][:, 1].astype(bool)})
        log("fixed_condition_ready", split=split_name, rate=rate, missing_slots=int(mask.sum()))
    return cases, mask_archive


def check_contract(raw, restored, valid, missing):
    for m in MODS:
        region = valid & missing[m].any(1, keepdims=True) if m == "text" else missing[m]
        assert restored[m].shape == raw[m].shape and restored[m].dtype == raw[m].dtype
        assert np.isfinite(restored[m]).all() and not np.any(restored[m][~valid])
        np.testing.assert_array_equal(restored[m][~region], raw[m][~region])
    return True


def reconstruction_metrics(bundle, clean, case, restored, csv_path=None):
    """Direct absolute/squared feature errors, with train-only scaling."""
    m = case["modality"]
    valid, missing = case["valid"], case["missing"][m]
    regions = {"primary": valid if m == "text" else missing, "missing": missing,
               "remaining": valid & ~missing, "full": valid}
    result, rows = {}, [{"id": str(x), "valid_slots": int(valid[i].sum()),
                         "missing_slots": int(missing[i].sum())} for i, x in enumerate(clean["id"])]
    for label, values in (("before", case["features"]), ("after", restored)):
        difference = np.asarray(values[m], np.float64) - np.asarray(clean[m], np.float64)
        scaled = difference / bundle.normalizer.stats[m]["std"]
        per_slot = {"mae": np.abs(scaled).mean(-1), "raw_mae": np.abs(difference).mean(-1),
                    "mse": np.square(scaled).mean(-1), "raw_mse": np.square(difference).mean(-1)}
        result[label] = {}
        for region_name, region in regions.items():
            counts = region.sum(1)
            count = int(counts.sum())
            result[label][region_name] = {"slots": count}
            for metric, values_per_slot in per_slot.items():
                sums = (values_per_slot * region).sum(1)
                result[label][region_name][metric] = float(sums.sum() / count) if count else None
                for i, row in enumerate(rows):
                    row[f"{label}_{region_name}_{metric}"] = float(sums[i] / counts[i]) if counts[i] else None
    if csv_path:
        write_csv(csv_path, rows)
    return result



def validate(bundle, clean, cases, batch_size):
    metrics = {}
    for case in cases:
        restored = bundle.reconstruct(case["features"], case["valid"], case["missing"], batch_size)
        check_contract(case["features"], restored, case["valid"], case["missing"])
        m = case["modality"]
        region = case["valid"] if m == "text" else case["missing"][m]
        part = rec.sample_error(restored[m], clean[m], bundle.normalizer.stats[m]["std"], region)
        if part["count"].sum() == 0:
            raise ValueError("Validation condition has no supervised slots")
        metrics[f"{m}_{case['percent']}"] = float(part["normalized_sum"].sum() / part["count"].sum())
    assert len(metrics) == 15
    return float(np.mean(list(metrics.values()))), metrics


def predict_inputs(case, ids, features=None):
    return {**(case["features"] if features is None else features),
            "text_bert": case["text_bert"], "id": ids}


def evaluate_test(bundle, saved, clean, cases, cfg, run):
    # Match historical ThisWork numerical backend, separately from training.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model, metadata = ev.load_experiment_model(cfg.this_work_checkpoint, device=cfg.device)
    ev.seed_everything(metadata["config"]["seed"])
    model.eval().requires_grad_(False)
    start_hash, reconstruct_hash = state_hash(model), state_hash(bundle.model)
    classes = np.asarray(metadata["class_values"])
    mapping = {float(v): i for i, v in enumerate(classes)}
    labels = np.asarray(clean["classification_labels"]).reshape(-1)
    targets = np.asarray([mapping[float(v)] for v in labels], dtype=np.int64)
    scores = np.asarray(clean["regression_labels"], dtype=np.float32).reshape(-1)
    ids = np.asarray(clean["id"], dtype=str)
    clean_inputs = {**{m: clean[m] for m in MODS}, "text_bert": clean["text_bert"], "id": clean["id"]}
    with torch.inference_mode():
        clean_prediction = ev.predict_split(model, clean_inputs, metadata, masks=None, batch_size=cfg.batch_size)
    ev.historical_check(ROOT / "results/second_question/this_work_robustness/run_20260925T090556_463095Z",
                        0, clean_prediction, targets, scores)
    clean_metrics = ev.metrics(clean_prediction, targets, scores, len(classes))
    summaries, details = [], {}
    for case in cases:
        m, rate = case["modality"], case["percent"]
        key = f"{m}_p{rate:03d}"
        raw = case["features"]
        original = ev.array_hashes({**raw, "text_bert": case["text_bert"]})
        with torch.inference_mode():
            before = ev.predict_split(model, predict_inputs(case, ids), metadata,
                                      masks=None, batch_size=cfg.batch_size)
            with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                    benchmark=torch.backends.cudnn.benchmark,
                    deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
                restored = bundle.reconstruct(raw, case["valid"], case["missing"], cfg.batch_size)
            check_contract(raw, restored, case["valid"], case["missing"])
            after = ev.predict_split(model, predict_inputs(case, ids, restored), metadata,
                                     masks=None, batch_size=cfg.batch_size)
        assert original == ev.array_hashes({**raw, "text_bert": case["text_bert"]})
        for pred in (before, after):
            np.testing.assert_array_equal(pred["ids"], ids)
            np.testing.assert_array_equal(pred["indices"], np.arange(len(ids)))
        reconstruction = reconstruction_metrics(bundle, clean, case, restored, run/f"{key}_errors.csv")
        bm, am = [ev.metrics(p, targets, scores, len(classes)) for p in (before, after)]
        mask = case["missing"][m]
        row = {"modality": m, "nominal_percent": rate,
               "actual_percent": float(mask.sum()/rec.eligible_slots(clean["text_bert"]).sum()*100),
               "samples": len(ids), "before_reconstruction_mae": reconstruction["before"]["primary"]["mae"],
               "after_reconstruction_mae": reconstruction["after"]["primary"]["mae"],
               "before_macro_f1": bm["macro_f1"], "after_macro_f1": am["macro_f1"],
               "macro_f1_gain": am["macro_f1"]-bm["macro_f1"],
               "before_mae": bm["mae"], "after_mae": am["mae"],
               "mae_reduction": bm["mae"]-am["mae"],
               "before_f1_drop_from_clean": clean_metrics["macro_f1"]-bm["macro_f1"],
               "after_f1_drop_from_clean": clean_metrics["macro_f1"]-am["macro_f1"],
               "before_mae_increase_from_clean": bm["mae"]-clean_metrics["mae"],
               "after_mae_increase_from_clean": am["mae"]-clean_metrics["mae"]}
        arrays = {f"{label}_{k}": p[k] for label, p in (("before", before), ("after", after))
                  for k in ("classification_logits", "predicted_class_indices", "predicted_labels", "regression_prediction")}
        np.savez_compressed(run/f"{key}_predictions.npz", **arrays, ids=ids,
                            classification_targets=targets, regression_targets=scores,
                            class_values=classes, corruption_count=mask.sum(1))
        pred_rows = [{"id": str(ids[i]), "true_class": float(labels[i]), "true_score": float(scores[i]),
                      "before_class": float(before["predicted_labels"][i]),
                      "after_class": float(after["predicted_labels"][i]),
                      "before_score": float(before["regression_prediction"][i]),
                      "after_score": float(after["regression_prediction"][i]),
                      "missing_slots": int(mask[i].sum())} for i in range(len(ids))]
        write_csv(run/f"{key}_predictions.csv", pred_rows)
        summaries.append(row)
        details[key] = {"summary": row, "reconstruction": reconstruction, "before": bm, "after": am,
                        "unaffected_modalities_exactly_preserved": True, "source_arrays_unchanged": True}
        log("test_result", **row)
    assert state_hash(model) == start_hash
    assert state_hash(bundle.model) == reconstruct_hash
    write_csv(run/"summary.csv", summaries)
    rec.write_json(run/"metrics.json", {"clean": clean_metrics, "conditions": details,
                   "best_epoch": saved["epoch"], "validation_score": saved["validation_score"],
                   "summary": summaries, "this_work_epoch": metadata["selected_epoch"],
                   "models_unchanged_during_evaluation": True})
    return summaries, clean_metrics


def run_experiment(cfg):
    if min(cfg.epochs, cfg.patience, cfg.batch_size, cfg.encoder_batch_size, cfg.cpu_threads) < 1:
        raise ValueError("Positive training sizes required")
    torch.set_num_threads(cfg.cpu_threads)
    rec.seed_all(cfg.seed)
    data_dir = Path(cfg.data_dir)
    paths = [data_dir/f"aligned_50_p{r:03d}.pkl" for r in (0, *RATES)]
    protected = {str(p): rec.sha256(p) for p in (*paths, Path(cfg.adapter_module),
                 Path(cfg.adapter_checkpoint), Path(cfg.this_work_checkpoint))}
    for folder in (ROOT/"question2", ROOT/"multi_fusion_model"):
        for p in folder.rglob("*.py"):
            protected[str(p.resolve())] = rec.sha256(p)
    data = rec.load_pickle(paths[0]);metadata = data["_metadata"]
    for key, path in (("adapter_module_sha256", cfg.adapter_module), ("checkpoint_sha256", cfg.adapter_checkpoint)):
        assert metadata[key] == protected[str(Path(path))]
    audit = rec.audit_splits(data)
    clean = {name: rec.select_fields(data[name]) for name in rec.SPLITS}
    # Labels are reserved separately for final evaluation, never training/selection.
    test_labels = {key: data["test"][key] for key in ("classification_labels", "regression_labels")}
    del data;gc.collect()
    normalizer = rec.Normalizer.fit(clean["train"], cfg.std_floor)
    adapter_module = gen.import_adapter(Path(cfg.adapter_module))
    adapter = adapter_module.load_adapter(cfg.adapter_checkpoint, device=cfg.device)
    adapter.eval().requires_grad_(False)
    adapter_hash = state_hash(adapter)
    fixed_valid, valid_masks = build_cases(clean["valid"], "valid", cfg, adapter, metadata)
    rec.seed_all(cfg.seed)
    model = SingleModalityRECAP(cfg).to(cfg.device)
    assert all(torch.count_nonzero(layer.weight)==0 and torch.count_nonzero(layer.bias)==0
               for layer in model.decode.values())
    initial_hash = state_hash(model)
    assert not set(map(id, model.parameters())) & set(map(id, adapter.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bundle = SingleModalityBundle(model, normalizer, cfg.device)
    run = Path(cfg.output_root)/datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    log("run_started", path=str(run), parameters=sum(p.numel() for p in model.parameters()))
    rec.write_json(run/"config.json", asdict(cfg))
    rec.write_json(run/"audit.json", audit)
    np.savez_compressed(run/"valid_masks.npz", **valid_masks, ids=np.asarray(clean["valid"]["id"],dtype=str))
    np.savez(run/"normalization.npz", **{f"{m}_{k}":normalizer.stats[m][k] for m in MODS for k in ("mean","std")})
    rec.write_json(run/"initial_provenance.json", {"protected_sha256": protected, "initial_model_sha256": initial_hash,
        "frozen_adapter_sha256": adapter_hash, "from_scratch": True,
        "reconstruction_cudnn_tf32": False, "this_work_cudnn_tf32": True})
    history=[];best=float("inf");best_epoch=0;bad=0
    for epoch in range(1, cfg.epochs+1):
        started=time.monotonic()
        adapter.eval()
        modes,rates,mask=training_masks(clean["train"],epoch,cfg.seed)
        features,tokens,missing=apply_corruption(clean["train"],modes,mask,adapter,cfg.encoder_batch_size)
        valid=tokens[:,1].astype(bool)
        order=np.random.default_rng(np.random.SeedSequence([cfg.seed,epoch,701])).permutation(len(modes))
        model.train();loss_sum=0.;samples=0;skipped=0
        for start in range(0,len(order),cfg.batch_size):
            ix=order[start:start+cfg.batch_size]
            # Empty mask batches cannot supervise any correction; skip AdamW too.
            if not mask[ix].any():
                skipped+=1
                continue
            x=normalizer.normalize({m:features[m][ix] for m in MODS},valid[ix],cfg.device)
            targets=normalizer.normalize({m:clean["train"][m][ix] for m in MODS},valid[ix],cfg.device)
            vt=torch.as_tensor(valid[ix],device=cfg.device)
            kt={m:torch.as_tensor(missing[m][ix],device=cfg.device) for m in MODS}
            optimizer.zero_grad(set_to_none=True)
            loss,_=model.training_objective(x,targets,vt,kt,
                    torch.as_tensor(modes[ix],device=cfg.device),torch.as_tensor(rates[ix],device=cfg.device))
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            loss_sum+=float(loss.detach())*len(ix);samples+=len(ix)
        del features,tokens,missing;gc.collect()
        score,conditions=validate(bundle,clean["valid"],fixed_valid,cfg.batch_size)
        record={"epoch":epoch,"train_loss":loss_sum/max(samples,1),"validation_score":score,
                "validation_conditions":conditions,"seconds":time.monotonic()-started,
                "skipped_empty_batches":skipped,
                "condition_counts":{f"{m}_{r}":int(((modes==mi)&(rates==r)).sum()) for mi,m in enumerate(MODS) for r in RATES},
                "mask_sha256":hashlib.sha256(mask.tobytes()).hexdigest(),
                "assignment_sha256":hashlib.sha256(modes.tobytes()+rates.tobytes()).hexdigest(),
                "realized_fraction":float(mask.sum()/rec.eligible_slots(clean["train"]["text_bert"]).sum())}
        history.append(record)
        rec.write_json(run/"history.json",history)
        if score<best-1e-8:
            best,best_epoch,bad=score,epoch,0
            torch.save({"format":FORMAT,"config":asdict(cfg),"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "normalizer":normalizer.checkpoint(),"epoch":epoch,"validation_score":score,
                "initial_state_sha256":initial_hash,"initialization":"random with original zero residual decoder",
                "training_conditions":[f"{m}_{r}" for m in MODS for r in RATES],
                "provenance_hashes":protected},run/"best.pt")
        else:bad+=1
        log("epoch",epoch=epoch,loss=record["train_loss"],validation_mse=score,best_epoch=best_epoch,seconds=record["seconds"])
        if bad>=cfg.patience:break
    assert state_hash(adapter)==adapter_hash and not adapter.training
    assert all(not p.requires_grad for p in adapter.parameters())
    del optimizer,model,bundle,fixed_valid;gc.collect()
    torch.cuda.empty_cache()
    bundle,saved=load_bundle(run/"best.pt",cfg.device)
    test_cases,test_masks=build_cases(clean["test"],"test",cfg,adapter,metadata)
    assert state_hash(adapter)==adapter_hash
    np.savez_compressed(run/"test_masks.npz",**test_masks,ids=np.asarray(clean["test"]["id"],dtype=str))
    del adapter;gc.collect();torch.cuda.empty_cache()
    # Targets and masks are now fixed; the test set did not select any checkpoint.
    clean["test"].update(test_labels)
    summaries,clean_metrics=evaluate_test(bundle,saved,clean["test"],test_cases,cfg,run)
    after={p:rec.sha256(p) for p in protected}
    if after!=protected:raise RuntimeError("Protected file changed")
    protocol={"protected_sha256_before":protected,"protected_sha256_after":after,"unchanged":True,
        "initial_model_sha256":initial_hash,"frozen_adapter_sha256":adapter_hash,
        "from_scratch":True,"bert_adapter_frozen":True,"this_work_frozen":True,
        "best_epoch":best_epoch,"epochs_run":len(history),"validation_score":best,
        "reconstructed_feature_datasets_saved":False,"perturbed_feature_datasets_saved":False,
        "new_python_files":[str(Path(__file__).resolve())],
        "selection":"Equal mean of 15 validation affected-modality standardized MSE values",
        "mask_protocol":"Independent fixed rate streams shared across modalities; fresh balanced training conditions each epoch"}
    rec.write_json(run/"provenance.json",protocol)
    write_report(run,cfg,summaries,clean_metrics,history,protocol)
    rec.write_json(run/"figure_qa.json",make_figures(run,summaries,clean_metrics))
    log("completed",run_dir=str(run),best_epoch=best_epoch)
    return run


def write_report(run,cfg,rows,clean,history,protocol):
    lines=["# 单模态局部缺失鲁棒性","",
       "统一RECAP从随机初始化训练；原网络结构和参数形状保持不变，仅改为每模态独立缺失标记、输出约束和条件平衡损失。BERT/adapter和ThisWork冻结。未经扰动的两路特征精确保留。",
       f"训练3395／验证728／测试727，seed={cfg.seed}，最多{cfg.epochs}轮。实际{len(history)}轮，选中第{protocol['best_epoch']}轮，15条件平均验证MSE={protocol['validation_score']:.8f}。",
       "训练在3种受扰模态×5档缺失率间均衡分配，每轮重新生成Bernoulli掩码。每个split/缺失率的固定验证测试掩码在三种模态间共享，便于配对比较；不同缺失率掩码不要求嵌套。",
       "训练批次先对每个出现的条件计算其区域总槽加权MSE，再在出现的条件间等权平均；验证集先分别计算15个条件的全量MSE，再等权平均。测试集不用于选模。",
       "报告的重建MAE使用训练集标准差归一化，仅评价受扰模态：文本统计全有效位置，音视频统计人工缺失位置。不可与此前三模态平均MSE直接混用，也不应把跨模态MSE差异直接当成模态重要性。",
       "完整模态直接复用原特征；仅文本受扰时替换[UNK]并重新编码完整文本序列。ThisWork继续使用原attention mask，不把人工缺失槽误当padding。",
       f"完整输入参考：Macro-F1={clean['macro_f1']:.8f}，MAE={clean['mae']:.8f}。",
       "","## 测试结果","",
       "| 受扰模态 | 缺失率 | 重建MAE前 | 重建MAE后 | Macro-F1前 | Macro-F1后 | 回归MAE前 | 回归MAE后 |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows,key=lambda x:(MODS.index(x["modality"]),x["nominal_percent"])):
        lines.append(f'| {row["modality"]} | {row["nominal_percent"]}% | {row["before_reconstruction_mae"]:.8f} | {row["after_reconstruction_mae"]:.8f} | {row["before_macro_f1"]:.4f} | {row["after_macro_f1"]:.4f} | {row["before_mae"]:.4f} | {row["after_mae"]:.4f} |')
    lines+=["","## 文件与解释","",
       "summary.csv包含15条件的完整数值、相对完整输入的下降以及重建带来的增益；每条件保存重建误差CSV及配对预测CSV/NPZ。valid_masks/test_masks.npz保存可复核掩码；不保存扰动或重建后的特征数据集。",
       "三张曲线分别回答分类鲁棒性、回归鲁棒性和特征修复效果。它们是一次训练种子和一次固定掩码抽样的描述性结果，不含置信区间或显著性结论。颜色代表受扰模态，虚线为直接输入，实线为重建后。",
       "0%点是共同干净输入的恒等参考；音视频的0%缺失区MAE数学上无观测，此图仅以恒等误差0作为曲线起点，非估计的缺失区平均绝对误差。",
       "text/audio/vision的性能下降衡量本模型对相应输入扰动的敏感程度，不是可加和的因果贡献百分比。",""]
    (run/"report.md").write_text("\n".join(lines),encoding="utf-8")

def self_test():
    """Small CPU checks of the experiment-specific intervention contract."""
    torch.set_num_threads(2)
    rec.seed_all(42)
    cfg = Config(device="cpu",d_model=8,heads=2,layers=1,kernels=(3,),dropout=0.)
    model = SingleModalityRECAP(cfg)
    old = rec.RECAPReconstructor(cfg)
    assert {k:tuple(v.shape) for k,v in model.state_dict().items()} == {
        k:tuple(v.shape) for k,v in old.state_dict().items()}
    b,l=4,6
    valid=torch.ones(b,l,dtype=torch.bool);valid[:,-1]=False
    missing={m:torch.zeros(b,l,dtype=torch.bool) for m in MODS}
    for i,m in enumerate(MODS):missing[m][i,2]=True
    x={m:torch.randn(b,l,rec.DIMS[m]).masked_fill(~valid[...,None],0) for m in MODS}
    target={m:v.clone() for m,v in x.items()}
    model.eval()
    initial=model(x,valid,missing)
    for m in MODS:torch.testing.assert_close(initial[m],x[m],rtol=0,atol=0)
    flags=[]
    hook=model.erasure.register_forward_pre_hook(lambda module,args:flags.append(args[0].clone()))
    model(x,valid,missing);hook.remove()
    for i,m in enumerate(MODS):torch.testing.assert_close(flags[i],missing[m].long())
    for layer in model.decode.values():
        nn.init.normal_(layer.weight,std=.01);nn.init.constant_(layer.bias,.03)
    y=model(x,valid,missing)
    regions=change_regions(valid,missing)
    for m in MODS:
        torch.testing.assert_close(y[m][~regions[m]],x[m][~regions[m]],rtol=0,atol=0)
        assert (y[m][regions[m]]-x[m][regions[m]]).abs().sum()>0
    modes=torch.tensor([0,1,2,0]);rates=torch.tensor([10,20,30,50])
    loss,_=model.training_objective(x,target,valid,missing,modes,rates)
    poisoned={m:target[m].clone() for m in MODS}
    for i,m in enumerate(MODS):poisoned[m][modes!=i]+=100
    lp,_=model.training_objective(x,poisoned,valid,missing,modes,rates)
    torch.testing.assert_close(lp,loss,rtol=0,atol=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    start=float(loss.detach())
    for _ in range(20):
        optimizer.zero_grad()
        loss,_=model.training_objective(x,target,valid,missing,modes,rates)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert float(loss.detach())<start
    stats={m:{"mean":np.full(rec.DIMS[m],.3,np.float32),
              "std":np.full(rec.DIMS[m],1.7,np.float32)} for m in MODS}
    norm=rec.Normalizer(stats)
    bundle=SingleModalityBundle(model,norm,"cpu")
    raw={m:x[m].numpy().astype(np.float64) for m in MODS}
    v=valid.numpy();k={m:missing[m].numpy() for m in MODS}
    restored=bundle.reconstruct(raw,v,k,2)
    check_contract(raw,restored,v,k)
    empty={m:np.zeros_like(v) for m in MODS}
    passthrough=bundle.reconstruct(raw,v,empty)
    for m in MODS:np.testing.assert_array_equal(raw[m],passthrough[m])
    # Roundtrip in memory: no test files or additional scripts.
    buffer=io.BytesIO()
    torch.save({"format":FORMAT,"config":asdict(cfg),"model":model.state_dict(),
                "normalizer":norm.checkpoint()},buffer)
    buffer.seek(0)
    loaded,_=load_bundle(buffer)
    rerun=loaded.reconstruct(raw,v,k,2)
    for m in MODS:np.testing.assert_array_equal(restored[m],rerun[m])
    legacy=io.BytesIO();torch.save({"model":model.state_dict()},legacy);legacy.seek(0)
    try:load_bundle(legacy)
    except ValueError:pass
    else:raise AssertionError("Legacy checkpoint accepted")
    invalid={m:a.clone() for m,a in missing.items()};invalid["audio"][0,1]=True
    try:validate_masks(valid,invalid)
    except ValueError:pass
    else:raise AssertionError("Multiple affected modalities accepted")
    tokens=np.zeros((b,3,l),dtype=np.int64)
    tokens[:,0,:5]=[101,200,201,202,102];tokens[:,1,:5]=1
    clean={**raw,"text_bert":tokens,"id":list(range(b))}
    class FakeEncoder:
        calls=0
        def encode_text_bert(self,tokens,**kwargs):
            self.calls+=1
            return np.repeat(tokens[:,0,:,None],768,axis=2).astype(np.float32)*tokens[:,1,:,None]
    enc=FakeEncoder()
    mask=np.zeros_like(v);mask[:,2]=True
    corrupted,toks,km=apply_corruption(clean,np.array([0,1,2,1]),mask,enc,2)
    assert enc.calls==1 and toks[0,0,2]==100
    for i,m in enumerate(MODS):
        for other in MODS:
            if m!=other:np.testing.assert_array_equal(corrupted[other][i],raw[other][i])
    enc.calls=0
    apply_corruption(clean,np.ones(b,dtype=int),mask,enc,2)
    assert enc.calls==0
    large={"id":list(range(151)),"text_bert":np.repeat(tokens[:1],151,axis=0)}
    a=training_masks(large,1,42);again=training_masks(large,1,42)
    for left,right in zip(a,again):np.testing.assert_array_equal(left,right)
    counts=[int(((a[0]==mi)&(a[1]==r)).sum()) for mi in range(3) for r in RATES]
    assert max(counts)-min(counts)<=1 and not a[2][:,[0,4,5]].any()
    assert not np.array_equal(a[2],training_masks(large,2,42)[2])
    log("self_test_passed", checks=["same_parameter_shapes","independent_erasure_flags",
        "mixed_batch_passthrough","identity_empty_masks","finite_gradient_learning",
        "unaffected_targets_excluded","checkpoint_roundtrip","old_checkpoint_rejected",
        "text_only_encoding","balanced_reproducible_training_masks"])

def make_figures(run_dir, summary_rows, clean_metrics, only_metrics=None):
    """Three single-panel quantitative robustness curves with source/QA metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=list(summary_rows)
    expected={(m,r) for m in MODS for r in RATES}
    indexed={(r["modality"],int(r["nominal_percent"])):r for r in rows}
    assert len(rows)==15 and set(indexed)==expected
    n={int(r["samples"]) for r in rows};assert len(n)==1
    sample_count=n.pop()
    colors=dict(text="#4477AA",audio="#DD8844",vision="#228866")
    markers=dict(text="o",audio="s",vision="^")
    names=dict(text="Text",audio="Audio",vision="Vision")
    out=Path(run_dir)/"figures";out.mkdir(exist_ok=True)
    metadata={"backend":"python","input_rows":15,"excluded_rows":0,
        "paired_test_samples":sample_count,
        "uncertainty":"One training seed and one fixed mask realization; no CI or significance tests.",
        "reconstruction_mae_support":{"text":"all valid slots","audio":"missing slots","vision":"missing slots"},
        "figures":[]}
    style={"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":8,
        "axes.labelsize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
        "legend.fontsize":7.2,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.65,"legend.frameon":False,"pdf.fonttype":42,
        "svg.fonttype":"none","svg.hashsalt":"single-modality-recap",
        "savefig.facecolor":"white","figure.facecolor":"white"}
    specs=[
        ("macro_f1","Classification under partial modality loss","Macro-F1 (higher is better)"),
        ("mae","Regression under partial modality loss","MAE (lower is better)"),
        ("reconstruction_mae","Feature reconstruction under partial modality loss","Standardized feature MAE (lower is better)")]
    x=np.array([0,*RATES])
    with plt.rc_context(style):
        for metric,title,ylabel in specs:
            if only_metrics is not None and metric not in only_metrics:continue
            fig,ax=plt.subplots(figsize=(180/25.4,110/25.4))
            fig.subplots_adjust(left=.125,right=.98,bottom=.37,top=.86)
            texts=[fig.text(.125,.965,title,fontsize=9,weight="bold",va="top"),
                   fig.text(.125,.905,"Only the named modality is perturbed; the other two remain observed.",fontsize=7.5,va="top")]
            baseline=0. if metric=="reconstruction_mae" else float(clean_metrics[metric])
            lines=[];series=[];values=[baseline]
            for m in MODS:
                for state in ("before","after"):
                    y=np.array([baseline]+[float(indexed[(m,r)][f"{state}_{metric}"]) for r in RATES])
                    assert np.isfinite(y).all() and (y>=0).all()
                    if metric=="macro_f1":assert (y<=1).all()
                    repaired=state=="after"
                    label="Repaired" if repaired else ("Perturbed input" if metric=="reconstruction_mae" else "Direct prediction")
                    line,=ax.plot(x,y,color=colors[m],ls="-" if repaired else (0,(4,2.2)),
                        lw=1.5 if repaired else 1.1,marker=markers[m],ms=3.8,
                        mfc=colors[m] if repaired else "white",mec=colors[m],mew=.85,
                        markevery=list(range(1,len(x))),zorder=3 if repaired else 2,
                        label=f"{names[m]}: {label}")
                    lines.append(line);values.extend(y.tolist())
                    series.append({"modality":m,"state":state,"x":x.tolist(),"y":y.tolist()})
            ax.plot([0],[baseline],marker="D",ms=4.3,mfc=".2",mec="white",mew=.5,ls="none",zorder=5,clip_on=False)
            lo,hi=min(values),max(values);pad=max((hi-lo)*.12,.012 if metric=="macro_f1" else .01)
            ax.set_ylim(0 if metric=="reconstruction_mae" else max(0,lo-pad),min(1,hi+pad) if metric=="macro_f1" else hi+pad)
            ax.set(xlim=(-1.5,51.5),xticks=x,xlabel="Nominal missing proportion of the named modality (%)",ylabel=ylabel)
            ax.tick_params(direction="out",length=3,width=.65);ax.grid(False)
            legend=fig.legend(handles=lines,loc="upper center",bbox_to_anchor=(.55,.265),
                ncol=3,fontsize=7.2,columnspacing=1.6,handlelength=2.8,handletextpad=.5,labelspacing=.6,borderaxespad=0)
            if metric=="reconstruction_mae":
                captions=[f"{sample_count} paired test samples; one seed/mask realization. Standardization uses the training set.",
                    "Text MAE: all valid slots; audio/vision MAE: missing slots only. Compare repair within a modality.",
                    "At 0%, MAE = 0 is an identity reference, not a missing-slot estimate. No uncertainty intervals."]
            else:
                captions=[f"{sample_count} paired test samples; one training seed and one fixed random-mask realization.",
                    "No uncertainty intervals or significance tests. At 0%, the diamond is the clean reference; repair is bypassed."]
            texts += [fig.text(.125,.115-i*.026,t,fontsize=7,va="top",color=".2") for i,t in enumerate(captions)]
            fig.canvas.draw();renderer=fig.canvas.get_renderer();page=fig.bbox
            for artist in [*texts,*legend.get_texts(),ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
                if not artist.get_visible() or not artist.get_text():continue
                box=artist.get_window_extent(renderer)
                if box.x0<page.x0-.5 or box.x1>page.x1+.5 or box.y0<page.y0-.5 or box.y1>page.y1+.5:
                    raise RuntimeError(f"Text outside figure: {metric}: {artist.get_text()}")
            paths={}
            for ext in ("pdf","svg","png"):
                path=out/f"robustness_{metric}.{ext}";fig.savefig(path,dpi=300);paths[ext]=str(path)
            bounds=ax.get_position();w,h=fig.get_size_inches()*72
            metadata["figures"].append({"metric":metric,"paths":paths,"series":series,
                "caption":" ".join(captions),"dimensions_mm":[180,110],"png_dpi":300,
                "alignment":{"status":"NOT APPLICABLE","reason":"One plot-area; no comparable panels.",
                    "axes_bbox_pt":[bounds.x0*w,bounds.y0*h,bounds.x1*w,bounds.y1*h]},
                "text_page_bounds_check":"PASS","pdf_text_audit":"PENDING","pdf_collision_audit":"PENDING","visual_audit":"PENDING"})
            plt.close(fig)
    return metadata


def recompute_reconstruction_mae(run, device="cuda:0"):
    """Re-evaluate reconstruction only; preserve weights, masks, downstream predictions."""
    import shutil
    run = Path(run).resolve()
    archive = run / "metric_archive_mse"
    archive.mkdir(exist_ok=True)
    for filename in ("summary.csv", "metrics.json", "report.md", "figure_qa.json",
                     "numerical_audit.json", "provenance.json"):
        path = run / filename
        if path.exists() and not (archive / filename).exists():
            shutil.copy2(path, archive / filename)
    metrics = json.loads((run / "metrics.json").read_text())
    original = json.loads((archive / "metrics.json").read_text())
    cfg = Config(**json.loads((run / "config.json").read_text()))
    cfg.device = device
    torch.set_num_threads(cfg.cpu_threads)
    rec.seed_all(cfg.seed)
    guarded = [run / "best.pt", run / "test_masks.npz", run / "valid_masks.npz",
               Path(cfg.this_work_checkpoint), Path(cfg.adapter_checkpoint)]
    guarded += list(run.glob("*_predictions.*"))
    guarded += [Path(cfg.data_dir) / f"aligned_50_p{r:03d}.pkl" for r in (0, *RATES)]
    before_hashes = {str(p): rec.sha256(p) for p in guarded}
    bundle, saved = load_bundle(run / "best.pt", device)
    data = rec.load_pickle(Path(cfg.data_dir) / "aligned_50_p000.pkl")
    clean = rec.select_fields(data["test"])
    del data
    gc.collect()
    valid = clean["text_bert"][:, 1].astype(bool)
    masks = np.load(run / "test_masks.npz")
    np.testing.assert_array_equal(masks["ids"], np.asarray(clean["id"], dtype=str))
    errors_stage = run / ".mae_evaluation_stage"
    errors_stage.mkdir(exist_ok=True)
    summaries, max_mse_difference = [], 0.
    for rate in RATES:
        mask = masks[f"p{rate:03d}"]
        expected = gen.make_corruption_mask(clean["text_bert"], rate/100., cfg.seed, "test")
        np.testing.assert_array_equal(mask, expected)
        cache = rec.load_pickle(Path(cfg.data_dir) / f"aligned_50_p{rate:03d}.pkl")
        cached = cache["test"]
        np.testing.assert_array_equal(cached["id"], clean["id"])
        np.testing.assert_array_equal(cached["corruption_mask"], mask)
        corrupted_tokens = clean["text_bert"].copy()
        corrupted_tokens[:, 0][mask] = 100
        np.testing.assert_array_equal(cached["text_bert"], corrupted_tokens)
        text = cached["text"].copy()
        del cached, cache
        gc.collect()
        for m in MODS:
            key = f"{m}_p{rate:03d}"
            features = {name: clean[name] for name in MODS}
            if m == "text":
                features[m] = text
            else:
                features[m] = clean[m].copy()
                features[m][mask] = 0
            missing = {name: mask if name == m else np.zeros_like(mask) for name in MODS}
            case = {"modality":m,"percent":rate,"features":features,"valid":valid,"missing":missing}
            restored = bundle.reconstruct(features, valid, missing, cfg.batch_size)
            check_contract(features, restored, valid, missing)
            result = reconstruction_metrics(bundle, clean, case, restored, errors_stage/f"{key}_errors.csv")
            old_result = original["conditions"][key]["reconstruction"]
            for stage in ("before", "after"):
                for region in ("primary", "missing", "remaining", "full"):
                    for metric in ("mse", "raw_mse"):
                        a, b = result[stage][region][metric], old_result[stage][region][metric]
                        if a is None or b is None:
                            assert a is b
                        else:
                            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
                            max_mse_difference = max(max_mse_difference, abs(a-b))
            detail = metrics["conditions"][key]
            old_row = detail["summary"]
            row = {}
            for k, v in old_row.items():
                if k == "before_mse":
                    row["before_reconstruction_mae"] = result["before"]["primary"]["mae"]
                elif k == "after_mse":
                    row["after_reconstruction_mae"] = result["after"]["primary"]["mae"]
                else:
                    row[k] = v
            # Idempotent repeated evaluation of an already migrated run.
            row["before_reconstruction_mae"] = result["before"]["primary"]["mae"]
            row["after_reconstruction_mae"] = result["after"]["primary"]["mae"]
            detail["summary"], detail["reconstruction"] = row, result
            summaries.append(row)
            log("reconstruction_mae", modality=m, percent=rate,
                before=row["before_reconstruction_mae"], after=row["after_reconstruction_mae"])
    after_hashes = {str(p): rec.sha256(p) for p in guarded}
    assert before_hashes == after_hashes, "Weights, predictions, data or masks changed"
    # Keep the original training/selection history honestly labelled MSE.
    revision = {
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"reconstruction evaluation/report column only; no retraining or downstream inference",
        "reported_reconstruction_metric":"standardized mean absolute feature error",
        "formula":"sum over selected slots and feature dimensions of abs(pred-clean)/train_std, divided by slots*dimensions",
        "regions":{"text":"all valid slots","audio":"artificially missing slots","vision":"artificially missing slots"},
        "raw_mae_also_saved":True, "selection_metric_unchanged":"mean of 15 validation MSE values",
        "best_epoch":saved["epoch"], "old_mse_max_abs_difference":max_mse_difference,
        "protected_hashes_before":before_hashes,"protected_hashes_after":after_hashes,
        "protected_files_unchanged":True,"source_sha256":rec.sha256(Path(__file__)),
        "previous_results":str(archive),
        "source_sha256_before":rec.sha256(archive/"source_before_mae.txt"),
        "authorized_source_edit":str(Path(__file__).resolve())}
    metrics["summary"] = summaries
    metrics["reconstruction_metric_revision"] = revision
    for path in errors_stage.glob("*_errors.csv"):
        destination=run/path.name
        if not (archive/path.name).exists():
            shutil.copy2(destination,archive/path.name)
        path.replace(destination)
    errors_stage.rmdir()
    rec.write_json(run/"metrics.json",metrics)
    write_csv(run/"summary.csv",summaries)
    protocol=json.loads((run/"provenance.json").read_text())
    old_protected=protocol["protected_sha256_after"]
    mismatches=[p for p,h in old_protected.items() if rec.sha256(Path(p))!=h]
    assert mismatches==[str(Path(__file__).resolve())], mismatches
    revision["original_protected_files_changed"]=mismatches
    revision["other_original_protected_files_unchanged"]=True
    protocol["metric_revision"]=revision
    rec.write_json(run/"provenance.json",protocol)
    write_report(run,cfg,summaries,metrics["clean"],json.loads((run/"history.json").read_text()),protocol)
    report=run/"report.md"
    report.write_text(report.read_text()+"\n## MAE口径修订\n\n本次从同一检查点重新重建，直接计算逐元素绝对误差；不是把MSE开平方，也不是只改列名。模型仍是按原验证MSE选出的第40轮，没有重新训练或按测试MAE选模。原预测文件、权重、掩码和数据的SHA256均未变化。原MSE版本存于metric_archive_mse。\n",encoding="utf-8")
    qa=json.loads((run/"figure_qa.json").read_text())
    qa["figures"]=[x for x in qa["figures"] if x["metric"]!="mse" and x["metric"]!="reconstruction_mae"]
    new_qa=make_figures(run,summaries,metrics["clean"],only_metrics={"reconstruction_mae"})
    qa["figures"].extend(new_qa["figures"])
    qa.pop("mse_support",None)
    qa["reconstruction_mae_support"]=new_qa["reconstruction_mae_support"]
    rec.write_json(run/"figure_qa.json",qa)
    old_figures=archive/"figures"
    old_figures.mkdir(exist_ok=True)
    for ext in ("pdf","svg","png"):
        old=run/"figures"/f"robustness_mse.{ext}"
        if old.exists():
            destination=old_figures/old.name
            if destination.exists():raise FileExistsError(destination)
            old.rename(destination)
    rec.write_json(run/"reconstruction_mae_revision.json",revision)
    log("reconstruction_mae_completed",run_dir=str(run),old_mse_max_abs_difference=max_mse_difference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("self-test","run","plot","reconstruction-mae"))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--run-dir",type=Path)
    args=parser.parse_args()
    if args.command=="self-test":self_test()
    elif args.command=="run":run_experiment(Config(device=args.device))
    elif args.command=="reconstruction-mae":
        if not args.run_dir:parser.error("--run-dir is required for reconstruction-mae")
        recompute_reconstruction_mae(args.run_dir,args.device)
    else:
        if not args.run_dir:parser.error("--run-dir is required for plot")
        metrics=json.loads((args.run_dir/"metrics.json").read_text())
        rec.write_json(args.run_dir/"figure_qa.json",make_figures(args.run_dir,metrics["summary"],metrics["clean"]))

if __name__=="__main__":
    main()

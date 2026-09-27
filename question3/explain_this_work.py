"""Frozen ThisWork hierarchical explanations for Attachment-4 samples 01--20.

All attribution baselines and interventions live in the checkpoint-normalized
feature space. Modality masks are fixed; absent coalitions mean reference-feature
replacement, not removal of slots. Word edits instead re-encode the full sentence.
This module never trains or modifies model inputs. The completion workflow also
evaluates the labelled validation split and creates post-hoc lexical time anchors
using the existing Q1 CTC alignment method; original feature timestamps remain
unavailable and are never fabricated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from multi_fusion_model.this_work import load_experiment_model, predict_split
from multi_fusion_model.weighted_sum_fusion import FeatureDataset, resolve_masks

MODALITIES = ("text", "audio", "vision")
HEADS = ("classification", "regression")
DIMENSIONS = {"text": 768, "audio": 74, "vision": 35}
DEFAULT_CHECKPOINT = ROOT / "results/second_question/this_work/20260925_150303_247629/best.pt"
DEFAULT_ADAPTER = ROOT / "experiments/bert_feature_adapter/run_20260925T075942_514937Z/adapter.pt"


def log(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def feature_input(split, metadata, device):
    masks = resolve_masks(split)
    dataset = FeatureDataset(split, masks, metadata["normalizers"],
                             class_values=metadata["class_values"], require_labels=False)
    item = dataset[0]
    return ({m: item["features"][m].unsqueeze(0).to(device) for m in MODALITIES},
            {m: item["masks"][m].unsqueeze(0).to(device) for m in MODALITIES})


def batch_masks(masks, size):
    return {m: masks[m].expand(size, -1) for m in MODALITIES}


def scalar_outputs(output, winner, runner):
    return torch.stack((output["classification_logits"][:, winner] -
                        output["classification_logits"][:, runner],
                        output["regression_prediction"].reshape(-1)), dim=1)


def evaluate_features(model, features, masks, winner, runner, batch_size=64):
    size = features["text"].shape[0]
    scores, logits, regression = [], [], []
    with torch.no_grad():
        for start in range(0, size, batch_size):
            batch = {m: value[start:start + batch_size] for m, value in features.items()}
            output = model(batch, batch_masks(masks, batch["text"].shape[0]))
            scores.append(scalar_outputs(output, winner, runner).cpu().numpy())
            logits.append(output["classification_logits"].cpu().numpy())
            regression.append(output["regression_prediction"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(logits), np.concatenate(regression)


def coalition_features(x, baseline, coalition):
    return {m: x[m] if coalition & (1 << index) else baseline[m]
            for index, m in enumerate(MODALITIES)}


def coalition_scores(model, x, baseline, masks, winner, runner):
    features = {m: torch.cat([coalition_features(x, baseline, c)[m] for c in range(8)])
                for m in MODALITIES}
    return evaluate_features(model, features, masks, winner, runner)[0].astype(np.float64)


def shapley_from_coalitions(scores):
    phi = np.zeros((2, 3), dtype=np.float64)
    for index in range(3):
        for subset in range(8):
            if subset & (1 << index):
                continue
            n = subset.bit_count()
            weight = math.factorial(n) * math.factorial(2 - n) / math.factorial(3)
            phi[:, index] += weight * (scores[subset | (1 << index)] - scores[subset])
    np.testing.assert_allclose(phi.sum(1), scores[7] - scores[0], atol=1e-12, rtol=1e-12)
    return phi



def softmax_probabilities(logits):
    """Stable NumPy softmax; retain float64 for probability-distance algebra."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=-1, keepdims=True)


def js_divergence_bits(probabilities, reference):
    """Jensen-Shannon divergence in bits (0 = identical; maximum 1)."""
    p, q = np.broadcast_arrays(np.asarray(probabilities, np.float64),
                               np.asarray(reference, np.float64))
    if (np.any(p < 0) or np.any(q < 0) or
        not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))):
        raise ValueError("Invalid probabilities for JS divergence")
    np.testing.assert_allclose(p.sum(axis=-1), 1., atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(q.sum(axis=-1), 1., atol=1e-10, rtol=1e-10)
    middle = (p + q) * 0.5
    def divergence(values):
        term = np.zeros_like(values)
        nonzero = values > 0
        term[nonzero] = values[nonzero] * np.log2(values[nonzero] / middle[nonzero])
        return term.sum(axis=-1)
    value = 0.5 * (divergence(p) + divergence(q))
    if np.any(value < -1e-12) or np.any(value > 1 + 1e-12):
        raise AssertionError("JS divergence outside [0,1]")
    return np.clip(value, 0., 1.)


def coalition_predictions(model, x, baseline, masks, winner, runner):
    features = {m: torch.cat([coalition_features(x, baseline, c)[m] for c in range(8)])
                for m in MODALITIES}
    scores, logits, regression = evaluate_features(model, features, masks, winner, runner)
    return (scores.astype(np.float64), logits.astype(np.float64),
            np.asarray(regression, np.float64).reshape(-1))


def coalition_behavior(scores, logits, regression, winner):
    probabilities = softmax_probabilities(logits)
    classification_js = js_divergence_bits(probabilities, probabilities[7])
    regression_deviation = np.abs(regression - regression[7])
    utilities = -np.stack((classification_js, regression_deviation), axis=1)
    preservation_phi = shapley_from_coalitions(utilities)
    np.testing.assert_allclose(preservation_phi.sum(1),
                               [classification_js[0], regression_deviation[0]],
                               atol=1e-12, rtol=1e-12)
    classes = np.argmax(logits, axis=1)
    # All coalition utilities share their own full-input row as reference,
    # avoiding tiny batch-size rounding discrepancies from direct inference.
    removal = {}
    for index, modality in enumerate(MODALITIES):
        c = 7 ^ (1 << index)
        removal[modality] = dict(
            coalition=c, class_index=int(classes[c]),
            class_flip=bool(classes[c] != winner),
            fixed_margin_signed_change=float(scores[7, 0] - scores[c, 0]),
            fixed_margin_absolute_change=float(abs(scores[7, 0] - scores[c, 0])),
            classification_js_bits=float(classification_js[c]),
            regression_signed_change=float(regression[7] - regression[c]),
            regression_absolute_change=float(regression_deviation[c]))
    return dict(
        coalition_predictions=dict(
            logits=logits.tolist(), probabilities=probabilities.tolist(),
            class_indices=classes.tolist(), regression=regression.tolist(),
            classification_js_bits=classification_js.tolist(),
            regression_absolute_deviation=regression_deviation.tolist(),
            utilities=utilities.tolist()),
        prediction_preservation_phi=preservation_phi.tolist(),
        prediction_preservation_completeness_error=(
            preservation_phi.sum(1) - (utilities[7]-utilities[0])).tolist(),
        whole_modality_removal=removal)


def finite_rank_correlation(x, y):
    def tied_ranks(values):
        values = np.asarray(values)
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), np.float64)
        first = 0
        while first < len(values):
            last = first + 1
            while last < len(values) and values[order[last]] == values[order[first]]:
                last += 1
            result[order[first:last]] = (first + last - 1) / 2
            first = last
        return result
    rx, ry = tied_ranks(x), tied_ranks(y)
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def tied_dominant_indices(values):
    values = np.asarray(values, np.float64)
    maximum = values.max()
    return np.flatnonzero(np.isclose(values, maximum, atol=1e-12, rtol=1e-7)).tolist()


def sample_reference_stability(record):
    content = np.asarray(record["content_mask"], dtype=bool)
    eligible = np.flatnonzero(content)
    mean, zero = [record["baselines"][b] for b in ("mean", "zero")]
    result = {}
    for head, name in enumerate(HEADS):
        rank_correlations, jaccards, top_slots = [], [], {}
        for m in MODALITIES:
            first = np.asarray(mean["rankings"][m]["time_mass"])[head, content]
            second = np.asarray(zero["rankings"][m]["time_mass"])[head, content]
            rank_correlations.append(finite_rank_correlation(first, second))
            selections = []
            for values in (first, second):
                selections.append(None if values.sum() <= 1e-12 else
                                  eligible[np.argsort(-values, kind="stable")[:5]].tolist())
            top_slots[m] = dict(mean=selections[0], zero=selections[1])
            if any(slots is None for slots in selections):
                jaccards.append(None)
            else:
                a, b = map(set, selections)
                jaccards.append(len(a & b)/len(a | b) if a | b else None)
        share_defined = mean["shares_defined"][head] and zero["shares_defined"][head]
        share_delta = (np.asarray(zero["shares"][head])-np.asarray(mean["shares"][head])
                       if share_defined else None)
        dominant_mean = (tied_dominant_indices(mean["shares"][head])
                         if mean["shares_defined"][head] else [])
        dominant_zero = (tied_dominant_indices(zero["shares"][head])
                         if zero["shares_defined"][head] else [])
        result[name] = dict(
            content_time_spearman=rank_correlations, top5_jaccard=jaccards,
            top5_slots=top_slots,
            shares_comparable=bool(share_defined),
            share_change_zero_minus_mean_percentage_points=(
                None if share_delta is None else (100 * share_delta).tolist()),
            dominant_mean=dominant_mean, dominant_zero=dominant_zero,
            dominant_set_agreement=(set(dominant_mean) == set(dominant_zero)
                                    if share_defined else None),
            ranking_note="Content slots only; constant rank arrays and zero-mass top-k are undefined.")
    return result


def summarize_all_samples(samples):
    """Descriptive summaries of these 20 cases; no ground-truth performance claims."""
    result = dict(
        sample_count=len(samples), sample_ids=[sample["id"] for sample in samples],
        scope="All 20 Attachment-4 aligned cases; descriptive finite-set summaries, not population estimates",
        labels_available=False, no_ground_truth_metrics=True,
        share_threshold=1e-6,
        uncertainty="Individual values and descriptive ranges are retained; no population confidence intervals",
        prediction_preservation="Shapley of negative JS divergence in bits and negative absolute regression deviation relative to each sample's full-input prediction; neither is prediction accuracy",
        baseline_order=["mean", "zero"], head_order=list(HEADS),
        modality_order=list(MODALITIES), baselines={})
    n = len(samples)
    for baseline_name in ("mean", "zero"):
        baselines = [s["baselines"][baseline_name] for s in samples]
        phi = np.asarray([b["phi"] for b in baselines], np.float64)
        preservation = np.asarray([b["prediction_preservation_phi"] for b in baselines])
        utilities = np.asarray([b["coalition_predictions"]["utilities"] for b in baselines])
        global_phi = shapley_from_coalitions(utilities.mean(axis=0))
        np.testing.assert_allclose(global_phi, preservation.mean(axis=0), atol=1e-12, rtol=1e-12)
        head_records = {}
        for head, head_name in enumerate(HEADS):
            defined = np.asarray([b["shares_defined"][head] for b in baselines], bool)
            shares = np.asarray([b["shares"][head] for b in baselines], np.float64)
            valid_shares = shares[defined]
            dominance = np.zeros(3)
            for row in valid_shares:
                indices = tied_dominant_indices(row)
                dominance[indices] += 1. / len(indices)
            pooled_denominator = np.abs(phi[:, head]).sum()
            head_records[head_name] = dict(
                sample_phi=phi[:, head].tolist(),
                sample_shares=[b["shares"][head] for b in baselines],
                share_defined=defined.tolist(),
                share_defined_count=int(defined.sum()),
                share_undefined_sample_ids=[samples[i]["id"] for i in np.flatnonzero(~defined)],
                mean_share=(valid_shares.mean(0).tolist() if defined.any() else [None]*3),
                median_share=(np.median(valid_shares, axis=0).tolist() if defined.any() else [None]*3),
                share_min=(valid_shares.min(0).tolist() if defined.any() else [None]*3),
                share_max=(valid_shares.max(0).tolist() if defined.any() else [None]*3),
                mean_absolute_phi=np.abs(phi[:, head]).mean(0).tolist(),
                pooled_absolute_phi_share=(
                    (np.abs(phi[:, head]).sum(0)/pooled_denominator).tolist()
                    if pooled_denominator > 1e-6 else [None]*3),
                mean_signed_phi=phi[:, head].mean(0).tolist(),
                positive_fraction=(phi[:, head] > 1e-6).mean(0).tolist(),
                negative_fraction=(phi[:, head] < -1e-6).mean(0).tolist(),
                near_zero_fraction=(np.abs(phi[:, head]) <= 1e-6).mean(0).tolist(),
                dominant_fraction=(dominance/defined.sum()).tolist() if defined.any() else [None]*3,
                dominant_fraction_note="Only defined shares; each tied winner receives 1 / number of tied modalities",
                sample_prediction_preservation_phi=preservation[:, head].tolist(),
                mean_prediction_preservation_phi=preservation[:, head].mean(0).tolist(),
                prediction_preservation_phi_min=preservation[:, head].min(0).tolist(),
                prediction_preservation_phi_max=preservation[:, head].max(0).tolist(),
                global_prediction_preservation_phi=global_phi[head].tolist(),
                preservation_aggregate_identity_error=float(
                    np.max(np.abs(global_phi[head]-preservation[:, head].mean(0)))),
                interpretation=("Sample-specific winner-versus-runner margin targets are aggregated"
                                if head == 0 else "Raw regression-score targets are aggregated"))
        combination_records = []
        for coalition in range(8):
            cp = [b["coalition_predictions"] for b in baselines]
            classes = [p["class_indices"][coalition] for p in cp]
            fullclasses = [p["class_indices"][7] for p in cp]
            js = np.asarray([p["classification_js_bits"][coalition] for p in cp])
            reg = np.asarray([p["regression"][coalition] for p in cp])
            fullreg = np.asarray([p["regression"][7] for p in cp])
            margin = np.asarray([b["coalitions"][coalition][0] for b in baselines])
            fullmargin = np.asarray([b["coalitions"][7][0] for b in baselines])
            combination_records.append(dict(
                coalition=coalition,
                modalities=[m for i,m in enumerate(MODALITIES) if coalition & (1 << i)],
                class_indices=classes,
                classification_agreement_fraction=float(np.mean(np.asarray(classes)==fullclasses)),
                classification_flip_count=int(np.sum(np.asarray(classes)!=fullclasses)),
                classification_js_bits=js.tolist(),
                classification_js_bits_mean=float(js.mean()),
                classification_js_bits_min=float(js.min()),
                classification_js_bits_max=float(js.max()),
                regression_predictions=reg.tolist(),
                regression_absolute_deviations=np.abs(reg-fullreg).tolist(),
                regression_absolute_deviation_mean=float(np.abs(reg-fullreg).mean()),
                regression_absolute_deviation_min=float(np.abs(reg-fullreg).min()),
                regression_absolute_deviation_max=float(np.abs(reg-fullreg).max()),
                fixed_margin_signed_changes=(fullmargin-margin).tolist(),
                fixed_margin_absolute_change_mean=float(np.abs(fullmargin-margin).mean()),
                regression_signed_change_mean=float((fullreg-reg).mean())))
        removal_summary = {}
        for m in MODALITIES:
            rows = [b["whole_modality_removal"][m] for b in baselines]
            removal_summary[m] = dict(
                classification_flip_fraction=float(np.mean([r["class_flip"] for r in rows])),
                classification_flip_sample_ids=[samples[i]["id"] for i,r in enumerate(rows) if r["class_flip"]],
                classification_js_bits_mean=float(np.mean([r["classification_js_bits"] for r in rows])),
                fixed_margin_absolute_change_mean=float(np.mean([r["fixed_margin_absolute_change"] for r in rows])),
                regression_absolute_change_mean=float(np.mean([r["regression_absolute_change"] for r in rows])))
        result["baselines"][baseline_name] = dict(
            heads=head_records, combinations=combination_records,
            whole_modality_removal=removal_summary)
    return result


def integrate_edge(model, x, baseline, masks, winner, runner, subset, modality,
                   steps, grad_batch):
    index = MODALITIES.index(modality)
    start_features = coalition_features(x, baseline, subset)
    difference = x[modality] - baseline[modality]
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    gradient_sum = torch.zeros((2, *difference.shape[1:]), dtype=torch.float64,
                               device=difference.device)
    with torch.enable_grad():
        for start in range(0, steps, grad_batch):
            n = min(grad_batch, steps - start)
            alpha = torch.as_tensor(nodes[start:start+n], dtype=difference.dtype,
                                    device=difference.device).reshape(n, 1, 1)
            quadrature_weights = torch.as_tensor(weights[start:start+n], dtype=torch.float64,
                                                 device=difference.device).reshape(n, 1, 1)
            changing = (baseline[modality] + alpha * difference).detach().requires_grad_(True)
            features = {m: start_features[m].expand(n, -1, -1) for m in MODALITIES}
            features[modality] = changing
            outputs = scalar_outputs(model(features, batch_masks(masks, n)), winner, runner)
            for head in range(2):
                gradient = torch.autograd.grad(outputs[:, head].sum(), changing,
                                               retain_graph=(head == 0))[0]
                gradient_sum[head] += (gradient.to(torch.float64) * quadrature_weights).sum(0)
    attribution = gradient_sum * difference[0].to(torch.float64).unsqueeze(0)
    attribution[:, ~masks[modality][0], :] = 0
    return attribution.cpu().numpy()


def conditional_ig(model, x, baseline, masks, winner, runner, scores, args):
    result = {m: np.zeros((2, *x[m].shape[1:]), np.float64) for m in MODALITIES}
    diagnostics = []
    for index, modality in enumerate(MODALITIES):
        for subset in range(8):
            if subset & (1 << index):
                continue
            marginal = scores[subset | (1 << index)] - scores[subset]
            tolerance = 1e-3 + 0.01 * np.abs(marginal)
            steps = args.ig_steps
            previous = None
            while True:
                attribution = integrate_edge(model, x, baseline, masks, winner, runner,
                                             subset, modality, steps, args.grad_batch)
                integral = attribution.sum((1, 2))
                residual = integral - marginal
                error = np.abs(residual)
                refinement_l1 = (None if previous is None else
                                 np.abs(attribution - previous).sum((1, 2)))
                refinement_tolerance = 0.01 + 0.02 * np.abs(attribution).sum((1, 2))
                # Completeness alone can conceal cancelling errors; require a
                # second quadrature order and convergence of the full array too.
                passed = bool(np.all(error <= tolerance) and previous is not None and
                              np.all(refinement_l1 <= refinement_tolerance))
                if passed or steps >= args.max_steps:
                    break
                previous = attribution
                steps = min(steps * 2, args.max_steps)
            weight = math.factorial(subset.bit_count()) * math.factorial(2 - subset.bit_count()) / 6
            result[modality] += weight * attribution
            record = dict(modality=modality, subset=subset, subset_modalities=[
                m for i, m in enumerate(MODALITIES) if subset & (1 << i)],
                weight=weight, steps=steps, marginal=marginal.tolist(),
                integrated=integral.tolist(), residual=residual.tolist(),
                absolute_error=error.tolist(), tolerance=tolerance.tolist(),
                refinement_l1=None if refinement_l1 is None else refinement_l1.tolist(),
                refinement_tolerance=refinement_tolerance.tolist(), passed=passed)
            diagnostics.append(record)
            log(f"IG {modality} subset={subset} steps={steps} error={error.round(6).tolist()} passed={passed}")
            if not passed:
                raise RuntimeError("Conditional IG failed numerical convergence; increase --max-steps. " +
                                   json.dumps(record))
    return result, diagnostics


def ranking_summary(attributions, content_mask):
    result = {}
    for modality, values in attributions.items():
        eligible = np.flatnonzero(content_mask)
        top_cells = []
        for head in range(2):
            mass = np.abs(values[head])
            candidates = [(int(t), int(d), float(values[head, eligible[t], d]))
                          for t, d in zip(*np.unravel_index(
                              np.argsort(mass[eligible].reshape(-1))[::-1][:10],
                              mass[eligible].shape))]
            # The unravelled first coordinate indexes eligible, not all 50 slots.
            candidates = [{"slot": int(eligible[t]), "dimension": d,
                           "contribution": score} for t, d, score in candidates]
            top_cells.append(candidates)
        result[modality] = dict(time_net=values.sum(2).tolist(),
                               time_mass=np.abs(values).sum(2).tolist(),
                               dimension_net=values[:, content_mask].sum(1).tolist(),
                               dimension_mass=np.abs(values[:, content_mask]).sum(1).tolist(),
                               top_cells=top_cells,
                               special_token_net=values[:, ~content_mask].sum((1, 2)).tolist())
    return result


def deletion_check(model, x, baseline, masks, winner, runner, original_scores,
                   attribution, content_mask, args, sample_number):
    eligible = np.flatnonzero(content_mask)
    counts = sorted(set(min(k, len(eligible)) for k in (0, 1, 2, 4, 8)))
    records = {}
    for head, head_name in enumerate(HEADS):
        records[head_name] = {}
        for group_index, group in enumerate(("all", *MODALITIES)):
            members = MODALITIES if group == "all" else (group,)
            salience = sum(np.abs(attribution[m][head]).sum(1) for m in members)
            top_order = eligible[np.argsort(-salience[eligible], kind="stable")]
            low_order = eligible[np.argsort(salience[eligible], kind="stable")]
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, sample_number, head, group_index, 619]))
            orders = [top_order, low_order] + [rng.permutation(eligible)
                                              for _ in range(args.random_repeats)]
            batches = {m: [] for m in MODALITIES}
            for order in orders:
                for count in counts:
                    selected = order[:count]
                    for m in MODALITIES:
                        changed = x[m].clone()
                        if m in members:
                            changed[:, selected] = baseline[m][:, selected]
                        batches[m].append(changed)
            batches = {m: torch.cat(values) for m, values in batches.items()}
            all_scores, all_logits, all_regression = evaluate_features(
                model, batches, masks, winner, runner)
            all_scores = all_scores.reshape(len(orders), len(counts), 2)
            all_logits = all_logits.reshape(len(orders), len(counts), -1)
            all_probabilities = softmax_probabilities(all_logits)
            all_classes = np.argmax(all_logits, axis=-1)
            all_regression = all_regression.reshape(len(orders), len(counts))
            predicted = all_scores[:, :, head]
            signed = original_scores[head] - predicted
            absolute = np.abs(signed)
            record = dict(counts=counts, eligible_slots=eligible.tolist(),
                          top_order=top_order.tolist(), low_order=low_order.tolist(),
                          random_orders=[order.tolist() for order in orders[2:]])
            for which, row in (("top", 0), ("low", 1)):
                record[which] = dict(score=predicted[row].tolist(),
                                     signed_change=signed[row].tolist(),
                                     absolute_change=absolute[row].tolist(),
                                     logits=all_logits[row].tolist(),
                                     probabilities=all_probabilities[row].tolist(),
                                     class_indices=all_classes[row].tolist(),
                                     class_flip=(all_classes[row] != winner).tolist(),
                                     regression_predictions=all_regression[row].tolist(),
                                     fixed_margin_signed_changes=(original_scores[0]-all_scores[row,:,0]).tolist(),
                                     regression_signed_changes=(original_scores[1]-all_regression[row]).tolist())
            record["random"] = dict(score_mean=predicted[2:].mean(0).tolist(),
                                   signed_change_mean=signed[2:].mean(0).tolist(),
                                   absolute_change_mean=absolute[2:].mean(0).tolist(),
                                   absolute_change_std=absolute[2:].std(0, ddof=1).tolist(),
                                   draws=predicted[2:].tolist(),
                                   class_flip_fraction=(all_classes[2:] != winner).mean(axis=0).tolist(),
                                   class_indices=all_classes[2:].tolist(),
                                   regression_predictions=all_regression[2:].tolist(),
                                   fixed_margin_signed_changes=(original_scores[0]-all_scores[2:,:,0]).tolist(),
                                   regression_signed_changes=(original_scores[1]-all_regression[2:]).tolist())
            records[head_name][group] = record
    return records


def load_existing_adapter(checkpoint, device):
    path = ROOT / "question2/bert_feature_adapter.py"
    spec = importlib.util.spec_from_file_location("_existing_frozen_bert_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_adapter(str(checkpoint), device=device)


def normalize_singleton(sample):
    required = ("text", "audio", "vision", "text_bert", "raw_text", "id")
    for key in required:
        if key not in sample:
            raise ValueError(f"Missing Attachment-4 key: {key}")
    split = {m: np.asarray(sample[m])[None] for m in MODALITIES}
    split["text_bert"] = np.asarray(sample["text_bert"])[None]
    split["id"] = [str(sample["id"])]
    for m in MODALITIES:
        if split[m].shape != (1, 50, DIMENSIONS[m]) or not np.isfinite(split[m]).all():
            raise ValueError(f"Invalid {m} shape or values")
    if split["text_bert"].shape != (1, 3, 50):
        raise ValueError("Expected text_bert [3,50]")

    if not np.array_equal(split["text_bert"], np.round(split["text_bert"])):
        raise ValueError("Non-integer BERT inputs")
    return split


def align_words(sample, tokenizer):
    raw_text = str(sample["raw_text"])
    encoded = tokenizer(raw_text, max_length=50, padding="max_length", truncation=True,
                        return_offsets_mapping=True, return_special_tokens_mask=True)
    observed = np.asarray(sample["text_bert"]).astype(np.int64)
    for key, row in (("input_ids", 0), ("attention_mask", 1), ("token_type_ids", 2)):
        np.testing.assert_array_equal(np.asarray(encoded[key]), observed[row],
                                      err_msg=f"Raw-text tokenizer mismatch: {key}")
    tokens = tokenizer.convert_ids_to_tokens(observed[0].tolist())
    content = (observed[1] == 1) & (np.asarray(encoded["special_tokens_mask"]) == 0)
    offsets = encoded["offset_mapping"]
    full = tokenizer(raw_text, add_special_tokens=False, return_offsets_mapping=True)
    truncated = len(full["input_ids"]) > int(content.sum())
    visible_end = max([end for (start, end), valid in zip(offsets, content) if valid], default=0)
    words = []
    pattern = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)
    for match in pattern.finditer(raw_text):
        slots = [slot for slot, (start, end) in enumerate(offsets)
                 if content[slot] and start < match.end() and end > match.start()]
        if not slots:
            continue
        incomplete = match.end() > visible_end
        words.append(dict(word=match.group(), tokens=slots, char_span=[match.start(), match.end()],
                          truncated=incomplete, word_index=len(words)))
    return tokens, content, words, {
        "token_ids_exact_match": True, "attention_exact_match": True,
        "token_types_exact_match": True, "input_truncated": bool(truncated),
        "visible_character_end": int(visible_end),
        "full_content_token_count": len(full["input_ids"]),
        "visible_content_token_count": int(content.sum()),
        "visible_content_token_fraction": float(content.sum()/max(1,len(full["input_ids"]))),
        "offsets": [list(pair) for pair in offsets]}


def word_validation(model, x, masks, metadata, split, words, attribution,
                    adapter, winner, runner, original_scores, args, sample_number):
    encoded_clean = adapter.encode_text_bert(split["text_bert"], batch_size=1,
                                            zero_padding=True, apply_adapter=True)
    valid = masks["text"][0].cpu().numpy()
    cached = split["text"][0]
    error = float(np.max(np.abs(encoded_clean[0, valid] - cached[valid])))
    if not np.allclose(encoded_clean[0, valid], cached[valid], rtol=2e-5, atol=1e-4):
        raise RuntimeError(f"Clean cached vs re-encoded text disagreement: max_abs={error}; "
                           "word validation requires a compatible encoder")
    normalizer = metadata["normalizers"]["text"]

    def model_text(encoded):
        normalized = (encoded.astype(np.float32) - normalizer["mean"]) / normalizer["std"]
        normalized[:, ~valid] = 0
        return torch.from_numpy(normalized).to(x["text"].device)

    clean_features = dict(x)
    clean_features["text"] = model_text(encoded_clean)
    clean_scores, _, _ = evaluate_features(model, clean_features, masks, winner, runner)
    for word in words:
        word["attribution_net"] = attribution["text"][:, word["tokens"], :].sum((1, 2)).tolist()
        word["attribution_mass"] = np.abs(attribution["text"][:, word["tokens"], :]).sum((1, 2)).tolist()
    eligible_words = [word for word in words if not word["truncated"]]
    orders = [sorted(eligible_words, key=lambda word: -word["attribution_mass"][head])
              for head in range(2)]
    top_indices = [{word["word_index"] for word in order[:args.top_words]} for order in orders]
    selected = sorted(set.union(*top_indices))
    available_controls = [word["word_index"] for word in eligible_words
                          if word["word_index"] not in selected]
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, sample_number, 331]))
    random_controls = (rng.choice(available_controls, size=min(5, len(available_controls)),
                                  replace=False).astype(int).tolist()
                       if available_controls else [])
    selected += random_controls
    candidates = []
    if selected:
        variants = np.repeat(split["text_bert"], len(selected), axis=0).copy()
        for row, index in enumerate(selected):
            variants[row, 0, words[index]["tokens"]] = 100
        np.testing.assert_array_equal(variants[:, 1:], np.repeat(split["text_bert"][:, 1:], len(selected), axis=0))
        encoded = adapter.encode_text_bert(variants, batch_size=32,
                                          zero_padding=True, apply_adapter=True)
        features = {m: x[m].expand(len(selected), -1, -1) for m in MODALITIES}
        features["text"] = model_text(encoded)
        scores, logits, regression = evaluate_features(model, features, masks, winner, runner)
        for row, index in enumerate(selected):
            word = words[index]
            roles = [f"top_{HEADS[h]}" for h in range(2) if index in top_indices[h]]
            if index in random_controls:
                roles.append("random")
            candidates.append(dict(
                **word, roles=roles,
                classification_rank=next(i+1 for i, w in enumerate(orders[0]) if w["word_index"] == index),
                regression_rank=next(i+1 for i, w in enumerate(orders[1]) if w["word_index"] == index),
                score=scores[row].tolist(), signed_change=(original_scores - scores[row]).tolist(),
                absolute_change=np.abs(original_scores - scores[row]).tolist(),
                change_from_reencoded_control=(clean_scores[0] - scores[row]).tolist(),
                logits=logits[row].tolist(),
                probabilities=softmax_probabilities(logits[row]).tolist(),
                regression=float(regression[row]),
                class_index=int(np.argmax(logits[row])),
                class_flip=bool(np.argmax(logits[row]) != winner)))
    return dict(encoding_control=dict(max_abs_feature_difference=error,
                                      original_score=original_scores.tolist(),
                                      reencoded_score=clean_scores[0].tolist(),
                                      score_difference=(clean_scores[0] - original_scores).tolist(),
                                      accepted_rtol=2e-5, accepted_atol=1e-4,
                                      offset_correction_applied=False),
                intervention="replace every BERT subtoken of one complete visible word by ID 100; "
                             "re-encode full sequence; retain audio/vision and all masks",
                candidates=candidates)


def training_means(path, metadata):
    with path.open("rb") as stream:
        dataset = pickle.load(stream)
    if dataset.get("_metadata", {}).get("nominal_probability") != 0:
        raise ValueError("Training-reference dataset must be the approved re-encoded p000 dataset")
    train = dataset["train"]
    masks = resolve_masks(train)
    means, counts = {}, {}
    for modality in MODALITIES:
        total = np.zeros(DIMENSIONS[modality], dtype=np.float64)
        count = 0
        stats = metadata["normalizers"][modality]
        for start in range(0, len(train[modality]), 128):
            values = np.asarray(train[modality][start:start+128], np.float32)
            selected = values[masks[modality][start:start+128]]
            selected = (selected - stats["mean"]) / stats["std"]
            if not np.isfinite(selected).all():
                raise ValueError("Non-finite training reference input")
            total += selected.astype(np.float64).sum(0)
            count += len(selected)
        means[modality] = (total / count).astype(np.float32)
        counts[modality] = count
    report = dict(dataset=str(path), samples=len(train["text"]), observed_positions=counts,
                  valid_mask="saved attention mask, including CLS/SEP; natural AV zeros retained",
                  split_used="train only", standardization="checkpoint normalizers; never refitted")
    del dataset, train
    gc.collect()
    return means, report


def numerical_self_test(device="cpu"):
    class Toy(torch.nn.Module):
        def forward(self, features, masks):
            t, a, v = [features[m].sum((1, 2)) for m in MODALITIES]
            first = 2*t + 3*a - v + 0.5*t*a
            second = -t + 0.25*a + 2*v + 0.2*a*v
            logits = torch.stack((first, torch.zeros_like(first), -torch.ones_like(first)), 1)
            return dict(classification_logits=logits, regression_prediction=second)
    model = Toy().to(device)
    x = {m: torch.tensor([[[value]]], device=device) for m, value in zip(MODALITIES, (1., 2., 3.))}
    b = {m: torch.zeros_like(value) for m, value in x.items()}
    masks = {m: torch.ones((1, 1), dtype=torch.bool, device=device) for m in MODALITIES}
    args = argparse.Namespace(ig_steps=4, max_steps=8, grad_batch=4)
    scores = coalition_scores(model, x, b, masks, 0, 1)
    phi = shapley_from_coalitions(scores)
    np.testing.assert_allclose(phi, [[2.5, 6.5, -3.], [-1., 1.1, 6.6]], atol=2e-6)
    attribution, diagnostics = conditional_ig(model, x, b, masks, 0, 1, scores, args)
    reconstructed = np.stack([attribution[m].sum((1, 2)) for m in MODALITIES], 1)
    np.testing.assert_allclose(reconstructed, phi, atol=2e-6)
    assert all(record["passed"] for record in diagnostics)
    p = np.asarray([[1.,0.],[0.,1.],[.3,.7]])
    q = np.asarray([[0.,1.],[1.,0.],[.6,.4]])
    np.testing.assert_allclose(js_divergence_bits(p,p), 0., atol=1e-15)
    np.testing.assert_allclose(js_divergence_bits(p,q), js_divergence_bits(q,p), atol=1e-15)
    np.testing.assert_allclose(js_divergence_bits(p,q)[:2], 1., atol=1e-15)
    rng = np.random.default_rng(391)
    utility_cases = -rng.random((3,8,2))
    utility_cases[:,7] = 0.
    local_phi = np.stack([shapley_from_coalitions(u) for u in utility_cases])
    np.testing.assert_allclose(local_phi.mean(0), shapley_from_coalitions(utility_cases.mean(0)),
                               atol=1e-12, rtol=1e-12)
    assert finite_rank_correlation([1,1,1],[2,3,4]) is None
    return {"linear_and_bilinear_exact": True, "maximum_error": float(np.abs(reconstructed-phi).max()),
            "js_identity_symmetry_and_one_bit_maximum": True,
            "preservation_shapley_local_global_linearity": True,
            "constant_rank_correlation_explicitly_undefined": True}


def run_analysis(args):
    """Return JSON-safe report and NumPy arrays; writes no output files."""
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Do not set model.train() to enable gradients: that would activate dropout.
    # Multihead attention's optimized eval fastpath can obscure grad control.
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    model, metadata = load_experiment_model(args.checkpoint, device=str(device))
    model.eval().requires_grad_(False)
    initial_state_hash = state_hash(model)
    self_test = numerical_self_test(str(device))
    input_directory = getattr(args, "input_dir", None)
    if input_directory is None:
        matches = list((ROOT / "datasets").glob("*4*/*4*/" + "\u5bf9\u9f50\u7248\u672c"))
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely locate Attachment-4 aligned inputs: {matches}")
        input_directory = matches[0]
    input_directory = Path(input_directory)
    train_path = Path(getattr(args, "train_dataset", ROOT / ("datasets/\u9644\u4ef62-\u540c\u6b65\u6270\u52a8\u7279\u5f81/aligned_50_p000.pkl")))
    adapter_checkpoint = Path(getattr(args, "adapter_checkpoint", DEFAULT_ADAPTER))
    paths = [input_directory / f"{number:02d}.pkl" for number in range(1, 21)]
    if sorted(p.name for p in input_directory.glob("*.pkl")) != [p.name for p in paths]:
        raise ValueError("Attachment-4 aligned input must contain exactly 01.pkl through 20.pkl")
    if not all(p.is_file() for p in paths):
        raise ValueError("Missing Attachment-4 aligned case")
    protected = {Path(args.checkpoint): sha256(args.checkpoint), train_path: sha256(train_path),
                 adapter_checkpoint: sha256(adapter_checkpoint),
                 ROOT / "question2/bert_feature_adapter.py": sha256(ROOT / "question2/bert_feature_adapter.py")}
    protected.update({path: sha256(path) for path in paths})
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (path.is_file() and path.suffix == ".py" and path.is_relative_to(ROOT) and
                not path.is_relative_to(ROOT / ".venv-align") and
                not path.is_relative_to(ROOT / "question3")):
                protected[path] = sha256(path)
    means, mean_provenance = training_means(train_path, metadata)
    log("Computed TRAIN-only reference means in checkpoint model-input space")
    checkpoint_info = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    bert_dir = Path(checkpoint_info["model_dir"])
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(str(bert_dir), local_files_only=True)
    for filename in ("vocab.txt", "tokenizer_config.json", "config.json", "model.safetensors"):
        path = bert_dir / filename
        if path.exists():
            protected[path] = sha256(path)
    adapter = load_existing_adapter(adapter_checkpoint, str(device))
    report = dict(
        format="this_work_hierarchical_explanations_v2_all20",
        created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(args.checkpoint), selected_epoch=int(metadata["selected_epoch"]),
        class_values=np.asarray(metadata["class_values"]).tolist(),
        head_order=list(HEADS), modality_order=list(MODALITIES),
        protocol=dict(classification="fixed original winner minus original runner-up logit",
                      regression="raw continuous regression output",
                      shapley="exact 8 three-modality coalitions, reference feature substitution",
                      features="12 conditional straight-line IG edges with Shapley weights; not feature-wise Shapley",
                      integration="adaptive Gauss-Legendre; marginal completeness AND full-array refinement",
                      baseline_mean="global TRAIN-only average of observed normalized feature slots, including specials",
                      baseline_zero="zero in model-input space, not necessarily zero raw acoustic/visual features",
                      masks="original attention masks held fixed for every intervention",
                      evidence_slots="content tokens only; padding excluded and specials reported separately",
                      negative_contribution="reduces fixed classification margin or continuous regression score",
                      strength_share="absolute modality Shapley divided by sum absolute Shapley",
                      deletion="ranking by absolute attribution mass; absolute score change and signed changes retained",

                      caveat="feature-space interventions and UNK word substitutions may be out of distribution; "
                             "correlated inputs prevent causal interpretation; embedding dimensions lack semantic labels",
                      scope="all 20 Attachment-4 aligned unlabeled samples, no accuracy/F1/MAE claim; no Whisper or reconstruction",
                      preservation="exact modality Shapley of negative JS divergence (bits) and negative absolute regression deviation relative to each full-input prediction",
                      near_zero_shares="absolute modality sum <=1e-6: shares undefined (JSON null); excluded only from share statistics",
                      global_statistics="equal-case descriptive statistics; no population confidence intervals",
                      input_limitations="Truncated inputs only explain visible encoded content; raw-zero modality observations are flagged without changing frozen preprocessing"),
        numerical_self_test=self_test, mean_reference=mean_provenance,
        seed=args.seed, random_repeats=args.random_repeats,
        integration_settings=dict(initial_steps=args.ig_steps, max_steps=args.max_steps,
                                  grad_batch=args.grad_batch, absolute_tolerance=1e-3,
                                  marginal_relative_tolerance=0.01,
                                  array_refinement_absolute_tolerance=0.01,
                                  array_refinement_relative_mass_tolerance=0.02),
        software=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__)),
        samples=[])
    arrays = {f"reference_mean_{m}": means[m] for m in MODALITIES}
    for number, path in enumerate(paths, 1):
        started = time.monotonic()
        with path.open("rb") as stream:
            sample = pickle.load(stream)
        if str(sample["id"]) != path.stem:
            raise ValueError(f"ID {sample['id']} does not match expected {path.stem}")
        if any("label" in key.lower() for key in sample):
            raise ValueError("Unexpected label field: revisit unlabeled-sample protocol")
        split = normalize_singleton(sample)
        tokens, content_mask, words, token_alignment = align_words(sample, tokenizer)
        x, masks = feature_input(split, metadata, device)
        with torch.no_grad():
            original = model(x, masks)
        logits = original["classification_logits"][0].cpu().numpy()
        order = np.argsort(-logits, kind="stable")
        winner, runner = map(int, order[:2])
        original_scores = scalar_outputs(original, winner, runner)[0].cpu().numpy().astype(np.float64)
        # Independently call the existing public inference API once per sample.
        public = predict_split(model, split, metadata, batch_size=1)
        np.testing.assert_allclose(public["classification_logits"][0], logits, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(public["regression_prediction"][0], original_scores[1], atol=1e-6, rtol=1e-6)
        valid = masks["text"][0].cpu().numpy()
        for modality in MODALITIES:
            np.testing.assert_array_equal(masks[modality][0].cpu().numpy(), valid)
        zero_content = {m: np.flatnonzero(content_mask & np.all(split[m][0] == 0, axis=1)).tolist()
                        for m in MODALITIES}
        all_zero_modalities = [m for m in MODALITIES
                               if len(zero_content[m]) == int(content_mask.sum())]
        quality_notes = []
        if token_alignment["input_truncated"]:
            quality_notes.append("Only the visible encoded prefix is explained; raw-text suffix beyond 48 content tokens has no model representation.")
        if all_zero_modalities:
            quality_notes.append("All raw content features are zero for " + ", ".join(all_zero_modalities) +
                                 "; original masks/normalization retained. Attribution describes the model's zero-input response, not observed content evidence.")
        data_quality = dict(
            input_truncated=token_alignment["input_truncated"],
            full_content_token_count=token_alignment["full_content_token_count"],
            visible_content_token_count=token_alignment["visible_content_token_count"],
            visible_content_token_fraction=token_alignment["visible_content_token_fraction"],
            raw_zero_content_slots=zero_content, all_zero_content_modalities=all_zero_modalities,
            notes=quality_notes)
        record = dict(id=str(sample["id"]), file=str(path), raw_text=str(sample["raw_text"]),
                      tokens=tokens, valid_mask=valid.tolist(), content_mask=content_mask.tolist(),
                      words=words, token_alignment=token_alignment, data_quality=data_quality,
                      prediction=dict(logits=logits.tolist(),
                                      probabilities=softmax_probabilities(logits).tolist(),
                                      class_index=winner, runnerup_index=runner,
                                      class_value=float(metadata["class_values"][winner]),
                                      regression=float(original_scores[1]), margin=float(original_scores[0])),
                      public_inference_matches=True, baselines={})
        mean_attribution = mean_baseline = None
        for baseline_name in ("mean", "zero"):
            baseline = {}
            for modality in MODALITIES:
                vector = means[modality] if baseline_name == "mean" else np.zeros(DIMENSIONS[modality], np.float32)
                base = torch.as_tensor(vector, device=device).reshape(1, 1, -1).expand_as(x[modality]).clone()
                base.masked_fill_(~masks[modality].unsqueeze(-1), 0)
                baseline[modality] = base
            log(f"Sample {number:02d} / {baseline_name}: explain winner={winner}, runner={runner}, score={original_scores.tolist()}")
            scores, coalition_logits, coalition_regression = coalition_predictions(
                model, x, baseline, masks, winner, runner)
            np.testing.assert_allclose(scores[7], original_scores, atol=3e-5, rtol=2e-5)
            np.testing.assert_allclose(coalition_logits[7], logits, atol=3e-5, rtol=2e-5)
            if int(np.argmax(coalition_logits[7])) != winner:
                raise RuntimeError("Batch-size rounding changed the original winner")
            behavior = coalition_behavior(scores, coalition_logits, coalition_regression, winner)
            phi = shapley_from_coalitions(scores)
            attribution, convergence = conditional_ig(model, x, baseline, masks, winner, runner, scores, args)
            totals = np.stack([attribution[m].sum((1, 2)) for m in MODALITIES], 1)
            denominator = np.abs(phi).sum(1, keepdims=True)
            shares_defined = denominator[:, 0] > 1e-6
            shares = np.divide(np.abs(phi), denominator, out=np.zeros_like(phi),
                               where=denominator > 1e-6)
            json_shares = [shares[h].tolist() if shares_defined[h] else [None]*3
                           for h in range(2)]
            record["baselines"][baseline_name] = dict(
                coalitions=scores.tolist(),
                coalition_bits={str(i): [m for j, m in enumerate(MODALITIES) if i & (1 << j)] for i in range(8)},
                phi=phi.tolist(), shares=json_shares,
                shares_defined=shares_defined.tolist(), share_denominator=denominator[:,0].tolist(),
                share_threshold=1e-6,
                **behavior,
                integrated_modality_totals=totals.tolist(),
                completeness_error=(totals-phi).tolist(),
                sum_absolute_modality_errors=np.abs(totals-phi).sum(1).tolist(),
                sum_absolute_edge_errors=sum(
                    entry["weight"] * np.asarray(entry["absolute_error"]) for entry in convergence).tolist(),
                global_sum_error=(totals.sum(1)-(scores[7]-scores[0])).tolist(),
                convergence=convergence, rankings=ranking_summary(attribution, content_mask))
            for modality in MODALITIES:
                arrays[f"s{number:02d}_{baseline_name}_{modality}"] = attribution[modality]
                assert not np.any(attribution[modality][:, ~valid])
            if baseline_name == "mean":
                mean_attribution, mean_baseline = attribution, baseline
        record["reference_stability"] = sample_reference_stability(record)
        record["deletion"] = deletion_check(model, x, mean_baseline, masks, winner, runner,
                                             original_scores, mean_attribution, content_mask, args, number)
        record["word_checks"] = word_validation(model, x, masks, metadata, split, words,
                                                mean_attribution, adapter, winner, runner,
                                                original_scores, args, number)
        record["elapsed_seconds"] = time.monotonic() - started
        report["samples"].append(record)
        log(f"Sample {number:02d} complete in {record['elapsed_seconds']:.1f}s; "
            f"{len(record['word_checks']['candidates'])} word interventions")
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Protected source or weight changed: {path}")
    if state_hash(model) != initial_state_hash:
        raise RuntimeError("Frozen ThisWork state changed")
    report["global_summary"] = summarize_all_samples(report["samples"])
    report["input_limitations"] = {
        "truncated_sample_ids": [s["id"] for s in report["samples"] if s["data_quality"]["input_truncated"]],
        "all_zero_modality_samples": [
            {"id":s["id"],"modalities":s["data_quality"]["all_zero_content_modalities"]}
            for s in report["samples"] if s["data_quality"]["all_zero_content_modalities"]],
        "no_case_excluded": True,
        "original_attention_masks_preserved": True}
    report["explanation_cautions"] = [
        "All 20 Attachment-4 aligned cases are included. No ground-truth labels are available; no accuracy, Macro-F1, MAE or performance-gain claim is made.",
        "Prediction-preservation Shapley measures agreement with the same model's full-input prediction, which is not ground truth.",
        "Raw-zero modalities are retained with original preprocessing; their attribution cannot be described as observed audio/video evidence.",
        "Truncated cases only explain the encoded visible prefix. Missing suffix content cannot receive attribution.",
        "Absolute Shapley shares depend on the reference; they are neither causal percentages nor trained gate weights.",
        "Audio/vision train means in normalized space are near zero, so the two-reference comparison mainly changes text.",
        "IG ranks describe contextual feature slots. Whole-word UNK edits re-encode the entire sentence and may change other contextual embeddings.",
        "Feature substitutions may be out of distribution; random word controls are descriptive and not matched for subtoken length.",
        "CLS/SEP contributions are separated from content; embedding dimensions are numeric IDs without asserted semantic meaning."]
    report["protected_sha256"] = {str(path): digest for path, digest in protected.items()}
    report["weights_unchanged"] = True
    report["model_state_sha256"] = initial_state_hash
    report["all_integrations_converged"] = all(
        entry["passed"] for sample in report["samples"]
        for baseline in sample["baselines"].values() for entry in baseline["convergence"])
    json.dumps(report, allow_nan=False)
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise AssertionError(f"Non-finite output array: {key}")
    return report, arrays




# ---- Static figures: all rendering uses Python / Matplotlib. ----
def make_figures(report, arrays, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    def rank_correlation(x, y):
        def ranks(v):
            order = np.argsort(v, kind="stable")
            r = np.empty(len(v), dtype=float)
            start = 0
            while start < len(v):
                stop = start + 1
                while stop < len(v) and v[order[stop]] == v[order[start]]:
                    stop += 1
                r[order[start:stop]] = (start + stop - 1) / 2
                start = stop
            return r
        rx, ry = ranks(x), ranks(y)
        return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else float("nan")
    output_dir = Path(output_dir)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    samples = report["samples"]
    modality_names = ("Text", "Audio", "Vision")
    modality_keys = ("text", "audio", "vision")
    head_names = ("Classification margin", "Regression score")
    head_keys = ("classification", "regression")
    colors = ("#4575A5", "#D99146", "#559C83")
    positive, negative = "#B44B4B", "#396CA6"
    figure_reports = []

    def attr(sample, baseline, modality):
        return np.asarray(arrays[f"s{sample['id']}_{baseline}_{modality}"])

    def decorate(axes):
        for i, ax in enumerate(np.asarray(axes).flat):
            ax.annotate(chr(97 + i), xy=(0, 1), xycoords="axes fraction",
                        xytext=(-27, 10), textcoords="offset points", weight="bold",
                        fontsize=9, ha="left", va="bottom")

    def save(fig, name, axes, claim, notes):
        selected = list(np.asarray(axes).flat)
        decorate(selected)
        fig.canvas.draw()
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            # Self-contained reruns retain measured rectangles. The delivery
            # run additionally executes the independent skill audit in memory.
            wh = fig.get_size_inches() * 72
            panels = []
            for i, ax in enumerate(selected):
                p = ax.get_position()
                panels.append({"id": chr(97 + i), "bbox_pt": [
                    float(p.x0 * wh[0]), float(p.y0 * wh[1]),
                    float(p.x1 * wh[0]), float(p.y1 * wh[1])]})
            alignment = {"status": "measured; independent audit not loaded",
                         "panels": panels}
        else:
            alignment = require_matplotlib_panel_alignment(
                fig, axes=selected, panel_ids=[chr(97 + i) for i in range(len(selected))],
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        fig.savefig(output_dir / f"{name}.png", dpi=360)
        fig.savefig(output_dir / f"{name}.pdf", dpi=360)
        figure_reports.append({"name": name, "question": claim, "notes": notes,
                               "alignment": alignment, "panels": len(selected)})
        plt.close(fig)

    # 1. Relative importance and its direction, with five individual samples.
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    fig.subplots_adjust(left=.10, right=.98, bottom=.09, top=.91, hspace=.53, wspace=.34)
    xx = np.arange(len(samples))
    for h in range(2):
        phi = np.asarray([s["baselines"]["mean"]["phi"][h] for s in samples])
        share = np.asarray([s["baselines"]["mean"]["shares"][h] for s in samples])
        ax = axes[0, h]
        for m in range(3):
            ax.bar(xx + (m - 1) * .24, phi[:, m], width=.23, color=colors[m], label=modality_names[m])
        ax.axhline(0, color=".35", linewidth=.6)
        ax.set(title=head_names[h], ylabel="Signed modality Shapley", xticks=xx,
               xticklabels=[s["id"] for s in samples], xlabel="Sample")
        ax = axes[1, h]
        bottom = np.zeros(len(samples))
        for m in range(3):
            ax.bar(xx, 100 * share[:, m], bottom=bottom, width=.64, color=colors[m])
            bottom += 100 * share[:, m]
        ax.set(ylim=(0, 100), ylabel="Absolute contribution share (%)",
               xticks=xx, xticklabels=[s["id"] for s in samples], xlabel="Sample")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=3, bbox_to_anchor=(.5, .997))
    fig.text(.5, .018, "Train-mean reference; includes CLS/SEP. Shares = |Shapley| / sum |Shapley|; reference-dependent.",
             ha="center", fontsize=7)
    save(fig, "01_modality_contributions", axes,
         "Which modality changes each prediction, and in which direction?",
         "Five fixed samples, train-mean reference. Shares normalize absolute Shapley values; not accuracy shares. Classification and regression have different units.")

    # 2-3. Signed time-slot evidence and absolute attribution mass.
    for h in range(2):
        fig, axes = plt.subplots(len(samples), 2, figsize=(10.2, 10.8), squeeze=False)
        fig.subplots_adjust(left=.095, right=.94, bottom=.11, top=.95, hspace=.95, wspace=.20)
        net = [np.stack([attr(s, "mean", m)[h].sum(-1) for m in modality_keys]) for s in samples]
        mass = [np.stack([np.abs(attr(s, "mean", m)[h]).sum(-1) for m in modality_keys]) for s in samples]
        net_lim = max(float(np.max(np.abs(v))) for v in net) or 1.
        mass_lim = max(float(np.max(v)) for v in mass) or 1.
        ims = [None, None]
        for row, s in enumerate(samples):
            valid = np.asarray(s["valid_mask"], bool)
            for col, values in enumerate((net[row], mass[row])):
                values = np.ma.array(values, mask=np.broadcast_to(~valid, values.shape))
                cmap = plt.get_cmap("RdBu_r" if col == 0 else "viridis").copy()
                cmap.set_bad("#EFEFEF")
                ax = axes[row, col]
                ims[col] = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap,
                                     vmin=-net_lim if col == 0 else 0, vmax=net_lim if col == 0 else mass_lim)
                ax.set(yticks=range(3), yticklabels=modality_names, xticks=range(0, 50, 5),
                       xlabel="Original token slot (0-based)",
                       title=f"Sample {s['id']} | " + ("Signed contribution" if col == 0 else "Contribution mass"))
                for pos, token in enumerate(s["tokens"]):
                    if valid[pos] and token in ("[CLS]", "[SEP]"):
                        ax.axvline(pos, color="black", lw=.65, linestyle=":")
        for col in range(2):
            bb = axes[-1, col].get_position()
            cax = fig.add_axes([bb.x0, .035, bb.width, .013])
            cb = fig.colorbar(ims[col], cax=cax, orientation="horizontal")
            cb.set_label(head_names[h] + (" units" if col == 0 else " attribution mass"))
            cb.ax.tick_params(labelsize=7)
        fig.text(.5, .989, "Train-mean reference | Gray: padding; dotted columns: CLS/SEP (excluded from word evidence).",
                 ha="center", va="top", fontsize=8)
        save(fig, f"0{2+h}_time_{head_keys[h]}", axes,
             "Where along the input does the model receive supporting or opposing evidence?",
             "Gray is padding; dotted columns denote CLS/SEP, not ordinary words. Net=sum(features); mass=sum(abs(features)). All valid slots retained, common scale across samples within each column.")

    # 4-5. Feature dimensions retain numerical identity, without semantic claims.
    for h in range(2):
        fig, axes = plt.subplots(len(samples), 3, figsize=(11.4, 12.0), squeeze=False)
        fig.subplots_adjust(left=.075, right=.98, bottom=.055, top=.925, hspace=.82, wspace=.34)
        for row, s in enumerate(samples):
            for m, mod in enumerate(modality_keys):
                a = attr(s, "mean", mod)[h][np.asarray(s["content_mask"], bool)]
                strength, net = np.abs(a).sum(0), a.sum(0)
                idx = np.argsort(-strength, kind="stable")[:8]
                yy = np.arange(len(idx))
                ax = axes[row, m]
                ax.barh(yy, strength[idx], height=.72, color="#DFE3E7", label="Absolute mass")
                ax.barh(yy, net[idx], height=.29, color=np.where(net[idx] >= 0, positive, negative), label="Signed net")
                ax.axvline(0, color=".3", linewidth=.55)
                ax.set(yticks=yy, yticklabels=[f"d{k}" for k in idx],
                       title=f"Sample {s['id']} | {modality_names[m]}", xlabel="Contribution")
                ax.invert_yaxis()
        fig.legend(handles=[Line2D([0], [0], color="#BDC4CB", lw=7, label="Absolute mass (rank criterion)"),
                            Line2D([0], [0], color=positive, lw=4, label="Positive net"),
                            Line2D([0], [0], color=negative, lw=4, label="Negative net")],
                   loc="upper center", ncol=3, bbox_to_anchor=(.5, .972))
        fig.suptitle("Feature-dimension contributions | " + head_names[h], y=.992, fontsize=11)
        fig.text(.5, .014, "Train-mean reference; content slots only. Top 8 dimensions by absolute mass; d = zero-based numeric feature index.",
                 ha="center", fontsize=7)
        save(fig, f"0{4+h}_features_{head_keys[h]}", axes,
             "Which numerical feature dimensions carry the largest attribution mass?",
             "Top 8 dimensions per modality/sample; zero-based dimension IDs. Aggregated over content slots; special-token contributions are reported separately. Full feature-by-slot arrays are saved; dimensions have no inferred semantic names.")

    # 6. Word evidence is checked by replacing ALL constituent pieces with UNK.
    fig, axes = plt.subplots(len(samples), 2, figsize=(10.6, 12.6), squeeze=False)
    fig.subplots_adjust(left=.18, right=.975, bottom=.055, top=.94, hspace=.80, wspace=.74)
    for row, s in enumerate(samples):
        for h in range(2):
            candidates = s["word_checks"]["candidates"]
            role = "top_" + head_keys[h]
            selected = sorted([w for w in candidates if role in w["roles"]],
                              key=lambda w: -w["attribution_mass"][h])[:5]
            yy = np.arange(len(selected))
            net = [w["attribution_net"][h] for w in selected]
            delta = [w["signed_change"][h] for w in selected]
            ax = axes[row, h]
            ax.barh(yy - .18, net, height=.32, color="#6E94B7")
            ax.barh(yy + .18, delta, height=.32, color="#D99146")
            ax.axvline(0, color=".3", linewidth=.6)
            labels = [str(w["word"])[:24] + f" [{','.join(map(str,w['tokens']))}]" for w in selected]
            ax.set(yticks=yy, yticklabels=labels, title=f"Sample {s['id']} | {head_names[h]}",
                   xlabel="Signed attribution / output change")
            ax.invert_yaxis()
    fig.legend(handles=[Line2D([0], [0], color="#6E94B7", lw=6, label="Cached-slot attribution"),
                        Line2D([0], [0], color="#D99146", lw=6, label="Output before minus UNK re-encoding")],
               loc="upper center", ncol=2, bbox_to_anchor=(.5, .997))
    fig.text(.5, .014, "Top 5 words by attribution mass; brackets = token slots. Whole-word UNK re-encoding is a separate intervention, not an attribution identity.",
             ha="center", fontsize=7)
    save(fig, "06_word_interventions", axes,
         "Do high-ranked textual representations also matter when their words are perturbed?",
         "Top 5 words per target. Brackets contain original token slots. Whole-sequence re-encoding, audio/video unchanged. Attribution and intervention are different estimands; disagreement is retained, not hidden. No word timestamps inferred.")

    # 7. Faithfulness: pooled-slot occlusion compared with low-ranked/random controls.
    fig, axes = plt.subplots(len(samples), 2, figsize=(9.9, 11.5), squeeze=False)
    fig.subplots_adjust(left=.11, right=.985, bottom=.06, top=.935, hspace=.75, wspace=.32)
    for row, s in enumerate(samples):
        for h in range(2):
            d = s["deletion"][head_keys[h]]["all"]
            ax = axes[row, h]
            counts = np.asarray(d["counts"])
            rm = np.asarray(d["random"]["absolute_change_mean"])
            rs = np.asarray(d["random"]["absolute_change_std"])
            ax.fill_between(counts, np.maximum(0, rm - rs), rm + rs, color="#BFC4C9", alpha=.45, linewidth=0)
            ax.plot(counts, rm, color="#7A8188", lw=1.3, label="Random: mean ± SD")
            ax.plot(counts, d["low"]["absolute_change"], color="#68A18D", lw=1.3, linestyle="--", label="Lowest attribution mass")
            ax.plot(counts, d["top"]["absolute_change"], color="#BA5B4D", lw=1.6, marker="o", ms=3, label="Highest attribution mass")
            ax.set(title=f"Sample {s['id']} | {head_names[h]}", xlabel="Number of jointly replaced content slots",
                   ylabel="Absolute output change", xticks=counts)
            ax.set_ylim(bottom=0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               ncol=3, bbox_to_anchor=(.5, .996))
    fig.text(.5, .014, f"Train-mean replacement; all three modalities at the same content slots. Random band: SD of {report['random_repeats']} controls (not a confidence interval).",
             ha="center", fontsize=7)
    save(fig, "07_occlusion_validation", axes,
         "Does replacing highly attributed evidence change the output more than controls?",
         "Mean-reference replacements at the same aligned slots in all three modalities; fixed masks/targets. Random band is SD over 20 seeded random rankings, not a confidence interval. All outcomes shown; stronger response is not guaranteed. Per-modality controls are in the numerical report.")

    # 8. Two reference choices test whether ranking conclusions are robust.
    rho = np.zeros((2, len(samples), 3))
    jac = np.zeros_like(rho)
    for h in range(2):
        for i, s in enumerate(samples):
            keep = np.asarray(s["content_mask"], bool)
            for m, mod in enumerate(modality_keys):
                first = np.abs(attr(s, "mean", mod)[h]).sum(-1)[keep]
                second = np.abs(attr(s, "zero", mod)[h]).sum(-1)[keep]
                rho[h, i, m] = rank_correlation(first, second)
                k = min(5, len(first))
                aa, bb = set(np.argsort(-first, kind="stable")[:k]), set(np.argsort(-second, kind="stable")[:k])
                jac[h, i, m] = len(aa & bb) / len(aa | bb) if aa | bb else np.nan
    mean_shares = np.asarray([s["baselines"]["mean"]["shares"] for s in samples]).transpose(1, 0, 2)
    zero_shares = np.asarray([s["baselines"]["zero"]["shares"] for s in samples]).transpose(1, 0, 2)
    share_delta = 100 * (zero_shares - mean_shares)
    dominant = []
    for h in range(2):
        rows = []
        for i, sample in enumerate(samples):
            defined = all(sample["baselines"][b]["shares_defined"][h] for b in ("mean", "zero"))
            a, b = int(np.argmax(mean_shares[h, i])), int(np.argmax(zero_shares[h, i]))
            rows.append({"sample_id": sample["id"], "defined": bool(defined),
                         "mean_top": modality_keys[a] if defined else None,
                         "zero_top": modality_keys[b] if defined else None,
                         "top_agrees": a == b if defined else None,
                         "absolute_share_l1_change": float(np.abs(share_delta[h, i]).sum() / 100) if defined else None})
        dominant.append(rows)
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 9.3))
    fig.subplots_adjust(left=.11, right=.96, bottom=.08, top=.95, hspace=.48, wspace=.28)
    titles = ("Time-rank correlation", "Top-5 time-slot overlap", "Modality share change (pp)")
    for row, values in enumerate((rho, jac, share_delta)):
        for h in range(2):
            ax = axes[row, h]
            limit = 100 if row == 2 else 1
            ax.imshow(values[h], aspect="auto", cmap="viridis" if row == 1 else "RdBu_r",
                      vmin=0 if row == 1 else -limit, vmax=limit)
            for i in range(len(samples)):
                for m in range(3):
                    value = values[h, i, m]
                    color = "white" if (value < .55 if row == 1 else abs(value) > .65 * limit) else "black"
                    label = (f"{value:+.1f}" if row == 2 else f"{value:.2f}") if np.isfinite(value) else "NA"
                    ax.text(m, i, label, ha="center", va="center", color=color, fontsize=8)
            ax.set(xticks=range(3), xticklabels=modality_names,
                   yticks=range(len(samples)), yticklabels=[s["id"] for s in samples],
                   title=head_names[h] + " | " + titles[row], ylabel="Sample")
    fig.text(.5, .025, "Zero minus train-mean reference; pp = percentage points. This comparison mainly varies the text reference.",
             ha="center", fontsize=7)
    save(fig, "08_reference_stability", axes,
         "Are modality importance and key time slots stable under a different attribution reference?",
         "Top: content-slot Spearman correlation; middle: top-5 Jaccard overlap; bottom: zero-reference minus mean-reference absolute modality share, in percentage points. AV train means in normalized input space are near zero, so this comparison mainly varies the text reference. These are sensitivity diagnostics, not statistical confidence.")
    def nullable(values):
        return [[[float(v) if np.isfinite(v) else None for v in row] for row in head] for head in values]
    report["explanation_cautions"] = [
        "Absolute Shapley shares are conditional on the reference and fixed masks; they are not causal percentages or accuracy contributions.",
        "The reused model treats zero audio/vision rows at CLS/SEP as valid observations. Its original preprocessing is preserved. Special-slot attributions must not be interpreted as spoken words or real video frames.",
        "Modality Shapley includes special slots; word and feature-dimension rankings exclude them. Net contributions can cancel although attribution mass is large.",
        "Agreement with interventions is sample-specific; inspect all controls, including cases where top-ranked slots do not outperform low or random slots.",
        "No semantic meaning is assigned to embedding dimension IDs; no real-time localization is performed."
    ]
    report["figures"] = figure_reports
    report["reference_stability"] = {"time_mass_spearman": nullable(rho), "top5_jaccard": nullable(jac),
                                    "modality_share_delta_percentage_points": nullable(share_delta),
                                    "dominant_modality": dominant,
                                    "limitation": "Normalized audio/vision train means are near zero; the alternate reference mainly changes text. Exact ties in top-k ranks use stable original-slot ordering.",
                                    "axis_order": ["head", "sample", "modality"]}
    return figure_reports


def make_individual_figures(report, arrays, output_dir):
    """Write three evidence plates per sample and three matching multipage PDFs."""
    import contextlib
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    png_dir = output_dir / "single_samples"
    png_dir.mkdir(exist_ok=True)
    samples = report["samples"]
    keys = ("text", "audio", "vision")
    names = ("Text", "Audio", "Vision")
    heads = ("classification", "regression")
    head_titles = ("Classification margin", "Regression score")
    colors = ("#4575A5", "#D99146", "#559C83")
    red, blue = "#B44B4B", "#396CA6"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    results = []

    def values(sample, reference, modality):
        return np.asarray(arrays[f"s{sample['id']}_{reference}_{modality}"])

    # Preserve a common color scale across all samples, separately by target.
    limits = {"net": np.zeros(2), "mass": np.zeros(2)}
    for sample in samples:
        valid = np.asarray(sample["valid_mask"], dtype=bool)
        for modality in keys:
            attribution = values(sample, "mean", modality)
            limits["net"] = np.maximum(
                limits["net"], np.max(np.abs(attribution.sum(-1)[:, valid]), axis=1))
            limits["mass"] = np.maximum(
                limits["mass"], np.max(np.abs(attribution).sum(-1)[:, valid], axis=1))
    limits = {key: np.maximum(value, 1e-12) for key, value in limits.items()}

    def header(fig, sample, label):
        prediction = sample["prediction"]
        probabilities = prediction.get("probabilities")
        if probabilities is None:
            logits = np.asarray(prediction["logits"], dtype=float)
            probabilities = np.exp(logits-logits.max())
            probabilities /= probabilities.sum()
        class_values = report.get("class_values", list(range(len(probabilities))))
        probability_text = ", ".join(
            f"{value:g}: {probability:.3f}"
            for value, probability in zip(class_values, probabilities))
        fig.text(.5, .985, f"Attachment 4 | Sample {sample['id']} | {label}",
                 ha="center", va="top", fontsize=14, weight="bold")
        fig.text(.5, .960,
                 f"Predicted class {prediction['class_value']:g}  |  "
                 f"Class probabilities [{probability_text}]  |  "
                 f"Regression {prediction['regression']:.4f}",
                 ha="center", va="top", fontsize=10)
        quality = sample.get("data_quality", {})
        warnings = []
        if quality.get("input_truncated"):
            warnings.append(
                f"Text truncated: {quality.get('visible_content_token_count', 48)} "
                f"of {quality.get('full_content_token_count', '?')} content tokens retained.")
        if "vision" in quality.get("all_zero_content_modalities", []):
            warnings.append("Raw vision is all zero; visual attribution explains zero-input behavior.")
        if warnings:
            fig.text(.5, .938, "\n".join(warnings), ha="center", va="top",
                     fontsize=9, color="#8D342D")

    def footer(fig, text):
        wrapped = "\n".join(textwrap.wrap(text, width=145))
        fig.text(.08, .018, wrapped, ha="left", va="bottom", fontsize=8, color=".25")

    def decorate(axes):
        for number, ax in enumerate(np.asarray(axes).flat):
            ax.annotate(chr(97 + number), xy=(0, 1), xycoords="axes fraction",
                        xytext=(-30, 10), textcoords="offset points",
                        fontsize=11, weight="bold", ha="left", va="bottom")

    def save(fig, sample, kind, axes, pdf, page, question, notes):
        main_axes = list(np.asarray(axes).flat)
        decorate(main_axes)
        fig.canvas.draw()
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            size_pt = fig.get_size_inches() * 72
            alignment = {
                "status": "measured; independent audit not loaded",
                "panels": [{"id": chr(97+i), "bbox_pt": [
                    float(ax.get_position().x0*size_pt[0]),
                    float(ax.get_position().y0*size_pt[1]),
                    float(ax.get_position().x1*size_pt[0]),
                    float(ax.get_position().y1*size_pt[1])]}
                    for i, ax in enumerate(main_axes)]}
        else:
            alignment = require_matplotlib_panel_alignment(
                fig, axes=main_axes,
                panel_ids=[chr(97+i) for i in range(len(main_axes))],
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        name = f"{sample['id']}_{kind}"
        png_path = png_dir / f"{name}.png"
        fig.savefig(png_path, dpi=300)
        pdf.savefig(fig, dpi=300)
        results.append({
            "name": name, "sample_id": sample["id"], "kind": kind,
            "png": str(png_path), "pdf": str(output_dir / f"individual_{kind}.pdf"),
            "pdf_page_1based": page, "question": question, "notes": notes,
            "panels": len(main_axes), "alignment": alignment,
        })
        plt.close(fig)

    def reference_bars(ax, data, ylabel, title):
        x = np.arange(3)
        for reference_index, reference in enumerate(("mean", "zero")):
            y = np.asarray(data[reference], dtype=float)
            for index in range(3):
                if np.isfinite(y[index]):
                    ax.bar(x[index] + (reference_index-.5)*.30, y[index],
                           width=.28, color=colors[index] if reference_index == 0 else "white",
                           edgecolor=colors[index], linewidth=1.5,
                           hatch=None)
        ax.axhline(0, color=".4", linewidth=.65)
        ax.set(xticks=x, xticklabels=names, ylabel=ylabel, title=title)
        ax.margins(y=.16)

    reference_legend = [
        Patch(facecolor=".5", edgecolor=".5", label="Train-mean reference"),
        Patch(facecolor="white", edgecolor=".5", hatch=None, label="Zero reference")]
    signed_cmap = plt.get_cmap("RdBu_r").copy()
    mass_cmap = plt.get_cmap("magma").copy()
    signed_cmap.set_bad("#E2E2E2")
    mass_cmap.set_bad("#E2E2E2")

    with contextlib.ExitStack() as stack:
        pdfs = {kind: stack.enter_context(PdfPages(output_dir / f"individual_{kind}.pdf"))
                for kind in ("modalities", "evidence", "validation")}
        for page, sample in enumerate(samples, 1):
            quality = sample.get("data_quality", {})
            content = np.asarray(sample["content_mask"], dtype=bool)
            valid = np.asarray(sample["valid_mask"], dtype=bool)
            special = np.flatnonzero(valid & ~content)
            baseline = sample["baselines"]["mean"]

            # Plate 1: model output and two distinct modality-level questions.
            fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.8))
            fig.subplots_adjust(left=.10, right=.97, bottom=.11, top=.855,
                                hspace=.68, wspace=.34)
            header(fig, sample, "Modality contributions")
            for head_index in range(2):
                reference_bars(
                    axes[0, head_index],
                    {r: sample["baselines"][r]["phi"][head_index] for r in ("mean", "zero")},
                    "Signed output contribution", head_titles[head_index])
                reference_bars(
                    axes[1, head_index],
                    {r: 100*np.asarray(sample["baselines"][r]["shares"][head_index], dtype=float)
                     for r in ("mean", "zero")},
                    "Absolute contribution share (%)", "Relative modality contribution")
                axes[1, head_index].set_ylim(0, 110)
                if any(not sample["baselines"][r]["shares_defined"][head_index]
                       for r in ("mean", "zero")):
                    axes[1, head_index].text(
                        .5, .93, "Undefined share omitted (near-zero total)",
                        transform=axes[1, head_index].transAxes,
                        fontsize=8, ha="center", va="top")
                reference_bars(
                    axes[2, head_index],
                    {r: sample["baselines"][r]["prediction_preservation_phi"][head_index]
                     for r in ("mean", "zero")},
                    "Reduced JS divergence (bits)" if head_index == 0 else
                    "Reduced absolute score deviation",
                    "Prediction-preservation Shapley")
            fig.legend(handles=reference_legend, loc="lower center",
                       bbox_to_anchor=(.5, .063), ncol=2, frameon=False)
            footer(fig, "Classification explains the fixed original winner-minus-runner-up logit. "
                   "Prediction preservation compares coalition predictions with the complete-input prediction; "
                   "the complete prediction is not ground truth. Modality totals include valid CLS/SEP slots. All masks remain fixed.")
            save(fig, sample, "modalities", axes, pdfs["modalities"], page,
                 "Which modalities move this output, and which preserve the complete prediction?",
                 ["Exact eight-coalition Shapley for two references.",
                  "Prediction-preservation values quantify behavior, not accuracy.",
                  *quality.get("notes", [])])

            # Plate 2: all original slots plus top numerical feature dimensions.
            fig, axes = plt.subplots(5, 2, figsize=(13.2, 18.2))
            fig.subplots_adjust(left=.115, right=.895, bottom=.075, top=.885,
                                hspace=.77, wspace=.55)
            header(fig, sample, "Time-step and feature evidence")
            for head_index in range(2):
                net = np.stack([values(sample, "mean", m)[head_index].sum(-1) for m in keys])
                mass = np.stack([np.abs(values(sample, "mean", m)[head_index]).sum(-1) for m in keys])
                for row, matrix, metric in ((0, net, "net"), (1, mass, "mass")):
                    masked = np.ma.array(matrix, mask=np.broadcast_to(~valid, matrix.shape))
                    ax = axes[row, head_index]
                    if metric == "net":
                        image = ax.imshow(masked, aspect="auto", interpolation="nearest",
                                          cmap=signed_cmap, vmin=-limits["net"][head_index],
                                          vmax=limits["net"][head_index])
                    else:
                        image = ax.imshow(masked, aspect="auto", interpolation="nearest",
                                          cmap=mass_cmap, vmin=0, vmax=limits["mass"][head_index])
                    for slot in special:
                        ax.axvline(slot, color=".55", linestyle=":", linewidth=.8)
                    ax.set(yticks=np.arange(3), yticklabels=names,
                           xticks=[0, 9, 19, 29, 39, 49],
                           xlabel="Original aligned slot (0-based)",
                           title=f"{head_titles[head_index]} | {'signed net' if row == 0 else 'absolute mass'}")
                    pos = ax.get_position()
                    color_axis = fig.add_axes([pos.x1+.012, pos.y0, .009, pos.height])
                    colorbar = fig.colorbar(image, cax=color_axis)
                    colorbar.ax.tick_params(labelsize=8)
                for modality_index, modality in enumerate(keys):
                    ax = axes[modality_index+2, head_index]
                    attribution = values(sample, "mean", modality)[head_index, content]
                    net_dim = attribution.sum(axis=0)
                    mass_dim = np.abs(attribution).sum(axis=0)
                    selected = np.argsort(-mass_dim, kind="stable")[:8]
                    y = np.arange(len(selected))
                    ax.barh(y, mass_dim[selected], color="#D3D5D8", height=.72,
                            label="Absolute mass")
                    ax.barh(y, net_dim[selected], height=.34,
                            color=np.where(net_dim[selected] >= 0, red, blue),
                            label="Signed net")
                    ax.axvline(0, color=".35", linewidth=.6)
                    ax.set(yticks=y, yticklabels=[f"d{d}" for d in selected],
                           xlabel="Contribution over content slots",
                           title=f"{names[modality_index]} | top 8 numerical dimensions")
                    ax.invert_yaxis()
                    ax.margins(x=.15)
            fig.legend(handles=[
                Patch(facecolor="#D3D5D8", label="Absolute contribution mass"),
                Patch(facecolor=red, label="Positive signed net"),
                Patch(facecolor=blue, label="Negative signed net")],
                loc="lower center", bbox_to_anchor=(.5, .037), ncol=3, frameon=False)
            footer(fig, "Train-mean reference. Heatmap scales are shared across all 20 samples for each target. "
                   "Gray cells: padding; dotted lines: CLS/SEP. Absolute mass is not a modality share. Dimension rankings exclude specials and padding. "
                   "Dimensions are numerical indices, not semantic labels or video timestamps.")
            save(fig, sample, "evidence", axes, pdfs["evidence"], page,
                 "Which slots and numerical dimensions carry this sample's attribution?",
                 ["Signed sums and absolute mass are distinct; cancellation is retained.",
                  "Heatmap color scales fixed over all analyzed samples.",
                  "Feature dimension rankings use content slots only.",
                  *quality.get("notes", [])])

            # Plate 3: direct interventions and robustness to reference choice.
            fig, axes = plt.subplots(4, 2, figsize=(13.2, 16.2))
            fig.subplots_adjust(left=.16, right=.97, bottom=.085, top=.875,
                                hspace=.83, wspace=.57)
            header(fig, sample, "Intervention and stability checks")
            for head_index, head in enumerate(heads):
                removed = {}
                for reference in ("mean", "zero"):
                    scores = np.asarray(sample["baselines"][reference]["coalitions"], dtype=float)
                    removed[reference] = np.asarray(
                        [scores[7, head_index]-scores[7 ^ (1 << m), head_index]
                         for m in range(3)])
                reference_bars(
                    axes[0, head_index], removed,
                    "Original minus replaced output", f"{head_titles[head_index]} | remove one modality")
                if head_index == 0:
                    axes[0, head_index].legend(handles=reference_legend, fontsize=7,
                                              loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False)
                deletion = sample["deletion"][head]["all"]
                ax = axes[1, head_index]
                counts = np.asarray(deletion["counts"])
                random_mean = np.asarray(deletion["random"]["absolute_change_mean"])
                random_sd = np.asarray(deletion["random"]["absolute_change_std"])
                ax.plot(counts, deletion["top"]["absolute_change"], "o-", color=red,
                        markersize=3.5, label="Highest attribution mass")
                ax.plot(counts, deletion["low"]["absolute_change"], "s-", color="#559C83",
                        markersize=3.5, label="Lowest attribution mass")
                ax.plot(counts, random_mean, "^-", color=".4",
                        markersize=3.5, label="Random mean")
                ax.fill_between(counts, np.maximum(0, random_mean-random_sd),
                                random_mean+random_sd, color=".65", alpha=.25)
                ax.set(xticks=counts, xlabel="Content slots replaced in all modalities",
                       ylabel="Absolute output change", title="Does the slot ranking survive replacement?")
                if head_index == 0:
                    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False, fontsize=7)

                ax = axes[2, head_index]
                rank_key = f"{head}_rank"
                candidates = [record for record in sample["word_checks"]["candidates"]
                              if f"top_{head}" in record.get("roles", [])]
                candidates.sort(key=lambda item: (
                    item.get(rank_key) if item.get(rank_key) is not None else 10**9))
                candidates = candidates[:5]
                y = np.arange(len(candidates))
                net_word = [record["attribution_net"][head_index] for record in candidates]
                changed_word = [record["signed_change"][head_index] for record in candidates]
                ax.barh(y-.18, net_word, height=.32, color="#4575A5", label="Cached-slot attribution")
                ax.barh(y+.18, changed_word, height=.32, color="#D99146", label="Whole-word UNK change")
                labels = [record["word"]+"\n["+ ",".join(map(str, record["tokens"]))+"]"
                          for record in candidates]
                ax.set(yticks=y, yticklabels=labels, xlabel="Signed contribution / output change",
                       title="Whole-word replacement after full re-encoding")
                ax.invert_yaxis()
                ax.axvline(0, color=".35", linewidth=.6)
                if head_index == 0:
                    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, fontsize=7, frameon=False)

                ax = axes[3, head_index]
                stability = sample["reference_stability"][head]
                correlation = np.asarray(stability["content_time_spearman"], dtype=float)
                overlap = np.asarray(stability["top5_jaccard"], dtype=float)
                x = np.arange(3)
                ax.bar(x-.16, correlation, width=.3, color="#4575A5",
                       label="Time-rank correlation")
                ax.bar(x+.16, overlap, width=.3, color="#D99146",
                       label="Top-5 overlap (Jaccard)")
                for modality_index, value in enumerate(correlation):
                    if not np.isfinite(value):
                        ax.text(x[modality_index]-.16, .05, "N/A", rotation=90, rotation_mode="anchor",
                                fontsize=8, ha="center", va="bottom")
                ax.set(xticks=x, xticklabels=names, ylim=(-1.10, 1.14),
                       ylabel="Reference agreement",
                       title="Train-mean versus zero reference")
                ax.axhline(0, color=".35", linewidth=.6)
                if head_index == 0:
                    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, frameon=False, fontsize=7)
            footer(fig, "Replacement keeps masks fixed. Slot curves use the train-mean reference; "
                   "random shading is +/- 1 SD over 20 draws. Word bars compare attribution with "
                   "original-minus-reencoded output, so they need not agree. These checks assess model behavior, "
                   "not correctness or real-world causality. N/A denotes an undefined rank correlation.")
            save(fig, sample, "validation", axes, pdfs["validation"], page,
                 "Do direct interventions and reference changes support this sample's explanation?",
                 ["Whole-modality scores preserve the fixed original classification contrast.",
                  "Random replacement controls are descriptive, with 20 draws.",
                  "Whole-word interventions re-encode the full sentence; audiovisual features remain fixed.",
                  "Reference rank agreement uses content slots and absolute attribution mass.",
                  *quality.get("notes", [])])
            log(f"Individual figures: sample {sample['id']} complete ({page}/{len(samples)})")
    return results



def make_attachment4_global_figures(report, arrays, output_dir):
    """Three descriptive global figures for the fixed, unlabeled Attachment-4 cohort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
                         "font.size":9,"axes.titlesize":10,"axes.labelsize":9,
                         "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.linewidth":.7,"pdf.fonttype":42,"svg.fonttype":"none",
                         "savefig.facecolor":"white"})
    out=Path(output_dir)
    samples=report["samples"]; n=len(samples)
    refs=("mean","zero"); ref_names=("Train-mean reference","Zero reference")
    mods=("Text","Audio","Vision"); colors=("#4575A5","#D99146","#559C83")
    heads=("Classification","Regression")
    shares=np.asarray([[s["baselines"][r]["shares"] for s in samples] for r in refs],float)
    defined=np.asarray([[s["baselines"][r]["shares_defined"] for s in samples] for r in refs],bool)
    preservation=np.asarray([[s["baselines"][r]["prediction_preservation_phi"] for s in samples] for r in refs],float)
    rows=[]
    ref_legend=[Patch(facecolor=".65",edgecolor=".3",label=ref_names[0]),
                Patch(facecolor="white",edgecolor=".3",hatch=None,label=ref_names[1])]

    def export(fig,name,axes,question,notes):
        chosen=list(np.asarray(axes).flat)
        for i,ax in enumerate(chosen):
            ax.annotate(chr(97+i),xy=(0,1),xycoords="axes fraction",xytext=(-29,11),
                        textcoords="offset points",ha="left",va="bottom",weight="bold",fontsize=10)
        fig.canvas.draw()
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            wh=fig.get_size_inches()*72
            alignment={"status":"measured; independent auditor not loaded",
                       "panels":[{"id":chr(97+i),"bbox_pt":[float(ax.get_position().x0*wh[0]),
                           float(ax.get_position().y0*wh[1]),float(ax.get_position().x1*wh[0]),
                           float(ax.get_position().y1*wh[1])]} for i,ax in enumerate(chosen)]}
        else:
            alignment=require_matplotlib_panel_alignment(
                fig,axes=chosen,panel_ids=[chr(97+i) for i in range(len(chosen))],
                tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        fig.savefig(out/f"{name}.png",dpi=360)
        fig.savefig(out/f"{name}.pdf",dpi=360)
        rows.append({"name":name,"scope":"global","question":question,"notes":notes,
                     "png":f"{name}.png","pdf":f"{name}.pdf","pdf_page":0,
                     "alignment":alignment,"size_inches":fig.get_size_inches().tolist()})
        plt.close(fig)

    # 1. Equal-case average reliance plus all observed case-level shares.
    fig,axes=plt.subplots(2,2,figsize=(10.5,8.2))
    fig.subplots_adjust(left=.11,right=.97,top=.83,bottom=.135,hspace=.60,wspace=.31)
    fig.suptitle("Global modality reliance",y=.978,fontsize=14,weight="bold")
    fig.text(.5,.937,f"Attachment 4 | All {n} cases | Classification and regression explained separately",
             ha="center",fontsize=9)
    fig.legend(handles=ref_legend,loc="upper center",bbox_to_anchor=(.5,.908),ncol=2)
    for h in range(2):
        ax=axes[0,h]
        for b in range(2):
            v=shares[b,:,h][defined[b,:,h]]
            mean=v.mean(0)*100 if len(v) else np.zeros(3)
            for m in range(3):
                y=m+(-.15 if b==0 else .15)
                ax.barh(y,mean[m],height=.25,color=colors[m] if b==0 else "white",
                        edgecolor=colors[m],hatch=None,linewidth=1)
                ax.text(mean[m]+1,y,f"{mean[m]:.1f}",ha="left",va="center",fontsize=8)
        ax.set(yticks=range(3),yticklabels=mods,ylim=(2.5,-.5),xlim=(0,105),
               xlabel="Equal-case mean absolute Shapley share (%)",
               title=heads[h]+" | Mean reliance")
        ax.grid(False);ax.set_axisbelow(True)
        ax=axes[1,h]
        for b in range(2):
            for m in range(3):
                v=shares[b,:,h,m][defined[b,:,h]]*100
                # Deterministic vertical separation for displaying every case;
                # the x coordinates always remain the observed share values.
                yy=m+(-.16 if b==0 else .16)+np.linspace(-.065,.065,len(v))
                ax.scatter(v,yy,s=23,marker="o" if b==0 else "D",
                           facecolors=colors[m] if b==0 else "none",
                           edgecolors=colors[m],linewidths=.9,alpha=.85)
        ax.set(yticks=range(3),yticklabels=mods,ylim=(2.6,-.6),xlim=(-2,102),
               xlabel="Individual-case absolute Shapley share (%)",
               title=heads[h]+" | All cases")
        ax.grid(False);ax.set_axisbelow(True)
    fig.text(.5,.078,"Bars: descriptive means. Dots: individual cases; circles = train mean, open diamonds = zero.",
             ha="center",fontsize=8)
    fig.text(.5,.049,"Shares describe dependence on the chosen reference; they are not accuracy or performance contributions.",
             ha="center",fontsize=8)
    fig.text(.5,.02,"No population confidence intervals: these are 20 provided cases. Valid special-token slots remain part of model input.",
             ha="center",fontsize=8)
    export(fig,"global_01_modality_reliance",axes,
           "Which modalities influence the model's predictions across Attachment 4, and how much do cases differ?",
           "Every defined case contributes equally to mean normalized absolute modality Shapley. Classification targets each case's original winner-runner margin. Raw magnitudes, shares, undefined counts and dominant-modal frequencies are in global_summary. Case dots are descriptive observations, not independent population draws.")

    # 2. Shapley of agreement with the complete model, not correctness.
    fig,axes=plt.subplots(1,2,figsize=(10.5,5.3))
    fig.subplots_adjust(left=.11,right=.97,top=.72,bottom=.28,wspace=.33)
    fig.suptitle("Modality contributions to prediction preservation",y=.973,fontsize=13,weight="bold")
    fig.text(.5,.919,"Reference target: the same frozen model with all three observed modalities",
             ha="center",fontsize=9)
    fig.legend(handles=ref_legend,loc="upper center",bbox_to_anchor=(.5,.87),ncol=2)
    for h,ax in enumerate(axes):
        mean=preservation[:,:,h,:].mean(1)
        values=mean.reshape(-1)
        spread=max(float(np.ptp(np.r_[values,0.])),.01)
        for b in range(2):
            for m in range(3):
                y=m+(-.15 if b==0 else .15)
                v=mean[b,m]
                ax.barh(y,v,height=.25,color=colors[m] if b==0 else "white",
                        edgecolor=colors[m],hatch=None,linewidth=1)
                ax.text(v+np.sign(v or 1)*.025*spread,y,f"{v:.4f}",
                        ha="left" if v>=0 else "right",va="center",fontsize=8)
        ax.axvline(0,color=".4",lw=.7)
        ax.set(yticks=range(3),yticklabels=mods,ylim=(2.5,-.5),
               xlim=(min(0,float(values.min()))-.28*spread,max(0,float(values.max()))+.32*spread),
               title="Classification | Probability preservation" if h==0 else "Regression | Score preservation",
               xlabel="Mean Shapley of negative JS divergence (bits)" if h==0 else "Mean Shapley of negative absolute deviation")
        ax.grid(False);ax.set_axisbelow(True)
    fig.text(.5,.179,"Positive: helps preserve the full-input prediction. Negative: increases divergence in the averaged coalition comparisons.",
             ha="center",fontsize=8)
    fig.text(.5,.126,"Classification compares all class probabilities; regression compares raw scores. Mean over all 20 cases, both references.",
             ha="center",fontsize=8)
    fig.text(.5,.073,"The full-input prediction is a comparison target, not a true label. These values do not measure Macro-F1 or MAE performance.",
             ha="center",fontsize=8)
    export(fig,"global_02_prediction_preservation",axes,
           "How much does each modality preserve the full-input model's own probability distribution and regression score?",
           "Local utilities are negative Jensen-Shannon divergence in bits and negative absolute regression deviation from full input. Exact modality Shapley is computed per case and averaged; this equals Shapley of mean utilities. No label, accuracy, Macro-F1 or ground-truth MAE is used. Descriptive full-cohort means, no inferential error bars.")

    # 3. Direct outcomes of eight coalitions and single-modality replacement.
    order=[0,1,2,4,3,5,6,7]
    labels=["All reference","Text","Audio","Vision","Text + Audio","Text + Vision","Audio + Vision","Full input"]
    agree=np.zeros((2,8)); drift=np.zeros((2,8))
    flip=np.zeros((2,3)); deviation=np.zeros((2,3))
    for b,r in enumerate(refs):
        for c in range(8):
            agree[b,c]=100*np.mean([s["baselines"][r]["coalition_predictions"]["class_indices"][c]==
                                   s["prediction"]["class_index"] for s in samples])
            drift[b,c]=np.mean([s["baselines"][r]["coalition_predictions"]["regression_absolute_deviation"][c]
                               for s in samples])
        for m in range(3):
            flip[b,m]=100-agree[b,7^(1<<m)]
            deviation[b,m]=drift[b,7^(1<<m)]
    fig,axes=plt.subplots(2,2,figsize=(11.2,9.1),gridspec_kw={"height_ratios":[1.55,1]})
    fig.subplots_adjust(left=.15,right=.97,top=.835,bottom=.14,hspace=.62,wspace=.32)
    fig.suptitle("Input combinations and prediction changes",y=.975,fontsize=14,weight="bold")
    fig.text(.5,.934,"Named modalities use observed features; others receive reference features. Model weights and masks stay fixed.",
             ha="center",fontsize=8.5)
    fig.legend(handles=ref_legend,loc="upper center",bbox_to_anchor=(.5,.91),ncol=2)
    rc=("#4575A5","#B87D47")
    for h in range(2):
        ax=axes[0,h]
        v=agree if h==0 else drift
        for b in range(2):
            y=np.arange(8)+(-.14 if b==0 else .14)
            ax.barh(y,v[b,order],height=.24,color=rc[b] if b==0 else "white",
                    edgecolor=rc[b],hatch=None,linewidth=1)
        ax.set(yticks=range(8),yticklabels=labels,ylim=(7.5,-.5),
               title="Classification | Agreement with full input" if h==0 else "Regression | Deviation from full input",
               xlabel="Cases retaining original predicted class (%)" if h==0 else "Mean absolute prediction change")
        if h==0:ax.set_xlim(0,105)
        else:ax.set_xlim(left=0);ax.margins(x=.06)
        ax.grid(False);ax.set_axisbelow(True)
        ax=axes[1,h]
        v=flip if h==0 else deviation
        span=max(float(v.max()),.01)
        for b in range(2):
            for m in range(3):
                y=m+(-.14 if b==0 else .14)
                ax.barh(y,v[b,m],height=.24,color=rc[b] if b==0 else "white",
                        edgecolor=rc[b],hatch=None,linewidth=1)
                ax.text(v[b,m]+.025*span,y,f"{v[b,m]:.0f}%" if h==0 else f"{v[b,m]:.3f}",
                        ha="left",va="center",fontsize=8)
        ax.set(yticks=range(3),yticklabels=["Replace "+m for m in mods],ylim=(2.5,-.5),
               xlim=(0,span*1.26),
               title="Whole-modality replacement | Classification" if h==0 else "Whole-modality replacement | Regression",
               xlabel="Predicted-class flip rate (%)" if h==0 else "Mean absolute prediction change")
        ax.grid(False);ax.set_axisbelow(True)
    fig.text(.5,.083,"All 20 cases included. Full input has 100% agreement and zero deviation by definition, not by accuracy.",
             ha="center",fontsize=8)
    fig.text(.5,.052,"Bottom panels replace one modality while retaining both others; these effects are conditional and are not additive.",
             ha="center",fontsize=8)
    fig.text(.5,.022,"Sample 13 has all-zero raw vision; samples 07 and 18 have truncated text. Both references mainly differ in text.",
             ha="center",fontsize=8)
    export(fig,"global_03_coalitions_and_replacements",axes,
           "Which input combinations change the full model's decisions, and how sensitive are decisions to each single modality?",
           "Classification agreement/flip rates compare predictions only, not labels. Regression drift is an absolute change in predictions, not error to ground truth. All-reference still preserves length/masks. No population generalization or confidence intervals from the 20 supplied cases.")
    return rows


def write_attachment4_reading_guide(report, out):
    samples = report["samples"]
    translate = {"text": "文本", "audio": "语音", "vision": "视觉"}
    lines = [
        "# 附件四全量预测与解释结果",
        "",
        "覆盖对齐版本 01—20 全部样本。没有真实分类或回归标签；所有一致率、变化量和贡献均解释模型行为，不代表预测正确率或真实情感因果。",
        "",
        "## 文件索引",
        "",
        "- analysis_report.json：每条样本的完整预测、8 种模态组合、贡献、特征排名、干预验证，以及总体汇总。",
        "- attributions.npz：两种参考下的逐模态、逐时间槽位、逐特征维度的原始归因数组。",
        "- global_01_modality_reliance.png / .pdf：总体依赖份额及全部 20 个样本的分布。",
        "- global_02_prediction_preservation.png / .pdf：各模态对保持完整输入预测的平均 Shapley 贡献。",
        "- global_03_coalitions_and_replacements.png / .pdf：8 种模态组合及整模态替换后的预测变化。",
        "- individual_modalities.pdf：20 页，每页一个样本的模态贡献、份额和预测保持贡献。",
        "- individual_evidence.pdf：20 页，每页一个样本的时间槽位和特征维度证据。",
        "- individual_validation.pdf：20 页，每页一个样本的整模态、关键槽位、完整词替换验证与参考敏感性。",
        "- single_samples/：对应的 60 张单样本 PNG，可按样本编号查找。",
        "",
        "## 各类数字如何阅读",
        "",
        "1. 分类归因目标固定为原预测第一名 logit 减第二名 logit；正值支持这一原始分类优势，负值削弱它。回归归因正值推高分数、负值降低分数；降低分数不等于降低预测质量。",
        "2. 模态份额为每路 Shapley 绝对值除以三路绝对值之和；分母接近零时标为未定义。总体份额先逐样本归一化再等权平均，不是准确率贡献百分比。",
        "3. 预测保持贡献以完整输入预测为参照：分类使用负 JS 散度（bits），回归使用负绝对预测差。正值表示加入该模态有助于恢复完整输入预测，负值保留原样。两头单位不同，不直接比较数值大小。",
        "4. 模态 Shapley 包括全部有效槽位。时间图保留原 50 槽位，标出特殊 token 和 padding；特征维度排名及词排名只统计有效内容槽位。净贡献可相互抵消，绝对归因量表示敏感强度，不等于因果证据。",
        "5. 条件 Integrated Gradients 将模态贡献追溯到时间槽位和特征维度。文本 768 维是上下文 embedding 坐标；没有特征定义表时，音频 74 维、视觉 35 维也仅按维度编号解释。",
        "6. 整模态与槽位替换使用模型输入空间中的训练均值参考；零参考另作敏感性分析。原掩码保持不变。被替换内容代表指定干预，不等同真实自然缺失。",
        "7. 关键槽位与低分、20 次固定随机对照比较，保留不符合预期的样本。词验证将整个词的所有子词替换为 UNK 并重新编码全文，避免仅清零某个上下文向量造成误解。",
        "8. Spearman 与 Top-5 overlap 比较均值、零参考下的排名；归一化后的音视频训练均值接近零，因此这项敏感性主要反映文本参考变化。",
        "",
        "## 输入限制",
        "",
        "- 07、18 的文本超出缓存长度；解释仅覆盖保留的 48 个内容 token，不能追踪被截断的后文。",
        "- 13 的对齐视觉特征全为原始零值。保留原模型预处理后，其视觉归因表示模型对该输入的响应，不能声称找到了真实画面证据，也不代表原视频不存在。",
        "- 特殊 token 的音视频零输入按原模型处理；相关归因不映射为真实语音或视频片段。",
        "- 不进行 Whisper 或真实时间戳映射；time slot 指模型输入槽位，不是秒数。",
        "- 全部结果是本组 20 条样本的描述，不附总体人群置信区间。",
        "",
        "## 全量预测速查",
        "",
        "class 是检查点保存的类别值，不自动赋予正负面语义。主导模态按训练均值参考的绝对 Shapley 计算；并列会列出全部。",
        "",
        "| ID | 预测类别 | 三类概率（按检查点类别顺序） | 回归分数 | 分类主导模态 | 回归主导模态 |",
        "|---|---:|---|---:|---|---|",
    ]
    for sample in samples:
        p = sample["prediction"]
        baseline = sample["baselines"]["mean"]
        dominant = []
        for h in range(2):
            if not baseline["shares_defined"][h]:
                dominant.append("未定义")
            else:
                values = np.abs(np.asarray(baseline["phi"][h], dtype=float))
                inds = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-12))
                dominant.append(" / ".join(translate[MODALITIES[int(i)]] for i in inds))
        probs = ", ".join(f"{v:.4f}" for v in p["probabilities"])
        lines.append(f'| {sample["id"]} | {p["class_value"]} | {probs} | {p["regression"]:.4f} | {dominant[0]} | {dominant[1]} |')
    lines.extend(["", "精确贡献及控制实验以 JSON / NPZ 为准。图中深浅、正负、绝对值份额使用不同定义，请按坐标轴和注释阅读。", ""])
    (out / "结果阅读说明.md").write_text("\n".join(lines), encoding="utf-8")


# Attachment-4 source-media mapping. These functions deliberately keep cached
# 74/35-dimensional feature attribution separate from post-hoc lexical anchors.
def q3_media_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def q3_media_source_rows(record):
    """Exact value matches identify candidate source rows, never source times."""
    aligned_path = Path(record["file"])
    original_path = aligned_path.parent.parent / "非对齐版本" / aligned_path.name
    if not original_path.is_file():
        # Actual supplied archive calls this directory 未对齐版本.
        original_path = aligned_path.parent.parent / "未对齐版本" / aligned_path.name
    result = {"status": "unavailable", "source_time_mapping_status": "unavailable",
              "source_time_reason": "Supplied cached feature arrays contain no source intervals."}
    if not original_path.is_file():
        return result
    with aligned_path.open("rb") as stream:
        aligned = pickle.load(stream)
    with original_path.open("rb") as stream:
        original = pickle.load(stream)
    mapped = {}
    for modality in ("audio", "vision"):
        a, b = np.asarray(aligned[modality]), np.asarray(original[modality])
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        rows = {}
        for slot in np.flatnonzero(np.asarray(record["content_mask"], bool)):
            if not np.any(a[slot]):
                rows[str(slot)] = {"status": "raw_zero", "matching_source_rows": []}
                continue
            matches = np.flatnonzero(np.all(b == a[slot], axis=1)).tolist()
            rows[str(slot)] = {
                "status": "exact_array_equality" if matches else "no_exact_match",
                "matching_source_rows": matches,
                "unique_source_row": matches[0] if len(matches) == 1 else None}
        mapped[modality] = rows
    result.update(status="checked", source_file=str(original_path),
                  source_sha256=sha256(original_path), modalities=mapped)
    return result


def q3_media_word_mapping(record, alignment, arrays):
    """Split a slot's mass only by observed raw-character overlap; never by time."""
    offsets = record["token_alignment"]["offsets"]
    content = np.asarray(record["content_mask"], bool)
    words = [dict(w) for w in alignment["words"]]
    visible_end = int(record["token_alignment"]["visible_character_end"])
    attrs = {m: np.asarray(arrays[f"s{int(record['id']):02d}_mean_{m}"], np.float64)
             for m in MODALITIES}
    for m, value in attrs.items():
        if value.shape[:2] != (2, len(content)) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid saved attribution shape or values: {m}")
    slot_map, assigned = [], np.zeros(len(content), np.float64)
    for word in words:
        word.update(token_slots=[], slot_weights=[], represented=False,
                    full_word_represented=False, model_visible_char_spans=[],
                    attribution_mass={h: {m: 0.0 for m in MODALITIES} for h in HEADS},
                    attribution_net={h: {m: 0.0 for m in MODALITIES} for h in HEADS})
    for slot in np.flatnonzero(content):
        start, end = map(int, offsets[slot])
        overlaps = [(j, max(0, min(end, w["raw_char_span"][1]) -
                                  max(start, w["raw_char_span"][0])))
                    for j, w in enumerate(words)]
        overlaps = [(j, n) for j, n in overlaps if n > 0]
        total = sum(n for _, n in overlaps)
        links = []
        for j, count in overlaps:
            weight = count / total
            w = words[j]
            w["token_slots"].append(int(slot))
            w["slot_weights"].append(weight)
            w["model_visible_char_spans"].append(
                [max(start, w["raw_char_span"][0]), min(end, w["raw_char_span"][1])])
            assigned[slot] += weight
            links.append({"word_index": int(w["word_index"]), "weight": weight,
                          "start_s": w["start_s"], "end_s": w["end_s"],
                          "alignment_status": w["status"], "alignment_score": w["score"]})
            for h, head in enumerate(HEADS):
                for modality in MODALITIES:
                    w["attribution_mass"][head][modality] += float(
                        np.abs(attrs[modality][h, slot]).sum() * weight)
                    w["attribution_net"][head][modality] += float(
                        attrs[modality][h, slot].sum() * weight)
        slot_map.append({
            "slot": int(slot), "token": record["tokens"][slot],
            "raw_char_span": [start, end], "raw_fragment": record["raw_text"][start:end],
            "word_links": links,
            "time_mapping_status": "lexical_anchor" if links else "unavailable_nonlexical",
            "source_feature_time_status": "unavailable",
            "attribution_mass": {head: {m: float(np.abs(attrs[m][h, slot]).sum())
                                      for m in MODALITIES} for h, head in enumerate(HEADS)},
            "attribution_net": {head: {m: float(attrs[m][h, slot].sum())
                                     for m in MODALITIES} for h, head in enumerate(HEADS)}})
    for w in words:
        w["represented"] = bool(w["token_slots"])
        w["full_word_represented"] = bool(
            w["represented"] and w["raw_char_span"][1] <= visible_end)
        w["model_evidence_status"] = ("no_model_representation" if not w["represented"]
                                      else "partial_word_truncation" if not w["full_word_represented"]
                                      else "represented")
        w["evidence_status_by_modality"] = {}
        for m in MODALITIES:
            zeros = set(record["data_quality"]["raw_zero_content_slots"].get(m, []))
            zero_slots = [s for s in w["token_slots"] if s in zeros]
            if not w["represented"]:
                status = "unavailable_no_model_representation"
            elif not w["full_word_represented"]:
                status = "unavailable_partial_word_truncation"
            elif len(zero_slots) == len(w["token_slots"]):
                status = "unavailable_raw_zero_input"
            elif zero_slots:
                status = "needs_review_partial_raw_zero_input"
            elif m == "text":
                status = "raw_text_located"
            elif w["status"] != "aligned":
                status = "needs_review_alignment" if w["start_s"] is not None else "unavailable_alignment"
            elif m == "vision" and not w["frame_indices"]:
                status = "unavailable_no_video_frame"
            else:
                status = "lexical_temporal_anchor_available"
            w["evidence_status_by_modality"][m] = status
    # The numerical balance explicitly retains punctuation and any lexical gaps.
    balance = {}
    for h, head in enumerate(HEADS):
        balance[head] = {}
        for m in MODALITIES:
            total = float(np.abs(attrs[m][h, content]).sum())
            mapped = sum(w["attribution_mass"][head][m] for w in words)
            unlocalized = float((np.abs(attrs[m][h]).sum(-1) * content * (1-assigned)).sum())
            if not np.isclose(mapped+unlocalized, total, rtol=1e-9, atol=1e-9):
                raise AssertionError("Lexical mapping does not conserve content attribution mass")
            balance[head][m] = {"content_mass": total, "lexically_mapped_mass": mapped,
                               "unlocalized_mass": unlocalized,
                               "mapped_fraction": mapped/total if total > 0 else None}
    return words, slot_map, balance


def q3_media_export_asset(q1, args, video, alignment, word, modality, directory):
    """Export only accepted lexical intervals; frame choice uses actual PTS."""
    directory.mkdir(parents=True, exist_ok=True)
    if modality == "audio":
        start, end = float(word["start_s"]), float(word["end_s"])
        name = f"word{word['word_index']:03d}_{round(start*1e6)}_{round(end*1e6)}.wav"
        target = directory / name
        if not target.is_file() or target.stat().st_size < 44:
            q1.run([args.ffmpeg, "-nostdin", "-v", "error", "-threads", "2",
                    "-y", "-copyts", "-i", str(video), "-map", "0:a:0", "-vn",
                    "-ac", "1", "-af",
                    f"aresample=16000:async=1:first_pts=0,atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(target)])
        import wave
        with wave.open(str(target), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
        if abs(duration-(end-start)) > 0.002:
            raise ValueError("Exported audio clip has unexpected duration")
        return {"file": str(target), "kind": "audio_wav", "start_s": start, "end_s": end,
                "duration_s": duration, "sample_rate": 16000,
                "interpretation": "Source audio in a post-hoc lexical interval; not a recovered cached-feature window."}
    pts, ends = alignment["video_pts_s"], alignment["video_frame_end_s"]
    indices = word["frame_indices"]
    midpoint = (float(word["start_s"])+float(word["end_s"]))/2
    chosen = min(indices, key=lambda i: (abs((pts[i]+ends[i])/2-midpoint), i))
    target = directory / f"frame{chosen:06d}.png"
    if not target.is_file() or target.stat().st_size == 0:
        q1.run([args.ffmpeg, "-nostdin", "-v", "error", "-threads", "2", "-y",
                "-i", str(video), "-map", "0:v:0", "-vf", f"select=eq(n\\,{chosen})",
                "-fps_mode", "passthrough", "-frames:v", "1", "-threads", "1", str(target)])
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError("Keyframe export produced no image")
    return {"file": str(target), "kind": "video_png", "decoded_frame_index": int(chosen),
            "frame_pts_s": float(pts[chosen]), "frame_end_s": float(ends[chosen]),
            "overlapping_decoded_frame_indices": list(indices),
            "selection": "Frame whose presentation interval midpoint is nearest the aligned word midpoint.",
            "interpretation": "Visual context at a post-hoc lexical interval; cached 35-D feature source time is unavailable."}


def q3_build_attachment4_evidence(args, report_path, out_dir):
    """Map saved Attachment-4 explanations to raw text and audited media anchors.

    args: device, ffmpeg, ffprobe; optional alignment_cache, alignment_model,
    alignment_min_score, evidence_topk, media_sample_ids. This does not train,
    change a cached prediction feature, or interpolate absent timestamps.
    """
    import csv
    from types import SimpleNamespace
    import importlib.metadata
    out_dir = Path(out_dir).resolve()
    allowed = (ROOT / "question3").resolve()
    if out_dir != allowed and allowed not in out_dir.parents:
        raise ValueError("All media results must stay inside question3")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    array_path = report_path.parent / "attributions.npz"
    q1_path = ROOT / "question1" / "first_question.py"
    spec = importlib.util.spec_from_file_location("q3_existing_question1_alignment", q1_path)
    q1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q1)
    minimum = float(getattr(args, "alignment_min_score", 0.5))
    topk = int(getattr(args, "evidence_topk", 3))
    if not 0 <= minimum <= 1 or topk < 1:
        raise ValueError("Invalid alignment score threshold or evidence_topk")
    model_name = getattr(args, "alignment_model", "WAV2VEC2_ASR_BASE_960H")
    cache = Path(getattr(args, "alignment_cache", ROOT / ".cache" / "alignment_models"))
    # The reused q1 character aligner expects the torchaudio model interface.
    weight = cache / "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
    if model_name != "WAV2VEC2_ASR_BASE_960H" or not weight.is_file():
        raise FileNotFoundError("Offline q1 alignment requires cached WAV2VEC2_ASR_BASE_960H weights")
    model_hash = sha256(weight)
    q1_hash = sha256(q1_path)
    signature_base = {"schema_version": 2, "model": model_name,
                      "model_sha256": model_hash, "q1_script_sha256": q1_hash,
                      "min_score": minimum, "interpolation": "none"}
    samples = report["samples"]
    selected_ids = getattr(args, "media_sample_ids", None)
    if selected_ids:
        selected_ids = {str(i).zfill(2) for i in selected_ids}
        samples = [s for s in samples if str(s["id"]).zfill(2) in selected_ids]
        if len(samples) != len(selected_ids):
            raise ValueError("Unknown media_sample_ids")
    model = metadata = None
    outputs, csv_rows, issues = [], [], []
    arrays = np.load(array_path, allow_pickle=False)
    try:
        for sample in samples:
            sid = str(sample["id"]).zfill(2)
            log(f"Attachment4 {sid}: source-media lexical anchors")
            video = Path(sample["file"]).parent / "videos" / f"{sid}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            video_hash = sha256(video)
            signature = dict(signature_base, source_sha256=video_hash, raw_text=sample["raw_text"])
            sample_dir = out_dir / "samples" / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            align_path = sample_dir / "alignment.json"
            alignment = None
            if align_path.is_file():
                cached = json.loads(align_path.read_text(encoding="utf-8"))
                if cached.get("cache_signature") == signature and cached.get("error") is None:
                    alignment = cached
            if alignment is None:
                if model is None:
                    from whisperx.alignment import load_align_model
                    model, metadata = load_align_model("en", str(args.device),
                                                       model_name=model_name, model_dir=str(cache))
                    model.eval()
                row = {"video_id": "", "clip_id": sid, "sample_id": sid,
                       "text": sample["raw_text"], "label": None, "annotation": None}
                options = SimpleNamespace(device=str(args.device), min_score=minimum,
                                          ffmpeg=str(args.ffmpeg), ffprobe=str(args.ffprobe))
                alignment = q1.process(row, video.parent, model, metadata, options)
                alignment.update(cache_signature=signature, source_video=str(video))
                q3_media_write_json(align_path, alignment)
            words, slots, balance = q3_media_word_mapping(sample, alignment, arrays)
            provenance = q3_media_source_rows(sample)
            selected = {}
            for head in HEADS:
                selected[head] = {}
                for modality in MODALITIES:
                    ranked = sorted((w for w in words if w["represented"]),
                                    key=lambda w: (-w["attribution_mass"][head][modality],
                                                   w["word_index"]))[:topk]
                    selected[head][modality] = []
                    for rank, word in enumerate(ranked, 1):
                        status = word["evidence_status_by_modality"][modality]
                        item = {"rank": rank, "word_index": word["word_index"], "word": word["word"],
                                "raw_char_span": word["raw_char_span"], "token_slots": word["token_slots"],
                                "model_visible_char_spans": word["model_visible_char_spans"],
                                "attribution_mass": word["attribution_mass"][head][modality],
                                "attribution_net": word["attribution_net"][head][modality],
                                "evidence_status": status, "alignment_status": word["status"],
                                "alignment_score": word["score"], "alignment_flags": word["flags"],
                                "start_s": word["start_s"], "end_s": word["end_s"],
                                "cached_feature_source_time_status": "unavailable" if modality != "text" else "not_applicable",
                                "asset": None}
                        if modality != "text" and status == "lexical_temporal_anchor_available":
                            try:
                                item["asset"] = q3_media_export_asset(
                                    q1, args, video, alignment, word, modality,
                                    sample_dir / ("audio_clips" if modality == "audio" else "keyframes"))
                                item["asset"]["file"] = str(Path(item["asset"]["file"]).relative_to(out_dir))
                            except Exception as error:
                                item["evidence_status"] = "unavailable_asset_export"
                                item["asset_error"] = f"{type(error).__name__}: {error}"
                                issues.append({"sample_id": sid, "head": head, "modality": modality,
                                               "word_index": word["word_index"], "error": item["asset_error"]})
                        selected[head][modality].append(item)
                        csv_rows.append({"sample_id": sid, "head": head, "modality": modality,
                                         **{k: v for k, v in item.items() if k != "asset"},
                                         "asset_file": item["asset"]["file"] if item["asset"] else None})
            output = {
                "schema_version": 2, "sample_id": sid, "source_video": str(video),
                "source_sha256": video_hash, "raw_text": sample["raw_text"],
                "prediction": sample["prediction"],
                "modality_phi": sample["baselines"]["mean"]["phi"],
                "modality_shares": sample["baselines"]["mean"]["shares"],
                "dominant_modalities": {
                    h: MODALITIES[int(np.argmax(sample["baselines"]["mean"]["shares"][i]))]
                    if sample["baselines"]["mean"]["shares_defined"][i] else None
                    for i, h in enumerate(HEADS)},
                "data_quality": sample["data_quality"],
                "alignment_file": str(align_path.relative_to(out_dir)),
                "alignment_error": alignment.get("error"),
                "acoustic_text_match_ratio": alignment.get("text_match_ratio"),
                "source_feature_provenance": provenance,
                "interpretation": {
                    "text": "Exact raw-character locations of saved model-visible BERT content slots.",
                    "audio_vision": "Post-hoc lexical time anchors obtained by q1 WhisperX/CTC alignment; temporal correspondence only.",
                    "limitation": "Cached 74-D audio / 35-D visual features have no source intervals. Exact matched source row indices do not establish source time. Media exports are auditable contextual anchors, not certified source windows of attributed feature values.",
                    "truncation": "Only saved content slots receive attribution. Unencoded suffix and special tokens are excluded.",
                    "raw_zero": "Raw-zero modality inputs retain numerical model attribution but are not observed media evidence.",
                    "review_policy": "No interpolated times; low score, partial alignment, unusual normalization, transcript mismatch and long intervals require review."},
                "words": words, "content_slot_mapping": slots, "attribution_mapping_balance": balance,
                "top_evidence": selected}
            q3_media_write_json(sample_dir / "evidence.json", output)
            outputs.append(output)
            if alignment.get("error"):
                issues.append({"sample_id": sid, "error": alignment["error"]})
    finally:
        arrays.close()
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {
        "schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path), "source_report_sha256": sha256(report_path),
        "source_attributions_sha256": sha256(array_path), "alignment": signature_base,
        "evidence_topk": topk, "sample_count": len(outputs),
        "versions": {p: importlib.metadata.version(p)
                     for p in ("whisperx", "torch", "torchaudio", "numpy", "num2words")},
        "cached_feature_source_time_recovered": False,
        "all_outputs_have_full_predictions_and_explanations": len(outputs) == len(samples),
        "all_audio_vision_evidence_has_certified_original_feature_times": False,
        "media_interpretation": "post-hoc lexical temporal correspondence; original cached-feature time provenance unavailable",
        "samples": outputs, "issues": issues}
    q3_media_write_json(out_dir / "attachment4_media_evidence.json", summary)
    if csv_rows:
        fields = sorted({k for row in csv_rows for k in row})
        with (out_dir / "attachment4_key_evidence.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
    log(f"Attachment4 media evidence: {len(outputs)} samples, {len(issues)} errors; {out_dir}")
    return summary




# ---- Q3 completion: labelled validation and raw-evidence delivery. ----
def q3_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def q3_write_csv(path, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def q3_prepare_runtime(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    model, metadata = load_experiment_model(args.checkpoint, device=args.device)
    model.eval().requires_grad_(False)
    return model, metadata


def q3_validation_data(args):
    with args.validation_dataset.open("rb") as stream:
        data = pickle.load(stream)
    if set(("train", "valid", "test")) - set(data):
        raise ValueError("Expected official train/valid/test splits")
    for split_name in ("train", "valid", "test"):
        target = np.asarray(data[split_name]["regression_labels"]).reshape(-1)
        classes = np.asarray(data[split_name]["classification_labels"]).reshape(-1)
        np.testing.assert_array_equal(classes, np.where(target < 0, 0, np.where(target > 0, 2, 1)))
    if len(data["valid"]["id"]) != 728:
        raise ValueError("Confirmed validation cohort must contain 728 cases")
    return data


def q3_metrics(y_class, y_regression, logits, regression):
    from multi_fusion_model.weighted_sum_fusion import classification_metrics, regression_metrics
    logits, regression = np.asarray(logits), np.asarray(regression).reshape(-1)
    cls = classification_metrics(np.asarray(y_class), logits.argmax(-1), logits.shape[-1])
    reg = regression_metrics(np.asarray(y_regression, np.float64), regression.astype(np.float64))
    probabilities = softmax_probabilities(logits)
    cls["cross_entropy"] = float(-np.log(np.maximum(probabilities[np.arange(len(logits)),
                                                               np.asarray(y_class, int)], 1e-300)).mean())
    return {"sample_count": len(logits), **cls, **reg}


def q3_tokenizer():
    from transformers import BertTokenizerFast
    info = torch.load(DEFAULT_ADAPTER, map_location="cpu", weights_only=True)
    return BertTokenizerFast.from_pretrained(str(info["model_dir"]), local_files_only=True)


def q3_raw_sample(split, index):
    sample = {m: np.asarray(split[m][index]) for m in MODALITIES}
    sample.update(id=str(split["id"][index]), raw_text=str(split["raw_text"][index]),
                  text_bert=np.asarray(split["text_bert"][index]))
    return sample


def q3_baseline(x, masks, means, name):
    output = {}
    for m in MODALITIES:
        vector = means[m] if name == "mean" else np.zeros(DIMENSIONS[m], np.float32)
        output[m] = torch.as_tensor(vector, device=x[m].device).reshape(1, 1, -1).expand_as(x[m]).clone()
        output[m].masked_fill_(~masks[m].unsqueeze(-1), 0)
    return output


def q3_cohort_coalitions(model, dataset, means, baseline_name, device, batch_cases=16):
    all_logits, all_regression = [], []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_cases):
            items = [dataset[i] for i in range(start, min(start + batch_cases, len(dataset)))]
            x = {m: torch.stack([item["features"][m] for item in items]).to(device) for m in MODALITIES}
            masks = {m: torch.stack([item["masks"][m] for item in items]).to(device) for m in MODALITIES}
            base = q3_baseline(x, masks, means, baseline_name)
            features = {m: torch.cat([x[m] if c & (1 << j) else base[m] for c in range(8)])
                        for j, m in enumerate(MODALITIES)}
            expanded_masks = {m: masks[m].repeat(8, 1) for m in MODALITIES}
            output = model(features, expanded_masks)
            all_logits.append(output["classification_logits"].cpu().numpy().reshape(8, len(items), -1).transpose(1, 0, 2))
            all_regression.append(output["regression_prediction"].cpu().numpy().reshape(8, len(items)).T)
    return np.concatenate(all_logits).astype(np.float64), np.concatenate(all_regression).astype(np.float64)


def q3_select_diagnostic_cases(records, seed):
    selected = []
    for truth in range(3):
        for correct in (True, False):
            pool = sorted([r for r in records if r["truth"]["class_index"] == truth and
                           r["evaluation"]["classification_correct"] == correct],
                          key=lambda r: (r["evaluation"]["regression_absolute_error"], r["dataset_id"]))
            if len(pool) < 10:
                raise ValueError("Confirmed class/correctness stratum contains fewer than 10 cases")
            bins = np.array_split(np.arange(len(pool)), 3)
            rng = np.random.default_rng(np.random.SeedSequence([seed, truth, int(correct), 731]))
            for error_bin, (indices, count) in enumerate(zip(bins, (3, 3, 4))):
                mandatory = [int(indices[-1])] if error_bin == 2 else []
                available = [int(v) for v in indices if int(v) not in mandatory]
                chosen = mandatory + rng.choice(available, size=count-len(mandatory), replace=False).astype(int).tolist()
                for j in sorted(chosen):
                    record = pool[j]
                    selected.append({"id": record["id"], "dataset_id": record["dataset_id"],
                                     "valid_index": record["valid_index"], "true_class": truth,
                                     "classification_correct": correct,
                                     "regression_absolute_error": record["evaluation"]["regression_absolute_error"],
                                     "regression_error_band_within_stratum": ("low", "middle", "high")[error_bin],
                                     "stratum_size": len(pool), "seed": seed,
                                     "mandatory_largest_error_in_stratum": j in mandatory})
    selected.sort(key=lambda r: r["valid_index"])
    if len(selected) != 60 or len({r["valid_index"] for r in selected}) != 60:
        raise AssertionError("Need 60 distinct diagnostic cases")
    return selected



def q3_validation_prepare(args, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cohort_path = out / "cohort.json"
    identity = {"checkpoint_sha256": sha256(args.checkpoint),
                "validation_dataset_sha256": sha256(args.validation_dataset),
                "reference_dataset_sha256": sha256(args.reference_dataset), "seed": args.seed}
    if cohort_path.exists():
        saved = json.loads(cohort_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Existing validation output belongs to different inputs/settings")
        log("Reusing verified complete 728-case coalition results")
        return saved
    data = q3_validation_data(args)
    valid = data["valid"]
    model, metadata = q3_prepare_runtime(args)
    initial_state = state_hash(model)
    means, provenance = training_means(args.reference_dataset, metadata)
    tokenizer = q3_tokenizer()
    public = predict_split(model, valid, metadata, batch_size=32)
    logits = np.asarray(public["classification_logits"], np.float64)
    regression = np.asarray(public["regression_prediction"], np.float64).reshape(-1)
    order = np.argsort(-logits, axis=1, kind="stable")
    y_class = np.asarray(valid["classification_labels"], int).reshape(-1)
    y_regression = np.asarray(valid["regression_labels"], np.float64).reshape(-1)
    metrics = q3_metrics(y_class, y_regression, logits, regression)
    old_metrics_path = args.checkpoint.parent / "metrics.json"
    previous = json.loads(old_metrics_path.read_text())["valid"]
    differences = {key: abs(metrics[key] - previous[key]) for key in ("accuracy", "macro_f1", "mae", "rmse", "pearson")}
    if any(value > 1e-5 for value in differences.values()):
        raise AssertionError("Full validation inference no longer matches saved checkpoint results: " + str(differences))
    dataset = FeatureDataset(valid, resolve_masks(valid), metadata["normalizers"],
                             class_values=metadata["class_values"], require_labels=False)
    records = []
    for i in range(len(valid["id"])):
        raw = q3_raw_sample(valid, i)
        tokens, content, words, alignment = align_words(raw, tokenizer)
        valid_mask = np.asarray(valid["text_bert"][i, 1], bool)
        zero = [m for m in MODALITIES if np.all(np.asarray(valid[m][i])[content] == 0)]
        winner, runner = map(int, order[i, :2])
        records.append({
            "id": f"v{i:04d}", "valid_index": i, "dataset_id": str(valid["id"][i]),
            "raw_text": raw["raw_text"], "tokens": tokens, "valid_mask": valid_mask.tolist(),
            "content_mask": content.tolist(), "words": words, "token_alignment": alignment,
            "truth": {"class_index": int(y_class[i]), "class_value": int(y_class[i]),
                      "polarity": ("negative", "neutral", "positive")[y_class[i]], "regression": float(y_regression[i])},
            "prediction": {"logits": logits[i].tolist(), "probabilities": softmax_probabilities(logits[i]).tolist(),
                           "class_index": winner, "runnerup_index": runner, "class_value": winner,
                           "polarity": ("negative", "neutral", "positive")[winner],
                           "regression": float(regression[i]), "margin": float(logits[i,winner]-logits[i,runner])},
            "evaluation": {"classification_correct": bool(winner == y_class[i]),
                           "regression_signed_error": float(regression[i]-y_regression[i]),
                           "regression_absolute_error": float(abs(regression[i]-y_regression[i]))},
            "data_quality": {"input_truncated": alignment["input_truncated"],
                             "all_zero_content_modalities": zero,
                             "raw_audio_video_available": False},
            "baselines": {}})
    for name in ("mean", "zero"):
        log(f"Validation {name}: exact coalitions for all 728 cases")
        c_logits, c_reg = q3_cohort_coalitions(model, dataset, means, name, args.device)
        np.testing.assert_allclose(c_logits[:, 7], logits, atol=4e-5, rtol=2e-5)
        np.testing.assert_allclose(c_reg[:, 7], regression, atol=4e-5, rtol=2e-5)
        for i, record in enumerate(records):
            winner, runner = order[i, :2]
            if int(np.argmax(c_logits[i,7])) != int(winner):
                raise AssertionError("Batch rounding changed prediction")
            scores = np.stack([c_logits[i,:,winner]-c_logits[i,:,runner], c_reg[i]], 1)
            phi = shapley_from_coalitions(scores)
            denominator = np.abs(phi).sum(1)
            defined = denominator > 1e-6
            shares = [(np.abs(phi[h])/denominator[h]).tolist() if defined[h] else [None]*3 for h in range(2)]
            record["baselines"][name] = {
                "phi": phi.tolist(), "shares": shares, "shares_defined": defined.tolist(),
                "share_denominator": denominator.tolist(), "coalitions": scores.tolist(),
                **coalition_behavior(scores, c_logits[i], c_reg[i], int(winner))}
            cp = record["baselines"][name]["coalition_predictions"]
            probs = np.asarray(cp["probabilities"])
            cp["ground_truth_cross_entropy"] = (-np.log(np.maximum(probs[:, y_class[i]], 1e-300))).tolist()
            cp["ground_truth_regression_absolute_error"] = np.abs(c_reg[i]-y_regression[i]).tolist()
    selected = q3_select_diagnostic_cases(records, args.seed)
    selected_ids = {r["id"] for r in selected}
    for record in records:
        record["selected_for_detailed_analysis"] = record["id"] in selected_ids
    summary = {
        "schema": "question3_labelled_validation_v1", "identity": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "sample_count": len(records),
        "metrics": metrics, "checkpoint_selected_epoch": int(metadata["selected_epoch"]),
        "validation_used_for_model_selection": True,
        "class_semantics": {"0": "negative", "1": "neutral", "2": "positive",
                            "evidence": "All supplied train/valid/test classes exactly match regression sign"},
        "existing_prediction_check": {"passed": True, "absolute_metric_differences": differences},
        "reference": provenance, "reference_means": {m: means[m].tolist() for m in MODALITIES},
        "scope": "All 728 labelled validation cases; descriptive same-checkpoint analysis",
        "selection": {"count": 60, "seed": args.seed,
                      "rule": "True class x original prediction correctness: 10 per each of 6 strata. Within each stratum sort by absolute regression error, split into three equal-count rank bands and select 3/3/4; highest band includes maximum error. Remaining selections are seeded without replacement.",
                      "interpretation": "Balanced diagnostic subset; unweighted detail averages do not estimate the full 728-case cohort.",
                      "selected": selected},
        "input_limitations": {"truncated_count": sum(r["data_quality"]["input_truncated"] for r in records),
                             "raw_audio_video_available": False,
                             "text_source": "Supplied raw_text; all saved BERT IDs/masks/types verified against re-tokenization"},
        "frozen_model_state_sha256": initial_state, "records": records}
    if state_hash(model) != initial_state:
        raise AssertionError("Model weights changed")
    q3_write_json(cohort_path, summary)
    q3_write_json(out / "metrics.json", {k: v for k,v in summary.items() if k not in ("records", "reference_means")})
    q3_write_csv(out / "selected_samples.csv", selected)
    q3_write_csv(out / "all_predictions.csv", [{
        "id": r["id"], "dataset_id": r["dataset_id"], "valid_index": r["valid_index"],
        "true_polarity": r["truth"]["polarity"], "predicted_polarity": r["prediction"]["polarity"],
        "true_class": r["truth"]["class_index"], "predicted_class": r["prediction"]["class_index"],
        "true_intensity": r["truth"]["regression"], "predicted_intensity": r["prediction"]["regression"],
        "probabilities": r["prediction"]["probabilities"], **r["evaluation"],
        "selected_for_detailed_analysis": r["selected_for_detailed_analysis"],
        "input_truncated": r["data_quality"]["input_truncated"], "raw_text": r["raw_text"]} for r in records])
    q3_write_csv(out / "modality_contributions.csv", [{
        "id": r["id"], "baseline": b, "head": head, "modality": m,
        "signed_shapley": r["baselines"][b]["phi"][h][j],
        "absolute_share": r["baselines"][b]["shares"][h][j],
        "prediction_preservation_shapley": r["baselines"][b]["prediction_preservation_phi"][h][j],
        "classification_correct": r["evaluation"]["classification_correct"],
        "true_class": r["truth"]["class_index"]}
        for r in records for b in ("mean", "zero") for h,head in enumerate(HEADS) for j,m in enumerate(MODALITIES)])
    del data, dataset, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary



def q3_validation_details(args, out):
    import copy
    out = Path(out)
    cohort = json.loads((out / "cohort.json").read_text(encoding="utf-8"))
    data = q3_validation_data(args)
    valid = data["valid"]
    means = {m: np.asarray(cohort["reference_means"][m], np.float32) for m in MODALITIES}
    model, metadata = q3_prepare_runtime(args)
    initial_state = state_hash(model)
    tokenizer = q3_tokenizer()
    adapter = load_existing_adapter(DEFAULT_ADAPTER, args.device)
    jobs = [r for r in cohort["records"] if r["selected_for_detailed_analysis"]]
    jobs = [r for j,r in enumerate(jobs) if j % args.worker_count == args.worker_index]
    for number, original_record in enumerate(jobs):
        path = out / "details" / original_record["id"]
        path.mkdir(parents=True, exist_ok=True)
        identity = {"cohort_identity": cohort["identity"], "id": original_record["id"],
                    "ig_steps": args.ig_steps, "max_steps": args.max_steps,
                    "random_repeats": args.random_repeats, "top_words": args.top_words}
        if (path / "analysis.json").exists():
            cached = json.loads((path / "analysis.json").read_text(encoding="utf-8"))
            if cached.get("detail_identity") == identity and (path / "attributions.npz").exists():
                log("Resume: " + original_record["id"] + " already complete")
                continue
            raise ValueError("Existing detail settings mismatch")
        started = time.monotonic()
        record = copy.deepcopy(original_record)
        sample = q3_raw_sample(valid, record["valid_index"])
        split = normalize_singleton(sample)
        x, masks = feature_input(split, metadata, args.device)
        content = np.asarray(record["content_mask"], bool)
        winner, runner = record["prediction"]["class_index"], record["prediction"]["runnerup_index"]
        full_scores = np.asarray([record["prediction"]["margin"], record["prediction"]["regression"]])
        arrays = {}
        for baseline_name in ("mean", "zero"):
            base = q3_baseline(x, masks, means, baseline_name)
            scores = coalition_scores(model, x, base, masks, winner, runner)
            np.testing.assert_allclose(scores, record["baselines"][baseline_name]["coalitions"], atol=5e-5, rtol=3e-5)
            attribution, convergence = conditional_ig(model, x, base, masks, winner, runner, scores, args)
            totals = np.stack([attribution[m].sum((1,2)) for m in MODALITIES], axis=1)
            phi = np.asarray(record["baselines"][baseline_name]["phi"])
            record["baselines"][baseline_name].update(
                rankings=ranking_summary(attribution, content), convergence=convergence,
                integrated_modality_totals=totals.tolist(), completeness_error=(totals-phi).tolist())
            for m in MODALITIES:
                arrays[baseline_name + "_" + m] = attribution[m]
            if baseline_name == "mean":
                mean_attribution, mean_base = attribution, base
        record["reference_stability"] = sample_reference_stability(record)
        record["deletion"] = deletion_check(model, x, mean_base, masks, winner, runner,
                                           full_scores, mean_attribution, content, args, record["valid_index"] + 1)
        record["word_checks"] = word_validation(model, x, masks, metadata, split, record["words"],
                                               mean_attribution, adapter, winner, runner, full_scores,
                                               args, record["valid_index"] + 1)
        # Ground-truth effects are separate from fixed-output changes.
        for check in record["word_checks"]["candidates"]:
            check["ground_truth_class_correct"] = check["class_index"] == record["truth"]["class_index"]
            check["ground_truth_regression_absolute_error"] = abs(check["regression"]-record["truth"]["regression"])
        record["detail_identity"] = identity
        record["elapsed_seconds"] = time.monotonic()-started
        record["all_integrations_converged"] = all(c["passed"] for b in record["baselines"].values() for c in b["convergence"])
        with (path / "attributions.npz").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        q3_write_json(path / "analysis.json", record)
        log(f"Validation detail {number+1}/{len(jobs)}: {record['id']} finished in {record['elapsed_seconds']:.1f}s")
    if state_hash(model) != initial_state:
        raise AssertionError("Frozen validation model changed")
    del data, model, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def q3_validation_summary(out):
    out = Path(out)
    cohort = json.loads((out/"cohort.json").read_text(encoding="utf-8"))
    records = cohort["records"]
    details = [json.loads((out/"details"/r["id"]/"analysis.json").read_text(encoding="utf-8"))
               for r in records if r["selected_for_detailed_analysis"]]
    if len(details) != 60:
        raise AssertionError("Need all 60 detailed cases")
    y = np.asarray([r["truth"]["class_index"] for r in records])
    yr = np.asarray([r["truth"]["regression"] for r in records])
    correct = np.asarray([r["evaluation"]["classification_correct"] for r in records], bool)
    summary = {"sample_count":len(records), "diagnostic_count":len(details), "baselines":{},
               "detail_scope":"60 deliberately balanced diagnostic cases; no full-cohort extrapolation",
               "all_integrations_converged":all(r["all_integrations_converged"] for r in details)}
    combination_rows, removal_rows, group_rows, detail_effects = [], [], [], []
    for b in ("mean","zero"):
        for coalition in range(8):
            cp = [r["baselines"][b]["coalition_predictions"] for r in records]
            logits = np.asarray([p["logits"][coalition] for p in cp])
            regression = np.asarray([p["regression"][coalition] for p in cp])
            row = {"baseline":b,"coalition":coalition,
                   "modalities":[m for j,m in enumerate(MODALITIES) if coalition & (1<<j)],
                   **q3_metrics(y,yr,logits,regression),
                   "agreement_with_full_prediction":float(np.mean(logits.argmax(-1)==
                            np.asarray([r["prediction"]["class_index"] for r in records])))}
            combination_rows.append(row)
        for j,m in enumerate(MODALITIES):
            c=7^(1<<j)
            predictions=np.asarray([r["baselines"][b]["coalition_predictions"]["class_indices"][c] for r in records])
            after_reg=np.asarray([r["baselines"][b]["coalition_predictions"]["regression"][c] for r in records])
            full_reg=np.asarray([r["prediction"]["regression"] for r in records])
            removal_rows.append({
                "baseline":b,"modality":m,"wrong_to_correct_count":int(np.sum(~correct & (predictions==y))),
                "originally_wrong_count":int((~correct).sum()),
                "correct_to_wrong_count":int(np.sum(correct & (predictions!=y))),
                "originally_correct_count":int(correct.sum()),
                "regression_error_change_mean":float(np.mean(np.abs(after_reg-yr)-np.abs(full_reg-yr))),
                "regression_error_change_wrong_group":float(np.mean((np.abs(after_reg-yr)-np.abs(full_reg-yr))[~correct])),
                "wrong_to_correct_ids":[r["id"] for i,r in enumerate(records) if not correct[i] and predictions[i]==y[i]],
                "correct_to_wrong_ids":[r["id"] for i,r in enumerate(records) if correct[i] and predictions[i]!=y[i]]})
        for h,head in enumerate(HEADS):
            for group,mask in [("all",np.ones(len(records),bool)),("correct",correct),("incorrect",~correct)] + [
                    (f"true_class_{c}",y==c) for c in range(3)]:
                selected=[r for i,r in enumerate(records) if mask[i] and r["baselines"][b]["shares_defined"][h]]
                shares=np.asarray([r["baselines"][b]["shares"][h] for r in selected])
                group_rows.append({"baseline":b,"head":head,"group":group,"sample_count":int(mask.sum()),
                                   "defined_share_count":len(selected),
                                   "mean_shares":shares.mean(0).tolist() if len(selected) else [None]*3,
                                   "median_shares":np.median(shares,axis=0).tolist() if len(selected) else [None]*3})
    for r in details:
        for h,head in enumerate(HEADS):
            for group in ("all",*MODALITIES):
                d=r["deletion"][head][group]
                k=min(4,max(d["counts"]))
                pos=d["counts"].index(k)
                detail_effects.append({"id":r["id"],"head":head,"group":group,"replaced_slots":k,
                      "top_absolute_change":d["top"]["absolute_change"][pos],
                      "low_absolute_change":d["low"]["absolute_change"][pos],
                      "random_mean_absolute_change":d["random"]["absolute_change_mean"][pos],
                      "top_minus_random":d["top"]["absolute_change"][pos]-d["random"]["absolute_change_mean"][pos],
                      "top_minus_low":d["top"]["absolute_change"][pos]-d["low"]["absolute_change"][pos],
                      "classification_correct":r["evaluation"]["classification_correct"]})
    summary.update(combinations=combination_rows,whole_modality_replacements=removal_rows,
                   share_groups=group_rows,detail_occlusion=detail_effects,
                   max_absolute_integral_error=max(abs(float(v)) for r in details for b in r["baselines"].values()
                       for row in b["completeness_error"] for v in row),
                   integration_edge_count=sum(len(b["convergence"]) for r in details for b in r["baselines"].values()))
    cm=np.asarray(cohort["metrics"]["confusion_matrix"]); off=cm.copy(); np.fill_diagonal(off,0)
    a,b=map(int,np.unravel_index(np.argmax(off),off.shape))
    conclusions=[
        f"附件二验证集共728条，当前固定模型分类准确率为{cohort['metrics']['accuracy']:.4%}，Macro-F1为{cohort['metrics']['macro_f1']:.4f}，强度预测MAE为{cohort['metrics']['mae']:.4f}。",
        f"混淆最集中的方向是真实类别{a}被预测为类别{b}，共{off[a,b]}条；类别0/1/2分别为负向/中性/正向。",
        "该验证集参与过检查点及适配器选择；本次属于固定模型验证分析，不是独立测试集泛化证明。"]
    for row in removal_rows:
        if row["baseline"]=="mean":
            conclusions.append(f"训练均值参考下，整体替换{row['modality']}后，原{row['originally_wrong_count']}条分类错误中有{row['wrong_to_correct_count']}条改判正确，同时原{row['originally_correct_count']}条正确预测中有{row['correct_to_wrong_count']}条变错。该结果说明特定输入替换与模型错误的关联，不能直接证明原模态内容是错误的因果来源。")
    for head in HEADS:
        values=[r for r in detail_effects if r["head"]==head and r["group"]=="all"]
        differences=np.asarray([r["top_minus_random"] for r in values])
        conclusions.append(f"60条平衡诊断样本中，{head}目标的联合高归因槽位替换在{int((differences>0).sum())}/60条样本上引起了比随机对照更大的绝对输出变化；高归因与随机对照的平均变化差为{differences.mean():.6f}。保留全部反例，不将本组比例外推至728条总体。")
    conclusions += [
        "60条样本按真实类别、分类对错及回归误差分层选取，未根据最终归因图或干预效果挑选。",
        "原文和分词映射可复核；验证集缺少原始音视频，因此音视频证据仅定位到缓存槽位，不生成没有来源的语音时段或画面。",
        "词替换通过原有编码器重新编码全文，可能改变其他上下文位置；特征均值替换和UNK替换可能偏离自然输入分布。",
        "不自动把负贡献解释为错误因素；回归贡献降低分数不等于降低预测质量。"]
    error_rows, case_rows, error_notes = q3_error_diagnostics(records,details,out)
    conclusions.extend(error_notes)
    summary["wrong_vs_true_error_contrast_count"]=len(error_rows)//2
    summary["diagnostic_case_error_count"]=len(case_rows)
    conclusions=[q3_chinese_note(v) for v in conclusions]
    summary["error_attribution_conclusions"]=conclusions
    q3_write_json(out/"analysis_summary.json",summary)
    q3_write_json(out/"error_attribution_conclusions.json",{
        "conclusions":conclusions,"basis":["all_predictions.csv","coalition_performance.csv","modality_replacement_errors.csv","diagnostic_occlusion.csv","wrong_vs_true_modality_contributions.csv","diagnostic_case_error_analysis.csv"],
        "scope":"Observed associations and input-intervention responses; not causal proof"})
    q3_write_csv(out/"coalition_performance.csv",combination_rows)
    q3_write_csv(out/"modality_replacement_errors.csv",removal_rows)
    q3_write_csv(out/"group_modality_shares.csv",group_rows)
    q3_write_csv(out/"diagnostic_occlusion.csv",detail_effects)
    return cohort,details,summary



def q3_completion_main(args):
    question3=Path(__file__).resolve().parent
    out=(args.output_dir or question3/datetime.now(timezone.utc).strftime("completion_%Y%m%dT%H%M%S_%fZ")).expanduser().resolve()
    if not out.is_relative_to(question3) or out==question3:
        raise ValueError("Completion outputs must use a child directory of question3")
    if args.ig_steps<2 or args.max_steps<2*args.ig_steps or args.random_repeats<2:
        raise ValueError("Invalid numerical settings")
    if not 0<=args.worker_index<args.worker_count:
        raise ValueError("Invalid worker assignment")
    out.mkdir(parents=True,exist_ok=True)
    phase={"phase":args.completion_stage,"started_utc":datetime.now(timezone.utc).isoformat(),
           "script_sha256":sha256(__file__),"checkpoint_sha256":sha256(args.checkpoint),
           "device":args.device,"worker_index":args.worker_index,"worker_count":args.worker_count}
    q3_write_json(out/f"phase_{args.completion_stage}_{args.worker_index}_started.json",phase)
    print("OUTPUT_DIR="+str(out),flush=True)
    if args.completion_stage in ("prepare","all"):
        q3_validation_prepare(args,out/"validation")
    if args.completion_stage in ("details","all"):
        q3_validation_details(args,out/"validation")
    if args.completion_stage in ("media","all"):
        q3_build_attachment4_evidence(args,args.attachment4_report,out/"attachment4")
    if args.completion_stage in ("figures","all"):
        q3_make_completion_figures(args,out)
    phase["finished_utc"]=datetime.now(timezone.utc).isoformat()
    q3_write_json(out/f"phase_{args.completion_stage}_{args.worker_index}_complete.json",phase)
    print("PHASE_COMPLETE="+args.completion_stage,flush=True)




_Q3_PANEL_AUDIT_SOURCE = "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"



def q3_alignment_auditor():
    import base64, zlib, types
    name="_q3_embedded_panel_auditor"
    if name not in sys.modules:
        module=types.ModuleType(name)
        sys.modules[name]=module
        source=zlib.decompress(base64.b64decode(_Q3_PANEL_AUDIT_SOURCE)).decode("utf-8")
        exec(compile(source,"<embedded-panel-auditor>","exec"),module.__dict__)
    return sys.modules[name].require_matplotlib_panel_alignment


Q3_ZH_ROOT = Path(__file__).resolve().parent / "completion_20260926" / "localization"
_Q3_ZH_CATALOG = {}
_Q3_ZH_LABELS = {
    "text":"文本", "audio":"语音", "vision":"视觉",
    "classification":"情感极性", "regression":"情感强度",
    "negative":"负向", "neutral":"中性", "positive":"正向",
    "mean":"训练均值参考", "zero":"零参考", "all":"三模态联合",
    "correct":"分类正确", "incorrect":"分类错误", "high":"高", "middle":"中", "low":"低",
    "train":"训练集", "valid":"验证集", "test":"测试集",
}


def q3_zh(value):
    return _Q3_ZH_LABELS.get(value,value)


def q3_zh_catalog(kind):
    key=(str(Q3_ZH_ROOT),kind)
    if key not in _Q3_ZH_CATALOG:
        path=Q3_ZH_ROOT/(kind+"_zh.json")
        if not path.is_file():
            raise FileNotFoundError("缺少中文展示资源："+str(path))
        _Q3_ZH_CATALOG[key]=json.loads(path.read_text(encoding="utf-8"))
    return _Q3_ZH_CATALOG[key]


def q3_zh_case(record):
    sid=record.get("id",record.get("sample_id"))
    kind="validation" if sid.startswith("v") else "attachment4"
    data=q3_zh_catalog(kind)
    item=data.get("samples",data)[sid]
    if item["raw_text"]!=record["raw_text"]:
        raise AssertionError("中文译文与原始文本不匹配："+sid)
    return item


def q3_zh_word(item,word,bilingual=True):
    translated=item["word_zh"][word]
    return translated+"（"+word+"）" if bilingual and translated!=word else translated


def q3_zh_status(value):
    if not value:
        return "无"
    data=q3_zh_catalog("attachment4")
    for key in ("evidence_status_zh","alignment_status_zh","provenance_status_zh","mapping_status_zh","alignment_flags_zh"):
        if value in data.get(key,{}):
            return data[key][value]
    extra={"needs_review_alignment":"对齐结果待复核","not_applicable":"不适用",
           "unavailable":"不可获得","post-hoc lexical temporal correspondence":"事后词语时间对应"}
    if value in extra:
        return extra[value]
    raise KeyError("尚未翻译的展示状态："+str(value))


def q3_wrap_display(value,max_units=112):
    import unicodedata
    lines=[];line="";units=0
    for char in str(value):
        if char=="\n":
            lines.append(line);line="";units=0;continue
        width=2 if unicodedata.east_asian_width(char) in ("W","F") else 1
        if units+width>max_units and line:
            lines.append(line);line="";units=0
        line+=char;units+=width
    if line:lines.append(line)
    return "\n".join(lines)


def q3_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    path=Q3_ZH_ROOT/"fonts"/"DroidSansFallbackFull.ttf"
    if not path.is_file():
        raise FileNotFoundError("缺少中文字体："+str(path))
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family":["DejaVu Sans","Droid Sans Fallback"],"font.sans-serif":["DejaVu Sans","Droid Sans Fallback"],
        "font.size":8,"axes.titlesize":9,"axes.labelsize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.7,
        "legend.frameon":False,"savefig.facecolor":"white","figure.facecolor":"white"})
    return plt


def q3_export_figure(fig, path, question, source, book=None, comparable_axes=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    axes=list(comparable_axes) if comparable_axes is not None else list(fig.axes)
    alignment=q3_alignment_auditor()(fig,json_out=path.with_suffix(".alignment.json"),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,axes=axes)
    # Fixed page bounds retain the audited physical layout.
    fig.savefig(path.with_suffix(".png"),dpi=360)
    if book is None:
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".svg"))
    else:
        book.savefig(fig)
    from matplotlib.text import Text
    meta={"rendered_text":[v.get_text() for v in fig.findobj(match=Text) if v.get_visible() and v.get_text()],"question":question,"source_data":source,"backend":"python-matplotlib","display_language":"zh-CN",
          "archetype":"quantitative grid" if len(axes)>1 else "single quantitative panel",
          "size_inches":fig.get_size_inches().tolist(),"minimum_requested_font_pt":7,
          "uncertainty":"Descriptive fixed-cohort observations. No population confidence intervals.",
          "alignment_verdict":alignment.get("verdict"),
          "color_semantics":{"text":"#3B6E8C","audio":"#BF8845","vision":"#5C8D78"},
          "pdf":"page in the associated PDF book" if book is not None else path.with_suffix(".pdf").name}
    q3_write_json(path.with_suffix(".figure.json"),meta)
    return meta


def q3_figure_label(ax, letter):
    ax.annotate(letter,(0,1),xycoords="axes fraction",xytext=(-31,12),
                textcoords="offset points",fontweight="bold",fontsize=9,fontfamily="DejaVu Sans",
                ha="left",va="bottom")



def q3_case_card(record, arrays, output_prefix, book, title_prefix):
    plt=q3_plot_style()
    fig=plt.figure(figsize=(8.27,11.69))
    prediction=record["prediction"]
    sid=record.get("id",record.get("sample_id"))
    translation=q3_zh_case(record)
    polarity=("负向","中性","正向")[prediction["class_index"]]
    prefix="验证集诊断样本" if sid.startswith("v") else "附件四样本"
    fig.text(.08,.965,f"{prefix} {sid}",fontsize=13,fontweight="normal")
    first=f"预测情感：{polarity}    |    预测强度：{prediction['regression']:+.3f}"
    if "truth" in record:
        first+=f"    |    真实情感：{q3_zh(record['truth']['polarity'])}，{record['truth']['regression']:+.3f}"
    fig.text(.08,.941,first,fontsize=9)
    visible_end=record["token_alignment"]["visible_character_end"]
    visible_text=record["raw_text"][:visible_end]
    import textwrap
    original=textwrap.fill("模型可见原文："+visible_text,width=108,break_long_words=False,break_on_hyphens=False)
    translated=q3_wrap_display("中文释义："+translation["visible_text_zh"],116)
    line_count=len(original.splitlines())
    fig.text(.08,.912,original,fontsize=8,va="top",linespacing=1.3)
    fig.text(.08,.912-(line_count*10.4+7)/841.68,translated,fontsize=8,va="top",linespacing=1.3)
    grid=fig.add_gridspec(3,2,left=.18,right=.965,bottom=.18,top=.75,wspace=.70,hspace=.80)
    colors=["#3B6E8C","#BF8845","#5C8D78"]
    eligible=np.asarray(record["content_mask"],bool)
    axes=[]
    for h,head in enumerate(HEADS):
        ax=fig.add_subplot(grid[0,h]); axes.append(ax)
        shares=record["baselines"]["mean"]["shares"][h]
        values=np.asarray([0 if v is None else 100*v for v in shares],float)
        ax.bar(range(3),values,color=colors,width=.65)
        ax.set(xticks=range(3),xticklabels=["文本","语音","视觉"],ylim=(0,105),
               ylabel="绝对 Shapley 份额（%）",title=q3_zh(head))
        defined=record["baselines"]["mean"]["shares_defined"][h]
        dom="、".join(q3_zh(MODALITIES[j]) for j in tied_dominant_indices(values)) if defined else "未定义"
        ax.set_xlabel("主要参考模态："+dom,fontsize=8)
        q3_figure_label(ax,chr(97+h))
        ax=fig.add_subplot(grid[1,h]); axes.append(ax)
        masses=np.stack([np.abs(arrays["mean_"+m][h]).sum(1) for m in MODALITIES])
        masses[:,~eligible]=0
        total=masses.sum()
        for j,m in enumerate(MODALITIES):
            values=100*masses[j]/total if total>1e-12 else masses[j]
            ax.plot(np.flatnonzero(eligible),values[eligible],color=colors[j],lw=1.1,
                    linestyle=("-", "--", ":")[j],label=q3_zh(m))
        ax.set(xlabel="编码后的内容位置",ylabel="绝对归因量占比（%）",
               title="局部重要性分布",xlim=(-1,50))
        q3_figure_label(ax,chr(99+h))
        ax=fig.add_subplot(grid[2,h]); axes.append(ax)
        words=[w for w in record["words"] if not w.get("truncated",False)]
        ranked=sorted(words,key=lambda w:-w.get("attribution_mass",[0,0])[h])[:5][::-1]
        if ranked:
            ax.barh(range(len(ranked)),[w["attribution_mass"][h] for w in ranked],color=colors[0],height=.62)
            labels=[translation["word_zh"][w["word"]]+"\n"+w["word"] for w in ranked]
            ax.set(yticks=range(len(ranked)),yticklabels=labels,
                   xlabel="文本绝对归因量",title="关键完整词语")
            ax.tick_params(axis="y",labelsize=7)
        q3_figure_label(ax,chr(101+h))
    notes=["参考输入采用训练均值；绝对份额表示输出归因的相对大小，不表示准确率贡献。",
           "曲线：文本（蓝色实线）、语音（赭色虚线）、视觉（绿色点线）。",
           "译文和词语释义仅供阅读；模型仍使用原始输入，关键词保留原词以便核对。",
           "曲线横轴为缓存内容位置；附件四的原始媒体时间对应见证据定位卡。"]
    if record["data_quality"].get("input_truncated"):
        notes.append("输入存在截断：未进入模型的文本后缀不参与解释。")
    zero=record["data_quality"].get("all_zero_content_modalities",[])
    if zero:
        notes.append("原始内容输入全零："+ "、".join(q3_zh(v) for v in zero)+"；数值归因不等于观察到的媒体证据。")
    if "deletion" in record:
        for head in HEADS:
            d=record["deletion"][head]["all"]; k=min(4,max(d["counts"]));j=d["counts"].index(k)
            notes.append(f"{q3_zh(head)}替换{k}个位置：绝对输出变化——高归因 {d['top']['absolute_change'][j]:.3f}，"
                         f"低归因 {d['low']['absolute_change'][j]:.3f}，随机均值 {d['random']['absolute_change_mean'][j]:.3f}。")
    fig.text(.08,.125,"\n".join(notes),fontsize=7,va="top",linespacing=1.5)
    meta=q3_export_figure(fig,output_prefix,"该样本的模态作用与局部输入重要性如何解释当前预测？",
                         "逐样本分析记录及归因数组",book,axes)
    plt.close(fig)
    return meta


def q3_plot_validation(cohort, details, summary, out):
    from matplotlib.backends.backend_pdf import PdfPages
    plt=q3_plot_style()
    out=Path(out)
    figures=out/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    records=cohort["records"]
    manifests=[]
    names=["负向","中性","正向"]
    # The first two figures establish labelled predictive performance.
    fig,ax=plt.subplots(figsize=(4.1,3.8))
    fig.subplots_adjust(left=.23,right=.95,bottom=.19,top=.88)
    cm=np.asarray(cohort["metrics"]["confusion_matrix"])
    row_percent=100*cm/cm.sum(1,keepdims=True)
    ax.imshow(row_percent,cmap="Blues",vmin=0,vmax=100,aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{cm[i,j]}\n{row_percent[i,j]:.1f}%",ha="center",va="center",
                    color="white" if row_percent[i,j]>60 else "#20272D",fontsize=9)
    ax.set(xticks=range(3),yticks=range(3),xticklabels=names,yticklabels=names,
           xlabel="预测情感极性",ylabel="真实情感极性",title="验证集情感极性混淆矩阵（728条）")
    manifests.append(q3_export_figure(fig,figures/"01_confusion","验证集主要出现哪些情感类别混淆？",
                                      "all_predictions.csv"))
    plt.close(fig)
    truth=np.asarray([r["truth"]["regression"] for r in records])
    predicted=np.asarray([r["prediction"]["regression"] for r in records])
    fig,ax=plt.subplots(figsize=(4.1,3.8))
    fig.subplots_adjust(left=.19,right=.95,bottom=.19,top=.88)
    ax.scatter(truth,predicted,s=8,color="#3B6E8C",alpha=.35,linewidths=0,rasterized=True)
    lo=min(-3,float(predicted.min()))-.15;hi=max(3,float(predicted.max()))+.15
    ax.plot([lo,hi],[lo,hi],color="#969CA2",ls="--",lw=.8)
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="真实情感强度",ylabel="预测情感强度",
           title=f"验证集情感强度预测（平均绝对误差 {cohort['metrics']['mae']:.3f}）")
    manifests.append(q3_export_figure(fig,figures/"02_regression","预测情感强度与真实值有何偏差？",
                                      "all_predictions.csv"))
    plt.close(fig)
    colors=["#3B6E8C","#BF8845","#5C8D78"]
    for h,head in enumerate(HEADS):
        fig,ax=plt.subplots(figsize=(6.9,3.6))
        fig.subplots_adjust(left=.12,right=.97,bottom=.22,top=.86)
        data=[];labels=[];boxcolors=[]
        for j,m in enumerate(MODALITIES):
            for flag,label in ((True,"分类正确"),(False,"分类错误")):
                data.append([100*r["baselines"]["mean"]["shares"][h][j] for r in records
                             if r["evaluation"]["classification_correct"]==flag and r["baselines"]["mean"]["shares_defined"][h]])
                labels.append(q3_zh(m)+"\n"+label);boxcolors.append(colors[j])
        bp=ax.boxplot(data,patch_artist=True,showfliers=True,widths=.52,
                      medianprops={"color":"#19262C","linewidth":1},
                      flierprops={"marker":".","markersize":2,"alpha":.25,"markeredgecolor":"#69747B"})
        for patch,color in zip(bp["boxes"],boxcolors):patch.set_facecolor(color);patch.set_alpha(.7)
        ax.set(xticks=range(1,7),xticklabels=labels,ylim=(-3,103),ylabel="绝对 Shapley 份额（%）",
               title=q3_zh(head)+"解释：分类正确与错误样本的模态作用")
        fig.text(.12,.055,"全部728条验证样本；按分类对错分组。箱体为四分位区间，中线为中位数，须线范围为1.5倍四分位距。",fontsize=7)
        manifests.append(q3_export_figure(fig,figures/f"03_{head}_modality_groups",
                "分类正确与错误样本的模态作用有何差异？",
                "cohort.json and group_modality_shares.csv"))
        plt.close(fig)
    # Ground-truth error changes are different from prediction preservation.
    fig,ax=plt.subplots(figsize=(5.5,3.7))
    fig.subplots_adjust(left=.17,right=.97,bottom=.30,top=.86)
    rows=[r for r in summary["whole_modality_replacements"] if r["baseline"]=="mean"]
    positions=np.arange(3);width=.34
    ax.bar(positions-width/2,[100*r["wrong_to_correct_count"]/r["originally_wrong_count"] for r in rows],
           width,label="原错误改判正确",color="#5C8D78")
    ax.bar(positions+width/2,[100*r["correct_to_wrong_count"]/r["originally_correct_count"] for r in rows],
           width,label="原正确变为错误",color="#B26B60")
    ax.set(xticks=positions,xticklabels=["文本","语音","视觉"],ylabel="占原始分组的比例（%）",
           title="整体模态替换后的分类变化")
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.56,.12),ncol=2,fontsize=7)
    fig.text(.17,.035,"采用训练均值替换；两组分母分别为267和461。\n结果描述模型响应，不作为错误的因果归因。",fontsize=7,va="bottom",linespacing=1.25)
    manifests.append(q3_export_figure(fig,figures/"04_modality_replacement_errors",
            "整体模态替换修正或引入了哪些分类错误？",
            "modality_replacement_errors.csv"))
    plt.close(fig)
    for head in HEADS:
        rows=[r for r in summary["detail_occlusion"] if r["head"]==head and r["group"]=="all"]
        x=np.asarray([r["random_mean_absolute_change"] for r in rows])
        y=np.asarray([r["top_absolute_change"] for r in rows])
        fig,ax=plt.subplots(figsize=(4.5,3.9))
        fig.subplots_adjust(left=.19,right=.96,bottom=.30,top=.88)
        hi=max(float(x.max()),float(y.max()),1e-6)*1.08
        ax.plot([0,hi],[0,hi],ls="--",lw=.8,color="#999999")
        flags=np.asarray([r["classification_correct"] for r in rows])
        for flag,color,label in ((True,"#3B6E8C","原分类正确"),(False,"#B26B60","原分类错误")):
            ax.scatter(x[flags==flag],y[flags==flag],s=17,color=color,alpha=.7,linewidths=0,label=label)
        ax.set(xlim=(-.025*hi,hi),ylim=(-.025*hi,hi),xlabel="随机位置：绝对输出变化的均值",
               ylabel="高归因位置：绝对输出变化",title=q3_zh(head)+"干预对照（60条诊断样本）")
        fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.56,.045),ncol=2,fontsize=7)
        manifests.append(q3_export_figure(fig,figures/f"05_{head}_occlusion",
                "相同数量的高归因位置替换是否比随机替换引起更大输出变化？",
                "diagnostic_occlusion.csv; random mean uses 20 permutations"))
        plt.close(fig)
    # Reference sensitivity, explicitly limited to the chosen diagnostic subset.
    fig,ax=plt.subplots(figsize=(5.5,3.8))
    fig.subplots_adjust(left=.16,right=.97,bottom=.20,top=.86)
    for h,head in enumerate(HEADS):
        values=[]
        for j in range(3):
            valid=[r["reference_stability"][head]["content_time_spearman"][j] for r in details]
            valid=[v for v in valid if v is not None]
            rng=np.random.default_rng(42+h*3+j)
            ax.scatter(np.full(len(valid),j+(h-.5)*.25)+rng.uniform(-.055,.055,len(valid)),
                       valid,s=10,alpha=.5,color=("#3B6E8C","#BF8845")[h],linewidths=0,
                       label=q3_zh(head) if j==0 else None)
    ax.axhline(0,color="#AAAAAA",lw=.7)
    ax.set(xticks=range(3),xticklabels=["文本","语音","视觉"],ylim=(-1.05,1.05),
           ylabel="局部重要性排序相关系数",title="训练均值参考与零参考的比较（60条）")
    ax.legend(loc="lower right")
    manifests.append(q3_export_figure(fig,figures/"06_reference_sensitivity",
            "两种指定参考输入下的局部重要性排序是否稳定？",
            "Per-case reference_stability records; undefined rank correlations omitted and recorded"))
    plt.close(fig)
    with PdfPages(out/"validation_explanation_cards.pdf") as book:
        for r in details:
            with np.load(out/"details"/r["id"]/"attributions.npz",allow_pickle=False) as saved:
                arrays={key:saved[key] for key in saved.files}
            manifests.append(q3_case_card(r,arrays,out/"cards"/r["id"],book,"Validation diagnostic"))
    manifests.append(q3_plot_error_contrast(out))
    q3_write_json(out/"figure_manifest.json",manifests)
    (out/"error_attribution_conclusions.txt").write_text("\n\n".join(summary["error_attribution_conclusions"])+"\n",encoding="utf-8")
    return manifests



def q3_prepare_visual_previews(args, out, media):
    """Export source-frame review/context previews without changing evidence.

    A preview is never promoted to confirmed evidence. Candidate word times and
    alignment flags remain unchanged; unavailable times receive explicitly
    unlocalized context frames from the video's presentation timeline.
    """
    import subprocess
    from PIL import Image
    out = Path(out).resolve()
    allowed = (ROOT / "question3").resolve()
    if out != allowed and allowed not in out.parents:
        raise ValueError("Visual review previews must remain inside question3")
    ffmpeg = str(getattr(args, "ffmpeg", ROOT / "work/ffmpeg-7.0.2-amd64-static/ffmpeg"))
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose_zh": "补充待复核画面及原始视频参考画面，不改变已确认的证据状态。",
        "confirmed_evidence_added": 0, "source_records_modified": False,
        "source_feature_time_recovered": False, "samples": {}, "source_videos": {},
        "role_definitions_zh": {
            "alignment_review": "按已有候选词语时间选取的原始帧；词语时间仍待复核。",
            "raw_context_only": "原始视频参考帧；原视觉输入特征全零，不代表模型实际观察到的视觉证据。",
            "unlocalized_context": "没有可靠的词语时间关联，按视频时长位置选取的普通原始参考帧。"},
        "errors": []}
    unique_assets = set()
    for sample in media["samples"]:
        sid = str(sample["sample_id"]).zfill(2)
        missing = {}
        for head in HEADS:
            for item in sample["top_evidence"][head]["vision"]:
                if not item.get("asset"):
                    missing.setdefault(int(item["word_index"]), item)
        if not missing:
            continue
        manifest["samples"][sid] = []
        video = Path(sample["source_video"])
        alignment_file = out / sample["alignment_file"]
        alignment = json.loads(alignment_file.read_text(encoding="utf-8"))
        video_hash = sha256(video)
        if (video_hash != sample["source_sha256"] or
                video_hash != alignment["source_sha256"]):
            raise ValueError(f"{sid}: source video hash differs from evidence/alignment")
        pts = list(map(float, alignment["video_pts_s"]))
        ends = list(map(float, alignment["video_frame_end_s"]))
        if not pts or len(pts) != len(ends) or any(b <= a for a, b in zip(pts, ends)):
            raise ValueError(f"{sid}: invalid cached video presentation timeline")
        manifest["source_videos"][sid] = {
            "file": str(video), "sha256": video_hash,
            "alignment_file": str(alignment_file.relative_to(out)),
            "alignment_sha256": sha256(alignment_file),
            "decoded_frame_count": len(pts), "first_pts_s": pts[0],
            "last_frame_end_s": ends[-1]}
        destination = out / "samples" / sid / "review_keyframes"
        destination.mkdir(parents=True, exist_ok=True)
        for word_index, item in sorted(missing.items()):
            start, end = item.get("start_s"), item.get("end_s")
            valid_time = (start is not None and end is not None and
                          np.isfinite(start) and np.isfinite(end) and 0 <= start < end)
            overlaps = ([i for i, (a, b) in enumerate(zip(pts, ends))
                         if b > start and a < end] if valid_time else [])
            status = item["evidence_status"]
            if overlaps:
                midpoint = (start + end) / 2
                choices = [(min(overlaps, key=lambda i: (abs((pts[i]+ends[i])/2-midpoint), i)),
                            None)]
                if status == "unavailable_raw_zero_input":
                    role = "raw_context_only"
                    label = "原始视频参考画面（视觉特征全零）"
                else:
                    role = "alignment_review"
                    label = "词语时间待复核"
            else:
                # Fractions are media context only, never fabricated word times.
                choices = [(min(range(len(pts)), key=lambda i: (
                            abs((pts[i]+ends[i])/2-(pts[0]+fraction*(ends[-1]-pts[0]))), i)),
                            fraction) for fraction in (0.25, 0.5, 0.75)]
                role, label = "unlocalized_context", "原始视频参考画面（未确认词语时间）"
            record = {
                "word_index": word_index, "word": item["word"],
                "start_s": start, "end_s": end,
                "evidence_status": status, "alignment_score": item.get("alignment_score"),
                "alignment_status": item.get("alignment_status"),
                "alignment_flags": list(item.get("alignment_flags", [])),
                "display_status_zh": label, "role": role,
                "candidate_word_interval_used": bool(overlaps),
                "confirmed_word_time": False,
                "counts_as_confirmed_evidence": False,
                "source_feature_time_status": "unavailable",
                "asset": None}
            try:
                assets = []
                for frame_index, fraction in choices:
                    path = destination / f"frame{frame_index:06d}_{video_hash[:12]}.png"
                    if not path.is_file() or path.stat().st_size == 0:
                        command = [ffmpeg, "-nostdin", "-v", "error", "-threads", "2",
                                   "-y", "-i", str(video), "-map", "0:v:0",
                                   "-vf", f"select=eq(n\\,{frame_index})",
                                   "-fps_mode", "passthrough", "-frames:v", "1",
                                   "-threads", "1", str(path)]
                        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
                    with Image.open(path) as frame:
                        width, height = frame.size
                        frame.verify()
                    if width <= 0 or height <= 0:
                        raise ValueError("Decoded preview frame has invalid dimensions")
                    relative = str(path.relative_to(out))
                    unique_assets.add(relative)
                    asset = {
                        "file": relative, "kind": "video_png",
                        "decoded_frame_index": frame_index, "frame_pts_s": pts[frame_index],
                        "frame_end_s": ends[frame_index], "width": width, "height": height,
                        "sha256": sha256(path), "source_video_sha256": video_hash,
                        "interpretation_zh": label,
                        "overlapping_candidate_frame_indices": overlaps}
                    if fraction is not None:
                        asset["video_context_fraction"] = fraction
                    assets.append(asset)
                record["asset"] = assets[0] if overlaps else assets[1]
                if not overlaps:
                    record["context_assets"] = assets
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if isinstance(error, subprocess.CalledProcessError):
                    message += " " + error.stderr.decode(errors="replace")[-1500:]
                record["preview_error"] = message
                manifest["errors"].append({"sample_id": sid, "word_index": word_index,
                                           "error": message})
            manifest["samples"][sid].append(record)
    counts = {}
    for records in manifest["samples"].values():
        for record in records:
            counts[record["role"]] = counts.get(record["role"], 0) + 1
    manifest["summary"] = {
        "sample_count": len(manifest["samples"]),
        "preview_item_count": sum(len(v) for v in manifest["samples"].values()),
        "unique_png_count": len(unique_assets), "role_counts": counts,
        "export_error_count": len(manifest["errors"])}
    q3_media_write_json(out / "visual_review_previews.json", manifest)
    return manifest


def q3_media_card(sample, out, book):
    """Show each task's original visual Top3 without merging task identities."""
    from PIL import Image
    plt=q3_plot_style()
    out=Path(out)
    fig=plt.figure(figsize=(11.69,11.69))
    sid=sample["sample_id"]
    translation=q3_zh_case(sample)
    preview_path=out/"visual_review_previews.json"
    previews=(json.loads(preview_path.read_text(encoding="utf-8")).get("samples",{}).get(sid,[])
              if preview_path.is_file() else [])
    by_word={row["word_index"]:row for row in previews}
    # fig.text(.06,.965,f"附件四样本 {sid}｜分类与回归的原始媒体定位",fontsize=14,fontweight="normal")
    # fig.text(.06,.935,"两项任务分别保留前三名；共有画面重复展示，并保留各自排名、归因数值及证据状态。",fontsize=9)
    candidates=[]
    for head in HEADS:
        rows=sample["top_evidence"][head]["vision"]
        if len(rows)!=3 or [item["rank"] for item in rows]!=[1,2,3]:
            raise AssertionError(f"{sid}/{head}: expected original ordered visual Top3")
        for item in rows:
            preview=None if item.get("asset") else by_word.get(item["word_index"])
            asset=preview.get("asset") if preview else item.get("asset")
            if not asset or not (out/asset["file"]).is_file():
                raise FileNotFoundError(f"{sid}/{head}/rank{item['rank']}: original Top3 image missing")
            candidates.append((head,item,preview,asset))
    with Image.open(out/candidates[0][3]["file"]) as first:
        shape=(first.height,first.width)
    grid=fig.add_gridspec(2,3,left=.06,right=.95,bottom=.35,top=.80,wspace=.16,hspace=1.0)
    axes=[]
    shown=[]
    for row,head in enumerate(HEADS):
        y=.895-row*.30
        heading=("情感极性分类" if head=="classification" else "情感强度回归")
        fig.text(.06,y,heading+"｜视觉关键位置前三名",fontsize=11,fontweight="normal")
        direction=("净贡献为正：增强原预测类别相对次高类别的优势；为负：削弱该优势。"
                   if head=="classification" else
                   "净贡献为正：提高情感强度预测分数；为负：降低预测分数。")
        fig.text(.06,y-.025,direction,fontsize=8.5,color="#47545B")
    for j,(head,item,preview,asset) in enumerate(candidates):
        ax=fig.add_subplot(grid[j//3,j%3]);axes.append(ax)
        ax.set_box_aspect(shape[0]/shape[1]);ax.set_axis_off()
        with Image.open(out/asset["file"]) as source:
            ax.imshow(np.asarray(source),interpolation="none")
        label=q3_zh_word(translation,item["word"])
        frame_label=f"{asset['frame_pts_s']:.3f}秒｜第{asset['decoded_frame_index']}帧"
        role=preview["role"] if preview else "accepted_lexical_anchor"
        color="#283B45"
        rank_label=f"第{item['rank']}位："
        if role=="raw_context_only":
            status="原始参考画面｜视觉特征全零"
            rank_label=f"候选第{item['rank']}位："
            color="#626A71"
        elif role=="unlocalized_context":
            status="原始参考画面｜词语时间未确认"
            color="#8B642D"
        elif role=="alignment_review":
            score=item.get("alignment_score")
            status=(f"待复核画面｜对齐分数 {score:.3f}" if score is not None
                    else "待复核画面｜对齐分数不可用")
            color="#8B642D"
        else:
            status="词语时间对应可用"
        mass=float(item["attribution_mass"]);net=float(item["attribution_net"])
        title=(q3_wrap_display(rank_label+label,44)+"\n"+frame_label+
               f"\n绝对归因量 {mass:.4f}｜净贡献 {net:+.4f}\n"+status)
        ax.set_title(title,fontsize=8,pad=8,color=color,linespacing=1.35)
        q3_figure_label(ax,chr(97+j))
        shown.append({"panel":chr(97+j),"head":head,"rank":item["rank"],
                      "word_index":item["word_index"],"word":item["word"],
                      "attribution_mass":mass,"attribution_net":net,
                      "role":role,"original_evidence_status":item["evidence_status"],
                      "asset":asset["file"],"frame_pts_s":asset["frame_pts_s"],
                      "decoded_frame_index":asset["decoded_frame_index"],
                      "counts_as_confirmed_evidence":preview is None})
    for j,head in enumerate(HEADS):
        left=.06+j*.48
        heading=("情感极性分类" if head=="classification" else "情感强度回归")
        # fig.text(left,.295,heading+"｜关键语音片段",fontsize=10,fontweight="normal")
        lines=[]
        for item in sample["top_evidence"][head]["audio"]:
            timing=(f"{item['start_s']:.3f}—{item['end_s']:.3f}秒"
                    if item["start_s"] is not None else "词语时间尚未确认")
            lines.append(f"{item['rank']}. {q3_zh_word(translation,item['word'])}｜{timing}")
            status=("已导出语音片段，可在结果索引中播放" if item.get("asset")
                    else q3_zh_status(item["evidence_status"]))
            lines.append("    "+status)
        fig.text(left,.267,"\n".join(lines),va="top",fontsize=8,linespacing=1.6)
    warnings=[
        "各任务独立按绝对归因量排序；同一画面可重复出现。采用训练集均值参考，两项任务的归因数值不直接比较。",
        "绝对归因量为该位置各维度归因绝对值之和；净贡献为有符号和，方向按各任务的解释目标理解。",
        "截图对应词语时间附近的视觉背景，不表示已恢复缓存视觉特征的原始提取窗口。",
        "待复核画面的词语时间关联尚未确认；原始参考画面不作为模型已观察到的视觉证据。",
        "原始视频帧未增强；帧号从0开始。原始预测、归因数值和证据状态均保留，中文释义仅供阅读。"]
    if sample.get("acoustic_text_match_ratio") is not None and sample["acoustic_text_match_ratio"]<.5:
        warnings.insert(0,"本例给定文本与声学转写匹配程度偏低，待复核画面不能视为已确认的词语定位。")
    if sample["data_quality"].get("all_zero_content_modalities"):
        warnings.insert(0,"本例原始视觉内容特征全零；六张画面仅供原始视频复核，所示位置归因不能解释为画面内容的贡献。")
    fig.text(.06,.145,"\n".join(warnings),fontsize=7.5,va="top",linespacing=1.65)
    path=out/"cards"/(sid+"_media")
    result=q3_export_figure(fig,path,
            "分类与回归分别将哪些视觉位置排在前三，其归因方向及原始媒体对应状态有何差异？",
            "附件四各任务独立的视觉前三位置、媒体证据及复核预览记录",book,axes)
    fig.savefig(path.with_suffix(".pdf"))
    result["pdf"]=path.with_suffix(".pdf").name
    result["archetype"]="image plate + quant"
    result["layout"]="two task rows, three original ranked visual positions per row"
    result["cross_head_deduplication"]=False
    result["displayed_visual_panels"]=shown
    result["preview_manifest"]="visual_review_previews.json"
    result["image_processing"]="original frames, no crop or enhancement; repeated across tasks when independently selected"
    q3_write_json(path.with_suffix(".figure.json"),result)
    plt.close(fig)
    return result


def q3_plot_attachment4(args,out):
    from matplotlib.backends.backend_pdf import PdfPages
    out=Path(out)
    report=json.loads(args.attachment4_report.read_text(encoding="utf-8"))
    media=json.loads((out/"attachment4_media_evidence.json").read_text(encoding="utf-8"))
    if media["sample_count"]!=20 or len(media["samples"])!=20:
        raise AssertionError("Need all 20 Attachment-4 evidence records")
    previews=q3_prepare_visual_previews(args,out,media)
    if previews["errors"]:
        raise RuntimeError("Visual preview exports need review: "+str(previews["errors"]))
    preview_rows=[]
    for sid,records in previews["samples"].items():
        loc=q3_zh_catalog("attachment4")["samples"][sid]
        for record in records:
            asset=record.get("asset") or {}
            preview_rows.append({"样本编号":sid,"原词":record["word"],
                "中文释义":loc["word_zh"].get(record["word"],record["word"]),
                "画面用途":record["display_status_zh"],"原证据状态":q3_zh_status(record["evidence_status"]),
                "候选开始时间秒":record["start_s"],"候选结束时间秒":record["end_s"],
                "对齐分数":record["alignment_score"],"视频帧时间秒":asset.get("frame_pts_s"),
                "解码帧序号":asset.get("decoded_frame_index"),"画面文件":asset.get("file"),
                "计入已确认定位":"否"})
    q3_write_csv(out.parent/"中文表格"/"附件四复核画面索引.csv",preview_rows)
    by_id={r["sample_id"]:r for r in media["samples"]}
    manifests=[];predictions=[]
    with np.load(args.attachment4_report.parent/"attributions.npz",allow_pickle=False) as saved, \
            PdfPages(out/"attachment4_explanation_cards.pdf") as book:
        for sample in report["samples"]:
            sid=sample["id"]
            arrays={b+"_"+m:saved[f"s{sid}_{b}_{m}"] for b in ("mean","zero") for m in MODALITIES}
            manifests.append(q3_case_card(sample,arrays,out/"cards"/(sid+"_analysis"),book,"Attachment 4"))
            manifests.append(q3_media_card(by_id[sid],out,book))
            p=sample["prediction"]
            row={"sample_id":sid,"predicted_class":p["class_index"],
                 "predicted_polarity":("negative","neutral","positive")[p["class_index"]],
                 "predicted_intensity":p["regression"],"class_probabilities":p["probabilities"],
                 "raw_text":sample["raw_text"],"input_truncated":sample["data_quality"]["input_truncated"],
                 "raw_zero_modalities":sample["data_quality"]["all_zero_content_modalities"],
                 "raw_feature_time_provenance":"unavailable",
                 "media_mapping":"post-hoc lexical temporal correspondence",
                 "details_json":f"samples/{sid}/evidence.json","alignment_json":f"samples/{sid}/alignment.json"}
            for h,head in enumerate(HEADS):
                shares=sample["baselines"]["mean"]["shares"][h]
                defined=sample["baselines"]["mean"]["shares_defined"][h]
                row[head+"_main_modalities"]=([MODALITIES[j] for j in tied_dominant_indices(shares)] if defined else [])
                for j,m in enumerate(MODALITIES):
                    row[head+"_"+m+"_share"]=shares[j]
                    row[head+"_"+m+"_signed_shapley"]=sample["baselines"]["mean"]["phi"][h][j]
            predictions.append(row)
    q3_write_csv(out/"attachment4_predictions_and_explanations.csv",predictions)
    q3_write_json(out/"figure_manifest.json",manifests)
    return media,manifests



def q3_export_chinese_validation_tables(out,cohort,details,summary):
    target=Path(out)/"中文表格"
    target.mkdir(parents=True,exist_ok=True)
    labels=("负向","中性","正向")
    def yes(value):return "是" if value else "否"
    metric_names={"accuracy":"分类准确率","macro_f1":"宏平均F1","weighted_f1":"加权平均F1",
                  "mae":"平均绝对误差","rmse":"均方根误差","pearson":"Pearson相关系数"}
    q3_write_csv(target/"验证集基础指标.csv",[{"指标":name,"数值":cohort["metrics"][key],"样本数":len(cohort["records"])}
                                            for key,name in metric_names.items()])
    predictions=[];contributions=[]
    for r in cohort["records"]:
        predictions.append({
            "样本编号":r["id"],"原始数据编号":r["dataset_id"],
            "真实情感":labels[r["truth"]["class_index"]],"预测情感":labels[r["prediction"]["class_index"]],
            "真实强度":r["truth"]["regression"],"预测强度":r["prediction"]["regression"],
            "分类正确":yes(r["evaluation"]["classification_correct"]),
            "强度绝对误差":r["evaluation"]["regression_absolute_error"],
            "进入详细诊断":yes(r["selected_for_detailed_analysis"]),
            "输入截断":yes(r["data_quality"]["input_truncated"]),"原始文本":r["raw_text"]})
        for baseline,b in r["baselines"].items():
            for h,head in enumerate(HEADS):
                for j,m in enumerate(MODALITIES):
                    contributions.append({"样本编号":r["id"],"参考输入":q3_zh(baseline),"解释任务":q3_zh(head),
                        "模态":q3_zh(m),"有符号Shapley值":b["phi"][h][j],
                        "绝对作用程度":b["shares"][h][j],"作用程度有效":yes(b["shares_defined"][h]),
                        "分类正确":yes(r["evaluation"]["classification_correct"])})
    q3_write_csv(target/"验证集全量预测.csv",predictions)
    q3_write_csv(target/"验证集模态贡献.csv",contributions)
    selected=[]
    for r in cohort["selection"]["selected"]:
        selected.append({"样本编号":r["id"],"原始数据编号":r["dataset_id"],"真实情感":labels[r["true_class"]],
            "分类正确":yes(r["classification_correct"]),"强度绝对误差":r["regression_absolute_error"],
            "组内误差层次":q3_zh(r["regression_error_band_within_stratum"]),"原始分组样本数":r["stratum_size"],
            "随机种子":r["seed"],"是否组内最大误差样本":yes(r["mandatory_largest_error_in_stratum"])})
    q3_write_csv(target/"诊断样本名单.csv",selected)
    rows=[]
    for r in summary["whole_modality_replacements"]:
        rows.append({"参考输入":q3_zh(r["baseline"]),"被替换模态":q3_zh(r["modality"]),
            "原错误改判正确":r["wrong_to_correct_count"],"原错误样本数":r["originally_wrong_count"],
            "原正确变为错误":r["correct_to_wrong_count"],"原正确样本数":r["originally_correct_count"],
            "全体平均绝对误差变化":r["regression_error_change_mean"],
            "原分类错误组平均绝对误差变化":r["regression_error_change_wrong_group"]})
    q3_write_csv(target/"模态替换结果.csv",rows)
    rows=[]
    for r in summary["detail_occlusion"]:
        rows.append({"样本编号":r["id"],"解释任务":q3_zh(r["head"]),"干预范围":q3_zh(r["group"]),
            "替换位置数":r["replaced_slots"],"高归因位置绝对输出变化":r["top_absolute_change"],
            "低归因位置绝对输出变化":r["low_absolute_change"],"随机位置绝对输出变化均值":r["random_mean_absolute_change"],
            "高归因减随机":r["top_minus_random"],"高归因减低归因":r["top_minus_low"],
            "原分类正确":yes(r["classification_correct"])})
    q3_write_csv(target/"局部干预对照.csv",rows)
    groups=[]
    for r in summary["share_groups"]:
        group=q3_zh(r["group"])
        if r["group"]=="all":group="全部样本"
        if r["group"].startswith("true_class_"):group="真实"+labels[int(r["group"][-1])]
        groups.append({"参考输入":q3_zh(r["baseline"]),"解释任务":q3_zh(r["head"]),"分类分组":group,
            "样本数":r["sample_count"],"有效作用程度样本数":r["defined_share_count"],
            **{q3_zh(m)+"平均作用程度":r["mean_shares"][j] for j,m in enumerate(MODALITIES)},
            **{q3_zh(m)+"作用程度中位数":r["median_shares"][j] for j,m in enumerate(MODALITIES)}})
    q3_write_csv(target/"分组模态作用程度.csv",groups)
    interpretations=[]
    for r in details:
        loc=q3_zh_case(r)
        interpretations.append({"样本编号":r["id"],"原始文本":r["raw_text"],"全文中文释义":loc["text_zh"],
            "模型可见部分中文释义":loc["visible_text_zh"],"输入截断":yes(r["data_quality"]["input_truncated"]),
            "真实情感":labels[r["truth"]["class_index"]],"预测情感":labels[r["prediction"]["class_index"]],
            "真实强度":r["truth"]["regression"],"预测强度":r["prediction"]["regression"]})
    q3_write_csv(target/"诊断样本中文释义.csv",interpretations)
    import csv
    wrong_rows=[]
    with (Path(out)/"validation/wrong_vs_true_modality_contributions.csv").open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            wrong_rows.append({"样本编号":r["id"],"原始数据编号":r["dataset_id"],"参考输入":q3_zh(r["baseline"]),
                "真实情感":labels[int(r["true_class"])],"错误预测情感":labels[int(r["wrong_predicted_class"])],
                "完整输入错类相对真类得分差":r["wrong_vs_true_logit_margin_full"],
                "参考输入错类相对真类得分差":r["wrong_vs_true_logit_margin_reference"],
                "文本Shapley值":r["text_shapley"],"语音Shapley值":r["audio_shapley"],"视觉Shapley值":r["vision_shapley"],
                "作用最大模态":"、".join(q3_zh(v) for v in json.loads(r["absolute_main_modalities"])),
                "符号含义":"正值增强错误类别相对真实类别的优势，负值抵消该优势；不是因果责任。"})
    q3_write_csv(target/"错误类别与真实类别对比归因.csv",wrong_rows)
    cases=json.loads((Path(out)/"validation/diagnostic_case_error_analysis.json").read_text(encoding="utf-8"))
    q3_write_csv(target/"逐例错误分析.csv",[{
        "样本编号":r["id"],"真实情感":labels[r["true_class"]],"预测情感":labels[r["predicted_class"]],
        "分类正确":yes(r["classification_correct"]),"强度绝对误差":r["regression_absolute_error"],
        "使分类恢复正确的已检验原词":r["word_substitution_recoveries"],
        "分析说明":q3_chinese_note(r["diagnostic_note"])} for r in cases])

    q3_write_json(target/"结果使用说明.json",{
        "展示语言":"简体中文","数据完整性":"仅翻译展示文本，不修改预测、归因、原文或原始媒体。",
        "作用程度":"绝对Shapley值归一化所得份额，范围0到1，不表示准确率贡献。",
        "误差变化":"替换后的平均绝对误差减去替换前，正值表示误差增大。",
        "局部干预":"比较固定解释目标的绝对变化，不表示准确率下降。",
        "原始文本":"英文属于原始输入证据；中文释义供阅读，不是新的模型输入。",
        "缓存位置":"验证集音视频归因定位到缓存位置，不冒充原始语音时段或视频帧。",
        "附件四时间":"以WhisperX建立事后词语时间对应，未恢复原缓存特征的精确提取窗口。"})



def q3_export_chinese_attachment_tables(out,media):
    import csv
    target=Path(out)/"中文表格"
    labels=("负向","中性","正向")
    predictions=[];evidence=[]
    for r in media["samples"]:
        sid=r["sample_id"];loc=q3_zh_case(r);p=r["prediction"]
        row={"样本编号":sid,"预测情感":labels[p["class_index"]],"预测强度":p["regression"],
            "原始文本":r["raw_text"],"全文中文释义":loc["text_zh"],"模型可见部分中文释义":loc["visible_text_zh"],
            "输入截断":"是" if r["data_quality"]["input_truncated"] else "否",
            "原始输入全零模态":"、".join(q3_zh(m) for m in r["data_quality"].get("all_zero_content_modalities",[])) or "无",
            "文本与声学转写匹配程度":r.get("acoustic_text_match_ratio"),
            "媒体时间说明":"WhisperX事后词语时间对应；原缓存特征提取窗口不可获得"}
        for h,head in enumerate(HEADS):
            shares=r["modality_shares"][h]
            row[q3_zh(head)+"主要参考模态"]="、".join(q3_zh(MODALITIES[j]) for j in tied_dominant_indices(shares))
            for j,m in enumerate(MODALITIES):
                row[q3_zh(head)+q3_zh(m)+"作用程度"]=shares[j]
                row[q3_zh(head)+q3_zh(m)+"有符号Shapley值"]=r["modality_phi"][h][j]
        predictions.append(row)
    with (Path(out)/"attachment4/attachment4_key_evidence.csv").open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            loc=q3_zh_catalog("attachment4")["samples"][r["sample_id"]]
            flags=json.loads(r["alignment_flags"])
            evidence.append({
                "样本编号":r["sample_id"],"解释任务":q3_zh(r["head"]),"模态":q3_zh(r["modality"]),
                "重要性名次":r["rank"],"原词":r["word"],"中文释义":loc["word_zh"][r["word"]],
                "原始字符范围":r["raw_char_span"],"模型可见字符范围":r["model_visible_char_spans"],
                "编码位置":r["token_slots"],"原词序号":r["word_index"],
                "绝对归因量":r["attribution_mass"],"有符号净归因":r["attribution_net"],
                "开始时间秒":r["start_s"],"结束时间秒":r["end_s"],
                "证据状态":q3_zh_status(r["evidence_status"]),"对齐状态":q3_zh_status(r["alignment_status"]),
                "对齐分数":r["alignment_score"],"对齐复核原因":"；".join(q3_zh_status(v) for v in flags) or "无",
                "原特征时间来源":q3_zh_status(r["cached_feature_source_time_status"]),
                "媒体文件相对路径":r["asset_file"]})
    q3_write_csv(target/"附件四全量预测与解释.csv",predictions)
    q3_write_csv(target/"附件四关键证据定位.csv",evidence)


def q3_chinese_note(value):
    import re
    for old,new in (("原winner-runner解释","原预测类别与次高类别之间的对比解释"),
                    ("错误预测类别logit减真实类别logit","错误预测类别得分减真实类别得分"),
                    ("winner-runner","原预测类别与次高类别的得分差"),("classification","情感极性"),
                    ("regression","情感强度"),("text","文本"),("audio","语音"),("vision","视觉"),
                    ("Macro-F1","宏平均F1值"),("RMSE","均方根误差"),("MAE","平均绝对误差"),
                    ("logit","类别得分"),("UNK","未知词标记")):
        value=re.sub(r"(?<![A-Za-z_])"+re.escape(old)+r"(?![A-Za-z_])",new,value)
    return value


def q3_write_result_index(out,cohort,summary,media):
    import html
    out=Path(out);esc=html.escape
    preview_path=out/"attachment4"/"visual_review_previews.json"
    preview_catalog=(json.loads(preview_path.read_text(encoding="utf-8")).get("samples",{})
                     if preview_path.is_file() else {})
    lines=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>问题三中文结果</title>',
        '<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:36px auto;line-height:1.7;color:#22313b}'
        'table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}'
        'img{max-width:100%;height:auto}a{color:#236b91}section{margin:36px 0}small{color:#586570}'
        '.notice{background:#f5f0e7;padding:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}</style>',
        '<h1>问题三：验证集评估与附件四全量解释</h1>',
        '<p>展示语言：简体中文。保留 Shapley、WhisperX 等专有名词；英文原文与原词作为输入证据保留，并为详细样本提供中文释义。</p>',
        '<p>固定现有预测模型：验证集728条全量预测与模态贡献、60条分层诊断、附件四20条全量预测与解释。</p>',
        '<p class="notice">中文释义不进入模型，不改变任何预测或归因数值。附件四媒体定位为WhisperX提供的事后词语时间对应，不能称为已恢复原缓存特征的提取窗口。原画面中的文字保持原样。</p>',
        '<h2>验证集结果与中文表格</h2><ul>']
    tables=["验证集基础指标","验证集全量预测","验证集模态贡献","诊断样本名单","诊断样本中文释义","模态替换结果","分组模态作用程度","局部干预对照","错误类别与真实类别对比归因","逐例错误分析"]
    lines+=['<li><a href="中文表格/'+esc(name)+'.csv">'+esc(name)+'</a></li>' for name in tables]
    lines+=['<li><a href="validation/validation_explanation_cards.pdf">60页验证集中文解释卡</a></li>',
            '<li><a href="validation/error_attribution_conclusions.txt">错误归因结论</a></li>',
            '<li><a href="中文表格/结果使用说明.json">中文结果使用说明</a></li></ul>',
            '<div class="grid"><img src="validation/figures/01_confusion.png"><img src="validation/figures/02_regression.png"></div>',
            '<h3>全部汇总图</h3><ul>']
    figure_names=[("01_confusion","情感极性混淆矩阵"),("02_regression","情感强度预测"),
        ("03_classification_modality_groups","情感极性解释的模态作用分布"),
        ("03_regression_modality_groups","情感强度解释的模态作用分布"),
        ("04_modality_replacement_errors","整体模态替换后的分类变化"),
        ("05_classification_occlusion","情感极性局部干预对照"),
        ("05_regression_occlusion","情感强度局部干预对照"),
        ("06_reference_sensitivity","参考输入敏感性"),
        ("07_wrong_vs_true_contrast","错误类别相对真实类别的模态贡献")]
    for filename,label in figure_names:
        lines.append('<li>'+esc(label)+'：<a href="validation/figures/'+filename+'.pdf">矢量图</a> / '
                     '<a href="validation/figures/'+filename+'.png">预览图</a></li>')
    lines+=['</ul><h3>验证集分析结论</h3><ol>']
    lines+=['<li>'+esc(q3_chinese_note(v))+'</li>' for v in summary["error_attribution_conclusions"]]
    lines+=['</ol><h2>附件四全量结果</h2><ul>',
        '<li><a href="中文表格/附件四全量预测与解释.csv">全量预测、模态作用与中文释义</a></li>',
        '<li><a href="中文表格/附件四关键证据定位.csv">全部关键证据定位</a></li>',
        '<li><a href="attachment4/attachment4_explanation_cards.pdf">40页中文解释卡与媒体定位卡</a></li>',
        '<li><a href="attachment4/attachment4_media_evidence.json">原始证据计算记录</a></li>',
        '<li><a href="中文表格/附件四复核画面索引.csv">待复核与原始参考画面的独立索引</a></li></ul>',
        '<p class="notice">媒体卡按分类、回归分为两排，各自完整保留排名前三的视觉位置；共有画面重复展示，分别标注排名、绝对归因量与净贡献。可用词语时间对应、待复核截图和原始参考画面分别标注；补充截图不改变原证据状态，也不计入已确认定位。</p>']
    for sample in media["samples"]:
        sid=sample["sample_id"];loc=q3_zh_case(sample)
        lines+=['<section><h3>样本 '+esc(sid)+'</h3>',
            '<p><strong>中文释义：</strong>'+esc(loc["text_zh"])+'</p>',
            '<p><strong>原始文本：</strong>'+esc(sample["raw_text"])+'</p>',
            '<p><a href="attachment4/samples/'+esc(sid)+'/evidence.json">详细解释与定位记录</a></p>',
            '<div class="grid"><img loading="lazy" src="attachment4/cards/'+sid+'_analysis.png">',
            '<img loading="lazy" src="attachment4/cards/'+sid+'_media.png"></div>',
            '<p><a href="attachment4/cards/'+sid+'_media.pdf">下载本样本媒体卡PDF</a></p>']
        if sid in preview_catalog:
            lines+=['<details><summary>查看待复核或原始参考画面的完整索引</summary><ul>']
            for preview in preview_catalog[sid]:
                asset=preview.get("asset")
                if asset:
                    lines.append('<li>'+esc(q3_zh_word(loc,preview["word"]))+'：'+
                                 esc(preview["display_status_zh"])+'，视频 '+f'{asset["frame_pts_s"]:.3f}'+
                                 ' 秒；<a href="attachment4/'+esc(asset["file"],quote=True)+'">查看原始尺寸截图</a></li>')
            lines.append('</ul></details>')
        if sample["data_quality"].get("input_truncated"):
            lines.append('<p class="notice">本例输入存在截断。模型可见部分的中文释义：'+esc(loc["visible_text_zh"])+'。完整原文的其余部分不参与解释。</p>')
        if sample.get("acoustic_text_match_ratio") is not None and sample["acoustic_text_match_ratio"]<.5:
            lines.append('<p class="notice">本例给定文本与声学转写的匹配程度偏低，语音和画面定位需复核；未将不可靠的时间对应导出为关键媒体片段。</p>')
        shown=set()
        for head in HEADS:
            lines.append('<h4>'+esc(q3_zh(head))+'：关键语音片段</h4>')
            for item in sample["top_evidence"][head]["audio"]:
                lines.append('<p>'+esc(q3_zh_word(loc,item["word"]))+' — '+esc(q3_zh_status(item["evidence_status"]))+'</p>')
                if item.get("asset"):
                    target=item["asset"]["file"]
                    if target not in shown:
                        lines.append('<audio controls preload="none" src="attachment4/'+esc(target,quote=True)+'"></audio>')
                        shown.add(target)
        lines.append('</section>')
    lines.append('</html>')
    (out/"结果索引.html").write_text("\n".join(lines),encoding="utf-8")


def q3_make_completion_figures(args,out):
    out=Path(out)
    cohort,details,summary=q3_validation_summary(out/"validation")
    contract={
        "backend":"python-matplotlib","prediction_scope":"All 728 validation cases plus all 20 Attachment-4 cases",
        "diagnostic_scope":"60 preselected validation cases, 10 per true-class/correctness stratum",
        "claims":["Quantify validation class confusion and continuous prediction errors",
                  "Describe modality dependence and labelled effects of fixed-reference replacements",
                  "Test local ranking against matched-budget low/random controls, retaining counterexamples",
                  "Show each Attachment-4 prediction with modality shares and auditable lexical temporal anchors"],
        "exports":["CSV","JSON","NPZ","PNG","SVG for quantitative figures","PDF explanation books","WAV","HTML result index"],
        "source_data":"All quantitative marks come from saved cohort/detail/evidence records",
        "exclusions":"No validation or Attachment-4 prediction omitted. Undefined ratios/ranks flagged. 60-case sampling explicitly authorised.",
        "error_bar_policy":"No population confidence intervals; boxplots display IQR and 1.5-IQR whiskers; random controls retain 20 draws",
        "media_integrity":"No enhancement of original frames. Audio clips use the source timeline. Missing source-feature times explicitly unavailable."}
    q3_write_json(out/"figure_contract.json",contract)
    vfigs=q3_plot_validation(cohort,details,summary,out/"validation")
    media,afigs=q3_plot_attachment4(args,out/"attachment4")
    q3_export_chinese_validation_tables(out,cohort,details,summary)
    q3_export_chinese_attachment_tables(out,media)
    q3_write_result_index(out,cohort,summary,media)
    files=[p for p in out.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    manifest={
        "schema":"q3_completion_delivery_v1","display_language":"zh-CN","created_utc":datetime.now(timezone.utc).isoformat(),
        "script":str(Path(__file__).resolve()),"script_sha256":sha256(__file__),
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":sha256(args.checkpoint),
        "validation_count":len(cohort["records"]),"validation_detail_count":len(details),"attachment4_count":media["sample_count"],
        "validation_metrics":cohort["metrics"],"all_validation_integrations_converged":summary["all_integrations_converged"],
        "validation_integration_edges":summary["integration_edge_count"],
        "max_validation_integral_error":summary["max_absolute_integral_error"],
        "attachment4_media_errors":media["issues"],"raw_feature_time_metadata_recovered":False,
        "entry_point":"结果索引.html","figures":{"validation":len(vfigs),"attachment4":len(afigs)},
        "reproduce_command":f"{sys.executable} -B {Path(__file__).resolve()} --complete-q3 --device {args.device} --output-dir {out}",
        "limitations":["Validation was used for model/adapter selection and is not an independent test set.",
                       "Raw validation audio/video is unavailable; its audio/vision attribution remains in feature-slot space.",
                       "Attachment-4 media is linked by post-hoc lexical timing; original cached feature time provenance is unavailable.",
                       "Truncated, raw-zero and low-confidence cases are kept and labelled."],
        "files":[{"file":str(p.relative_to(out)),"bytes":p.stat().st_size} for p in sorted(files)]}
    if manifest["checkpoint_sha256"]!=cohort["identity"]["checkpoint_sha256"]:
        raise AssertionError("Checkpoint changed after validation preparation")
    q3_write_json(out/"completion_manifest.json",manifest)
    log("Completion figures, explanation books and result index written")


def q3_error_diagnostics(records,details,out):
    rows=[];case_rows=[]
    for r in records:
        if r["evaluation"]["classification_correct"]:
            continue
        winner=r["prediction"]["class_index"]; truth=r["truth"]["class_index"]
        for baseline in ("mean","zero"):
            cp=r["baselines"][baseline]["coalition_predictions"]
            logits=np.asarray(cp["logits"],float)
            contrast=logits[:,winner]-logits[:,truth]
            scores=np.stack([contrast,np.asarray(cp["regression"])],axis=1)
            phi=shapley_from_coalitions(scores)[0]
            rows.append({"id":r["id"],"dataset_id":r["dataset_id"],"baseline":baseline,
                         "true_class":truth,"wrong_predicted_class":winner,
                         "wrong_vs_true_logit_margin_full":float(contrast[7]),
                         "wrong_vs_true_logit_margin_reference":float(contrast[0]),
                         "text_shapley":float(phi[0]),"audio_shapley":float(phi[1]),"vision_shapley":float(phi[2]),
                         "absolute_main_modalities":[MODALITIES[j] for j in tied_dominant_indices(np.abs(phi))],
                         "interpretation":"Positive increases wrong-class advantage over the true class relative to this reference; not causal blame."})
    lookup={r["id"]:r for r in rows if r["baseline"]=="mean"}
    for r in details:
        wrong=not r["evaluation"]["classification_correct"]
        winner=r["prediction"]["class_index"];truth=r["truth"]["class_index"]
        full_logits=np.asarray(r["prediction"]["logits"])
        edited=[]
        for word in r["word_checks"]["candidates"]:
            logits=np.asarray(word["logits"])
            edited.append({"word":word["word"],"roles":word["roles"],"predicted_class_after":word["class_index"],
                           "correct_after":word["class_index"]==truth,
                           "wrong_vs_true_margin_drop":float((full_logits[winner]-full_logits[truth])-(logits[winner]-logits[truth])) if wrong else None,
                           "regression_absolute_error_after":float(abs(word["regression"]-r["truth"]["regression"]))})
        recoveries=[w["word"] for w in edited if wrong and w["correct_after"]]
        supporting=sorted((w for w in r["words"] if not w.get("truncated")),
                           key=lambda w:-w["attribution_net"][0])[:3]
        terms=[w["word"] for w in supporting if w["attribution_net"][0]>0]
        contrast=lookup.get(r["id"])
        parts=[f"真实类别{truth}，预测类别{winner}，强度绝对误差{r['evaluation']['regression_absolute_error']:.3f}。"]
        if wrong and contrast:
            named=", ".join(f"{m}={contrast[m+'_shapley']:+.4f}" for m in MODALITIES)
            parts.append("错误类别相对真实类别的优势，其训练均值参考Shapley贡献为："+named+"。")
            parts.append("这里的正值支持错误类别相对真实类别的优势，负值抵消这种优势。")
        if terms:
            parts.append("对原预测第一名与第二名优势具有较大正向文本净贡献的词包括："+", ".join(terms)+"。")
        if wrong:
            parts.append("完整词UNK替换后改判正确的已检验词："+(", ".join(recoveries) if recoveries else "无")+"。")
        parts.append("词替换会重新编码全文；以上为具体操作下的可复核模型响应，不据此断言词语语义本身是错误原因。")
        case_rows.append({"id":r["id"],"dataset_id":r["dataset_id"],"true_class":truth,
                          "predicted_class":winner,"classification_correct":not wrong,
                          "regression_absolute_error":r["evaluation"]["regression_absolute_error"],
                          "error_contrast":contrast,"original_margin_supporting_words":terms,
                          "word_substitution_recoveries":recoveries,"tested_word_responses":edited,
                          "diagnostic_note":"".join(parts)})
    q3_write_csv(Path(out)/"wrong_vs_true_modality_contributions.csv",rows)
    q3_write_csv(Path(out)/"diagnostic_case_error_analysis.csv",case_rows)
    q3_write_json(Path(out)/"diagnostic_case_error_analysis.json",case_rows)
    wrong=[r for r in rows if r["baseline"]=="mean"]
    notes=[]
    if wrong:
        means={m:float(np.mean([r[m+"_shapley"] for r in wrong])) for m in MODALITIES}
        notes.append(f"对全部{len(wrong)}条分类错误，另以“错误预测类别logit减真实类别logit”构建错误对比目标。在训练均值参考下，三模态平均Shapley贡献为"+
                     "，".join(f"{m}={means[m]:+.4f}" for m in MODALITIES)+"。该指标明确针对错类与真类的竞争，与原winner-runner解释分开保存。")
    repaired=[r for r in case_rows if r["word_substitution_recoveries"]]
    notes.append(f"60条诊断样本中的30条分类错误里，{len(repaired)}条至少存在一个已检验完整词，其UNK替换使预测改为真实类别。具体词、类别变化及错误对比优势变化均保存在逐例错误分析文件；这不是自然删除词语的因果效果。")
    return rows,case_rows,notes



def q3_plot_error_contrast(out):
    import csv
    plt=q3_plot_style()
    out=Path(out)
    with (out/"wrong_vs_true_modality_contributions.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if r["baseline"]=="mean"]
    fig,ax=plt.subplots(figsize=(5.1,3.8))
    fig.subplots_adjust(left=.17,right=.97,bottom=.22,top=.86)
    data=[[float(r[m+"_shapley"]) for r in rows] for m in MODALITIES]
    bp=ax.boxplot(data,patch_artist=True,showfliers=True,widths=.5,
                  medianprops={"color":"#19262C","linewidth":1},
                  flierprops={"marker":".","markersize":2,"alpha":.25,"markeredgecolor":"#69747B"})
    for patch,color in zip(bp["boxes"],["#3B6E8C","#BF8845","#5C8D78"]):
        patch.set_facecolor(color);patch.set_alpha(.7)
    ax.axhline(0,color="#8C9295",ls="--",lw=.8)
    ax.set(xticks=range(1,4),xticklabels=["文本","语音","视觉"],ylabel="错类相对真类得分差的 Shapley 值",
           title=f"错误预测的模态贡献分析（{len(rows)}条）")
    fig.text(.17,.055,"采用训练均值参考；正值增强错误类别相对真实类别的得分优势。",fontsize=7)
    result=q3_export_figure(fig,out/"figures"/"07_wrong_vs_true_contrast",
                    "哪些模态增强了错误类别相对真实类别的优势？",
                    "wrong_vs_true_modality_contributions.csv")
    plt.close(fig)
    return result


def q3_overview_source(args, sample_id):
    """Load and verify saved data for a task-specific original-media overview."""
    sid=str(sample_id).zfill(2)
    completion=Path(args.completion_root).resolve()
    media_root=completion/"attachment4"
    destination=(Path(args.output_dir).resolve() if args.output_dir else
                 media_root/"figures"/("sample"+sid))
    if not destination.is_relative_to(ROOT/"question3"):
        raise ValueError("Overview outputs must remain inside question3")
    destination.mkdir(parents=True,exist_ok=True)
    report_path=Path(args.attachment4_report)
    report=json.loads(report_path.read_text(encoding="utf-8"))
    record=next(row for row in report["samples"] if row["id"]==sid)
    evidence_path=media_root/"samples"/sid/"evidence.json"
    evidence=json.loads(evidence_path.read_text(encoding="utf-8"))
    alignment_path=media_root/evidence["alignment_file"]
    alignment=json.loads(alignment_path.read_text(encoding="utf-8"))
    words=[w for w in evidence["words"] if w["represented"] and w["full_word_represented"]]
    if len(words)!=len(evidence["words"]):
        raise ValueError("This overview requires all words to be represented; handle truncation explicitly")
    valid=np.asarray(record["valid_mask"],bool)
    content=np.asarray(record["content_mask"],bool)
    special=np.flatnonzero(valid & ~content).tolist()
    if len(special)!=2 or special[0]!=0 or special[-1]!=int(np.flatnonzero(valid)[-1]):
        raise ValueError("Overview expects one starting and one ending special position")
    with np.load(report_path.parent/"attributions.npz",allow_pickle=False) as z:
        arrays={m:np.asarray(z[f"s{sid}_mean_{m}"],np.float64) for m in MODALITIES}
    heat=np.zeros((2,3,len(words)+2),np.float64)
    masses=np.zeros_like(heat)
    for hi,head in enumerate(HEADS):
        for mi,m in enumerate(MODALITIES):
            if np.any(arrays[m][hi,~valid]):
                raise ValueError("Nonzero padding attribution cannot be silently omitted")
            heat[hi,mi,0]=arrays[m][hi,special[0]].sum()
            heat[hi,mi,-1]=arrays[m][hi,special[1]].sum()
            masses[hi,mi,0]=np.abs(arrays[m][hi,special[0]]).sum()
            masses[hi,mi,-1]=np.abs(arrays[m][hi,special[1]]).sum()
            for wi,w in enumerate(words):
                value=sum(float(arrays[m][hi,slot].sum())*weight
                          for slot,weight in zip(w["token_slots"],w["slot_weights"]))
                mass=sum(float(np.abs(arrays[m][hi,slot]).sum())*weight
                         for slot,weight in zip(w["token_slots"],w["slot_weights"]))
                np.testing.assert_allclose(value,w["attribution_net"][head][m],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(mass,w["attribution_mass"][head][m],rtol=1e-10,atol=1e-10)
                heat[hi,mi,wi+1]=value;masses[hi,mi,wi+1]=mass
            np.testing.assert_allclose(heat[hi,mi].sum(),arrays[m][hi].sum(),rtol=1e-10,atol=1e-10)
    base=record["baselines"]["mean"]
    reference=np.asarray(base["coalitions"][0],float)
    full=np.asarray(base["coalitions"][7],float)
    phi=np.asarray(base["phi"],float)
    np.testing.assert_allclose(reference+phi.sum(1),full,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(heat.sum(2),base["integrated_modality_totals"],rtol=1e-10,atol=1e-10)
    if not all(w["start_s"] is not None and w["end_s"]>w["start_s"] for w in words):
        raise ValueError("Overview requires observed word intervals")
    limits=[max(float(np.abs(heat[h]).max()),1e-12) for h in range(2)]
    columns=[{"label":"[CLS]","kind":"special","slot":special[0]}]
    columns += [{"label":w["word"],"kind":"word","word_index":w["word_index"],
                 "start_s":w["start_s"],"end_s":w["end_s"],"token_slots":w["token_slots"],
                 "slot_weights":w["slot_weights"],"raw_char_span":w["raw_char_span"]} for w in words]
    columns += [{"label":"[SEP]","kind":"special","slot":special[-1]}]
    source={
        "sample_id":sid,"baseline":"training_mean","prediction":record["prediction"],
        "reference_output":reference.tolist(),"full_coalition_output":full.tolist(),
        "signed_modality_shapley":phi.tolist(),"modality_shares":base["shares"],
        "columns":columns,"heatmap_signed_net":heat.tolist(),"absolute_attribution_mass":masses.tolist(),
         "heatmap_color_limits":{head:[-limits[h],limits[h]] for h,head in enumerate(HEADS)},"heatmap_modality_sums":heat.sum(2).tolist(),
        "shapley_completeness_residual":(heat.sum(2)-phi).tolist(),
        "source_feature_windows_recovered":False,
        "time_mapping":"WhisperX post-hoc word intervals; no assigned media time for CLS/SEP",
        "top_evidence":evidence["top_evidence"],
        "source_files":{str(v):sha256(v) for v in
            [report_path,report_path.parent/"attributions.npz",evidence_path,alignment_path]},
        "video_start_s":alignment["video_pts_s"][0],
        "video_end_s":alignment["video_frame_end_s"][-1],
        "padding_positions_omitted":np.flatnonzero(~valid).tolist(),
        "padding_attribution_exactly_zero":True}
    q3_write_json(destination/(sid+"_overview_source_data.json"),source)
    return destination,media_root,record,evidence,alignment,words,heat,source


def q3_plot_sample_overview(args, sample_id):
    """Draw separate target overviews from saved values and original media."""
    from PIL import Image
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle, Polygon
    destination,media_root,record,evidence,alignment,words,heat,source=q3_overview_source(args,sample_id)
    sid=record["id"]
    plt=q3_plot_style()
    modalities=list(MODALITIES)
    modality_colors=["#3B6E8C","#BF8845","#5C8D78"]
    translation=q3_zh_case(record)
    gloss={"Replacing":"更换","these":"这些","wear":"磨损","components":"部件","when":"当……时",
           "replacing":"更换","the":"该","timing":"正时","belt":"皮带","is":"是","essential":"至关重要",
           "to":"以；用于","ensuring":"确保","new":"新的","performs":"发挥性能","its":"其",
           "mileage":"使用里程","requirements":"要求"}
    labels=["[CLS]\n起始标记"]+[w["word"]+"\n"+gloss.get(w["word"],translation["word_zh"].get(w["word"],w["word"]))
                                    for w in words]+["[SEP]\n结束标记"]
    column_by_word={w["word_index"]:i+1 for i,w in enumerate(words)}
    ncol=len(labels)
    cmap=plt.get_cmap("RdBu")
    start=float(source["video_start_s"]);end=float(source["video_end_s"])
    duration=end-start
    outputs=[]
    book_path=destination/(sid+"_attribution_overview.pdf")
    with PdfPages(book_path) as book:
        for hi,head in enumerate(HEADS):
            limit=source["heatmap_color_limits"][head][1]
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
            fig=plt.figure(figsize=(13,7.5))
            base=record["baselines"]["mean"]
            phi=np.asarray(base["phi"][hi],float)
            shares=np.asarray(base["shares"][hi],float)
            reference=float(source["reference_output"][hi])
            final=float(source["full_coalition_output"][hi])
            title_head="情感极性分类" if head=="classification" else "情感强度回归"
            main= q3_zh(modalities[int(np.argmax(shares))])
            polarity=("负向","中性","正向")[record["prediction"]["class_index"]]
            fig.text(.075,.920,f"预测：{polarity}    情感强度：{record['prediction']['regression']:+.3f}"
                     f"    主要参考模态：{main}（{100*max(shares):.2f}%）",fontsize=10)
            axes=[]
            water=fig.add_axes([.075,.660,.260,.210]);axes.append(water)
            water.set_title("模态贡献的加和分解",fontsize=10,pad=12)
            running=reference
            levels=[reference]
            water.bar(0,reference,bottom=0,width=.63,color="#AFB3AC",edgecolor="#848A84",linewidth=.7)
            water.annotate(f"{reference:+.4f}",(0,reference),xytext=(0,7 if reference>=0 else -10),
                           textcoords="offset points",ha="center",va="bottom" if reference>=0 else "top",fontsize=8.5)
            for mi,value in enumerate(phi):
                previous=running;running+=float(value);levels.append(running)
                water.bar(mi+1,float(value),bottom=previous,width=.63,
                          color=modality_colors[mi],edgecolor=modality_colors[mi],linewidth=.8)
                water.plot([mi+.315,mi+.685],[previous,previous],color="#909898",lw=.65,ls="--")
                water.annotate(f"{value:+.4f}",(mi+1,running),xytext=(0,7 if value>=0 else -10),
                               textcoords="offset points",ha="center",va="bottom" if value>=0 else "top",fontsize=8.5)
            water.bar(4,final,bottom=0,width=.63,color="#424A4D",edgecolor="#424A4D",linewidth=.7)
            water.plot([3.315,3.685],[final,final],color="#909898",lw=.65,ls="--")
            water.annotate(f"{final:+.4f}",(4,final),xytext=(0,7 if final>=0 else -10),
                           textcoords="offset points",ha="center",va="bottom" if final>=0 else "top",fontsize=8.5)
            low=min(0,*levels,final);high=max(0,*levels,final);span=max(high-low,1e-3)
            water.set_ylim(low-.25*span,high+.28*span)
            water.set_xlim(-.6,4.6)
            water.axhline(0,color="#C8CDCD",lw=.7,zorder=0)
            water.set_xticks(range(5),["参考\n输入"]+
                [q3_zh(m)+f"\n{100*shares[i]:.2f}%" for i,m in enumerate(modalities)]+["完整\n输入"])
            water.tick_params(axis="both",labelsize=8)
            winner=("负向","中性","正向")[record["prediction"]["class_index"]]
            runner=("负向","中性","正向")[record["prediction"]["runnerup_index"]]
            water.set_ylabel(f"{winner}相对{runner}的得分差" if hi==0 else "情感强度分数",fontsize=9)
            frame_grid=fig.add_gridspec(1,3,left=.405,right=.965,bottom=.648,top=.87,wspace=.10)
            frame_records=[]
            for j,item in enumerate(evidence["top_evidence"][head]["vision"]):
                asset=item.get("asset")
                if not asset: raise ValueError("Overview needs the original ranked frame asset")
                ax=fig.add_subplot(frame_grid[0,j]);axes.append(ax)
                with Image.open(media_root/asset["file"]) as image:
                    raw=np.asarray(image)
                    ax.imshow(raw,interpolation="none")
                    ax.set_box_aspect(image.height/image.width)
                ax.set_anchor("N");ax.set_axis_off()
                ax.set_title(f"视觉关键帧 {j+1}",fontsize=10,pad=12)
                word_label=q3_zh_word(translation,item["word"])
                ax.text(.5,-.07,word_label+f"\n{asset['frame_pts_s']:.3f}秒｜第{asset['decoded_frame_index']}帧",
                        transform=ax.transAxes,ha="center",va="top",fontsize=8.5,linespacing=1.5)
                frame_records.append({"rank":item["rank"],"word_index":item["word_index"],"word":item["word"],
                    "asset":asset["file"],"asset_sha256":sha256(media_root/asset["file"]),
                    "frame_pts_s":asset["frame_pts_s"],"frame_index":asset["decoded_frame_index"]})
            axh=fig.add_axes([.075,.425,.89,.115]);axes.append(axh)
            mesh=axh.pcolormesh(np.arange(ncol+1),np.arange(4),heat[hi],cmap=cmap,norm=norm,
                               edgecolors="#FFFFFF",linewidth=.5,shading="flat")
            axh.set_xlim(0,ncol);axh.set_ylim(3,0)
            axh.set_xticks(np.arange(ncol)+.5,labels,rotation=90,ha="right",va="center",rotation_mode="anchor",fontsize=7.5)
            axh.set_yticks(np.arange(3)+.5,["文本","语音","视觉"],fontsize=9)
            axh.tick_params(axis="both",length=0,pad=6)
            for tick,color in zip(axh.get_yticklabels(),modality_colors):tick.set_color(color)
            for spine in axh.spines.values():spine.set_visible(False)
            for mi,m in enumerate(modalities):
                for item in evidence["top_evidence"][head][m]:
                    col=column_by_word[item["word_index"]]
                    axh.add_patch(Rectangle((col+.045,mi+.045),.91,.91,fill=False,
                                            edgecolor="#222B30",linewidth=1.3))
            axh.set_title("三模态局部归因",loc="left",fontsize=10,pad=12)
            cbax=fig.add_axes([.783,.568,.182,.012])
            colorbar=fig.colorbar(mesh,cax=cbax,orientation="horizontal")
            colorbar.set_ticks([-limit,0,limit],labels=[f"{-limit:.3f}","0",f"{limit:.3f}"])
            colorbar.ax.tick_params(labelsize=7.5,length=2,pad=2)
            colorbar.outline.set_linewidth(.4)

            # fig.text(.783,.591,"净贡献：红色为负，蓝色为正",fontsize=8)

            axt=fig.add_axes([.075,.115,.89,.150]);axes.append(axt)
            axt.set_xlim(start,end);axt.set_ylim(-.02,1.02)
            for wi,w in enumerate(words):
                col=wi+1
                left=start+col/ncol*duration;right=start+(col+1)/ncol*duration
                axt.add_patch(Polygon([(left,1),(right,1),(w["end_s"],.61),(w["start_s"],.61)],
                                     closed=True,facecolor="#DFE9EF",edgecolor="white",linewidth=.55,alpha=.85))
                axt.add_patch(Rectangle((w["start_s"],.0),w["end_s"]-w["start_s"],.265,
                                       facecolor="#EDF2F4",edgecolor="white",linewidth=.55))
            for mi,m in enumerate(("audio","text")):
                band_y=.015 if m=="audio" else .145
                color=modality_colors[modalities.index(m)]
                for item in evidence["top_evidence"][head][m]:
                    axt.add_patch(Rectangle((item["start_s"],band_y),
                                            item["end_s"]-item["start_s"],.105,
                                            facecolor=color+"30",edgecolor=color,linewidth=1.25))
            for item in frame_records:
                t=item["frame_pts_s"]
                axt.scatter([t],[.34],marker="v",s=34,color=modality_colors[2],zorder=5)
                axt.text(t,.43,str(item["rank"]),ha="center",va="bottom",fontsize=8.5,
                         color="#356851",zorder=6)
            axt.set_yticks([.1975,.0675],["文本","语音"],fontsize=8)
            axt.tick_params(axis="y",length=0,pad=7)
            axt.set_xticks(np.arange(int(np.ceil(start)),int(np.floor(end))+1))
            axt.tick_params(axis="x",labelsize=8)
            axt.set_xlabel("原始视频时间（秒）",fontsize=9,labelpad=5)
            for side in ["top","right","left"]:axt.spines[side].set_visible(False)
            axt.spines["bottom"].set_color("#98A5AB")
            fig.text(.075,.295,"WhisperX 词语时间对应",fontsize=10)
            # fig.text(.51,.295,"蓝框：文本前三｜橙框：语音前三｜绿标：视觉前三帧",fontsize=8.5)
            # target_note=(f"分类解释目标为“{winner}相对{runner}的得分差”；正贡献增大该差值，负贡献减小该差值。"
            #              if hi==0 else "回归解释目标为情感强度分数；正贡献提高分数，负贡献降低分数。")
            # fig.text(.075,.045,target_note+"同图三模态共用色标；两任务数值不直接比较。",fontsize=7.5)
            # fig.text(.075,.019,"起始、结束标记参与归因但不对应媒体时间；连带表示词语时间对应，不代表缓存特征的原始提取窗口。",fontsize=7.5)
            prefix=destination/(sid+"_"+head+"_overview")
            fig.canvas.draw()
            alignment_report=q3_alignment_auditor()(fig,json_out=prefix.with_suffix(".alignment.json"),
                tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,
                axes=axes,panel_ids=list("abcdef"),row_groups=[["b","c","d"]],column_groups=[["e","f"]])
            fig.savefig(prefix.with_suffix(".pdf"))
            fig.savefig(prefix.with_suffix(".svg"))
            fig.savefig(prefix.with_suffix(".png"),dpi=360)
            book.savefig(fig)
            metadata={"sample_id":sid,"head":head,"size_inches":[13,7.5],"dpi":360,
                      "source_sha256":sha256(__file__),"source_data":sid+"_overview_source_data.json",
                      "heatmap_columns":ncol,"heatmap_signed_net":heat[hi].tolist(),
                      "reference_output":reference,"signed_shapley":phi.tolist(),"full_output":final,
                      "frames":frame_records,"alignment_verdict":alignment_report["verdict"],
                      "rendered_text":[t.get_text() for t in fig.findobj(match=__import__("matplotlib").text.Text)
                                       if t.get_visible() and t.get_text()],
                      "formats":["pdf","svg","png"],"prediction_values_unchanged":True,
                      "image_processing":"original frames; no crop or enhancement",
                      "color_limits":source["heatmap_color_limits"][head],"special_positions_mapped_to_time":False}
            q3_write_json(prefix.with_suffix(".figure.json"),metadata)
            outputs.append({"head":head,"stem":prefix.name,"frames":frame_records})
            plt.close(fig)
    q3_write_json(destination/(sid+"_overview_manifest.json"),
        {"status":"rendered_pending_qa","sample_id":sid,"source_sha256":sha256(__file__),
         "source_data":sid+"_overview_source_data.json","outputs":outputs,"combined_pdf":book_path.name,
         "reproduce_command":f"{sys.executable} -B {Path(__file__).resolve()} --sample-overview {sid}",
         "previous_cards_modified":False})
    print("OVERVIEW_RENDERED="+str(destination),flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--grad-batch", type=int, default=32)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--top-words", type=int, default=5)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures-only", type=Path, help="Existing analysis_report.json; recompute figures only.")
    parser.add_argument("--complete-q3", action="store_true", help="Complete labelled validation and raw-media evidence.")
    parser.add_argument("--completion-stage", choices=("all","prepare","details","media","figures"), default="all")
    parser.add_argument("--validation-dataset", type=Path, default=ROOT / "datasets/附件2-数据集特征文件/aligned_50.pkl")
    parser.add_argument("--reference-dataset", type=Path, default=ROOT / "datasets/附件2-同步扰动特征/aligned_50_p000.pkl")
    parser.add_argument("--attachment4-report", type=Path, default=ROOT / "question3/attachment4_all_20260925T152718_260737Z/analysis_report.json")
    parser.add_argument("--worker-index",type=int,default=0)
    parser.add_argument("--worker-count",type=int,default=1)
    parser.add_argument("--ffmpeg",default=str(ROOT / "work/ffmpeg-7.0.2-amd64-static/ffmpeg"))
    parser.add_argument("--ffprobe",default=str(ROOT / "work/ffmpeg-7.0.2-amd64-static/ffprobe"))
    parser.add_argument("--alignment-cache",type=Path,default=ROOT / ".cache/alignment_models")
    parser.add_argument("--alignment-model",default="WAV2VEC2_ASR_BASE_960H")
    parser.add_argument("--alignment-min-score",type=float,default=0.5)
    parser.add_argument("--evidence-topk",type=int,default=3)
    parser.add_argument("--media-sample-ids",nargs="*")
    parser.add_argument("--sample-overview", help="Draw separate classification/regression media overviews for a saved sample ID.")
    parser.add_argument("--completion-root", type=Path, default=ROOT/"question3/completion_20260926")
    args = parser.parse_args()
    if args.sample_overview:
        return q3_plot_sample_overview(args,args.sample_overview)
    if args.complete_q3:
        return q3_completion_main(args)
    if args.ig_steps < 2 or args.max_steps < 2*args.ig_steps or args.grad_batch < 1:
        parser.error("Need ig-steps>=2, max-steps>=2*ig-steps, grad-batch>=1")
    if args.random_repeats < 2 or args.top_words < 1:
        parser.error("Need at least two random controls and one top word")
    if args.figures_only:
        report_path = args.figures_only.expanduser().resolve()
        out = report_path.parent
        if not out.is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("All outputs must stay within question3")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("quality_assurance", None)
        with np.load(out / "attributions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
    else:
        out = (args.output_dir or (Path(__file__).resolve().parent /
               datetime.now(timezone.utc).strftime("attachment4_all_%Y%m%dT%H%M%S_%fZ"))).expanduser().resolve()
        question3 = Path(__file__).resolve().parent
        if not out.is_relative_to(question3):
            raise ValueError("All new outputs must stay within question3")
        out.mkdir(parents=True, exist_ok=False)
        print(f"OUTPUT_DIR={out}", flush=True)
        script_before = sha256(__file__)
        report, arrays = run_analysis(args)
        if sha256(__file__) != script_before:
            raise RuntimeError("Explanation script changed during numerical analysis")
        report["analysis_script_sha256"] = script_before
        report["sample_selection"] = "All Attachment-4 aligned samples 01--20; no ground-truth labels."
        report["numerical_files"] = ["attributions.npz", "analysis_report.json"]
        with (out / "attributions.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(out / "attributions.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise RuntimeError("Serialized array inventory mismatch")
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        report_path = out / "analysis_report.json"
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if "global_summary" in report:
        report["figures"] = (
            make_attachment4_global_figures(report, arrays, out) +
            make_individual_figures(report, arrays, out)
        )
        report["figure_contract"] = {
            "dataset": "Attachment-4 aligned 01--20, all samples",
            "global_figures": 3, "individual_pngs": 60, "individual_pdf_books": 3,
            "pages_per_individual_pdf": 20,
            "purpose": "Descriptive model-behavior explanation without true labels",
            "no_population_confidence_intervals": True,
            "classification_target": "Fixed original winner-minus-runner-up logit margin",
            "regression_target": "Original raw regression output",
        }
        write_attachment4_reading_guide(report, out)
    else:
        make_figures(report, arrays, out)
    report["figure_backend"] = "python-matplotlib"
    report["figure_script_sha256"] = sha256(__file__)
    report["figure_formats"] = ["png", "pdf"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("COMPLETE: all requested numerical results and figures in " + str(out), flush=True)


if __name__ == "__main__":
    main()

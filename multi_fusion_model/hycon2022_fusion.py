"""Dedicated HyCon arXiv-v1 feature adaptation with joint contrastive training.

This entry uses the v1 intra-modal, inter-modal and semi-contrastive objectives
alongside the supervised task. It does not reproduce the final journal version's
hard-pair selection. Cached contextual text features, aligned within-sample slots
and a nonnegative projection are explicit adaptations, not original raw encoders.
The ratio objectives can legitimately be negative; they are not InfoNCE losses.

Classification retains each supplied class's identity, mapped to consecutive
indices by the shared runtime, without assuming positive/negative sentiment IDs.
For regression, contrastive grouping uses score >= 0; the task head still predicts
the original continuous score and is trained with MAE. Semi-contrastive means
same-sample cross-modal consistency, not a semi-supervised training stage.
All objectives are trained jointly; this dedicated entry has no pretraining.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real

from multimodal_suite import (
    Config as _BaseConfig, fit_experiment as _fit_experiment,
    explain_feature_groups as _explain_feature_groups,
    load_experiment_model, predict_split,
)
from multimodal_suite.robust_pretraining import HyCon2022Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data


AUXILIARY_NAMES = ("hycon_intra", "hycon_inter", "hycon_semi")
_DEFAULT_AUXILIARY_WEIGHTS = {name: 1.0 for name in AUXILIARY_NAMES}
_DEFAULT_MODEL_OPTIONS = {"hycon_margin": 0.8, "hycon_refinement_weight": 1.0}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="hycon2022", init=False)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 1
    dropout: float = 0.2
    pretrain_epochs: int = 0
    auxiliary_weights: dict = field(default_factory=lambda: dict(_DEFAULT_AUXILIARY_WEIGHTS))
    pretrain_auxiliary_weights: dict = field(default_factory=dict)
    model_options: dict = field(default_factory=lambda: dict(_DEFAULT_MODEL_OPTIONS))
    output_root: str = "results/second_question/hycon2022_fusion"

    def __post_init__(self):
        self._merge_options()

    def _merge_options(self):
        if not isinstance(self.model_options, dict):
            raise ValueError("model_options must be a dictionary of recognized HyCon settings")
        unknown = set(self.model_options) - set(_DEFAULT_MODEL_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown HyCon model_options: {list(unknown)!r}")
        self.model_options = {**_DEFAULT_MODEL_OPTIONS, **self.model_options}
        if not isinstance(self.auxiliary_weights, dict):
            raise ValueError("auxiliary_weights must be a dictionary")
        unknown = set(self.auxiliary_weights) - set(AUXILIARY_NAMES)
        if unknown:
            raise ValueError(f"Unknown HyCon objectives in auxiliary_weights: {list(unknown)!r}")
        self.auxiliary_weights = {**_DEFAULT_AUXILIARY_WEIGHTS, **self.auxiliary_weights}
        if not isinstance(self.pretrain_auxiliary_weights, dict) or self.pretrain_auxiliary_weights:
            raise ValueError("pretrain_auxiliary_weights must be {}; HyCon uses joint supervised training")
        self.pretrain_auxiliary_weights = dict(self.pretrain_auxiliary_weights)

    def validate(self):
        self._merge_options()
        if self.model_name != "hycon2022":
            raise ValueError("Use model_name='hycon2022' for this entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs != 0):
            raise ValueError("pretrain_epochs must be integer 0; HyCon contrastive training uses task labels jointly")
        self.pretrain_epochs = 0
        margin = self.model_options["hycon_margin"]
        if (not isinstance(margin, Real) or isinstance(margin, bool)
                or not math.isfinite(margin) or not 0 < margin <= 1):
            raise ValueError("hycon_margin must be finite and in (0, 1]")
        self.model_options["hycon_margin"] = float(margin)
        refinement = self.model_options["hycon_refinement_weight"]
        if (not isinstance(refinement, Real) or isinstance(refinement, bool)
                or not math.isfinite(refinement) or refinement < 0):
            raise ValueError("hycon_refinement_weight must be finite and nonnegative")
        self.model_options["hycon_refinement_weight"] = float(refinement)
        for objective, value in self.auxiliary_weights.items():
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"auxiliary_weights[{objective!r}] must be finite and nonnegative")
            self.auxiliary_weights[objective] = float(value)
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    """Train joint task/contrastive objectives, selecting the model on validation."""
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's HyCon Config so joint-training options are validated")
    config.validate()
    experiment = _fit_experiment(data, config, mask_overrides=mask_overrides)
    for record in experiment.history:
        record["train_loss"] = record["train"]["total"]
        record["train_task_loss"] = record["train"]["supervised"]
    (experiment.run_dir / "history.json").write_text(
        json.dumps(experiment.history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return experiment


def explain_feature_groups(experiment, split="test", index=0, target_class=None,
                           baseline=0.0, perturb_batch_size=32):
    """Measure full-model prediction changes under cached-feature replacement.

    Masks and the target/reference class remain fixed. Contrastive losses are
    training objectives, not per-word attribution scores or contribution shares.
    Cached BERT context can spread a word's information across positions; these
    feature ablations are not raw-word removal or raw-media causal effects.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)

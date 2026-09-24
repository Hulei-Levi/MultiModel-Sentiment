"""MEmoBERT-inspired feature reconstruction followed by task-only fine-tuning.

This is a feature-space adaptation, not original pretrained MEmoBERT. Text inputs
are cached contextual BERT outputs, not the paper's BERT token embedding layer.
No original BERT checkpoint, vocabulary MLM or facial-teacher KL task is loaded.
Learned soft prompts, a query vector and a new prediction head do not reproduce
the original hard "I am [MASK]" prompt or vocabulary-based emotion verbalizer.

Defaults use three reconstruction-only warmup epochs, then supervised task-only
fine-tuning. Nonzero supervised reconstruction weights enable an explicit joint
training experiment. Context already present in cached text can reveal masked
content through other positions; these feature objectives are not whole-word MLM.
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
from multimodal_suite.robust_pretraining import MEmoBERT2022Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data


AUXILIARY_NAMES = ("memobert_masked_text", "memobert_masked_audio", "memobert_masked_vision")
_SUPERVISED_WEIGHTS = {name: 0.0 for name in AUXILIARY_NAMES}
_PRETRAIN_WEIGHTS = {name: 1.0 for name in AUXILIARY_NAMES}
_MODEL_OPTIONS = {"memobert_mask_probability": 0.15, "memobert_span_length": 3}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="memobert2022", init=False)
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    pretrain_epochs: int = 3
    auxiliary_weights: dict = field(default_factory=lambda: dict(_SUPERVISED_WEIGHTS))
    pretrain_auxiliary_weights: dict = field(default_factory=lambda: dict(_PRETRAIN_WEIGHTS))
    model_options: dict = field(default_factory=lambda: dict(_MODEL_OPTIONS))
    output_root: str = "results/second_question/memobert2022_fusion"

    def __post_init__(self):
        self._merge_options()

    def _merge_options(self):
        if not isinstance(self.model_options, dict):
            raise ValueError("model_options must be a dictionary of recognized MEmoBERT settings")
        unknown = set(self.model_options) - set(_MODEL_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown MEmoBERT model_options: {list(unknown)!r}")
        self.model_options = {**_MODEL_OPTIONS, **self.model_options}
        for name, defaults in (("auxiliary_weights", _SUPERVISED_WEIGHTS),
                               ("pretrain_auxiliary_weights", _PRETRAIN_WEIGHTS)):
            weights = getattr(self, name)
            if not isinstance(weights, dict):
                raise ValueError(f"{name} must be a dictionary")
            unknown = set(weights) - set(AUXILIARY_NAMES)
            if unknown:
                raise ValueError(f"Unknown MEmoBERT objectives in {name}: {list(unknown)!r}")
            setattr(self, name, {**defaults, **weights})

    def validate(self):
        self._merge_options()
        if self.model_name != "memobert2022":
            raise ValueError("Use model_name='memobert2022' for this entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs < 0):
            raise ValueError("pretrain_epochs must be a nonnegative integer; explicit 0 skips warmup")
        self.pretrain_epochs = int(self.pretrain_epochs)
        probability = self.model_options["memobert_mask_probability"]
        if (not isinstance(probability, Real) or isinstance(probability, bool)
                or not math.isfinite(probability) or not 0 < probability <= 1):
            raise ValueError("memobert_mask_probability must be finite and in (0, 1]")
        self.model_options["memobert_mask_probability"] = float(probability)
        span = self.model_options["memobert_span_length"]
        if not isinstance(span, Integral) or isinstance(span, bool) or span <= 0:
            raise ValueError("memobert_span_length must be a positive integer")
        self.model_options["memobert_span_length"] = int(span)
        for name in ("auxiliary_weights", "pretrain_auxiliary_weights"):
            for objective, value in getattr(self, name).items():
                if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}[{objective!r}] must be finite and nonnegative")
                getattr(self, name)[objective] = float(value)
        if self.pretrain_epochs and not any(value > 0 for value in self.pretrain_auxiliary_weights.values()):
            raise ValueError("Warmup requires at least one positive pretrain_auxiliary_weights value")
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    """Use training-only feature warmup and validation-selected task fine-tuning."""
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's MEmoBERT Config so phase defaults and options are validated")
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
    """Full-model response to cached-feature replacement with fixed masks/classes.

    This is not raw-token removal, a word-level causal effect or a decomposition
    of Transformer attention. Cached BERT context can distribute word information
    across multiple positions. Prediction uses the uncorrupted inference path;
    synthetic reconstruction masks are not applied during ordinary explanation.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)

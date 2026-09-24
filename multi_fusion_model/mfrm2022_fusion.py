"""Dedicated entry for an MFRM-inspired, principle-level adaptation baseline.

The original full equations were not verified from the accessible abstract and
author README. Seven Hadamard interaction channels, the acoustic intensity score
and the residual update equations are implementation choices, not established
copies of the published MFRM equations. This entry does not claim exact paper
reproduction or original benchmark accuracy.

Training uses the shared runtime without auxiliary objectives or gradient
pretraining. Learned intensity weights route information and are not contribution
percentages. Explanations perturb cached feature groups; contextual BERT features
already mix token information, so this is not raw-word/audio/frame causality.
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
from multimodal_suite.memory_translation import MFRM2022Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data


_DEFAULT_MODEL_OPTIONS = {"mfrm_intensity_gain": 1.0}


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="mfrm2022", init=False)
    d_model: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    pretrain_epochs: int = 0
    model_options: dict = field(default_factory=lambda: dict(_DEFAULT_MODEL_OPTIONS))
    output_root: str = "results/second_question/mfrm2022_fusion"

    def __post_init__(self):
        self._merge_model_options()

    def _merge_model_options(self):
        if not isinstance(self.model_options, dict):
            raise ValueError("model_options must be a dictionary of recognized MFRM settings")
        unknown = set(self.model_options) - set(_DEFAULT_MODEL_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown MFRM model_options: {list(unknown)!r}")
        self.model_options = {**_DEFAULT_MODEL_OPTIONS, **self.model_options}

    def validate(self):
        self._merge_model_options()
        if self.model_name != "mfrm2022":
            raise ValueError("Use model_name='mfrm2022' for the dedicated MFRM entry")
        if (not isinstance(self.pretrain_epochs, Integral) or isinstance(self.pretrain_epochs, bool)
                or self.pretrain_epochs != 0):
            raise ValueError("MFRM pretrain_epochs must be integer 0: this model has no auxiliary pretraining objective")
        self.pretrain_epochs = 0
        for name in ("auxiliary_weights", "pretrain_auxiliary_weights"):
            weights = getattr(self, name)
            if not isinstance(weights, dict) or weights:
                raise ValueError(f"MFRM defines no auxiliary objectives; {name} must be an empty dictionary")
        gain = self.model_options["mfrm_intensity_gain"]
        if not isinstance(gain, Real) or isinstance(gain, bool) or not math.isfinite(gain) or gain < 0:
            raise ValueError("mfrm_intensity_gain must be finite and nonnegative")
        self.model_options["mfrm_intensity_gain"] = float(gain)
        super().validate()


def fit_experiment(data, config=None, mask_overrides=None):
    """Train the declared adaptation baseline and select solely on validation."""
    config = config or Config()
    if not isinstance(config, Config):
        raise ValueError("Use this entry's MFRM Config so model-specific restrictions are validated")
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
    """Perturb cached groups with fixed masks; not raw-media causal attribution.

    The model's acoustic intensity weights are information-routing weights, not
    percentages of evidence or final-score contributions. The output describes
    the full model response to feature replacement, including its learned gates.
    """
    return _explain_feature_groups(experiment, split, index, target_class, baseline, perturb_batch_size)

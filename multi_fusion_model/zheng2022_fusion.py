"""Zheng MCWSA-CMHA adaptation for cached aligned features and sample labels.

Three reconstruction channels share the same cascade fusion network. Their
encoders and decoders are independent. Defaults separate reconstruction-only
warmup from task-only fine-tuning; all prediction-path parameters are fine-tuned.
"""
import json
from dataclasses import dataclass, field

from multimodal_suite import (
    Config as _BaseConfig, fit_experiment as _fit_experiment,
    explain_feature_groups, load_experiment_model, predict_split,
)
from multimodal_suite.attention_models import Zheng2022Model as Model
from multi_fusion_model.weighted_sum_fusion import audit_data


@dataclass
class Config(_BaseConfig):
    model_name: str = field(default="zheng2022", init=False)
    pretrain_epochs: int = 3
    pretrain_auxiliary_weights: dict = field(default_factory=lambda: {"reconstruction": 1.0})
    auxiliary_weights: dict = field(default_factory=lambda: {"reconstruction": 0.0})
    output_root: str = "results/second_question/zheng2022_fusion"

    def validate(self):
        super().validate()
        if self.model_name != "zheng2022":
            raise ValueError("Use model_name='zheng2022' for this entry")


def fit_experiment(data, config=None, mask_overrides=None):
    config = config or Config()
    if config.model_name != "zheng2022":
        raise ValueError("Use a Zheng Config for this model entry")
    experiment = _fit_experiment(data, config, mask_overrides=mask_overrides)
    for record in experiment.history:
        record["train_loss"] = record["train"]["total"]
    (experiment.run_dir / "history.json").write_text(
        json.dumps(experiment.history, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return experiment

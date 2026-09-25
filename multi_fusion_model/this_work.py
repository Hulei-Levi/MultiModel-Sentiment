"""Notebook entry for ThisWork: joint Macro-F1/MAE experiments.

Each batch contains classification_target and regression_target. One model and
one selected checkpoint return classification_logits and regression_prediction.
"""
from multimodal_suite.multitask_runtime import (
    Config, Experiment, MultiTaskFeatureDataset, compute_task_losses,
    fit_experiment, load_experiment_model, predict_split, explain_feature_groups,
    audit_data,
)
from multimodal_suite.attention_models import ThisWork as Model

__all__ = ["Config", "Experiment", "Model", "MultiTaskFeatureDataset",
           "compute_task_losses", "fit_experiment", "load_experiment_model",
           "predict_split", "explain_feature_groups", "audit_data"]

"""Project package."""

"""Baseline human activity recognition components."""

from src.activity_recognition.label_encoder import (
    ActivityLabelEncoder,
    KTH_ACTIVITY_CLASSES,
)
from src.activity_recognition.lstm_model import build_lstm_model
from src.activity_recognition.predictor import ActivityPredictor
from src.activity_recognition.sequence_builder import (
    FeatureRecording,
    SequenceBuilder,
    SequenceDataset,
)
from src.activity_recognition.trainer import (
    ActivityTrainer,
    TrainingResult,
)

__all__ = [
    "ActivityLabelEncoder",
    "KTH_ACTIVITY_CLASSES",
    "build_lstm_model",
    "ActivityPredictor",
    "FeatureRecording",
    "SequenceBuilder",
    "SequenceDataset",
    "ActivityTrainer",
    "TrainingResult",
]
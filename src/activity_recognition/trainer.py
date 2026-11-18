
"""Train and save the baseline LSTM activity recognition model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from src.config import MODELS_DIR, RANDOM_SEED
from src.activity_recognition.lstm_model import build_lstm_model

@dataclass
class TrainingResult:
    """Paths and training history returned after successful training."""

    model_path: Path
    metadata_path: Path
    history: dict[str, list[float]]

class ActivityTrainer:
    """Train, validate, and save an LSTM activity classifier."""

    def __init__(
        self,
        sequence_length: int = 30,
        feature_dim: int = 283,
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.seed = seed

    def _validate_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate and normalize training and validation arrays."""

        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)

        if X_train.ndim != 3 or X_val.ndim != 3:
            raise ValueError("Feature arrays must be three-dimensional.")

        expected_shape = (self.sequence_length, self.feature_dim)

        if X_train.shape[1:] != expected_shape:
            raise ValueError(
                f"Training input must have shape (samples, "
                f"{expected_shape[0]}, {expected_shape[1]})."
            )

        if X_val.shape[1:] != expected_shape:
            raise ValueError(
                f"Validation input must have shape (samples, "
                f"{expected_shape[0]}, {expected_shape[1]})."
            )

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                "Training and validation datasets cannot be empty."
            )

        if y_train.ndim != 1 or y_val.ndim != 1:
            raise ValueError("Label arrays must be one-dimensional.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "Training feature and label counts do not match."
            )

        if len(X_val) != len(y_val):
            raise ValueError(
                "Validation feature and label counts do not match."
            )

        if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
            raise ValueError("Feature arrays contain NaN or infinite values.")

        if not np.issubdtype(y_train.dtype, np.integer):
            raise ValueError("Training labels must contain integers.")

        if not np.issubdtype(y_val.dtype, np.integer):
            raise ValueError("Validation labels must contain integers.")

        if np.any(y_train < 0) or np.any(y_train >= num_classes):
            raise ValueError("Training labels contain invalid class indices.")

        if np.any(y_val < 0) or np.any(y_val >= num_classes):
            raise ValueError("Validation labels contain invalid class indices.")

        return (
            X_train,
            y_train.astype(np.int64),
            X_val,
            y_val.astype(np.int64),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_names: Sequence[str],
        output_dir: Optional[Path] = None,
        epochs: int = 30,
        batch_size: int = 32,
        patience: int = 5,
        verbose: int = 1,
    ) -> TrainingResult:
        """Train the LSTM and save its model and metadata."""

        if epochs <= 0 or batch_size <= 0 or patience <= 0:
            raise ValueError(
                "epochs, batch_size, and patience must be greater than zero."
            )

        normalized_classes = tuple(
            str(name).strip().lower() for name in class_names
        )

        if len(normalized_classes) < 2:
            raise ValueError("At least two activity classes are required.")

        if any(not name for name in normalized_classes):
            raise ValueError("Activity class names cannot be empty.")

        if len(set(normalized_classes)) != len(normalized_classes):
            raise ValueError("Activity class names must be unique.")

        X_train, y_train, X_val, y_val = self._validate_arrays(
            X_train,
            y_train,
            X_val,
            y_val,
            len(normalized_classes),
        )

        tf.keras.utils.set_random_seed(self.seed)

        model = build_lstm_model(
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim,
            num_classes=len(normalized_classes),
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        destination = Path(output_dir) if output_dir else MODELS_DIR
        destination.mkdir(parents=True, exist_ok=True)

        model_path = destination / "activity_lstm.keras"
        metadata_path = destination / "activity_lstm_metadata.json"

        model.save(model_path)

        metadata = {
            "model_name": "human_activity_lstm",
            "model_type": "LSTM",
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "num_classes": len(normalized_classes),
            "class_names": list(normalized_classes),
            "random_seed": self.seed,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        return TrainingResult(
            model_path=model_path,
            metadata_path=metadata_path,
            history={
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        )
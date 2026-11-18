
"""Load a trained LSTM and predict human activities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

class ActivityPredictor:
    """Make activity predictions using a saved LSTM model."""

    def __init__(
        self,
        model: tf.keras.Model,
        class_names: list[str],
        sequence_length: int,
        feature_dim: int,
    ) -> None:
        if not class_names:
            raise ValueError("At least one activity class is required.")

        if sequence_length <= 0 or feature_dim <= 0:
            raise ValueError("Input dimensions must be positive.")

        self.model = model
        self.class_names = tuple(class_names)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        metadata_path: str | Path,
    ) -> "ActivityPredictor":
        """Load a model and its metadata from disk."""

        model_file = Path(model_path)
        metadata_file = Path(metadata_path)

        if not model_file.is_file():
            raise FileNotFoundError(f"Model not found: {model_file}")

        if not metadata_file.is_file():
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_file}"
            )

        metadata = json.loads(
            metadata_file.read_text(encoding="utf-8")
        )

        model = tf.keras.models.load_model(model_file)

        return cls(
            model=model,
            class_names=metadata["class_names"],
            sequence_length=int(metadata["sequence_length"]),
            feature_dim=int(metadata["feature_dim"]),
        )

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Predict the activity represented by one sequence or a batch."""

        features = np.asarray(features, dtype=np.float32)

        if features.ndim == 2:
            features = np.expand_dims(features, axis=0)
        elif features.ndim != 3:
            raise ValueError(
                "Input must have shape (frames, features) or "
                "(samples, frames, features)."
            )

        expected_shape = (self.sequence_length, self.feature_dim)

        if features.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected input shape (samples, "
                f"{expected_shape[0]}, {expected_shape[1]})."
            )

        if len(features) == 0:
            raise ValueError("Prediction input cannot be empty.")

        if not np.isfinite(features).all():
            raise ValueError("Prediction input contains invalid values.")

        probabilities = self.model.predict(features, verbose=0)

        if probabilities.shape != (len(features), len(self.class_names)):
            raise ValueError("Model output does not match the class metadata.")

        results = []

        for sample_probabilities in probabilities:
            predicted_index = int(np.argmax(sample_probabilities))

            results.append(
                {
                    "predicted_activity": self.class_names[predicted_index],
                    "confidence": float(
                        sample_probabilities[predicted_index]
                    ),
                    "probabilities": {
                        class_name: float(probability)
                        for class_name, probability in zip(
                            self.class_names,
                            sample_probabilities,
                        )
                    },
                }
            )

        return results[0] if len(results) == 1 else {"predictions": results}
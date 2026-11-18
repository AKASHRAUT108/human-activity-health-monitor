
"""Tests for activity training and prediction components."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.activity_recognition.lstm_model import build_lstm_model
from src.activity_recognition.predictor import ActivityPredictor
from src.activity_recognition.trainer import ActivityTrainer

class TestActivityTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = ActivityTrainer(
            sequence_length=10,
            feature_dim=283,
            lstm_units=16,
        )

    def test_rejects_wrong_input_dimensions(self):
        X_train = np.ones((4, 10, 100), dtype=np.float32)
        y_train = np.array([0, 1, 0, 1])
        X_val = np.ones((2, 10, 100), dtype=np.float32)
        y_val = np.array([0, 1])

        with self.assertRaises(ValueError):
            self.trainer._validate_arrays(
                X_train, y_train, X_val, y_val, 2
            )

    def test_rejects_empty_training_data(self):
        X_train = np.empty((0, 10, 283), dtype=np.float32)
        y_train = np.array([], dtype=np.int64)
        X_val = np.ones((2, 10, 283), dtype=np.float32)
        y_val = np.array([0, 1])

        with self.assertRaises(ValueError):
            self.trainer._validate_arrays(
                X_train, y_train, X_val, y_val, 2
            )

    def test_rejects_mismatched_labels(self):
        X_train = np.ones((4, 10, 283), dtype=np.float32)
        y_train = np.array([0, 1])
        X_val = np.ones((2, 10, 283), dtype=np.float32)
        y_val = np.array([0, 1])

        with self.assertRaises(ValueError):
            self.trainer._validate_arrays(
                X_train, y_train, X_val, y_val, 2
            )

    def test_rejects_invalid_class_index(self):
        X_train = np.ones((4, 10, 283), dtype=np.float32)
        y_train = np.array([0, 1, 0, 3])
        X_val = np.ones((2, 10, 283), dtype=np.float32)
        y_val = np.array([0, 1])

        with self.assertRaises(ValueError):
            self.trainer._validate_arrays(
                X_train, y_train, X_val, y_val, 2
            )

class TestActivityPredictor(unittest.TestCase):

    def setUp(self):
        self.class_names = ["walking", "running", "boxing"]
        self.model = build_lstm_model(
            sequence_length=10,
            feature_dim=283,
            num_classes=3,
            lstm_units=16,
        )
        self.predictor = ActivityPredictor(
            model=self.model,
            class_names=self.class_names,
            sequence_length=10,
            feature_dim=283,
        )

    def test_predict_single_sequence(self):
        features = np.ones((10, 283), dtype=np.float32)

        result = self.predictor.predict(features)

        self.assertIn("predicted_activity", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["predicted_activity"], self.class_names)
        self.assertEqual(len(result["probabilities"]), 3)

    def test_predict_batch(self):
        features = np.ones((2, 10, 283), dtype=np.float32)

        result = self.predictor.predict(features)

        self.assertEqual(len(result["predictions"]), 2)

    def test_invalid_input_dimensions(self):
        features = np.ones((10, 100), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.predictor.predict(features)

    def test_nan_input_is_rejected(self):
        features = np.ones((10, 283), dtype=np.float32)
        features[0, 0] = np.nan

        with self.assertRaises(ValueError):
            self.predictor.predict(features)

    def test_load_saved_model_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            model_path = directory / "test_model.keras"
            metadata_path = directory / "metadata.json"

            self.model.save(model_path)

            metadata = {
                "class_names": self.class_names,
                "sequence_length": 10,
                "feature_dim": 283,
            }

            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            loaded = ActivityPredictor.load(
                model_path,
                metadata_path,
            )

            self.assertEqual(loaded.class_names, tuple(self.class_names))

            result = loaded.predict(
                np.ones((10, 283), dtype=np.float32)
            )

            self.assertIn(result["predicted_activity"], self.class_names)

if __name__ == "__main__":
    unittest.main()
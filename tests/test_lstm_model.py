
"""Tests for the baseline LSTM model."""

import unittest

import numpy as np
import tensorflow as tf

from src.activity_recognition.lstm_model import build_lstm_model

class TestLSTMModel(unittest.TestCase):

    def test_model_input_and_output_shapes(self):
        model = build_lstm_model(
            sequence_length=10,
            feature_dim=283,
            num_classes=6,
            lstm_units=16,
        )

        self.assertEqual(model.input_shape, (None, 10, 283))
        self.assertEqual(model.output_shape, (None, 6))

    def test_model_prediction_probabilities(self):
        model = build_lstm_model(
            sequence_length=10,
            feature_dim=283,
            num_classes=6,
            lstm_units=16,
        )

        sample = np.random.random((2, 10, 283)).astype(np.float32)
        predictions = model.predict(sample, verbose=0)

        self.assertEqual(predictions.shape, (2, 6))
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertTrue(np.all(predictions <= 1.0))

        np.testing.assert_allclose(
            predictions.sum(axis=1),
            np.ones(2),
            atol=1e-5,
        )

    def test_model_contains_lstm_layer(self):
        model = build_lstm_model(
            sequence_length=10,
            feature_dim=283,
            num_classes=6,
            lstm_units=16,
        )

        self.assertTrue(
            any(
                isinstance(layer, tf.keras.layers.LSTM)
                for layer in model.layers
            )
        )

    def test_invalid_sequence_length(self):
        with self.assertRaises(ValueError):
            build_lstm_model(sequence_length=0)

    def test_invalid_feature_dimension(self):
        with self.assertRaises(ValueError):
            build_lstm_model(feature_dim=0)

    def test_invalid_number_of_classes(self):
        with self.assertRaises(ValueError):
            build_lstm_model(num_classes=1)

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            build_lstm_model(dropout_rate=1.0)

if __name__ == "__main__":
    unittest.main()
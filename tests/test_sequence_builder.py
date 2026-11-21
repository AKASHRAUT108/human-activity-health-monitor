
"""Tests for activity sequence preparation."""

import unittest

import numpy as np

from src.activity_recognition.label_encoder import ActivityLabelEncoder
from src.activity_recognition.sequence_builder import (
    FeatureRecording,
    SequenceBuilder,
)

class TestSequenceBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = SequenceBuilder(
            sequence_length=4,
            stride=2,
            feature_dim=283,
        )

    def test_builds_expected_number_of_sequences(self):
        recording = FeatureRecording(
            features=np.ones((10, 283), dtype=np.float32),
            label="walking",
            sample_id="video01",
        )

        dataset = self.builder.build([recording])

        self.assertEqual(dataset.X.shape, (4, 4, 283))
        self.assertEqual(len(dataset.labels), 4)

    def test_sequence_stride_is_respected(self):
        recording = FeatureRecording(
            features=np.arange(10 * 283, dtype=np.float32).reshape(10, 283),
            label="walking",
            sample_id="video01",
        )

        dataset = self.builder.build([recording])

        np.testing.assert_array_equal(
            dataset.X[1],
            recording.features[2:6],
        )

    def test_short_recording_is_skipped(self):
        recording = FeatureRecording(
            features=np.ones((3, 283), dtype=np.float32),
            label="walking",
            sample_id="short_video",
        )

        with self.assertRaises(ValueError):
            self.builder.build([recording])

    def test_multiple_recordings_do_not_mix_sequences(self):
        first = FeatureRecording(
            features=np.ones((4, 283), dtype=np.float32),
            label="walking",
            sample_id="video01",
        )
        second = FeatureRecording(
            features=np.full((4, 283), 2.0, dtype=np.float32),
            label="running",
            sample_id="video02",
        )

        dataset = self.builder.build([first, second])

        self.assertEqual(len(dataset), 2)
        self.assertTrue(np.all(dataset.X[0] == 1.0))
        self.assertTrue(np.all(dataset.X[1] == 2.0))
        self.assertEqual(
            dataset.labels.tolist(),
            ["walking", "running"],
        )

    def test_preserves_recording_metadata(self):
        recording = FeatureRecording(
            features=np.ones((4, 283), dtype=np.float32),
            label="walking",
            sample_id="video01",
            subject_id="person01",
            site_id="home01",
            split="train",
        )

        dataset = self.builder.build([recording])

        self.assertEqual(dataset.subject_ids, ("person01",))
        self.assertEqual(dataset.site_ids, ("home01",))
        self.assertEqual(dataset.splits, ("train",))

    def test_select_split(self):
        recordings = [
            FeatureRecording(
                features=np.ones((4, 283), dtype=np.float32),
                label="walking",
                sample_id="train01",
                split="train",
            ),
            FeatureRecording(
                features=np.ones((4, 283), dtype=np.float32),
                label="running",
                sample_id="test01",
                split="test",
            ),
        ]

        dataset = self.builder.build(recordings)
        X_train, y_train = dataset.select_split("train")

        self.assertEqual(X_train.shape, (1, 4, 283))
        self.assertEqual(y_train.tolist(), ["walking"])

    def test_unknown_split_is_rejected(self):
        recording = FeatureRecording(
            features=np.ones((4, 283), dtype=np.float32),
            label="walking",
            sample_id="video01",
            split="holdout",
        )

        with self.assertRaises(ValueError):
            self.builder.build([recording])

    def test_invalid_feature_dimension_is_rejected(self):
        recording = FeatureRecording(
            features=np.ones((4, 100), dtype=np.float32),
            label="walking",
            sample_id="video01",
        )

        with self.assertRaises(ValueError):
            self.builder.build([recording])

    def test_nan_features_are_rejected(self):
        features = np.ones((4, 283), dtype=np.float32)
        features[0, 0] = np.nan

        recording = FeatureRecording(
            features=features,
            label="walking",
            sample_id="video01",
        )

        with self.assertRaises(ValueError):
            self.builder.build([recording])

    def test_label_encoder(self):
        encoder = ActivityLabelEncoder()

        encoded = encoder.transform(
            ["walking", "running", "boxing"]
        )

        self.assertEqual(encoded.tolist(), [5, 4, 0])
        self.assertEqual(
            encoder.inverse_transform(encoded),
            ["walking", "running", "boxing"],
        )

    def test_unknown_label_is_rejected(self):
        encoder = ActivityLabelEncoder()

        with self.assertRaises(ValueError):
            encoder.transform(["swimming"])

    def test_duplicate_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            ActivityLabelEncoder(("walking", "walking"))

if __name__ == "__main__":
    unittest.main()
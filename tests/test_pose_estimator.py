
"""Tests for MediaPipe pose estimation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.pose_estimation.pose_estimator import (
    LANDMARK_NAMES,
    PoseEstimator,
)


def make_fake_landmark(
    x=0.5,
    y=0.5,
    z=0.0,
    visibility=0.99,
    presence=0.99,
):
    return SimpleNamespace(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
        presence=presence,
    )


def make_fake_result(number_of_poses=1):
    poses = []

    for _ in range(number_of_poses):
        poses.append(
            [
                make_fake_landmark(
                    x=0.1 + index * 0.01,
                    y=0.2 + index * 0.01,
                )
                for index in range(33)
            ]
        )

    world_poses = [
        [make_fake_landmark() for _ in range(33)]
        for _ in range(number_of_poses)
    ]

    return SimpleNamespace(
        pose_landmarks=poses,
        pose_world_landmarks=world_poses,
    )


class TestPoseEstimator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.model_path = Path(self.temp_dir.name) / "test_model.task"
        self.model_path.write_bytes(b"test model placeholder")

        self.fake_landmarker = SimpleNamespace(
            detect=lambda image: make_fake_result(),
            close=lambda: None,
        )

        self.create_patcher = patch.object(
            PoseEstimator,
            "_create_landmarker",
            return_value=self.fake_landmarker,
        )

        self.create_patcher.start()

        self.estimator = PoseEstimator(
            model_path=self.model_path
        )

        self.frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

    def tearDown(self):
        self.create_patcher.stop()
        self.temp_dir.cleanup()

    def test_model_file_is_required(self):
        with self.assertRaises(FileNotFoundError):
            PoseEstimator(
                model_path=Path(self.temp_dir.name) / "missing.task"
            )

    def test_invalid_num_poses(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PoseEstimator(
                        model_path=self.model_path,
                        num_poses=value,
                    )

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            PoseEstimator(
                model_path=self.model_path,
                min_detection_confidence=1.5,
            )

    def test_landmark_names_contains_33_entries(self):
        self.assertEqual(len(LANDMARK_NAMES), 33)

    def test_detect_returns_33_landmarks(self):
        poses = self.estimator.detect(self.frame)

        self.assertIsNotNone(poses)
        self.assertEqual(len(poses), 1)
        self.assertEqual(len(poses[0]["landmarks"]), 33)

    def test_landmark_contains_expected_fields(self):
        poses = self.estimator.detect(self.frame)
        landmark = poses[0]["landmarks"][0]

        expected_fields = {
            "id",
            "name",
            "x",
            "y",
            "z",
            "visibility",
            "presence",
        }

        self.assertEqual(set(landmark.keys()), expected_fields)

    def test_landmark_names_are_correctly_assigned(self):
        poses = self.estimator.detect(self.frame)

        self.assertEqual(
            poses[0]["landmarks"][11]["name"],
            "left_shoulder",
        )

        self.assertEqual(
            poses[0]["landmarks"][23]["name"],
            "left_hip",
        )

    def test_detect_returns_none_when_no_person_found(self):
        self.fake_landmarker.detect = lambda image: SimpleNamespace(
            pose_landmarks=[],
            pose_world_landmarks=[],
        )

        poses = self.estimator.detect(self.frame)

        self.assertIsNone(poses)

    def test_detect_supports_multiple_poses(self):
        self.fake_landmarker.detect = lambda image: make_fake_result(2)

        poses = self.estimator.detect(self.frame)

        self.assertEqual(len(poses), 2)

    def test_detect_rejects_none_frame(self):
        with self.assertRaises(ValueError):
            self.estimator.detect(None)

    def test_detect_rejects_invalid_dimensions(self):
        invalid_frame = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.estimator.detect(invalid_frame)

    def test_detect_rejects_non_uint8_frame(self):
        invalid_frame = np.zeros(
            (100, 100, 3),
            dtype=np.float32,
        )

        with self.assertRaises(TypeError):
            self.estimator.detect(invalid_frame)

    def test_draw_landmarks_returns_same_frame_size(self):
        poses = self.estimator.detect(self.frame)

        annotated = self.estimator.draw_landmarks(
            self.frame,
            poses,
        )

        self.assertEqual(annotated.shape, self.frame.shape)

    def test_draw_landmarks_does_not_modify_original(self):
        poses = self.estimator.detect(self.frame)
        original = self.frame.copy()

        self.estimator.draw_landmarks(self.frame, poses)

        np.testing.assert_array_equal(self.frame, original)

    def test_draw_without_poses_returns_copy(self):
        annotated = self.estimator.draw_landmarks(
            self.frame,
            None,
        )

        np.testing.assert_array_equal(annotated, self.frame)
        self.assertIsNot(annotated, self.frame)

    def test_draw_rejects_invalid_visibility_threshold(self):
        with self.assertRaises(ValueError):
            self.estimator.draw_landmarks(
                self.frame,
                None,
                visibility_threshold=2.0,
            )

    def test_close_releases_landmarker(self):
        close_mock = patch.object(
            self.fake_landmarker,
            "close",
        ).start()

        self.estimator.close()

        close_mock.assert_called_once()
        self.assertIsNone(self.estimator.landmarker)

        patch.stopall()

    def test_context_manager_closes_landmarker(self):
        close_mock = patch.object(
            self.fake_landmarker,
            "close",
        ).start()

        with self.estimator:
            pass

        close_mock.assert_called_once()

        patch.stopall()


if __name__ == "__main__":
    unittest.main()

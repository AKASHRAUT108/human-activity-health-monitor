
"""Unit tests for MediaPipe pose feature extraction."""

import unittest

import numpy as np

from src.feature_extraction import (
    LANDMARK_NAMES,
    NUM_LANDMARKS,
    PoseFeatureExtractor,
)


def make_pose():
    """Create a synthetic pose with 33 image and world landmarks."""

    landmarks = []
    world_landmarks = []

    for index in range(NUM_LANDMARKS):
        x = index * 0.01
        y = index * 0.02
        z = index * 0.001

        landmarks.append(
            {
                "id": index,
                "name": LANDMARK_NAMES[index],
                "x": x,
                "y": y,
                "z": z,
                "visibility": 0.99,
                "presence": 0.99,
            }
        )

        world_landmarks.append(
            {
                "x": x,
                "y": y,
                "z": z,
            }
        )

    # Define stable shoulder and hip positions for normalization.
    positions = {
        11: (-0.2, 0.5, 0.0),
        12: (0.2, 0.5, 0.0),
        23: (-0.15, 0.0, 0.0),
        24: (0.15, 0.0, 0.0),
        13: (-0.4, 0.4, 0.0),
        14: (0.4, 0.4, 0.0),
        15: (-0.5, 0.3, 0.0),
        16: (0.5, 0.3, 0.0),
        25: (-0.15, -0.4, 0.0),
        26: (0.15, -0.4, 0.0),
        27: (-0.15, -0.8, 0.0),
        28: (0.15, -0.8, 0.0),
        31: (-0.15, -0.85, 0.1),
        32: (0.15, -0.85, 0.1),
    }

    for index, position in positions.items():
        world_landmarks[index] = {
            "x": position[0],
            "y": position[1],
            "z": position[2],
        }
        landmarks[index]["x"] = position[0]
        landmarks[index]["y"] = position[1]
        landmarks[index]["z"] = position[2]

    return {
        "landmarks": landmarks,
        "world_landmarks": world_landmarks,
    }


class TestPoseFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = PoseFeatureExtractor()
        self.pose = make_pose()

    def test_feature_names_are_unique(self):
        self.assertEqual(
            len(self.extractor.feature_names),
            len(set(self.extractor.feature_names)),
        )

    def test_feature_vector_has_expected_shape(self):
        result = self.extractor.extract_frame(self.pose)

        self.assertEqual(
            result["feature_vector"].shape,
            (len(self.extractor.feature_names),),
        )

    def test_feature_vector_is_float32(self):
        result = self.extractor.extract_frame(self.pose)

        self.assertEqual(
            result["feature_vector"].dtype,
            np.float32,
        )

    def test_valid_pose_is_detected(self):
        result = self.extractor.extract_frame(self.pose)

        self.assertTrue(result["pose_present"])

    def test_none_pose_returns_zero_features(self):
        result = self.extractor.extract_frame(None)

        self.assertFalse(result["pose_present"])
        self.assertTrue(
            np.all(result["feature_vector"] == 0)
        )

    def test_landmark_count_must_be_33(self):
        invalid_pose = make_pose()
        invalid_pose["landmarks"] = invalid_pose["landmarks"][:-1]

        with self.assertRaises(ValueError):
            self.extractor.extract_frame(invalid_pose)

    def test_missing_world_landmarks_uses_image_coordinates(self):
        pose = make_pose()
        pose["world_landmarks"] = None

        result = self.extractor.extract_frame(pose)

        self.assertTrue(result["pose_present"])
        self.assertTrue(
            np.all(np.isfinite(result["feature_vector"]))
        )

    def test_low_visibility_landmark_is_not_used_for_coordinates(self):
        pose = make_pose()
        pose["landmarks"][0]["visibility"] = 0.1

        result = self.extractor.extract_frame(pose)
        vector = result["feature_vector"]

        # The first three values represent nose x, y and z.
        np.testing.assert_array_equal(
            vector[:3],
            np.zeros(3, dtype=np.float32),
        )

    def test_confidence_values_are_preserved(self):
        pose = make_pose()
        pose["landmarks"][0]["visibility"] = 0.1

        result = self.extractor.extract_frame(pose)
        visibility_index = NUM_LANDMARKS * 3

        self.assertAlmostEqual(
            float(result["feature_vector"][visibility_index]),
            0.1,
            places=5,
        )

    def test_angle_calculation_returns_90_degrees(self):
        angle = self.extractor.calculate_angle(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        self.assertAlmostEqual(angle, 90.0, places=5)

    def test_angle_calculation_returns_180_degrees(self):
        angle = self.extractor.calculate_angle(
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(angle, 180.0, places=5)

    def test_degenerate_angle_returns_none(self):
        angle = self.extractor.calculate_angle(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertIsNone(angle)

    def test_angle_points_must_have_three_coordinates(self):
        with self.assertRaises(ValueError):
            self.extractor.calculate_angle(
                np.array([1.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )

    def test_multiple_pose_extraction(self):
        results = self.extractor.extract_multiple(
            [self.pose, self.pose]
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["pose_present"])
        self.assertTrue(results[1]["pose_present"])

    def test_no_detected_people_returns_empty_list(self):
        self.assertEqual(
            self.extractor.extract_multiple(None),
            [],
        )

    def test_invalid_visibility_threshold(self):
        with self.assertRaises(ValueError):
            PoseFeatureExtractor(visibility_threshold=1.5)

    def test_sequence_has_expected_shape(self):
        sequence = self.extractor.extract_sequence(
            [self.pose, self.pose, self.pose],
            fps=30.0,
        )

        self.assertEqual(
            sequence.shape,
            (
                3,
                len(self.extractor.sequence_feature_names),
            ),
        )

    def test_sequence_features_are_float32(self):
        sequence = self.extractor.extract_sequence(
            [self.pose, self.pose],
            fps=30.0,
        )

        self.assertEqual(sequence.dtype, np.float32)

    def test_first_frame_velocity_is_zero(self):
        sequence = self.extractor.extract_sequence(
            [self.pose, self.pose],
            fps=30.0,
        )

        velocity_count = NUM_LANDMARKS * 3

        np.testing.assert_array_equal(
            sequence[0, -velocity_count:],
            np.zeros(velocity_count, dtype=np.float32),
        )

    def test_moving_landmark_has_nonzero_velocity(self):
        first_pose = make_pose()
        second_pose = make_pose()

        second_pose["world_landmarks"][0]["x"] += 0.1
        second_pose["landmarks"][0]["x"] += 0.1

        sequence = self.extractor.extract_sequence(
            [first_pose, second_pose],
            fps=30.0,
        )

        # The first velocity value belongs to nose x.
        nose_x_velocity = sequence[1, -NUM_LANDMARKS * 3]

        self.assertGreater(nose_x_velocity, 0.0)

    def test_missing_frame_has_zero_features(self):
        sequence = self.extractor.extract_sequence(
            [self.pose, None, self.pose],
            fps=30.0,
        )

        self.assertTrue(
            np.all(sequence[1] == 0)
        )

    def test_invalid_fps_raises_error(self):
        with self.assertRaises(ValueError):
            self.extractor.extract_sequence(
                [self.pose],
                fps=0,
            )

    def test_empty_sequence_returns_correct_shape(self):
        sequence = self.extractor.extract_sequence(
            [],
            fps=30.0,
        )

        self.assertEqual(
            sequence.shape,
            (0, len(self.extractor.sequence_feature_names)),
        )


if __name__ == "__main__":
    unittest.main()

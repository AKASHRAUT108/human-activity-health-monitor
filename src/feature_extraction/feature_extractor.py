
"""Extract machine-learning features from MediaPipe Pose Landmarker output."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# MediaPipe Pose landmark indices.
LANDMARK_NAMES = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

NUM_LANDMARKS = len(LANDMARK_NAMES)
COORDINATE_COUNT = NUM_LANDMARKS * 3

# Each tuple is (first point, joint, third point).
ANGLE_TRIPLETS = {
    "left_elbow_angle": (11, 13, 15),
    "right_elbow_angle": (12, 14, 16),
    "left_shoulder_angle": (13, 11, 23),
    "right_shoulder_angle": (14, 12, 24),
    "left_hip_angle": (11, 23, 25),
    "right_hip_angle": (12, 24, 26),
    "left_knee_angle": (23, 25, 27),
    "right_knee_angle": (24, 26, 28),
    "left_ankle_angle": (25, 27, 31),
    "right_ankle_angle": (26, 28, 32),
}

# Each tuple is (first point, second point).
DISTANCE_PAIRS = {
    "shoulder_width": (11, 12),
    "hip_width": (23, 24),
    "left_arm_reach": (11, 15),
    "right_arm_reach": (12, 16),
    "ankle_width": (27, 28),
    "torso_length": (11, 23),
    "left_lower_leg_length": (25, 27),
    "right_lower_leg_length": (26, 28),
}


class PoseFeatureExtractor:
    """Convert one person's MediaPipe pose into numerical features.

    Expected pose format:
        {
            "landmarks": [
                {
                    "id": 0,
                    "name": "nose",
                    "x": 0.5,
                    "y": 0.2,
                    "z": 0.0,
                    "visibility": 0.99,
                    "presence": 0.99,
                },
                ...
            ],
            "world_landmarks": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                ...
            ],
        }

    World coordinates are preferred for geometry. Image coordinates
    are used as a fallback when world coordinates are unavailable.
    """

    def __init__(self, visibility_threshold: float = 0.5):
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError(
                "visibility_threshold must be between 0.0 and 1.0."
            )

        self.visibility_threshold = float(visibility_threshold)

        self.feature_names = self._build_feature_names()
        self.sequence_feature_names = (
            self.feature_names
            + tuple(
                f"velocity_{landmark}_{axis}"
                for landmark in LANDMARK_NAMES
                for axis in ("x", "y", "z")
            )
        )

    @staticmethod
    def _build_feature_names() -> tuple[str, ...]:
        names = []

        for landmark in LANDMARK_NAMES:
            for axis in ("x", "y", "z"):
                names.append(f"{landmark}_{axis}")

        for landmark in LANDMARK_NAMES:
            names.append(f"{landmark}_visibility")
            names.append(f"{landmark}_presence")

        names.extend(ANGLE_TRIPLETS.keys())
        names.extend(DISTANCE_PAIRS.keys())
        names.append("pose_present")

        return tuple(names)

    @staticmethod
    def calculate_angle(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> Optional[float]:
        """Calculate angle ABC in degrees, returning None if invalid."""

        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        c = np.asarray(point_c, dtype=np.float64)

        if a.shape != (3,) or b.shape != (3,) or c.shape != (3,):
            raise ValueError("Each point must contain exactly 3 coordinates.")

        if not (
            np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
            and np.all(np.isfinite(c))
        ):
            return None

        vector_a = a - b
        vector_c = c - b

        norm_a = np.linalg.norm(vector_a)
        norm_c = np.linalg.norm(vector_c)

        if norm_a < 1e-8 or norm_c < 1e-8:
            return None

        cosine = np.dot(vector_a, vector_c) / (norm_a * norm_c)
        cosine = np.clip(cosine, -1.0, 1.0)

        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _read_confidence(
        landmark: dict,
        key: str,
        default: float,
    ) -> float:
        value = landmark.get(key, default)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _read_coordinate(landmark: dict) -> Optional[np.ndarray]:
        try:
            coordinate = np.array(
                [
                    float(landmark["x"]),
                    float(landmark["y"]),
                    float(landmark["z"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            return None

        if not np.all(np.isfinite(coordinate)):
            return None

        return coordinate

    def _prepare_landmarks(
        self,
        pose: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
        image_landmarks = pose.get("landmarks")

        if not isinstance(image_landmarks, (list, tuple)):
            raise ValueError("Pose must contain a landmarks list.")

        if len(image_landmarks) != NUM_LANDMARKS:
            raise ValueError(
                f"Expected {NUM_LANDMARKS} landmarks, "
                f"received {len(image_landmarks)}."
            )

        world_landmarks = pose.get("world_landmarks")

        if world_landmarks is not None:
            if not isinstance(world_landmarks, (list, tuple)):
                raise ValueError("world_landmarks must be a list or tuple.")

            if len(world_landmarks) != NUM_LANDMARKS:
                raise ValueError(
                    f"Expected {NUM_LANDMARKS} world landmarks, "
                    f"received {len(world_landmarks)}."
                )

        coordinates = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
        visibility = np.zeros(NUM_LANDMARKS, dtype=np.float32)
        presence = np.zeros(NUM_LANDMARKS, dtype=np.float32)
        valid = np.zeros(NUM_LANDMARKS, dtype=bool)

        for index, landmark in enumerate(image_landmarks):
            if not isinstance(landmark, dict):
                continue

            visibility[index] = self._read_confidence(
                landmark, "visibility", 1.0
            )
            presence[index] = self._read_confidence(
                landmark, "presence", visibility[index]
            )

            if (
                visibility[index] < self.visibility_threshold
                or presence[index] < self.visibility_threshold
            ):
                continue

            coordinate = None

            if world_landmarks is not None:
                world_point = world_landmarks[index]

                if isinstance(world_point, dict):
                    coordinate = self._read_coordinate(world_point)

            if coordinate is None:
                coordinate = self._read_coordinate(landmark)

            if coordinate is None:
                visibility[index] = 0.0
                presence[index] = 0.0
                continue

            coordinates[index] = coordinate
            valid[index] = True

        return coordinates, visibility, presence, valid

    @staticmethod
    def _normalize_coordinates(
        coordinates: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Center coordinates on the hips and scale by shoulder width."""

        normalized = np.zeros_like(coordinates)

        if not np.any(valid):
            return normalized

        if valid[23] and valid[24]:
            center = (coordinates[23] + coordinates[24]) / 2.0
        elif valid[11] and valid[12]:
            center = (coordinates[11] + coordinates[12]) / 2.0
        else:
            center = np.mean(coordinates[valid], axis=0)

        if valid[11] and valid[12]:
            scale = np.linalg.norm(coordinates[11] - coordinates[12])
        elif valid[23] and valid[24]:
            scale = np.linalg.norm(coordinates[23] - coordinates[24])
        elif valid[11] and valid[23]:
            scale = np.linalg.norm(coordinates[11] - coordinates[23])
        else:
            scale = 1.0

        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0

        normalized[valid] = (coordinates[valid] - center) / scale

        return normalized

    def extract_frame(self, pose: Optional[dict]) -> dict:
        """Extract static features for one person in one frame."""

        vector = np.zeros(len(self.feature_names), dtype=np.float32)

        if pose is None:
            return {
                "feature_vector": vector,
                "feature_names": self.feature_names,
                "pose_present": False,
            }

        if not isinstance(pose, dict):
            raise ValueError("pose must be a dictionary or None.")

        coordinates, visibility, presence, valid = (
            self._prepare_landmarks(pose)
        )

        if not np.any(valid):
            return {
                "feature_vector": vector,
                "feature_names": self.feature_names,
                "pose_present": False,
            }

        normalized = self._normalize_coordinates(coordinates, valid)

        cursor = 0

        # 1. Body-centered, scale-normalized XYZ coordinates.
        coordinate_values = normalized.astype(np.float32).flatten()
        vector[cursor:cursor + COORDINATE_COUNT] = coordinate_values
        cursor += COORDINATE_COUNT

        # 2. Landmark confidence information.
        for index in range(NUM_LANDMARKS):
            vector[cursor] = visibility[index]
            vector[cursor + 1] = presence[index]
            cursor += 2

        # 3. Joint angles, normalized to the range [0, 1].
        for first, joint, third in ANGLE_TRIPLETS.values():
            if valid[first] and valid[joint] and valid[third]:
                angle = self.calculate_angle(
                    normalized[first],
                    normalized[joint],
                    normalized[third],
                )

                if angle is not None:
                    vector[cursor] = angle / 180.0

            cursor += 1

        # 4. Distances between selected body landmarks.
        for first, second in DISTANCE_PAIRS.values():
            if valid[first] and valid[second]:
                distance = np.linalg.norm(
                    normalized[first] - normalized[second]
                )
                vector[cursor] = float(distance)

            cursor += 1

        # 5. Pose-present indicator.
        vector[cursor] = 1.0

        return {
            "feature_vector": vector,
            "feature_names": self.feature_names,
            "pose_present": True,
        }

    def extract_multiple(self, poses: Optional[list]) -> list[dict]:
        """Extract features for every detected person in one frame.

        The returned list follows the order of the pose detections.
        These indices are not persistent person IDs; tracking is a
        separate future pipeline stage.
        """

        if poses is None:
            return []

        if not isinstance(poses, (list, tuple)):
            raise ValueError("poses must be a list, tuple, or None.")

        return [self.extract_frame(pose) for pose in poses]

    def extract_sequence(
        self,
        pose_sequence: Sequence[Optional[dict]],
        fps: float,
    ) -> np.ndarray:
        """Extract static and temporal features for one person's track.

        Args:
            pose_sequence: One pose dictionary (or None) per frame.
                All entries must belong to the same tracked person.
            fps: Frame rate of the source video.

        Returns:
            float32 array with shape (number_of_frames, number_of_features).

        Velocity features are normalized-coordinate changes per second.
        The first frame has zero velocity because no previous frame exists.
        """

        try:
            fps = float(fps)
        except (TypeError, ValueError):
            raise ValueError("fps must be a positive finite number.")

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number.")

        if not isinstance(pose_sequence, (list, tuple)):
            raise ValueError("pose_sequence must be a list or tuple.")

        if len(pose_sequence) == 0:
            return np.empty(
                (0, len(self.sequence_feature_names)),
                dtype=np.float32,
            )

        static_vectors = [
            self.extract_frame(pose)["feature_vector"]
            for pose in pose_sequence
        ]

        static_matrix = np.stack(static_vectors).astype(np.float32)

        coordinate_matrix = static_matrix[:, :COORDINATE_COUNT]
        confidence_start = COORDINATE_COUNT

        valid_matrix = np.zeros(
            (len(pose_sequence), NUM_LANDMARKS),
            dtype=bool,
        )

        for index in range(NUM_LANDMARKS):
            visibility_index = confidence_start + (index * 2)
            presence_index = visibility_index + 1

            valid_matrix[:, index] = (
                (static_matrix[:, visibility_index] >= self.visibility_threshold)
                & (static_matrix[:, presence_index] >= self.visibility_threshold)
            )

        velocities = np.zeros_like(coordinate_matrix)

        for frame_index in range(1, len(pose_sequence)):
            current_valid = valid_matrix[frame_index]
            previous_valid = valid_matrix[frame_index - 1]
            both_valid = current_valid & previous_valid

            current = coordinate_matrix[frame_index].reshape(
                NUM_LANDMARKS, 3
            )
            previous = coordinate_matrix[frame_index - 1].reshape(
                NUM_LANDMARKS, 3
            )

            frame_velocity = np.zeros(
                (NUM_LANDMARKS, 3),
                dtype=np.float32,
            )

            frame_velocity[both_valid] = (
                (current[both_valid] - previous[both_valid]) * fps
            )

            velocities[frame_index] = frame_velocity.flatten()

        return np.concatenate(
            [static_matrix, velocities],
            axis=1,
        ).astype(np.float32)

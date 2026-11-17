
"""Human pose estimation using the MediaPipe Tasks API."""

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.config import MODELS_DIR


# MediaPipe Pose Landmarker returns 33 landmarks in this order.
LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


# Connections between landmark indices for skeleton visualization.
POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (15, 17),
    (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (28, 32),
)


class PoseEstimator:
    """Detect human body landmarks and draw pose skeletons."""

    def __init__(
        self,
        model_path: str | Path = MODELS_DIR / "pose_landmarker_lite.task",
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {self.model_path}. "
                "Download pose_landmarker_lite.task into the models directory."
            )

        if (
            not isinstance(num_poses, int)
            or isinstance(num_poses, bool)
            or num_poses <= 0
        ):
            raise ValueError("num_poses must be a positive integer.")

        confidences = (
            min_detection_confidence,
            min_presence_confidence,
            min_tracking_confidence,
        )

        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= value <= 1.0
            for value in confidences
        ):
            raise ValueError("Confidence values must be between 0 and 1.")

        self.num_poses = num_poses
        self.landmarker = self._create_landmarker(
            num_poses=num_poses,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _create_landmarker(
        self,
        num_poses: int,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
    ):
        """Create the MediaPipe Pose Landmarker."""

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate an OpenCV BGR frame."""

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")

        if frame.size == 0:
            raise ValueError("Input frame cannot be empty.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Input frame must have three color channels."
            )

        if frame.dtype != np.uint8:
            raise TypeError("Input frame must have uint8 data type.")

    def detect(self, frame: np.ndarray) -> Optional[list[dict]]:
        """
        Detect poses in one BGR video frame.

        Returns:
            A list containing one dictionary per detected person.
            Each dictionary contains 33 image landmarks and world landmarks.
            Returns None when no person is detected.
        """

        self._validate_frame(frame)

        # MediaPipe Tasks expects an RGB image.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.landmarker.detect(mp_image)

        detected_poses = []

        for pose_index, pose_landmarks in enumerate(
            result.pose_landmarks
        ):
            landmarks = []

            for index, landmark in enumerate(pose_landmarks):
                landmarks.append(
                    {
                        "id": index,
                        "name": LANDMARK_NAMES[index],
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "visibility": float(
                            getattr(landmark, "visibility", 0.0)
                        ),
                        "presence": float(
                            getattr(landmark, "presence", 0.0)
                        ),
                    }
                )

            world_landmarks = []

            if pose_index < len(
                getattr(result, "pose_world_landmarks", [])
            ):
                for landmark in result.pose_world_landmarks[pose_index]:
                    world_landmarks.append(
                        {
                            "x": float(landmark.x),
                            "y": float(landmark.y),
                            "z": float(landmark.z),
                        }
                    )

            detected_poses.append(
                {
                    "landmarks": landmarks,
                    "world_landmarks": world_landmarks,
                }
            )

        return detected_poses if detected_poses else None

    def draw_landmarks(
        self,
        frame: np.ndarray,
        poses: Optional[list[dict]],
        visibility_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Draw pose landmarks and skeleton connections on a copy of a frame.

        Returns:
            An annotated BGR image.
        """

        self._validate_frame(frame)

        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError(
                "visibility_threshold must be between 0 and 1."
            )

        annotated = frame.copy()

        if not poses:
            return annotated

        height, width = annotated.shape[:2]

        for pose in poses:
            landmarks = pose["landmarks"]

            visible_points = {}

            for landmark in landmarks:
                if landmark["visibility"] < visibility_threshold:
                    continue

                x = int(landmark["x"] * width)
                y = int(landmark["y"] * height)

                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))

                visible_points[landmark["id"]] = (x, y)

            # Draw skeleton connections.
            for start_id, end_id in POSE_CONNECTIONS:
                if (
                    start_id in visible_points
                    and end_id in visible_points
                ):
                    cv2.line(
                        annotated,
                        visible_points[start_id],
                        visible_points[end_id],
                        (0, 255, 0),
                        2,
                    )

            # Draw individual landmarks.
            for point in visible_points.values():
                cv2.circle(
                    annotated,
                    point,
                    4,
                    (0, 0, 255),
                    -1,
                )

        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

"""Video inference service for the Human Activity Health Monitor."""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import MODELS_DIR
from src.pose_estimation.pose_estimator import PoseEstimator
from src.feature_extraction.feature_extractor import PoseFeatureExtractor
from src.activity_recognition.predictor import ActivityPredictor


class VideoInferenceService:
    """Extract pose features and predict activities from uploaded videos."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        pose_model_path: str | Path | None = None,
        frame_stride: int = 1,
        sequence_stride: int = 15,
    ) -> None:
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")

        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive.")

        self.model_dir = Path(model_dir or MODELS_DIR / "kth_tuned_conf03")
        self.model_path = self.model_dir / "activity_lstm.keras"
        self.metadata_path = self.model_dir / "activity_lstm_metadata.json"

        self.pose_model_path = Path(
            pose_model_path or MODELS_DIR / "pose_landmarker_lite.task"
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Activity model not found: {self.model_path}")

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Model metadata not found: {self.metadata_path}"
            )

        if not self.pose_model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {self.pose_model_path}"
            )

        self.predictor = ActivityPredictor.load(
            model_path=self.model_path,
            metadata_path=self.metadata_path,
        )

        self.pose_estimator = PoseEstimator(
            model_path=self.pose_model_path,
            num_poses=1,
        )

        self.feature_extractor = PoseFeatureExtractor()

        self.sequence_length = self.predictor.sequence_length
        self.feature_dim = self.predictor.feature_dim
        self.class_names = list(self.predictor.class_names)

        self.frame_stride = frame_stride
        self.sequence_stride = sequence_stride

        # MediaPipe's pose estimator is shared, so serialize video inference.
        self._inference_lock = threading.Lock()

    @staticmethod
    def _normalize_probabilities(
        probabilities: Any,
        class_names: list[str],
    ) -> dict[str, float]:
        """Convert model probabilities into a class-to-score dictionary."""

        if isinstance(probabilities, dict):
            result = {
                name: float(probabilities.get(name, 0.0))
                for name in class_names
            }
        else:
            values = np.asarray(probabilities, dtype=np.float64).reshape(-1)

            if len(values) != len(class_names):
                raise ValueError(
                    "Prediction probability count does not match class count."
                )

            result = {
                name: float(value)
                for name, value in zip(class_names, values)
            }

        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Model returned non-finite probabilities.")

        return result

    def predict_video(self, video_path: str | Path) -> dict[str, Any]:
        """Predict activities across overlapping windows in a video."""

        path = Path(video_path)

        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        with self._inference_lock:
            return self._predict_video_locked(path)

    def _predict_video_locked(self, path: Path) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError("Could not open the uploaded video.")

        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            reported_frame_count = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if not np.isfinite(source_fps) or source_fps <= 0:
                raise ValueError("The video has an invalid or unavailable FPS.")

            effective_fps = source_fps / self.frame_stride

            pose_sequence = []
            processed_frame_count = 0
            detected_frame_count = 0
            original_frame_index = 0

            while True:
                success, frame = capture.read()

                if not success:
                    break

                if original_frame_index % self.frame_stride == 0:
                    detected_poses = self.pose_estimator.detect(frame)

                    pose = (
                        detected_poses[0]
                        if detected_poses
                        else None
                    )

                    pose_sequence.append(pose)
                    processed_frame_count += 1

                    if pose is not None:
                        detected_frame_count += 1

                original_frame_index += 1

        finally:
            capture.release()

        if processed_frame_count == 0:
            raise ValueError("The uploaded video contains no readable frames.")

        if detected_frame_count == 0:
            raise ValueError(
                "No person was detected in the uploaded video."
            )

        if processed_frame_count < self.sequence_length:
            raise ValueError(
                f"The video contains only {processed_frame_count} sampled "
                f"frames. At least {self.sequence_length} are required."
            )

        # Calculate velocities across the entire video before windowing.
        features = self.feature_extractor.extract_sequence(
            pose_sequence,
            fps=effective_fps,
        )

        if features.shape != (processed_frame_count, self.feature_dim):
            raise ValueError(
                f"Unexpected feature shape {features.shape}; expected "
                f"({processed_frame_count}, {self.feature_dim})."
            )

        if not np.isfinite(features).all():
            raise ValueError("Extracted features contain invalid values.")

        window_predictions = []

        probability_totals = {
            name: 0.0 for name in self.class_names
        }

        activity_counts = Counter()

        for start in range(
            0,
            processed_frame_count - self.sequence_length + 1,
            self.sequence_stride,
        ):
            end = start + self.sequence_length
            window = features[start:end]

            prediction = self.predictor.predict(window)

            activity = str(prediction["predicted_activity"])
            confidence = float(prediction["confidence"])

            probabilities = self._normalize_probabilities(
                prediction["probabilities"],
                self.class_names,
            )

            activity_counts[activity] += 1

            for name, probability in probabilities.items():
                probability_totals[name] += probability

            window_predictions.append(
                {
                    "window_index": len(window_predictions) + 1,
                    "start_frame": start * self.frame_stride,
                    "end_frame": (end - 1) * self.frame_stride,
                    "predicted_activity": activity,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            )

        if not window_predictions:
            raise ValueError("No complete prediction windows were generated.")

        number_of_windows = len(window_predictions)

        average_probabilities = {
            name: total / number_of_windows
            for name, total in probability_totals.items()
        }

        dominant_activity = max(
            average_probabilities,
            key=average_probabilities.get,
        )

        duration_seconds = original_frame_index / source_fps

        return {
            "video_name": path.name,
            "model_name": "human_activity_lstm",
            "model_type": "LSTM",
            "dominant_activity": dominant_activity,
            "confidence": average_probabilities[dominant_activity],
            "average_probabilities": average_probabilities,
            "activity_window_counts": dict(activity_counts),
            "video_info": {
                "fps": source_fps,
                "duration_seconds": duration_seconds,
                "total_frames_read": original_frame_index,
                "sampled_frames": processed_frame_count,
                "frames_with_detected_person": detected_frame_count,
                "frame_stride": self.frame_stride,
                "sequence_length": self.sequence_length,
                "sequence_stride": self.sequence_stride,
                "feature_dimension": self.feature_dim,
            },
            "window_predictions": window_predictions,
            "disclaimer": (
                "Activity predictions are estimates and are not medical "
                "diagnoses or a substitute for professional assessment."
            ),
        }

    def close(self) -> None:
        """Release the pose estimator."""

        self.pose_estimator.close()

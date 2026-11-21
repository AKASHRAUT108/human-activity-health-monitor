
"""Extract pose-based feature sequences from annotated video segments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from src.feature_extraction.feature_extractor import PoseFeatureExtractor
from src.pose_estimation.pose_estimator import PoseEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"


class VideoSequenceFeatureExtractor:
    """Extract 283-dimensional pose features from video sequence samples."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        visibility_threshold: float = 0.5,
        pose_estimator: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        capture_factory: Optional[Callable] = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else PoseFeatureExtractor(
                visibility_threshold=visibility_threshold
            )
        )

        self.capture_factory = capture_factory or cv2.VideoCapture

        self._owns_pose_estimator = pose_estimator is None
        self.pose_estimator = pose_estimator

        if self._owns_pose_estimator:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Pose model not found: {self.model_path}"
                )

            self.pose_estimator = PoseEstimator(
                model_path=str(self.model_path),
                num_poses=1,
                min_detection_confidence=self.min_detection_confidence,
                min_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    @staticmethod
    def _resolve_video_path(video_path: str | Path) -> Path:
        """Resolve relative manifest paths against the project root."""
        path = Path(video_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    @staticmethod
    def _safe_sample_id(sample_id: str) -> str:
        """Convert a sample ID into a safe filename."""
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))
        safe_id = safe_id.strip("._")

        if not safe_id:
            raise ValueError(
                "Sample ID cannot produce a valid filename."
            )

        return safe_id

    @staticmethod
    def _validate_sample(sample: dict[str, Any]) -> None:
        required = (
            "sample_id",
            "video_path",
            "frame_start",
            "frame_end",
        )

        missing = [key for key in required if key not in sample]

        if missing:
            raise ValueError(
                f"Sample is missing required fields: {missing}"
            )

        start = int(sample["frame_start"])
        end = int(sample["frame_end"])

        if start < 1:
            raise ValueError("frame_start must be at least 1.")

        if end < start:
            raise ValueError("frame_end must be >= frame_start.")

    def extract_sample(
        self,
        sample: dict[str, Any],
        frame_stride: int = 1,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract features from one annotated video segment.

        Frame numbers in the manifest are 1-based and inclusive.
        """

        self._validate_sample(sample)

        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")

        video_path = self._resolve_video_path(sample["video_path"])
        start = int(sample["frame_start"])
        end = int(sample["frame_end"])

        capture = self.capture_factory(str(video_path))

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)

            source_fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )

            if not np.isfinite(source_fps) or source_fps <= 0:
                raise RuntimeError(
                    f"Invalid FPS ({source_fps}) for {video_path}"
                )

            pose_sequence = []
            decoded_frames = 0

            for frame_number in range(start, end + 1):
                success, frame = capture.read()

                if not success or frame is None:
                    raise RuntimeError(
                        "Video ended before the annotated sequence "
                        f"finished: {video_path}, "
                        f"expected frame {frame_number}, end frame {end}"
                    )

                decoded_frames += 1

                if (frame_number - start) % frame_stride != 0:
                    continue

                detected_poses = self.pose_estimator.detect(frame)

                if detected_poses:
                    pose_sequence.append(detected_poses[0])
                else:
                    pose_sequence.append(None)

            if not pose_sequence:
                raise RuntimeError(
                    f"No frames were sampled from {sample['sample_id']}."
                )

            effective_fps = source_fps / frame_stride

            features = self.feature_extractor.extract_sequence(
                pose_sequence,
                fps=effective_fps,
            )

            features = np.asarray(
                features,
                dtype=np.float32,
            )

            if features.ndim != 2:
                raise ValueError(
                    "Feature extractor must return a 2D array; "
                    f"received shape {features.shape}."
                )

            if features.shape[0] != len(pose_sequence):
                raise ValueError(
                    "Feature row count does not match sampled frame count."
                )

            if features.shape[1] != 283:
                raise ValueError(
                    "Expected 283 features per frame, "
                    f"received {features.shape[1]}."
                )

            if not np.isfinite(features).all():
                raise ValueError(
                    "Extracted features contain NaN or infinity."
                )

            detected_count = sum(
                pose is not None for pose in pose_sequence
            )

            metadata = {
                **sample,
                "video_path": str(video_path),
                "frame_start": start,
                "frame_end": end,
                "source_fps": source_fps,
                "effective_fps": effective_fps,
                "frame_stride": frame_stride,
                "annotated_frame_count": end - start + 1,
                "decoded_frame_count": decoded_frames,
                "sampled_frame_count": len(pose_sequence),
                "pose_detected_frames": detected_count,
                "pose_detection_rate": (
                    detected_count / len(pose_sequence)
                ),
                "feature_shape": list(features.shape),
                "feature_dimension": int(features.shape[1]),
            }

            return features, metadata

        finally:
            capture.release()

    @staticmethod
    def _load_existing_manifest(
        manifest_path: Path,
    ) -> dict[str, dict[str, Any]]:
        """Load existing metadata, indexed by sample ID."""
        records: dict[str, dict[str, Any]] = {}

        if not manifest_path.is_file():
            return records

        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)

                sample_id = record.get("sample_id")

                if sample_id is None:
                    continue

                records[str(sample_id)] = record

        return records

    @staticmethod
    def _validate_existing_feature(
        feature_path: Path,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Validate an existing array without modifying it."""
        existing_features = np.load(
            feature_path,
            allow_pickle=False,
        )

        if existing_features.ndim != 2:
            raise ValueError(
                "Existing feature file must contain a 2D array; "
                f"received {existing_features.shape}."
            )

        if existing_features.shape[1] != 283:
            raise ValueError(
                "Existing feature file does not contain 283 features "
                f"per frame: {existing_features.shape}."
            )

        if existing_features.shape != expected_shape:
            raise ValueError(
                "Existing feature shape does not match the newly "
                f"extracted sequence: existing={existing_features.shape}, "
                f"expected={expected_shape}."
            )

        if not np.isfinite(existing_features).all():
            raise ValueError(
                "Existing feature file contains NaN or infinity."
            )

    def extract_manifest(
        self,
        manifest_path: str | Path,
        output_dir: str | Path,
        max_samples: Optional[int] = None,
        frame_stride: int = 1,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Process a JSONL manifest and save feature arrays.

        Existing feature files are skipped when overwrite=False.

        Existing manifest metadata is preserved. If an existing feature
        file has no manifest entry, metadata is reconstructed and the
        existing array is validated without being overwritten.
        """

        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Sequence manifest not found: {manifest_path}"
            )

        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")

        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be at least 1.")

        output_dir.mkdir(parents=True, exist_ok=True)

        feature_dir = output_dir / "sequences"
        feature_dir.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("r", encoding="utf-8") as file:
            samples = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if max_samples is not None:
            samples = samples[:max_samples]

        feature_manifest = (
            output_dir / "pose_feature_manifest.jsonl"
        )

        # Preserve metadata from previous extraction runs.
        previous_records = self._load_existing_manifest(
            feature_manifest
        )

        successful_records = []
        manifest_records = []
        failures = []
        skipped_existing = 0
        recovered_metadata = 0

        for index, sample in enumerate(samples, start=1):
            sample_id = str(
                sample.get("sample_id", f"sample_{index}")
            )

            try:
                safe_id = self._safe_sample_id(sample_id)

                feature_path = (
                    feature_dir / f"{safe_id}.npy"
                )

                relative_feature_path = str(
                    feature_path.relative_to(output_dir)
                )

                previous_record = previous_records.get(sample_id)

                if feature_path.exists() and not overwrite:
                    skipped_existing += 1

                    if previous_record is not None:
                        # Reuse the existing metadata.
                        metadata = dict(previous_record)
                        metadata["feature_file"] = relative_feature_path

                        manifest_records.append(metadata)

                        print(
                            f"[{index}/{len(samples)}] "
                            f"Skipped existing feature and preserved "
                            f"metadata: {sample_id}"
                        )

                        continue

                    # The feature exists, but its metadata is missing.
                    # Reconstruct metadata without replacing the array.
                    features, metadata = self.extract_sample(
                        sample,
                        frame_stride=frame_stride,
                    )

                    self._validate_existing_feature(
                        feature_path,
                        expected_shape=features.shape,
                    )

                    metadata["feature_file"] = relative_feature_path

                    manifest_records.append(metadata)
                    recovered_metadata += 1

                    print(
                        f"[{index}/{len(samples)}] "
                        f"Recovered missing metadata without replacing "
                        f"feature: {sample_id}"
                    )

                    continue

                # Extract and save a new feature array.
                features, metadata = self.extract_sample(
                    sample,
                    frame_stride=frame_stride,
                )

                temporary_path = feature_path.with_suffix(".tmp")

                with temporary_path.open("wb") as file:
                    np.save(
                        file,
                        features,
                        allow_pickle=False,
                    )

                temporary_path.replace(feature_path)

                metadata["feature_file"] = relative_feature_path

                successful_records.append(metadata)
                manifest_records.append(metadata)

                print(
                    f"[{index}/{len(samples)}] "
                    f"Processed {sample_id}: {features.shape}"
                )

            except Exception as exc:
                failures.append({
                    "sample_id": sample_id,
                    "error": str(exc),
                })

                print(
                    f"[{index}/{len(samples)}] "
                    f"FAILED {sample_id}: {exc}"
                )

                # Preserve old metadata if processing failed and an
                # existing feature and metadata record are available.
                if (
                    not overwrite
                    and previous_records.get(sample_id) is not None
                    and feature_path.exists()
                ):
                    preserved = dict(
                        previous_records[sample_id]
                    )
                    preserved["feature_file"] = str(
                        feature_path.relative_to(output_dir)
                    )

                    manifest_records.append(preserved)

        # Write the complete manifest atomically.
        temporary_manifest = feature_manifest.with_suffix(".tmp")

        with temporary_manifest.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in manifest_records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        temporary_manifest.replace(feature_manifest)

        report = {
            "dataset_name": "KTH",
            "source_manifest": str(manifest_path),
            "total_samples_requested": len(samples),
            "successful_samples": len(successful_records),
            "failed_samples": len(failures),
            "skipped_existing_samples": skipped_existing,
            "recovered_missing_metadata": recovered_metadata,
            "manifest_records_written": len(manifest_records),
            "frame_stride": frame_stride,
            "feature_dimension": 283,
            "feature_directory": str(feature_dir),
            "feature_manifest": str(feature_manifest),
            "failures": failures,
        }

        report_path = output_dir / "pose_feature_report.json"
        temporary_report = report_path.with_suffix(".tmp")

        with temporary_report.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=2)

        temporary_report.replace(report_path)

        return report

    def close(self) -> None:
        """Release the pose estimator if this class created it."""
        if (
            self._owns_pose_estimator
            and self.pose_estimator is not None
        ):
            self.pose_estimator.close()
            self.pose_estimator = None

    def __enter__(self) -> "VideoSequenceFeatureExtractor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()


"""Build fixed-length activity sequences from pose feature recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

@dataclass
class FeatureRecording:
    """Pose features and metadata belonging to one video recording."""

    features: np.ndarray
    label: str
    sample_id: str
    subject_id: Optional[str] = None
    site_id: Optional[str] = None
    split: Optional[str] = None

@dataclass
class SequenceDataset:
    """Collection of fixed-length feature sequences and their metadata."""

    X: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    subject_ids: tuple[Optional[str], ...]
    site_ids: tuple[Optional[str], ...]
    splits: tuple[Optional[str], ...]

    def __len__(self) -> int:
        return len(self.X)

    def select_split(self, split_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Select sequences belonging to a named dataset split."""
        indices = [
            index
            for index, split in enumerate(self.splits)
            if split == split_name
        ]

        if not indices:
            raise ValueError(
                f"No sequences found for split {split_name!r}."
            )

        return self.X[indices], self.labels[indices]

class SequenceBuilder:
    """Convert per-video frame features into fixed-length sequences."""

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        sequence_length: int = 30,
        stride: int = 15,
        feature_dim: int = 283,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")

        if stride <= 0:
            raise ValueError("stride must be greater than zero.")

        if feature_dim <= 0:
            raise ValueError("feature_dim must be greater than zero.")

        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_dim = feature_dim

    def build(
        self,
        recordings: Sequence[FeatureRecording],
    ) -> SequenceDataset:
        """Create overlapping fixed-length sequences from recordings."""
        if not recordings:
            raise ValueError("At least one recording is required.")

        sequences = []
        labels = []
        sample_ids = []
        subject_ids = []
        site_ids = []
        splits = []

        for recording in recordings:
            features = np.asarray(recording.features, dtype=np.float32)

            if features.ndim != 2:
                raise ValueError(
                    f"Recording {recording.sample_id!r} must contain "
                    "a 2D feature array."
                )

            if features.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Recording {recording.sample_id!r} has "
                    f"{features.shape[1]} features per frame; "
                    f"expected {self.feature_dim}."
                )

            if not np.isfinite(features).all():
                raise ValueError(
                    f"Recording {recording.sample_id!r} contains "
                    "NaN or infinite feature values."
                )

            if not recording.sample_id.strip():
                raise ValueError("sample_id cannot be empty.")

            label = recording.label.strip().lower()

            if not label:
                raise ValueError("Activity label cannot be empty.")

            split = recording.split

            if split is not None and split not in self.VALID_SPLITS:
                raise ValueError(
                    f"Invalid split {split!r}. "
                    f"Expected one of {sorted(self.VALID_SPLITS)}."
                )

            if len(features) < self.sequence_length:
                continue

            last_start = len(features) - self.sequence_length

            for start in range(0, last_start + 1, self.stride):
                end = start + self.sequence_length

                sequences.append(features[start:end])
                labels.append(label)
                sample_ids.append(
                    f"{recording.sample_id}_frame_{start}_{end}"
                )
                subject_ids.append(recording.subject_id)
                site_ids.append(recording.site_id)
                splits.append(split)

        if not sequences:
            raise ValueError(
                "No sequences could be created. "
                "Check the recording lengths and sequence_length."
            )

        return SequenceDataset(
            X=np.asarray(sequences, dtype=np.float32),
            labels=np.asarray(labels, dtype=str),
            sample_ids=tuple(sample_ids),
            subject_ids=tuple(subject_ids),
            site_ids=tuple(site_ids),
            splits=tuple(splits),
        )
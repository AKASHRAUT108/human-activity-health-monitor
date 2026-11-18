
"""Encode activity names into stable integer class labels."""

from __future__ import annotations

import numpy as np

KTH_ACTIVITY_CLASSES = (
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking",
)

class ActivityLabelEncoder:
    """Convert supported activity labels to integer class indices."""

    def __init__(
        self,
        classes: tuple[str, ...] = KTH_ACTIVITY_CLASSES,
    ) -> None:
        normalized = tuple(label.strip().lower() for label in classes)

        if not normalized:
            raise ValueError("At least one activity class is required.")

        if any(not label for label in normalized):
            raise ValueError("Activity class names cannot be empty.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Activity class names must be unique.")

        self.classes = normalized
        self._class_to_index = {
            label: index for index, label in enumerate(self.classes)
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def transform(self, labels: list[str] | tuple[str, ...]) -> np.ndarray:
        """Convert activity names into integer indices."""
        encoded = []

        for label in labels:
            normalized = str(label).strip().lower()

            if normalized not in self._class_to_index:
                raise ValueError(f"Unsupported activity label: {label!r}")

            encoded.append(self._class_to_index[normalized])

        return np.asarray(encoded, dtype=np.int64)

    def inverse_transform(
        self,
        encoded_labels: list[int] | tuple[int, ...] | np.ndarray,
    ) -> list[str]:
        """Convert integer indices back into activity names."""
        decoded = []

        for value in encoded_labels:
            index = int(value)

            if index < 0 or index >= self.num_classes:
                raise ValueError(f"Invalid activity class index: {index}")

            decoded.append(self.classes[index])

        return decoded
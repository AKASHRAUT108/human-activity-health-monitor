
"""Map dataset-specific activity labels to common activity labels."""

import re


class LabelMapper:
    """Normalize compatible activity names across datasets."""

    LABEL_ALIASES = {
        "walking": "walking",
        "jogging": "jogging",
        "running": "running",
        "boxing": "boxing",
        "boxingspeedbag": "boxing",
        "boxingpunchingbag": "boxing",
        "handwaving": "hand_waving",
        "hand_waving": "hand_waving",
        "handclapping": "hand_clapping",
        "hand_clapping": "hand_clapping",
    }

    CANONICAL_LABELS = tuple(sorted(set(LABEL_ALIASES.values())))

    @staticmethod
    def normalize(label: str) -> str:
        """Normalize a label for case- and punctuation-insensitive lookup."""

        if not isinstance(label, str) or not label.strip():
            raise ValueError("Activity label must be a non-empty string.")

        return re.sub(r"[^a-z0-9]", "", label.lower())

    def map_label(self, raw_label: str) -> str | None:
        """
        Return a canonical label, or None when no safe mapping exists.

        Unknown activities are deliberately not forced into another class.
        """

        normalized = self.normalize(raw_label)
        return self.LABEL_ALIASES.get(normalized)

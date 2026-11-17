
"""Create reproducible, subject-aware dataset splits."""

import random
from dataclasses import replace

from src.dataset_preparation.metadata import SampleRecord


class SplitManager:
    """Split samples while keeping each subject in only one split."""

    VALID_SPLITS = ("train", "validation", "test")

    def assign_group_splits(
        self,
        records: list[SampleRecord],
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        group_field: str = "subject_id",
    ) -> list[SampleRecord]:
        """Assign whole subject groups to train, validation, or test."""

        ratios = (train_ratio, validation_ratio, test_ratio)

        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            for value in ratios
        ):
            raise ValueError("All split ratios must be positive numbers.")

        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError("Split ratios must add up to 1.0.")

        if group_field not in ("subject_id", "site_id"):
            raise ValueError(
                "group_field must be 'subject_id' or 'site_id'."
            )

        if not records:
            raise ValueError("Cannot split an empty dataset.")

        groups = {}

        for record in records:
            group_value = getattr(record, group_field)

            if not group_value:
                raise ValueError(
                    f"Missing {group_field} for sample {record.sample_id}. "
                    "A group-aware split cannot be guaranteed."
                )

            groups.setdefault(group_value, []).append(record)

        group_names = list(groups)

        if len(group_names) < 3:
            raise ValueError(
                "At least three distinct groups are required to create "
                "train, validation, and test splits."
            )

        random.Random(seed).shuffle(group_names)

        group_count = len(group_names)

        test_count = max(1, round(group_count * test_ratio))
        validation_count = max(
            1,
            round(group_count * validation_ratio),
        )

        while test_count + validation_count >= group_count:
            if validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("Not enough groups for three splits.")

        train_count = group_count - validation_count - test_count

        assignments = {}

        for group in group_names[:train_count]:
            assignments[group] = "train"

        validation_start = train_count
        validation_end = train_count + validation_count

        for group in group_names[validation_start:validation_end]:
            assignments[group] = "validation"

        for group in group_names[validation_end:]:
            assignments[group] = "test"

        return [
            replace(
                record,
                split=assignments[getattr(record, group_field)],
            )
            for record in records
        ]

    def preserve_existing_splits(
        self,
        records: list[SampleRecord],
    ) -> list[SampleRecord]:
        """Keep official dataset splits unchanged."""

        if not records:
            raise ValueError("Cannot preserve splits for an empty dataset.")

        for record in records:
            if record.split not in self.VALID_SPLITS:
                raise ValueError(
                    f"Invalid or missing split for {record.sample_id}: "
                    f"{record.split}"
                )

        return list(records)

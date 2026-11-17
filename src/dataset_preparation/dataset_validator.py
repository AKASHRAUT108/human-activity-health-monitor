
"""Validate dataset records before model training."""

from pathlib import Path

from src.config import SUPPORTED_VIDEO_EXTENSIONS
from src.dataset_preparation.metadata import SampleRecord


class DatasetValidator:
    """Check dataset records for common data-quality problems."""

    def validate(
        self,
        records: list[SampleRecord],
        require_canonical_label: bool = False,
    ) -> dict:
        errors = []
        warnings = []
        seen_sample_ids = set()
        seen_paths = set()

        for index, record in enumerate(records):

            if record.sample_id in seen_sample_ids:
                errors.append(
                    f"Duplicate sample ID at index {index}: "
                    f"{record.sample_id}"
                )
            seen_sample_ids.add(record.sample_id)

            resolved_path = str(
                Path(record.video_path).resolve()
            ).casefold()

            if resolved_path in seen_paths:
                errors.append(
                    f"Duplicate video path at index {index}: "
                    f"{record.video_path}"
                )
            seen_paths.add(resolved_path)

            video_path = Path(record.video_path)

            if not video_path.is_file():
                errors.append(
                    f"Video file does not exist: {record.video_path}"
                )
            elif video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                errors.append(
                    f"Unsupported video extension: {record.video_path}"
                )

            if not record.raw_label.strip():
                errors.append(
                    f"Missing raw label at index {index}"
                )

            if not record.canonical_label:
                message = (
                    f"Unmapped activity label at index {index}: "
                    f"{record.raw_label}"
                )

                if require_canonical_label:
                    errors.append(message)
                else:
                    warnings.append(message)

            if not record.subject_id:
                warnings.append(
                    f"Missing subject ID at index {index}: "
                    f"{record.sample_id}"
                )

        return {
            "valid": len(errors) == 0,
            "total_samples": len(records),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

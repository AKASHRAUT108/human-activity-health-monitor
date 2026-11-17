
"""Dataset adapter for KTH human action videos."""

import re
from pathlib import Path

from src.config import SUPPORTED_VIDEO_EXTENSIONS
from src.dataset_preparation.label_mapper import LabelMapper
from src.dataset_preparation.metadata import SampleRecord


class KTHAdapter:
    """Read KTH videos and extract action and subject information."""

    DATASET_NAME = "KTH"

    FILENAME_PATTERN = re.compile(
        r"person(?P<subject>\d+)_"
        r"(?P<action>boxing|handclapping|handwaving|jogging|running|walking)_",
        re.IGNORECASE,
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.label_mapper = LabelMapper()

    def scan(self) -> list[SampleRecord]:
        """Discover KTH videos recursively."""

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"KTH dataset directory not found: {self.root}"
            )

        records = []

        for video_path in sorted(self.root.rglob("*")):
            if (
                not video_path.is_file()
                or video_path.suffix.lower()
                not in SUPPORTED_VIDEO_EXTENSIONS
            ):
                continue

            match = self.FILENAME_PATTERN.search(video_path.stem)

            if match is None:
                continue

            raw_label = match.group("action").lower()
            subject_id = f"person{match.group('subject')}"

            canonical_label = self.label_mapper.map_label(raw_label)

            relative_path = video_path.relative_to(self.root)

            records.append(
                SampleRecord.from_video(
                    dataset_name=self.DATASET_NAME,
                    video_path=video_path,
                    sample_id=f"KTH:{relative_path.as_posix()}",
                    raw_label=raw_label,
                    canonical_label=canonical_label,
                    subject_id=subject_id,
                    source_type="public_dataset",
                )
            )

        return records

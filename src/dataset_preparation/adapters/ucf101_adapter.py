
"""Dataset adapter for UCF101 human action videos."""

from pathlib import Path

from src.config import SUPPORTED_VIDEO_EXTENSIONS
from src.dataset_preparation.label_mapper import LabelMapper
from src.dataset_preparation.metadata import SampleRecord


class UCF101Adapter:
    """Read UCF101 class folders and official split files."""

    DATASET_NAME = "UCF101"

    def __init__(
        self,
        root: str | Path,
        split_number: int = 1,
    ):
        self.root = Path(root)

        if (
            not isinstance(split_number, int)
            or isinstance(split_number, bool)
            or split_number not in (1, 2, 3)
        ):
            raise ValueError("split_number must be 1, 2, or 3.")

        self.split_number = split_number
        self.label_mapper = LabelMapper()

    @staticmethod
    def _normalize_relative_path(value: str) -> str:
        """Normalize paths from official split-list files."""

        value = value.strip().replace("\\", "/")

        if value.startswith("UCF-101/"):
            value = value[len("UCF-101/"):]

        return value.casefold()

    def _load_split_file(self, filename: str) -> dict:
        """Read an official UCF101 train or test list."""

        split_directory = self.root / "ucfTrainTestlist"
        split_path = split_directory / filename

        if not split_path.is_file():
            return {}

        split_mapping = {}

        with split_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if not parts:
                    continue

                relative_path = self._normalize_relative_path(parts[0])
                split_mapping[relative_path] = (
                    "train" if filename.startswith("train") else "test"
                )

        return split_mapping

    def scan(self) -> list[SampleRecord]:
        """Discover UCF101 videos and apply official splits if available."""

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"UCF101 dataset directory not found: {self.root}"
            )

        video_root = self.root / "UCF-101"

        if not video_root.is_dir():
            video_root = self.root

        split_filename_train = (
            f"trainlist{self.split_number:02d}.txt"
        )
        split_filename_test = (
            f"testlist{self.split_number:02d}.txt"
        )

        split_mapping = {}
        split_mapping.update(
            self._load_split_file(split_filename_train)
        )
        split_mapping.update(
            self._load_split_file(split_filename_test)
        )

        records = []

        for video_path in sorted(video_root.rglob("*")):
            if (
                not video_path.is_file()
                or video_path.suffix.lower()
                not in SUPPORTED_VIDEO_EXTENSIONS
            ):
                continue

            try:
                relative_path = video_path.relative_to(video_root)
            except ValueError:
                continue

            if len(relative_path.parts) < 2:
                continue

            raw_label = relative_path.parts[0]

            split_key = self._normalize_relative_path(
                relative_path.as_posix()
            )

            records.append(
                SampleRecord.from_video(
                    dataset_name=self.DATASET_NAME,
                    video_path=video_path,
                    sample_id=f"UCF101:{relative_path.as_posix()}",
                    raw_label=raw_label,
                    canonical_label=self.label_mapper.map_label(
                        raw_label
                    ),
                    split=split_mapping.get(split_key),
                    source_type="public_dataset",
                )
            )

        return records

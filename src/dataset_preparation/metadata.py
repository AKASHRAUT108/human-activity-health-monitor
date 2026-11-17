
"""Standard metadata structure for activity-recognition datasets."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SampleRecord:
    """Represent one video sample and its metadata."""

    sample_id: str
    dataset_name: str
    video_path: str
    raw_label: str
    canonical_label: Optional[str] = None
    subject_id: Optional[str] = None
    site_id: Optional[str] = None
    split: Optional[str] = None
    source_type: str = "public_dataset"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_video(
        cls,
        dataset_name: str,
        video_path: str | Path,
        raw_label: str,
        canonical_label: Optional[str] = None,
        subject_id: Optional[str] = None,
        site_id: Optional[str] = None,
        split: Optional[str] = None,
        source_type: str = "public_dataset",
        sample_id: Optional[str] = None,
    ):
        path = Path(video_path)

        return cls(
            sample_id=(
                sample_id
                if sample_id is not None
                else f"{dataset_name}:{path.as_posix()}"
            ),
            dataset_name=dataset_name,
            video_path=str(path.resolve()),
            raw_label=raw_label,
            canonical_label=canonical_label,
            subject_id=subject_id,
            site_id=site_id,
            split=split,
            source_type=source_type,
        )

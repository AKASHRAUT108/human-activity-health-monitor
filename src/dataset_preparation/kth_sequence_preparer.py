
"""Prepare and validate sequence-level samples from the KTH dataset."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import cv2


KTH_ACTIVITIES = (
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking",
)

KTH_CONDITIONS = ("d1", "d2", "d3", "d4")

OFFICIAL_SUBJECT_SPLITS = {
    "train": tuple(f"person{i:02d}" for i in range(11, 19)),
    "validation": (
        "person19", "person20", "person21", "person23",
        "person24", "person25", "person01", "person04",
    ),
    "test": (
        "person22", "person02", "person03", "person05",
        "person06", "person07", "person08", "person09", "person10",
    ),
}

VIDEO_ID_PATTERN = re.compile(
    r"^(?P<subject>person\d{2})_"
    r"(?P<activity>boxing|handclapping|handwaving|jogging|running|walking)_"
    r"(?P<condition>d[1-4])$"
)

FRAME_RANGE_PATTERN = re.compile(r"^(\d+)\s*-\s*(\d+)$")

EXPECTED_MISSING_ANNOTATIONS = ("person13_handclapping_d3",)


@dataclass(frozen=True)
class KTHAnnotation:
    """Annotations for one original KTH source video."""

    video_id: str
    subject_id: str
    activity: str
    condition: str
    frame_ranges: tuple[tuple[int, int], ...]
    is_missing: bool = False


@dataclass(frozen=True)
class KTHSequenceSample:
    """One annotated action repetition within a source video."""

    sample_id: str
    dataset_name: str
    source_video: str
    video_path: str
    subject_id: str
    site_id: str | None
    recording_condition: str
    raw_label: str
    canonical_label: str
    split: str
    sequence_index: int
    frame_start: int
    frame_end: int
    source_type: str = "public_dataset"

    def to_dict(self) -> dict:
        return asdict(self)


class KTHAnnotationParser:
    """Parse the official 00sequences.txt annotation file."""

    RECORD_PATTERN = re.compile(
        r"^(?P<video_id>person\d{2}_(?:boxing|handclapping|handwaving|"
        r"jogging|running|walking)_d[1-4])\s+(?P<payload>.+?)\s*$"
    )

    @staticmethod
    def _split_video_id(video_id: str) -> tuple[str, str, str]:
        match = VIDEO_ID_PATTERN.fullmatch(video_id)
        if match is None:
            raise ValueError(f"Invalid KTH video identifier: {video_id}")
        return (
            match.group("subject"),
            match.group("activity"),
            match.group("condition"),
        )

    @classmethod
    def parse_lines(cls, lines) -> list[KTHAnnotation]:
        """Parse annotation lines and preserve explicitly missing entries."""
        annotations = []
        seen_ids = set()

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line.startswith("person"):
                continue

            match = cls.RECORD_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Malformed KTH annotation at line {line_number}: {line}"
                )

            video_id = match.group("video_id")
            payload = match.group("payload").strip()

            if video_id in seen_ids:
                raise ValueError(
                    f"Duplicate KTH annotation for {video_id} "
                    f"at line {line_number}"
                )
            seen_ids.add(video_id)

            subject_id, activity, condition = cls._split_video_id(video_id)

            if payload == "*missing*":
                annotations.append(
                    KTHAnnotation(
                        video_id=video_id,
                        subject_id=subject_id,
                        activity=activity,
                        condition=condition,
                        frame_ranges=(),
                        is_missing=True,
                    )
                )
                continue

            if not payload.startswith("frames"):
                raise ValueError(
                    f"Expected 'frames' or '*missing*' for {video_id} "
                    f"at line {line_number}"
                )

            ranges_text = payload[len("frames"):].strip()
            if not ranges_text:
                raise ValueError(f"No frame ranges provided for {video_id}")

            frame_ranges = []
            for item in ranges_text.split(","):
                range_match = FRAME_RANGE_PATTERN.fullmatch(item.strip())
                if range_match is None:
                    raise ValueError(
                        f"Invalid frame range for {video_id} "
                        f"at line {line_number}: {item.strip()}"
                    )

                start, end = map(int, range_match.groups())
                if start < 1 or end < start:
                    raise ValueError(
                        f"Invalid 1-based frame range {start}-{end} "
                        f"for {video_id}"
                    )
                frame_ranges.append((start, end))

            annotations.append(
                KTHAnnotation(
                    video_id=video_id,
                    subject_id=subject_id,
                    activity=activity,
                    condition=condition,
                    frame_ranges=tuple(frame_ranges),
                )
            )

        return annotations

    @classmethod
    def parse_file(cls, annotation_file: Path) -> list[KTHAnnotation]:
        annotation_file = Path(annotation_file)
        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"KTH annotation file not found: {annotation_file}"
            )
        with annotation_file.open("r", encoding="utf-8-sig") as file:
            return cls.parse_lines(file)


class KTHDatasetPreparer:
    """Extract KTH videos, validate them, and create a JSONL manifest."""

    DATASET_NAME = "KTH"

    def __init__(
        self,
        project_root: Path,
        downloads_dir: Path | None = None,
        annotations_file: Path | None = None,
        videos_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.downloads_dir = Path(
            downloads_dir or self.project_root / "data/raw/kth/downloads"
        ).resolve()
        self.annotations_file = Path(
            annotations_file
            or self.project_root / "data/raw/kth/annotations/00sequences.txt"
        ).resolve()
        self.videos_dir = Path(
            videos_dir or self.project_root / "data/raw/kth/videos"
        ).resolve()
        self.output_dir = Path(
            output_dir or self.project_root / "data/processed/kth"
        ).resolve()
        self.manifest_path = self.output_dir / "kth_sequences.jsonl"
        self.report_path = self.output_dir / "kth_preparation_report.json"

    def _archive_paths(self) -> dict[str, Path]:
        archives = {}
        for activity in KTH_ACTIVITIES:
            archive = self.downloads_dir / f"{activity}.zip"
            if not archive.is_file():
                raise FileNotFoundError(f"Missing KTH archive: {archive}")
            archives[activity] = archive
        return archives

    @staticmethod
    def _video_id_from_archive_name(filename: str) -> str:
        stem = Path(filename).stem
        if stem.endswith("_uncomp"):
            stem = stem[:-len("_uncomp")]
        if VIDEO_ID_PATTERN.fullmatch(stem) is None:
            raise ValueError(f"Unexpected KTH AVI filename: {filename}")
        return stem

    def extract_archives(self) -> dict:
        """Safely extract expected AVI files and skip complete files."""
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        archives = self._archive_paths()
        extracted = 0
        already_present = 0
        extracted_ids = set()
        activity_counts = Counter()

        for expected_activity, archive_path in archives.items():
            try:
                with ZipFile(archive_path, "r") as archive:
                    corrupt_member = archive.testzip()
                    if corrupt_member is not None:
                        raise BadZipFile(
                            f"Corrupt member in {archive_path.name}: "
                            f"{corrupt_member}"
                        )

                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        member_path = PurePosixPath(info.filename)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            raise ValueError(
                                f"Unsafe archive member path: {info.filename}"
                            )
                        if member_path.suffix.lower() != ".avi":
                            continue

                        filename = member_path.name
                        if not filename:
                            raise ValueError(
                                f"Invalid archive filename: {info.filename}"
                            )
                        video_id = self._video_id_from_archive_name(filename)
                        match = VIDEO_ID_PATTERN.fullmatch(video_id)
                        if match.group("activity") != expected_activity:
                            raise ValueError(
                                f"{filename} is in {archive_path.name}, "
                                f"but its activity is {match.group('activity')}"
                            )
                        if video_id in extracted_ids:
                            raise ValueError(
                                f"Duplicate video across KTH archives: {video_id}"
                            )

                        extracted_ids.add(video_id)
                        activity_counts[expected_activity] += 1
                        destination = self.videos_dir / filename

                        if (
                            destination.is_file()
                            and destination.stat().st_size == info.file_size
                        ):
                            already_present += 1
                            continue

                        temporary_path = destination.with_name(
                            destination.name + ".part"
                        )
                        try:
                            with archive.open(info, "r") as source:
                                with temporary_path.open("wb") as target:
                                    shutil.copyfileobj(source, target)
                            if temporary_path.stat().st_size != info.file_size:
                                raise IOError(
                                    f"Extracted size mismatch: {filename}"
                                )
                            temporary_path.replace(destination)
                            extracted += 1
                        finally:
                            if temporary_path.exists():
                                temporary_path.unlink()

            except (BadZipFile, ValueError):
                raise
            except OSError as exc:
                raise RuntimeError(
                    f"Could not extract {archive_path}: {exc}"
                ) from exc

        return {
            "extracted": extracted,
            "already_present": already_present,
            "total_archive_videos": len(extracted_ids),
            "activity_video_counts": dict(activity_counts),
        }

    @staticmethod
    def _official_split(subject_id: str) -> str:
        for split_name, subjects in OFFICIAL_SUBJECT_SPLITS.items():
            if subject_id in subjects:
                return split_name
        raise ValueError(
            f"No official KTH split defined for subject {subject_id}"
        )

    def _relative_video_path(self, video_path: Path) -> str:
        try:
            return Path(video_path).resolve().relative_to(
                self.project_root
            ).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Path is outside project root: {video_path}"
            ) from exc

    def build_samples(
        self,
        annotations: list[KTHAnnotation],
    ) -> tuple[list[KTHSequenceSample], dict]:
        """Create one manifest record per annotated action repetition."""
        samples = []
        missing_annotation_ids = []
        missing_video_ids = []
        matched_video_ids = set()

        for annotation in annotations:
            if annotation.is_missing:
                missing_annotation_ids.append(annotation.video_id)
                continue

            video_path = (
                self.videos_dir / f"{annotation.video_id}_uncomp.avi"
            )
            if not video_path.is_file():
                alternative = self.videos_dir / f"{annotation.video_id}.avi"
                if alternative.is_file():
                    video_path = alternative
                else:
                    missing_video_ids.append(annotation.video_id)
                    continue

            matched_video_ids.add(annotation.video_id)
            split = self._official_split(annotation.subject_id)

            for sequence_index, (frame_start, frame_end) in enumerate(
                annotation.frame_ranges, start=1
            ):
                sample_id = (
                    f"KTH:{annotation.video_id}:sequence{sequence_index:02d}"
                )
                samples.append(
                    KTHSequenceSample(
                        sample_id=sample_id,
                        dataset_name=self.DATASET_NAME,
                        source_video=annotation.video_id,
                        video_path=self._relative_video_path(video_path),
                        subject_id=annotation.subject_id,
                        site_id=None,
                        recording_condition=annotation.condition,
                        raw_label=annotation.activity,
                        canonical_label=annotation.activity,
                        split=split,
                        sequence_index=sequence_index,
                        frame_start=frame_start,
                        frame_end=frame_end,
                    )
                )

        report = {
            "annotation_records": len(annotations),
            "matched_source_videos": len(matched_video_ids),
            "missing_annotation_ids": sorted(missing_annotation_ids),
            "missing_video_ids": sorted(missing_video_ids),
            "sequence_samples": len(samples),
            "split_counts": dict(Counter(s.split for s in samples)),
            "activity_counts": dict(Counter(s.canonical_label for s in samples)),
            "recording_condition_counts": dict(
                Counter(s.recording_condition for s in samples)
            ),
        }
        return samples, report

    def validate_videos(self, samples: list[KTHSequenceSample]) -> dict:
        """
        Decode each source once and exclude only sequences not fully decodable.

        An unopened/empty/invalid video remains a fatal validation error.
        A truncated source may still contribute sequences whose complete
        annotated frame ranges are present in the decoded portion.
        """
        samples_by_video = {}
        for sample in samples:
            samples_by_video.setdefault(sample.source_video, []).append(sample)

        valid_samples = []
        video_details = []
        exclusions = []
        failures = []
        total_decoded_frames = 0
        videos_with_decode_errors = 0

        for video_id, video_samples in sorted(samples_by_video.items()):
            video_path = self.project_root / video_samples[0].video_path
            capture = cv2.VideoCapture(str(video_path))

            try:
                if not capture.isOpened():
                    failures.append({
                        "video_id": video_id,
                        "reason": "OpenCV could not open the video",
                    })
                    continue

                fps = float(capture.get(cv2.CAP_PROP_FPS))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                decoded_frames = 0
                empty_frame_error = None
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    if frame is None or frame.size == 0:
                        empty_frame_error = (
                            "Decoder returned an empty frame at "
                            f"decoded frame {decoded_frames + 1}"
                        )
                        break
                    decoded_frames += 1

                if empty_frame_error:
                    failures.append({
                        "video_id": video_id,
                        "reason": empty_frame_error,
                    })
                    continue
                if decoded_frames == 0:
                    failures.append({
                        "video_id": video_id,
                        "reason": "Video contains no decodable frames",
                    })
                    continue
                if width <= 0 or height <= 0:
                    failures.append({
                        "video_id": video_id,
                        "reason": "Video has invalid dimensions",
                    })
                    continue

                total_decoded_frames += decoded_frames
                is_truncated = (
                    reported_frames > 0 and decoded_frames < reported_frames
                )
                if is_truncated:
                    videos_with_decode_errors += 1

                valid_for_video = 0
                for sample in video_samples:
                    if sample.frame_start < 1:
                        reason = "Annotated frame range starts before frame 1"
                    elif sample.frame_end > decoded_frames:
                        reason = (
                            "Annotated frame range exceeds decoded video length"
                        )
                    else:
                        valid_samples.append(sample)
                        valid_for_video += 1
                        continue

                    exclusions.append({
                        "sample_id": sample.sample_id,
                        "video_id": video_id,
                        "sequence_index": sample.sequence_index,
                        "frame_start": sample.frame_start,
                        "frame_end": sample.frame_end,
                        "decoded_frames": decoded_frames,
                        "reported_frames": reported_frames,
                        "reason": reason,
                    })

                video_details.append({
                    "video_id": video_id,
                    "video_path": video_samples[0].video_path,
                    "reported_frames": reported_frames,
                    "decoded_frames": decoded_frames,
                    "missing_frame_count": max(
                        reported_frames - decoded_frames, 0
                    ),
                    "decode_complete": not is_truncated,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "sequence_count": len(video_samples),
                    "sequences_validated": valid_for_video,
                    "sequences_excluded": len(video_samples) - valid_for_video,
                })
            finally:
                capture.release()

        if failures:
            raise ValueError(
                "KTH video validation encountered "
                f"{len(failures)} unusable source video(s). "
                f"First failure: {failures[0]}"
            )

        if not valid_samples:
            raise ValueError(
                "KTH video validation produced no usable annotated sequences."
            )

        return {
            "valid_samples": valid_samples,
            "videos_checked": len(samples_by_video),
            "videos_validated": sum(
                1 for item in video_details if item["decode_complete"]
            ),
            "videos_with_decode_errors": videos_with_decode_errors,
            "videos_failed": len(failures),
            "sequences_checked": len(samples),
            "sequences_validated": len(valid_samples),
            "sequences_excluded": len(exclusions),
            "total_decoded_frames": total_decoded_frames,
            "exclusions": exclusions,
            "failures": failures,
            "video_details": video_details,
        }

    def write_outputs(self, samples: list[KTHSequenceSample], report: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for sample in samples:
                file.write(
                    json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
                )
        with self.report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

    def prepare(
        self,
        extract: bool = True,
        validate_videos: bool = True,
    ) -> dict:
        """Prepare a manifest containing only fully decodable sequences."""
        extraction_report = None
        if extract:
            extraction_report = self.extract_archives()
        elif not self.videos_dir.is_dir():
            raise FileNotFoundError(
                f"Video directory does not exist: {self.videos_dir}"
            )

        annotations = KTHAnnotationParser.parse_file(self.annotations_file)
        samples, report = self.build_samples(annotations)

        report["dataset_name"] = self.DATASET_NAME
        report["extraction"] = extraction_report
        report["expected_missing_annotations"] = list(
            EXPECTED_MISSING_ANNOTATIONS
        )

        unexpected_missing_annotations = sorted(
            set(report["missing_annotation_ids"])
            - set(EXPECTED_MISSING_ANNOTATIONS)
        )
        absent_expected_annotations = sorted(
            set(EXPECTED_MISSING_ANNOTATIONS)
            - set(report["missing_annotation_ids"])
        )
        if unexpected_missing_annotations or absent_expected_annotations:
            raise ValueError(
                "KTH annotation missing-entry check failed. "
                f"Unexpected: {unexpected_missing_annotations}; "
                f"absent expected entries: {absent_expected_annotations}"
            )
        if report["missing_video_ids"]:
            raise ValueError(
                "KTH source videos are missing: "
                f"{report['missing_video_ids'][:10]}"
            )

        if validate_videos:
            validation_report = self.validate_videos(samples)
            samples = validation_report.pop("valid_samples")
            report["video_validation"] = validation_report
        else:
            report["video_validation"] = {
                "performed": False,
                "reason": "Video decoding validation was disabled",
                "sequences_validated": len(samples),
                "sequences_excluded": 0,
                "exclusions": [],
            }

        report["sequence_samples_before_video_validation"] = (
            report["sequence_samples"]
        )
        report["sequence_samples"] = len(samples)
        report["split_counts"] = dict(Counter(s.split for s in samples))
        report["activity_counts"] = dict(
            Counter(s.canonical_label for s in samples)
        )
        report["recording_condition_counts"] = dict(
            Counter(s.recording_condition for s in samples)
        )
        report["manifest_path"] = self._relative_video_path(self.manifest_path)
        report["frame_number_convention"] = (
            "1-based inclusive; convert to zero-based indices when "
            "seeking with OpenCV"
        )

        self.write_outputs(samples, report)
        return report

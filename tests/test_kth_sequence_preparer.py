
"""Tests for KTH sequence-level dataset preparation and validation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from src.dataset_preparation.kth_sequence_preparer import (
    KTHAnnotationParser,
    KTHDatasetPreparer,
    OFFICIAL_SUBJECT_SPLITS,
)


class FakeVideoCapture:
    """Small OpenCV VideoCapture substitute for deterministic tests."""

    def __init__(
        self, frames=185, opened=True, width=64, height=48, fps=25,
        reported_frames=None,
    ):
        self.frames = frames
        self.opened = opened
        self.width = width
        self.height = height
        self.fps = fps
        self.reported_frames = (
            frames if reported_frames is None else reported_frames
        )
        self.position = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        import cv2

        values = {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FRAME_COUNT: self.reported_frames,
        }
        return values.get(property_id, 0)

    def read(self):
        import numpy as np

        if self.position >= self.frames:
            return False, None
        self.position += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class TestKTHAnnotationParser(unittest.TestCase):

    def test_parse_one_video_with_four_sequences(self):
        lines = [
            "person01_boxing_d1 frames 1-95, 96-185, 186-245, 246-360"
        ]
        annotations = KTHAnnotationParser.parse_lines(lines)
        self.assertEqual(len(annotations), 1)
        annotation = annotations[0]
        self.assertEqual(annotation.video_id, "person01_boxing_d1")
        self.assertEqual(annotation.subject_id, "person01")
        self.assertEqual(annotation.activity, "boxing")
        self.assertEqual(annotation.condition, "d1")
        self.assertEqual(
            annotation.frame_ranges,
            ((1, 95), (96, 185), (186, 245), (246, 360)),
        )
        self.assertFalse(annotation.is_missing)

    def test_parse_explicit_missing_annotation(self):
        annotations = KTHAnnotationParser.parse_lines(
            ["person13_handclapping_d3 *missing*"]
        )
        self.assertEqual(len(annotations), 1)
        self.assertTrue(annotations[0].is_missing)
        self.assertEqual(annotations[0].frame_ranges, ())

    def test_parse_ignores_header_lines(self):
        lines = [
            "This database contains sequences.",
            "Training: person11, person12",
            "person01_walking_d1 frames 1-75, 76-150",
        ]
        annotations = KTHAnnotationParser.parse_lines(lines)
        self.assertEqual(len(annotations), 1)

    def test_parse_rejects_invalid_frame_range(self):
        with self.assertRaises(ValueError):
            KTHAnnotationParser.parse_lines(
                ["person01_boxing_d1 frames 95-1"]
            )

    def test_parse_rejects_zero_frame_number(self):
        with self.assertRaises(ValueError):
            KTHAnnotationParser.parse_lines(
                ["person01_boxing_d1 frames 0-95"]
            )

    def test_parse_rejects_malformed_range(self):
        with self.assertRaises(ValueError):
            KTHAnnotationParser.parse_lines(
                ["person01_boxing_d1 frames 1 to 95"]
            )

    def test_parse_rejects_duplicate_video_annotation(self):
        lines = [
            "person01_boxing_d1 frames 1-95",
            "person01_boxing_d1 frames 96-185",
        ]
        with self.assertRaises(ValueError):
            KTHAnnotationParser.parse_lines(lines)

    def test_parse_rejects_unknown_activity(self):
        with self.assertRaises(ValueError):
            KTHAnnotationParser.parse_lines(
                ["person01_dancing_d1 frames 1-95"]
            )

    def test_parse_rejects_unknown_payload(self):
        with self.assertRaises(ValueError):
            KTHAnnotationParser.parse_lines(
                ["person01_boxing_d1 unknown"]
            )


class TestKTHDatasetPreparer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.downloads = self.root / "downloads"
        self.annotations_dir = self.root / "annotations"
        self.videos_dir = self.root / "videos"
        self.output_dir = self.root / "processed"

        self.downloads.mkdir()
        self.annotations_dir.mkdir()

        for activity in (
            "boxing", "handclapping", "handwaving",
            "jogging", "running", "walking",
        ):
            with ZipFile(self.downloads / f"{activity}.zip", "w") as archive:
                if activity == "boxing":
                    archive.writestr(
                        "person01_boxing_d1_uncomp.avi",
                        b"test-video-data",
                    )

        self.annotation_file = self.annotations_dir / "00sequences.txt"
        self.annotation_file.write_text(
            "\n".join([
                "Training: person11, person12",
                "person01_boxing_d1 frames 1-95, 96-185",
                "person13_handclapping_d3 *missing*",
            ]),
            encoding="utf-8",
        )

        self.preparer = KTHDatasetPreparer(
            project_root=self.root,
            downloads_dir=self.downloads,
            annotations_file=self.annotation_file,
            videos_dir=self.videos_dir,
            output_dir=self.output_dir,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _samples(self):
        self.preparer.extract_archives()
        annotations = KTHAnnotationParser.parse_file(self.annotation_file)
        return self.preparer.build_samples(annotations)[0]

    def test_official_subject_splits_cover_25_subjects(self):
        all_subjects = [
            subject
            for subjects in OFFICIAL_SUBJECT_SPLITS.values()
            for subject in subjects
        ]
        self.assertEqual(len(all_subjects), 25)
        self.assertEqual(len(set(all_subjects)), 25)

    def test_official_split_assignments(self):
        self.assertEqual(self.preparer._official_split("person11"), "train")
        self.assertEqual(self.preparer._official_split("person19"), "validation")
        self.assertEqual(self.preparer._official_split("person02"), "test")

    def test_unknown_subject_has_no_split(self):
        with self.assertRaises(ValueError):
            self.preparer._official_split("person26")

    def test_extract_archives(self):
        report = self.preparer.extract_archives()
        self.assertEqual(report["extracted"], 1)
        self.assertEqual(report["total_archive_videos"], 1)
        self.assertTrue(
            (self.videos_dir / "person01_boxing_d1_uncomp.avi").is_file()
        )

    def test_extraction_is_idempotent(self):
        first = self.preparer.extract_archives()
        second = self.preparer.extract_archives()
        self.assertEqual(first["extracted"], 1)
        self.assertEqual(second["extracted"], 0)
        self.assertEqual(second["already_present"], 1)

    def test_extraction_rejects_wrong_activity_archive(self):
        with ZipFile(self.downloads / "boxing.zip", "w") as archive:
            archive.writestr(
                "person01_walking_d1_uncomp.avi",
                b"wrong-activity",
            )
        with self.assertRaises(ValueError):
            self.preparer.extract_archives()

    def test_build_samples_creates_one_record_per_sequence(self):
        samples = self._samples()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].frame_start, 1)
        self.assertEqual(samples[0].frame_end, 95)
        self.assertEqual(samples[1].frame_start, 96)
        self.assertEqual(samples[1].frame_end, 185)
        self.assertEqual(samples[0].split, "validation")
        self.assertEqual(samples[0].subject_id, "person01")
        self.assertIsNone(samples[0].site_id)
        self.assertEqual(samples[0].recording_condition, "d1")
        self.assertEqual(samples[0].canonical_label, "boxing")
        self.assertEqual(
            samples[0].sample_id,
            "KTH:person01_boxing_d1:sequence01",
        )

    def test_missing_annotation_is_reported(self):
        self.preparer.extract_archives()
        annotations = KTHAnnotationParser.parse_file(self.annotation_file)
        _, report = self.preparer.build_samples(annotations)
        self.assertEqual(report["matched_source_videos"], 1)
        self.assertEqual(
            report["missing_annotation_ids"],
            ["person13_handclapping_d3"],
        )

    def test_missing_video_is_reported(self):
        annotations = KTHAnnotationParser.parse_lines(
            ["person02_walking_d1 frames 1-100"]
        )
        samples, report = self.preparer.build_samples(annotations)
        self.assertEqual(samples, [])
        self.assertEqual(report["missing_video_ids"], ["person02_walking_d1"])

    def test_video_validation_counts_decoded_frames(self):
        samples = self._samples()
        fake_capture = FakeVideoCapture(frames=185)
        with patch(
            "src.dataset_preparation.kth_sequence_preparer.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            report = self.preparer.validate_videos(samples)

        self.assertEqual(report["videos_checked"], 1)
        self.assertEqual(report["videos_validated"], 1)
        self.assertEqual(report["videos_with_decode_errors"], 0)
        self.assertEqual(report["sequences_validated"], 2)
        self.assertEqual(report["sequences_excluded"], 0)
        self.assertEqual(report["total_decoded_frames"], 185)
        self.assertEqual(len(report["valid_samples"]), 2)
        self.assertTrue(fake_capture.released)

    def test_video_validation_excludes_only_sequence_past_decoded_end(self):
        samples = self._samples()
        fake_capture = FakeVideoCapture(
            frames=100,
            reported_frames=185,
        )
        with patch(
            "src.dataset_preparation.kth_sequence_preparer.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            report = self.preparer.validate_videos(samples)

        self.assertEqual(report["videos_checked"], 1)
        self.assertEqual(report["videos_validated"], 0)
        self.assertEqual(report["videos_with_decode_errors"], 1)
        self.assertEqual(report["sequences_checked"], 2)
        self.assertEqual(report["sequences_validated"], 1)
        self.assertEqual(report["sequences_excluded"], 1)
        self.assertEqual(
            report["valid_samples"][0].sample_id,
            "KTH:person01_boxing_d1:sequence01",
        )
        self.assertEqual(
            report["exclusions"][0]["sample_id"],
            "KTH:person01_boxing_d1:sequence02",
        )
        self.assertTrue(fake_capture.released)

    def test_video_validation_rejects_unopenable_video(self):
        samples = self._samples()
        fake_capture = FakeVideoCapture(opened=False)
        with patch(
            "src.dataset_preparation.kth_sequence_preparer.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            with self.assertRaises(ValueError):
                self.preparer.validate_videos(samples)
        self.assertTrue(fake_capture.released)

    def test_video_validation_rejects_video_with_no_decodable_frames(self):
        samples = self._samples()
        fake_capture = FakeVideoCapture(frames=0, reported_frames=185)
        with patch(
            "src.dataset_preparation.kth_sequence_preparer.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            with self.assertRaises(ValueError):
                self.preparer.validate_videos(samples)
        self.assertTrue(fake_capture.released)

    def test_prepare_writes_only_validated_sequences_and_exclusion_report(self):
        fake_capture = FakeVideoCapture(frames=100, reported_frames=185)
        with patch(
            "src.dataset_preparation.kth_sequence_preparer.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            report = self.preparer.prepare(extract=True)

        manifest_lines = self.preparer.manifest_path.read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(manifest_lines), 1)
        manifest_sample = json.loads(manifest_lines[0])
        self.assertEqual(manifest_sample["frame_start"], 1)
        self.assertEqual(manifest_sample["frame_end"], 95)

        saved_report = json.loads(
            self.preparer.report_path.read_text(encoding="utf-8")
        )
        self.assertEqual(saved_report["sequence_samples_before_video_validation"], 2)
        self.assertEqual(saved_report["sequence_samples"], 1)
        self.assertEqual(
            saved_report["expected_missing_annotations"],
            ["person13_handclapping_d3"],
        )
        self.assertEqual(
            saved_report["video_validation"]["sequences_excluded"], 1
        )
        self.assertEqual(
            saved_report["video_validation"]["exclusions"][0]["frame_start"],
            96,
        )
        self.assertEqual(report["sequence_samples"], 1)

    def test_prepare_without_video_validation_keeps_all_samples(self):
        self.preparer.extract_archives()
        report = self.preparer.prepare(
            extract=False,
            validate_videos=False,
        )
        self.assertEqual(report["sequence_samples"], 2)
        self.assertEqual(
            len(self.preparer.manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()),
            2,
        )

    def test_prepare_rejects_unexpected_missing_annotation(self):
        self.annotation_file.write_text(
            "person01_boxing_d1 frames 1-95",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self.preparer.prepare(
                extract=True,
                validate_videos=False,
            )


if __name__ == "__main__":
    unittest.main()

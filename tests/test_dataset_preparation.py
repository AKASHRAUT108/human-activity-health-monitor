
"""Tests for dataset preparation components."""

import tempfile
import unittest
from pathlib import Path

from src.dataset_preparation.metadata import SampleRecord
from src.dataset_preparation.label_mapper import LabelMapper
from src.dataset_preparation.dataset_validator import DatasetValidator
from src.dataset_preparation.split_manager import SplitManager
from src.dataset_preparation.adapters.kth_adapter import KTHAdapter
from src.dataset_preparation.adapters.ucf101_adapter import UCF101Adapter
from src.dataset_preparation.dataset_registry import DatasetRegistry


class TestLabelMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = LabelMapper()

    def test_map_walking(self):
        self.assertEqual(self.mapper.map_label("Walking"), "walking")

    def test_map_hand_waving(self):
        self.assertEqual(
            self.mapper.map_label("handwaving"),
            "hand_waving",
        )

    def test_map_boxing_alias(self):
        self.assertEqual(
            self.mapper.map_label("BoxingSpeedBag"),
            "boxing",
        )

    def test_unknown_label_returns_none(self):
        self.assertIsNone(self.mapper.map_label("PlayingGuitar"))

    def test_empty_label_raises_error(self):
        with self.assertRaises(ValueError):
            self.mapper.map_label("")


class TestSampleRecord(unittest.TestCase):

    def test_create_record(self):
        record = SampleRecord.from_video(
            dataset_name="KTH",
            video_path="person01_walking.avi",
            raw_label="walking",
            canonical_label="walking",
            subject_id="person01",
        )

        self.assertEqual(record.dataset_name, "KTH")
        self.assertEqual(record.subject_id, "person01")
        self.assertEqual(record.canonical_label, "walking")

    def test_convert_record_to_dictionary(self):
        record = SampleRecord(
            sample_id="sample-1",
            dataset_name="KTH",
            video_path="sample.avi",
            raw_label="walking",
        )

        self.assertEqual(record.to_dict()["sample_id"], "sample-1")


class TestKTHAdapter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_kth_videos(self):
        video = self.root / "person01_walking_d1_uncomp.avi"
        video.touch()

        records = KTHAdapter(self.root).scan()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].raw_label, "walking")
        self.assertEqual(records[0].canonical_label, "walking")
        self.assertEqual(records[0].subject_id, "person01")

    def test_ignore_unrecognized_filename(self):
        (self.root / "unknown_video.avi").touch()

        records = KTHAdapter(self.root).scan()

        self.assertEqual(records, [])

    def test_missing_root_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            KTHAdapter(self.root / "missing").scan()


class TestUCF101Adapter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.video_root = self.root / "UCF-101"
        self.split_root = self.root / "ucfTrainTestlist"

        self.video_root.mkdir()
        self.split_root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_class_folder(self):
        class_dir = self.video_root / "Walking"
        class_dir.mkdir()

        (class_dir / "v_Walking_g01_c01.avi").touch()

        records = UCF101Adapter(self.root).scan()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].raw_label, "Walking")
        self.assertEqual(records[0].canonical_label, "walking")

    def test_read_official_split_files(self):
        class_dir = self.video_root / "Walking"
        class_dir.mkdir()

        video = class_dir / "v_Walking_g01_c01.avi"
        video.touch()

        (self.split_root / "trainlist01.txt").write_text(
            "Walking/v_Walking_g01_c01.avi 1\n",
            encoding="utf-8",
        )

        (self.split_root / "testlist01.txt").write_text(
            "",
            encoding="utf-8",
        )

        records = UCF101Adapter(self.root).scan()

        self.assertEqual(records[0].split, "train")

    def test_split_number_validation(self):
        with self.assertRaises(ValueError):
            UCF101Adapter(self.root, split_number=4)

    def test_missing_root_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            UCF101Adapter(self.root / "missing").scan()


class TestDatasetValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video = self.root / "sample.avi"
        self.video.touch()

        self.validator = DatasetValidator()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_record(self):
        record = SampleRecord(
            sample_id="sample-1",
            dataset_name="KTH",
            video_path=str(self.video),
            raw_label="walking",
            canonical_label="walking",
            subject_id="person01",
        )

        result = self.validator.validate([record])

        self.assertTrue(result["valid"])
        self.assertEqual(result["error_count"], 0)

    def test_missing_video_is_invalid(self):
        record = SampleRecord(
            sample_id="sample-1",
            dataset_name="KTH",
            video_path=str(self.root / "missing.avi"),
            raw_label="walking",
            canonical_label="walking",
        )

        result = self.validator.validate([record])

        self.assertFalse(result["valid"])
        self.assertGreater(result["error_count"], 0)

    def test_unmapped_label_is_warning_by_default(self):
        record = SampleRecord(
            sample_id="sample-1",
            dataset_name="UCF101",
            video_path=str(self.video),
            raw_label="PlayingGuitar",
            subject_id="subject01",
        )

        result = self.validator.validate([record])

        self.assertTrue(result["valid"])
        self.assertEqual(result["warning_count"], 1)
    def test_unmapped_label_can_be_error(self):
        record = SampleRecord(
            sample_id="sample-1",
            dataset_name="UCF101",
            video_path=str(self.video),
            raw_label="PlayingGuitar",
        )

        result = self.validator.validate(
            [record],
            require_canonical_label=True,
        )

        self.assertFalse(result["valid"])

    def test_duplicate_sample_id_is_invalid(self):
        record = SampleRecord(
            sample_id="duplicate",
            dataset_name="KTH",
            video_path=str(self.video),
            raw_label="walking",
            canonical_label="walking",
        )

        result = self.validator.validate([record, record])

        self.assertFalse(result["valid"])


class TestSplitManager(unittest.TestCase):

    def setUp(self):
        self.manager = SplitManager()

        self.records = [
            SampleRecord(
                sample_id=f"sample-{subject}-{index}",
                dataset_name="KTH",
                video_path=f"video-{subject}-{index}.avi",
                raw_label="walking",
                canonical_label="walking",
                subject_id=f"person{subject}",
            )
            for subject in range(1, 11)
            for index in range(2)
        ]

    def test_create_subject_aware_splits(self):
        result = self.manager.assign_group_splits(self.records)

        self.assertEqual(len(result), len(self.records))

        subject_splits = {}

        for record in result:
            subject_splits.setdefault(
                record.subject_id,
                set(),
            ).add(record.split)

        self.assertTrue(
            all(len(splits) == 1 for splits in subject_splits.values())
        )

        self.assertEqual(
            {record.split for record in result},
            {"train", "validation", "test"},
        )

    def test_splits_are_reproducible(self):
        first = self.manager.assign_group_splits(
            self.records,
            seed=42,
        )

        second = self.manager.assign_group_splits(
            self.records,
            seed=42,
        )

        self.assertEqual(
            [record.split for record in first],
            [record.split for record in second],
        )

    def test_missing_subject_id_raises_error(self):
        record = SampleRecord(
            sample_id="sample-1",
            dataset_name="KTH",
            video_path="video.avi",
            raw_label="walking",
        )

        with self.assertRaises(ValueError):
            self.manager.assign_group_splits([record])

    def test_invalid_ratios_raise_error(self):
        with self.assertRaises(ValueError):
            self.manager.assign_group_splits(
                self.records,
                train_ratio=0.8,
                validation_ratio=0.3,
                test_ratio=0.1,
            )

    def test_preserve_existing_splits(self):
        records = [
            SampleRecord(
                sample_id="train-1",
                dataset_name="UCF101",
                video_path="train.avi",
                raw_label="walking",
                split="train",
            ),
            SampleRecord(
                sample_id="test-1",
                dataset_name="UCF101",
                video_path="test.avi",
                raw_label="walking",
                split="test",
            ),
        ]

        result = self.manager.preserve_existing_splits(records)

        self.assertEqual(result[0].split, "train")
        self.assertEqual(result[1].split, "test")


class TestDatasetRegistry(unittest.TestCase):

    def test_register_and_scan_adapter(self):
        registry = DatasetRegistry()

        class FakeAdapter:
            def scan(self):
                return ["sample"]

        registry.register("Example", FakeAdapter())

        self.assertEqual(registry.scan("example"), ["sample"])

    def test_unknown_dataset_raises_error(self):
        registry = DatasetRegistry()

        with self.assertRaises(KeyError):
            registry.scan("unknown")

    def test_duplicate_registration_raises_error(self):
        registry = DatasetRegistry()

        class FakeAdapter:
            def scan(self):
                return []

        registry.register("Example", FakeAdapter())

        with self.assertRaises(ValueError):
            registry.register("Example", FakeAdapter())


if __name__ == "__main__":
    unittest.main()

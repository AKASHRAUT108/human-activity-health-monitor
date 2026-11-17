
"""Unit tests for the VideoReader class."""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.video_processing.video_reader import VideoReader


class TestVideoReader(unittest.TestCase):

    def setUp(self):
        """Create a temporary video file for testing."""

        self.temp_directory = tempfile.TemporaryDirectory()

        self.video_path = (
            Path(self.temp_directory.name) / "sample.mp4"
        )

        self.video_path.touch()

    def tearDown(self):
        """Remove temporary test files."""

        self.temp_directory.cleanup()

    def create_mock_capture(self, opened=True):
        """Create a mocked OpenCV video capture."""

        capture = MagicMock()

        capture.isOpened.return_value = opened

        metadata = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FRAME_COUNT: 300.0,
        }

        capture.get.side_effect = lambda prop: metadata[prop]

        return capture

    def test_video_opens_successfully(self):

        capture = self.create_mock_capture()

        with patch(
            "src.video_processing.video_reader.cv2.VideoCapture",
            return_value=capture
        ):
            reader = VideoReader(self.video_path)

            self.assertTrue(reader.is_opened())

            reader.release()

    def test_video_metadata(self):

        capture = self.create_mock_capture()

        with patch(
            "src.video_processing.video_reader.cv2.VideoCapture",
            return_value=capture
        ):
            reader = VideoReader(self.video_path)

            metadata = reader.get_metadata()

            self.assertEqual(metadata["fps"], 30.0)
            self.assertEqual(metadata["width"], 640)
            self.assertEqual(metadata["height"], 480)
            self.assertEqual(metadata["frame_count"], 300)
            self.assertEqual(metadata["duration_seconds"], 10.0)

            reader.release()

    def test_read_frame_successfully(self):

        capture = self.create_mock_capture()

        expected_frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        capture.read.return_value = (True, expected_frame)

        with patch(
            "src.video_processing.video_reader.cv2.VideoCapture",
            return_value=capture
        ):
            reader = VideoReader(self.video_path)

            frame = reader.read_frame()

            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))

            reader.release()

    def test_read_frame_returns_none_at_end(self):

        capture = self.create_mock_capture()

        capture.read.return_value = (False, None)

        with patch(
            "src.video_processing.video_reader.cv2.VideoCapture",
            return_value=capture
        ):
            reader = VideoReader(self.video_path)

            frame = reader.read_frame()

            self.assertIsNone(frame)

            reader.release()

    def test_missing_video_file(self):

        missing_path = (
            Path(self.temp_directory.name) / "missing.mp4"
        )

        with self.assertRaises(FileNotFoundError):
            VideoReader(missing_path)

    def test_unsupported_video_extension(self):

        unsupported_path = (
            Path(self.temp_directory.name) / "sample.txt"
        )

        unsupported_path.touch()

        with self.assertRaises(ValueError):
            VideoReader(unsupported_path)

    def test_video_open_failure(self):

        capture = self.create_mock_capture(opened=False)

        with patch(
            "src.video_processing.video_reader.cv2.VideoCapture",
            return_value=capture
        ):
            with self.assertRaises(RuntimeError):
                VideoReader(self.video_path)

            capture.release.assert_called_once()

    def test_context_manager_releases_video(self):

        capture = self.create_mock_capture()

        with patch(
            "src.video_processing.video_reader.cv2.VideoCapture",
            return_value=capture
        ):
            with VideoReader(self.video_path) as reader:
                self.assertTrue(reader.is_opened())

            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

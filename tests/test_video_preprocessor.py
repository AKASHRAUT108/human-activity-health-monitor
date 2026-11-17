
"""Unit tests for VideoPreprocessor."""

import unittest

import numpy as np

from src.video_processing.video_preprocessor import VideoPreprocessor


class MockVideoReader:
    """Simple fake video reader for testing."""

    def __init__(self, frames):
        self.frames = iter(frames)

    def read_frame(self):
        return next(self.frames, None)


class TestVideoPreprocessor(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        self.preprocessor = VideoPreprocessor()

    def test_default_configuration(self):
        self.assertEqual(
            self.preprocessor.target_size,
            (224, 224),
        )
        self.assertEqual(
            self.preprocessor.color_format,
            "RGB",
        )
        self.assertTrue(self.preprocessor.normalize)

    def test_frame_resize(self):
        result = self.preprocessor.preprocess_frame(self.frame)

        self.assertEqual(result.shape, (224, 224, 3))

    def test_frame_normalization(self):
        frame = np.full(
            (100, 100, 3),
            255,
            dtype=np.uint8,
        )

        result = self.preprocessor.preprocess_frame(frame)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, 1.0)

    def test_rgb_color_conversion(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :] = [10, 20, 30]

        result = self.preprocessor.preprocess_frame(frame)

        expected = np.array([30, 20, 10]) / 255.0

        np.testing.assert_allclose(
            result[0, 0],
            expected,
        )

    def test_bgr_color_format(self):
        preprocessor = VideoPreprocessor(
            color_format="BGR",
            normalize=False,
        )

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :] = [10, 20, 30]

        result = preprocessor.preprocess_frame(frame)

        np.testing.assert_array_equal(
            result[0, 0],
            [10, 20, 30],
        )

    def test_normalization_can_be_disabled(self):
        preprocessor = VideoPreprocessor(normalize=False)

        result = preprocessor.preprocess_frame(self.frame)

        self.assertEqual(result.dtype, np.uint8)

    def test_none_frame_raises_error(self):
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_frame(None)

    def test_invalid_frame_dimensions(self):
        invalid_frame = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_frame(invalid_frame)

    def test_invalid_target_size(self):
        with self.assertRaises(ValueError):
            VideoPreprocessor(target_size=(0, 224))

    def test_invalid_color_format(self):
        with self.assertRaises(ValueError):
            VideoPreprocessor(color_format="XYZ")

    def test_process_video(self):
        frames = [
            self.frame,
            self.frame,
            self.frame,
        ]

        reader = MockVideoReader(frames)

        results = list(
            self.preprocessor.process_video(reader)
        )

        self.assertEqual(len(results), 3)

        for result in results:
            self.assertEqual(result.shape, (224, 224, 3))

    def test_process_video_max_frames(self):
        frames = [
            self.frame,
            self.frame,
            self.frame,
        ]

        reader = MockVideoReader(frames)

        results = list(
            self.preprocessor.process_video(
                reader,
                max_frames=2,
            )
        )

        self.assertEqual(len(results), 2)

    def test_invalid_max_frames(self):
        reader = MockVideoReader([self.frame])

        with self.assertRaises(ValueError):
            list(
                self.preprocessor.process_video(
                    reader,
                    max_frames=0,
                )
            )


if __name__ == "__main__":
    unittest.main()


"""Video frame preprocessing for machine learning."""

from typing import Iterator, Optional

import cv2
import numpy as np


class VideoPreprocessor:
    """Resize, convert, and normalize video frames."""

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        color_format: str = "RGB",
        normalize: bool = True,
    ) -> None:

        if (
            not isinstance(target_size, tuple)
            or len(target_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                or value <= 0
                for value in target_size
            )
        ):
            raise ValueError(
                "target_size must contain two positive integers."
            )

        if color_format not in ("RGB", "BGR"):
            raise ValueError(
                "color_format must be either 'RGB' or 'BGR'."
            )

        self.target_size = target_size
        self.color_format = color_format
        self.normalize = normalize

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess one OpenCV video frame."""

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Input frame must have three color channels."
            )

        if frame.size == 0:
            raise ValueError("Input frame cannot be empty.")

        # Resize the frame.
        processed = cv2.resize(
            frame,
            self.target_size,
            interpolation=cv2.INTER_AREA,
        )

        # OpenCV uses BGR by default.
        if self.color_format == "RGB":
            processed = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2RGB,
            )

        # Normalize pixel values to the range [0, 1].
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        return np.ascontiguousarray(processed)

    def process_video(
        self,
        video_reader,
        max_frames: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """
        Process a video one frame at a time.

        The caller remains responsible for releasing video_reader.
        """

        if max_frames is not None:
            if (
                not isinstance(max_frames, int)
                or isinstance(max_frames, bool)
                or max_frames <= 0
            ):
                raise ValueError(
                    "max_frames must be a positive integer or None."
                )

        processed_count = 0

        while max_frames is None or processed_count < max_frames:

            frame = video_reader.read_frame()

            if frame is None:
                break

            yield self.preprocess_frame(frame)

            processed_count += 1

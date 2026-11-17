
"""Video input module for reading video files using OpenCV."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import SUPPORTED_VIDEO_EXTENSIONS


class VideoReader:
    """Open, inspect, read, and release video files."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)

        # 1. Validate file existence.
        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}"
            )

        # 2. Validate the file extension.
        if self.video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {self.video_path.suffix}"
            )

        # 3. Open the video.
        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f"Unable to open video: {self.video_path}"
            )

        # 4. Extract video metadata.
        self.fps = float(
            self.capture.get(cv2.CAP_PROP_FPS)
        )

        self.width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.frame_count = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def is_opened(self) -> bool:
        """Check whether the video is currently open."""
        return self.capture.isOpened()

    def get_metadata(self) -> dict:
        """Return the basic metadata of the video."""

        duration = (
            self.frame_count / self.fps
            if self.fps > 0
            else 0.0
        )

        return {
            "video_path": str(self.video_path),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_seconds": duration,
        }

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the video.

        Returns:
            A NumPy array containing the frame,
            or None when no frame can be read.
        """

        if not self.is_opened():
            return None

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self) -> None:
        """Release the video capture resource."""

        if self.capture is not None:
            self.capture.release()

    def __enter__(self):
        """Support the Python with statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Automatically release the video resource."""
        self.release()

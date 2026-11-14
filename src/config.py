"""Central configuration for the Human Activity Health Monitor project."""

from pathlib import Path

# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Project metadata
PROJECT_NAME = "Human Activity Health Monitor"
PROJECT_VERSION = "0.1.0"

# Reproducibility
RANDOM_SEED = 42

# Supported video formats
SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
)

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

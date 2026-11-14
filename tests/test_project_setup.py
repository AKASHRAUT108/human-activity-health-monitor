import unittest

from src.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    DOCS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
)


class TestProjectSetup(unittest.TestCase):

    def test_project_root_exists(self):
        self.assertTrue(PROJECT_ROOT.exists())

    def test_requirements_file_exists(self):
        self.assertTrue((PROJECT_ROOT / "requirements.txt").is_file())

    def test_required_directories_exist(self):
        for directory in [
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            MODELS_DIR,
            DOCS_DIR,
        ]:
            with self.subTest(directory=directory):
                self.assertTrue(directory.is_dir())

    def test_video_extensions_are_defined(self):
        self.assertIn(".mp4", SUPPORTED_VIDEO_EXTENSIONS)
        self.assertIn(".avi", SUPPORTED_VIDEO_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()

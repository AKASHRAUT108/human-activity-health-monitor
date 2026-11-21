
"""Extract pose-based features from the prepared KTH sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the project root to Python's import search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.video_feature_pipeline import (
    VideoSequenceFeatureExtractor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 283-dimensional pose features from KTH videos."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "kth"
        / "kth_sequences.jsonl",
        help="Path to the KTH sequence JSONL manifest.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "kth"
        / "pose_features",
        help="Directory where extracted features will be saved.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process. Default: all.",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run pose estimation on every Nth frame. Default: 1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace feature files that already exist.",
    )

    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.5
    )
    parser.add_argument(
        "--min-presence-confidence", type=float, default=0.5
    )
    parser.add_argument(
        "--min-tracking-confidence", type=float, default=0.5
    )

    args = parser.parse_args()
    for name in (
        "min_detection_confidence",
        "min_presence_confidence",
        "min_tracking_confidence",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace(chr(95), chr(45))} must be between 0 and 1.")

    if not args.manifest.is_absolute():
        args.manifest = PROJECT_ROOT / args.manifest

    if not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output

    with VideoSequenceFeatureExtractor(
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as extractor:
        report = extractor.extract_manifest(
            manifest_path=args.manifest,
            output_dir=args.output,
            max_samples=args.max_samples,
            frame_stride=args.frame_stride,
            overwrite=args.overwrite,
        )

    print("\nFeature extraction summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

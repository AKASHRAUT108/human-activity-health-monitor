
"""Command-line entry point for KTH dataset preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_preparation.kth_sequence_preparer import (  # noqa: E402
    KTHDatasetPreparer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the KTH human action dataset."
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Use already extracted videos instead of extracting ZIP archives.",
    )

    args = parser.parse_args()

    preparer = KTHDatasetPreparer(project_root=PROJECT_ROOT)

    try:
        report = preparer.prepare(extract=not args.no_extract)

    except Exception as exc:
        print(f"\nKTH preparation failed: {exc}")
        return 1

    print("\nKTH DATASET PREPARATION COMPLETED")
    print("=" * 45)
    print(json.dumps(report, indent=4))

    print("\nGenerated files:")
    print(f"Manifest: {preparer.manifest_path}")
    print(f"Report:   {preparer.report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Prepare, train, and evaluate the KTH LSTM activity recognition model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RANDOM_SEED
from src.activity_recognition.label_encoder import (
    ActivityLabelEncoder,
    KTH_ACTIVITY_CLASSES,
)
from src.activity_recognition.sequence_builder import (
    FeatureRecording,
    SequenceBuilder,
)
from src.activity_recognition.trainer import ActivityTrainer


FEATURE_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "kth" / "pose_features"
)
MANIFEST_PATH = FEATURE_ROOT / "pose_feature_manifest.jsonl"

SEQUENCE_LENGTH = 30
SEQUENCE_STRIDE = 15
FEATURE_DIM = 283

EXPECTED_SPLITS = {"train", "validation", "test"}


def load_manifest() -> list[dict]:
    """Load and validate the KTH pose feature manifest."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"KTH feature manifest not found: {MANIFEST_PATH}"
        )

    rows = []

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on manifest line {line_number}."
                ) from exc

            rows.append(row)

    if not rows:
        raise ValueError("The KTH feature manifest is empty.")

    return rows


def validate_split_integrity(rows: list[dict]) -> None:
    """Reject missing metadata and subject/video split leakage."""
    subjects_by_split = defaultdict(set)
    videos_by_split = defaultdict(set)
    seen_sample_ids = set()

    for row in rows:
        required_fields = (
            "sample_id",
            "source_video",
            "subject_id",
            "canonical_label",
            "split",
            "feature_file",
        )

        for field in required_fields:
            if field not in row or row[field] is None:
                raise ValueError(
                    f"Manifest record is missing required field {field!r}: "
                    f"{row.get('sample_id', '<unknown>')}"
                )

        sample_id = str(row["sample_id"]).strip()
        subject_id = str(row["subject_id"]).strip()
        source_video = str(row["source_video"]).strip()
        split = str(row["split"]).strip().lower()

        if not sample_id or not subject_id or not source_video:
            raise ValueError(
                "Manifest contains an empty sample, subject, or video ID."
            )

        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")

        seen_sample_ids.add(sample_id)

        if split not in EXPECTED_SPLITS:
            raise ValueError(
                f"Unsupported split {split!r} for sample {sample_id!r}."
            )

        subjects_by_split[split].add(subject_id)
        videos_by_split[split].add(source_video)

    missing_splits = EXPECTED_SPLITS - set(subjects_by_split)
    if missing_splits:
        raise ValueError(
            f"Manifest is missing dataset splits: {sorted(missing_splits)}"
        )

    split_names = sorted(EXPECTED_SPLITS)

    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            subject_overlap = (
                subjects_by_split[first] & subjects_by_split[second]
            )
            video_overlap = (
                videos_by_split[first] & videos_by_split[second]
            )

            if subject_overlap:
                raise ValueError(
                    f"Subject leakage between {first} and {second}: "
                    f"{sorted(subject_overlap)}"
                )

            if video_overlap:
                raise ValueError(
                    f"Video leakage between {first} and {second}: "
                    f"{sorted(video_overlap)}"
                )

    print("\nSplit integrity check: PASSED")
    for split in split_names:
        print(
            f"  {split}: "
            f"{len(subjects_by_split[split])} subjects, "
            f"{len(videos_by_split[split])} videos"
        )

    print("  Subject overlap: none")
    print("  Video overlap: none")


def load_recordings(rows: list[dict]) -> list[FeatureRecording]:
    """Load feature arrays and attach their manifest metadata."""
    recordings = []
    label_counts = Counter()
    split_counts = Counter()
    skipped_short = 0

    for index, row in enumerate(rows, start=1):
        feature_path = FEATURE_ROOT / row["feature_file"]

        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Feature file is missing: {feature_path}"
            )

        features = np.load(feature_path, allow_pickle=False)

        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Invalid feature shape for {row['sample_id']}: "
                f"{features.shape}; expected (frames, {FEATURE_DIM})."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite feature values found in {feature_path}"
            )

        label = str(row["canonical_label"]).strip().lower()
        if label not in KTH_ACTIVITY_CLASSES:
            raise ValueError(
                f"Unsupported KTH activity label {label!r} "
                f"in {row['sample_id']}"
            )

        split = str(row["split"]).strip().lower()

        if len(features) < SEQUENCE_LENGTH:
            skipped_short += 1

        recordings.append(
            FeatureRecording(
                features=features,
                label=label,
                sample_id=str(row["sample_id"]),
                subject_id=str(row["subject_id"]),
                site_id=row.get("site_id"),
                split=split,
            )
        )

        label_counts[label] += 1
        split_counts[split] += 1

        if index % 250 == 0:
            print(f"Loaded {index}/{len(rows)} feature recordings...")

    print("\nFeature recordings loaded:", len(recordings))
    print("Recording counts by split:", dict(split_counts))
    print("Recording counts by class:", dict(label_counts))
    print("Recordings shorter than 30 frames:", skipped_short)

    return recordings


def encode_split(
    labels: np.ndarray,
    encoder: ActivityLabelEncoder,
    split_name: str,
) -> np.ndarray:
    """Encode labels and verify that every KTH class is represented."""
    present = set(str(label) for label in labels)
    expected = set(encoder.classes)

    missing = expected - present
    if missing:
        raise ValueError(
            f"The {split_name} split is missing activity classes: "
            f"{sorted(missing)}"
        )

    return encoder.transform(labels.tolist())


def main() -> None:
    global FEATURE_ROOT, MANIFEST_PATH
    parser = argparse.ArgumentParser(
        description="Train and evaluate the KTH LSTM activity model."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing KTH model.",
    )
    parser.add_argument(
        "--feature-root", type=Path, default=FEATURE_ROOT,
        help="Directory containing pose features and their manifest.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODELS_DIR / "kth",
        help="Directory for model and evaluation outputs.",
    )
    args = parser.parse_args()

    FEATURE_ROOT = args.feature_root
    if not FEATURE_ROOT.is_absolute():
        FEATURE_ROOT = PROJECT_ROOT / FEATURE_ROOT
    FEATURE_ROOT = FEATURE_ROOT.resolve()
    MANIFEST_PATH = FEATURE_ROOT / "pose_feature_manifest.jsonl"

    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if args.patience <= 0:
        parser.error("--patience must be greater than zero.")

    output_dir = args.output_dir
    expected_outputs = (
        output_dir / "activity_lstm.keras",
        output_dir / "activity_lstm_metadata.json",
        output_dir / "kth_test_evaluation.json",
    )

    existing_outputs = [
        path for path in expected_outputs if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "KTH model output already exists:\n"
            + "\n".join(str(path) for path in existing_outputs)
            + "\nUse --overwrite only if you intend to replace it."
        )

    print("=" * 65)
    print("KTH HUMAN ACTIVITY RECOGNITION - LSTM TRAINING")
    print("=" * 65)

    rows = load_manifest()
    print("Manifest records:", len(rows))

    validate_split_integrity(rows)

    recordings = load_recordings(rows)

    builder = SequenceBuilder(
        sequence_length=SEQUENCE_LENGTH,
        stride=SEQUENCE_STRIDE,
        feature_dim=FEATURE_DIM,
    )

    print("\nBuilding fixed-length sequences...")
    dataset = builder.build(recordings)

    print("Total sequences:", len(dataset))
    print("Sequence tensor shape:", dataset.X.shape)

    encoder = ActivityLabelEncoder(KTH_ACTIVITY_CLASSES)

    X_train, y_train_text = dataset.select_split("train")
    X_val, y_val_text = dataset.select_split("validation")
    X_test, y_test_text = dataset.select_split("test")

    y_train = encode_split(y_train_text, encoder, "train")
    y_val = encode_split(y_val_text, encoder, "validation")
    y_test = encode_split(y_test_text, encoder, "test")

    print("\nSequence counts:")
    print("  Training:", len(X_train))
    print("  Validation:", len(X_val))
    print("  Test:", len(X_test))

    print("\nClass mapping:")
    for index, class_name in enumerate(encoder.classes):
        print(f"  {index}: {class_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = ActivityTrainer(
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        seed=RANDOM_SEED,
    )

    print("\nStarting model training...")
    print("The test split will not be used during training.")

    result = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_names=encoder.classes,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=1,
    )

    history_path = output_dir / "kth_training_history.json"
    history_path.write_text(
        json.dumps(result.history, indent=4),
        encoding="utf-8",
    )

    print("\nTraining finished.")
    print("Model saved:", result.model_path)
    print("Model metadata:", result.metadata_path)
    print("Training history:", history_path)

    # The held-out test split is evaluated only after training is complete.
    import tensorflow as tf

    print("\nEvaluating on the held-out test split...")
    model = tf.keras.models.load_model(result.model_path)

    probabilities = model.predict(
        X_test,
        batch_size=args.batch_size,
        verbose=1,
    )

    predictions = np.argmax(probabilities, axis=1)

    accuracy = float(accuracy_score(y_test, predictions))

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(encoder.num_classes)),
        target_names=list(encoder.classes),
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(encoder.num_classes)),
    )

    evaluation = {
        "dataset": "KTH",
        "model_type": "LSTM",
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
        "feature_dimension": FEATURE_DIM,
        "class_names": list(encoder.classes),
        "class_mapping": {
            name: index
            for index, name in enumerate(encoder.classes)
        },
        "test_sequence_count": int(len(X_test)),
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(encoder.classes),
        "training_subjects": sorted({
            subject
            for subject, split in zip(
                dataset.subject_ids, dataset.splits
            )
            if split == "train"
        }),
        "validation_subjects": sorted({
            subject
            for subject, split in zip(
                dataset.subject_ids, dataset.splits
            )
            if split == "validation"
        }),
        "test_subjects": sorted({
            subject
            for subject, split in zip(
                dataset.subject_ids, dataset.splits
            )
            if split == "test"
        }),
    }

    evaluation_path = output_dir / "kth_test_evaluation.json"
    evaluation_path.write_text(
        json.dumps(evaluation, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("KTH TEST RESULTS")
    print("=" * 65)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nPer-class metrics:")

    for class_name in encoder.classes:
        metrics = report[class_name]
        print(
            f"  {class_name:15s} "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1-score']:.4f}"
        )

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Classes:", list(encoder.classes))
    print(matrix)

    print("\nEvaluation saved:", evaluation_path)
    print("\nKTH training and evaluation completed successfully.")


if __name__ == "__main__":
    main()

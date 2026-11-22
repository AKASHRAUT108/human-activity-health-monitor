# Stage 8: Real-Video Pose Feature Extraction

## Purpose

This stage extracts pose-based numerical features from annotated video sequences. It connects the KTH sequence manifest with the pose estimation and feature engineering pipelines.

## Input

Default input manifest:

`data/processed/kth/kth_sequences.jsonl`

Each sequence record preserves:

- Sample ID and dataset name.
- Source video and relative video path.
- Subject ID and site ID.
- Original and canonical activity labels.
- Official dataset split.
- Sequence index.
- 1-based inclusive start and end frame numbers.
- Source type.

## Processing Pipeline

For every sequence:

1. Open the source video using OpenCV.
2. Seek to the annotated starting frame.
3. Read the annotated frame range.
4. Run MediaPipe Pose Landmarker on sampled frames.
5. Extract 283 numerical features per frame.
6. Save the feature sequence as a NumPy array.
7. Record sample metadata and extraction diagnostics.

The feature extractor combines body-centered pose coordinates, landmark confidence, joint angles, body distances, pose presence, and temporal velocity.

A missing pose is represented using the existing zero-feature behavior.

## Output Structure

Default output directory:

`data/processed/kth/pose_features/`

Expected structure:

    pose_features/
        sequences/
            sample_sequence_001.npy
            sample_sequence_002.npy
            ...
        pose_feature_manifest.jsonl
        pose_feature_report.json

Each NumPy file contains a two-dimensional feature array:

`(number_of_sampled_frames, 283)`

The JSONL manifest records the relationship between extracted feature files and their original video sequences.

The JSON report summarizes requested, successful, failed, and reused samples.

## Processing Results

Recorded KTH processing results:

- Requested samples: 2,390.
- Successfully processed samples: 2,385.
- Failed samples: 0.
- Existing samples reused: 5.
- Features per frame: 283.
- Frame stride: 1.

## Data Integrity

The extraction pipeline preserves source sample identifiers, activity labels, subject identifiers, and dataset split metadata.

Training, validation, and testing assignments must remain consistent with the original sequence manifest.

Existing feature files can be reused to avoid unnecessary repeated extraction.

## Main Implementation

The feature extraction implementation uses the video sequence feature extraction pipeline and the existing pose estimation and feature extraction modules.

Main script:

`scripts/extract_kth_pose_features.py`

## Limitations

- Pose estimation can fail because of occlusion, camera angle, lighting, or poor video quality.
- Pose landmarks are estimates, not exact anatomical measurements.
- Missing poses can reduce the quality of downstream predictions.
- Extracted movement features are not medical measurements or diagnoses.

## Future Improvements

- Extend extraction to additional public activity datasets.
- Preserve reliable recording-site metadata when available.
- Evaluate feature quality across different cameras and environments.
- Improve handling of missing and low-confidence poses.
- Support larger-scale multi-dataset feature extraction.

## Conclusion

This stage converts annotated video sequences into numerical pose-feature sequences suitable for temporal activity recognition. It preserves sample metadata and dataset splits for reproducible model training and evaluation.

# Stage 8: Real-Video Pose Feature Extraction

## Purpose

This stage extracts pose-based numerical features from annotated video
sequences. It builds on the existing video reader, pose estimator,
feature extractor, and KTH sequence manifest.

## Input

Default input:

`data/processed/kth/kth_sequences.jsonl`

Each sequence record preserves:

- sample ID
- source video and video path
- subject ID
- site ID
- original activity label
- canonical activity label
- official dataset split
- sequence index
- 1-based inclusive start and end frame numbers
- source type

## Processing

For every sequence:

1. Open the source video.
2. Seek to `frame_start - 1` using OpenCV.
3. Read through `frame_end`, inclusive.
4. Run MediaPipe pose estimation on sampled frames.
5. Use the existing `PoseFeatureExtractor` to produce 283 features
   per sampled frame.
6. Save the resulting float32 NumPy array.
7. Record sequence metadata and extraction diagnostics.

A missing pose is represented using the existing feature extractor's
zero-feature behavior.

## Output

Default output directory:

`data/processed/kth/pose_features/`

Structure:

```text
pose_features/
    sequences/
        person01_boxing_d1_seq_001.npy
        person01_boxing_d1_seq_002.npy
        ...
    pose_feature_manifest.jsonl
    pose_feature_report.json
# KTH Dataset Preparation

## Purpose
The KTH Human Action Dataset is used to train and evaluate the activity recognition model. It contains six activities: boxing, handclapping, handwaving, jogging, running, and walking.

## Dataset Structure
Input:
- data/raw/kth/downloads/ - six activity ZIP archives.
- data/raw/kth/annotations/00sequences.txt - official frame annotations.
- data/raw/kth/videos/ - extracted AVI recordings.

Output:
- data/processed/kth/kth_sequences.jsonl
- data/processed/kth/kth_preparation_report.json

## Annotation Processing
The KTHAnnotationParser reads the official 00sequences.txt file. It identifies the subject, activity, recording condition, and annotated frame ranges. It rejects malformed and duplicate annotations and preserves explicitly missing entries.

Each annotated action repetition becomes an independent sequence-level manifest record. Frame ranges use 1-based inclusive numbering.

## Video Extraction
KTHDatasetPreparer extracts videos from six activity ZIP archives. It checks archive integrity, expected filenames, activity-to-archive matching, and unsafe archive paths. Existing complete video files are reused.

## Subject-Based Dataset Splitting
The implementation uses subject-independent splits:

- Training: person11 to person18.
- Validation: person19, person20, person21, person23, person24, person25, person01, person04.
- Testing: person22, person02, person03, person05, person06, person07, person08, person09, person10.

This prevents the same subject from appearing in multiple splits.

## Sequence Manifest
Each JSONL record contains:
- Sample ID and dataset name.
- Source video and relative video path.
- Subject ID and site ID.
- Recording condition.
- Original and canonical activity labels.
- Dataset split and sequence index.
- Starting and ending frame numbers.
- Source type.

The KTH site_id is null because site information is not available in these records.

## Video Validation
OpenCV checks whether videos can be opened, whether dimensions are valid, and whether frames can be decoded. Each annotated frame range is checked against the decoded video length.

Sequences whose complete annotated ranges cannot be decoded are excluded. Unusable source videos cause validation to fail.

## Preparation Report
The generated kth_preparation_report.json records annotation counts, matched videos, missing entries, sequence counts, activity counts, split counts, extraction statistics, validation results, and excluded sequences.

The implementation checks the expected missing annotation: person13_handclapping_d3.

## Main Implementation
Source file: src/dataset_preparation/kth_sequence_preparer.py

Main components:
- KTHAnnotationParser: parses official annotations.
- KTHDatasetPreparer: extracts, prepares, validates, and saves samples.
- KTHAnnotation: represents annotations for one source video.
- KTHSequenceSample: represents one annotated action repetition.

## Recorded Processing Results
- Original recordings: 2,390.
- Successfully extracted feature samples: 2,385.
- Existing feature samples reused: 5.
- Generated temporal sequences: 11,733.
- Features per frame: 283.
- Sequence length: 30 frames.
- Sequence stride: 15 frames.

Feature extraction and sequence preparation are different pipeline stages.

## Data Integrity
The pipeline preserves source identifiers, official annotations, subject-based splits, and validation results. Overlapping sequences from one recording must not be distributed across training and testing.

## Limitations
KTH contains only six activity classes and does not provide site identifiers in this implementation. Real-world camera angles, lighting, backgrounds, and recording conditions may affect model performance.

## Future Improvements
- Integrate additional public activity datasets.
- Add data from multiple recording sites.
- Standardize labels across datasets.
- Evaluate cross-dataset generalization.
- Test on independent real-world videos.

## Conclusion
The KTH preparation pipeline converts original recordings into validated sequence-level samples for pose feature extraction and deep learning. It preserves annotations, metadata, and subject-based splits for reliable activity recognition experiments.

# Human Activity Health Monitor - Project Overview

## 1. Introduction

Human Activity Health Monitor is an AI-based project that analyzes human movement from video recordings.

The system uses computer vision and deep learning to recognize physical activities and provide basic movement-related health indicators.

The project is designed to help users understand movement patterns through automated video analysis.

**Important:** This project is for educational and activity-monitoring purposes. It does not diagnose medical conditions or replace professional medical advice.

## 2. Main Objectives

- Recognize human activities from video recordings.
- Extract human body pose landmarks using MediaPipe.
- Convert pose landmarks into numerical features.
- Analyze movement patterns using a deep learning model.
- Display predictions and confidence scores through an interactive dashboard.
- Generate reports for analyzed videos.
- Build a foundation for future multi-dataset and multi-site training.

## 3. Supported Activities

The current KTH-trained activity recognition model supports six classes:

1. Boxing
2. Handclapping
3. Handwaving
4. Jogging
5. Running
6. Walking

## 4. System Workflow

The project follows this processing pipeline:

1. Video Input: The user uploads a video.
2. Video Processing: Frames are read and prepared for analysis.
3. Pose Estimation: MediaPipe detects human body landmarks.
4. Feature Extraction: Landmark coordinates, confidence values, joint angles, body distances, and movement velocities are calculated.
5. Sequence Preparation: Frame-level features are grouped into temporal sequences.
6. Activity Recognition: The trained LSTM model predicts the activity.
7. Health Monitoring: Movement-related indicators are presented as estimates.
8. Dashboard and Reports: Results are displayed and can be included in reports.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video reading and frame processing |
| MediaPipe | Human pose estimation |
| NumPy | Numerical calculations |
| TensorFlow / Keras | Deep learning and LSTM model |
| Scikit-learn | Label encoding and evaluation |
| Streamlit | Interactive web dashboard |
| Plotly | Data visualization |
| Git and GitHub | Version control and project hosting |

## 6. Dataset

The current activity recognition pipeline uses the KTH Human Action Dataset.

Dataset processing results:

- Original recordings: 2,390
- Successfully extracted feature samples: 2,385
- Existing feature samples reused: 5
- Features per frame: 283
- Sequence length: 30 frames
- Sequence stride: 15 frames
- Generated sequences: 11,733

The current trained model is based on KTH data. Training with additional datasets and multiple recording sites remains a future expansion.

## 7. Pose Feature Representation

Each video frame is represented using 283 numerical features:

- 99 normalized landmark coordinates
- 66 landmark confidence values
- 10 joint angles
- 8 body distances
- 1 pose-presence indicator
- 99 temporal velocity features

These features describe body position, pose confidence, body geometry, and movement between frames.

## 8. Activity Recognition Model

The current activity recognition system uses a Long Short-Term Memory (LSTM) neural network.

LSTM is suitable for sequential data because it can learn patterns that develop over multiple video frames.

The model receives sequences with a shape of 30 frames by 283 features and predicts one of six activity classes.

## 9. Model Performance

The current KTH test results are:

| Model | Test Accuracy |
|---|---:|
| Baseline LSTM | 84.85% |
| Tuned LSTM | 86.11% |

Performance differs between activity classes. In particular, running recognition remains a limitation and requires further improvement.

These results describe evaluation on the available KTH test split. They do not guarantee the same performance on every real-world video.

## 10. Application Features

The deployed Streamlit dashboard includes:

- Overview: Project introduction and system information.
- Video Analysis: Upload and analyze a video.
- Model Performance: Review model evaluation results.
- Reports & About: Review project information and generate reports.

Live application:

https://human-activity-health-monitor-4hlskw3fmysc2yywqjqpbe.streamlit.app/

GitHub repository:

https://github.com/AKASHRAUT108/human-activity-health-monitor

## 11. Current Limitations

- The current trained activity model uses the KTH dataset.
- Recognition performance may change with different camera angles, lighting, backgrounds, and recording conditions.
- Some activities are harder to distinguish than others.
- Running recognition needs further improvement.
- Health-related indicators are estimates derived from visible movement, not clinical measurements.
- Additional datasets and recording sites are needed to improve generalization.

## 12. Future Improvements

- Train using all three planned categories of data sources.
- Add data from multiple recording sites and environments.
- Improve recognition of activities with lower evaluation scores.
- Expand movement-related health indicators.
- Improve robustness across camera positions and recording conditions.
- Add further evaluation using independent real-world videos.
- Improve reporting and visualization features.

## 13. Conclusion

Human Activity Health Monitor combines computer vision, pose estimation, feature engineering, and deep learning to recognize human activities from video.

The current implementation includes a complete KTH-based activity recognition pipeline and a deployed interactive dashboard. The project provides a foundation for further development using additional datasets, recording sites, and improved movement analysis.

# Human Activity Health Monitor

**Live Demo:** [Open Human Activity Health Monitor](https://human-activity-health-monitor-4hlskw3fmysc2yywqjqpbe.streamlit.app/)


An AI-powered human activity recognition system that analyzes videos, identifies human movements, and provides movement-related insights using computer vision and deep learning.

## 1. About the Project

**Human Activity Health Monitor - Recognizing human movement through computer vision and deep learning.**

The system analyzes body posture and movement over time. It uses pose estimation to identify important body landmarks and an LSTM deep learning model to classify the activity being performed.

The project also presents the analysis through an interactive web dashboard, making the results easier to understand.

The main goal is to build a system that can recognize common human activities and provide basic movement-related information from video.

> Important: This project provides AI-based movement estimates. It is not a medical diagnostic system and must not be used as a replacement for professional medical advice.

## 2. Main Features

- **Video-based activity recognition:** Upload a video and analyze the movement shown in it.
- **Human pose estimation:** Detect body landmarks using MediaPipe Pose.
- **Movement feature extraction:** Extract body coordinates, confidence values, joint angles, distances, and movement information.
- **Deep learning classification:** Use an LSTM model to recognize activities across a sequence of video frames.
- **Activity confidence:** Display the model's estimated confidence for its predictions.
- **Interactive dashboard:** Explore video analysis, model performance, and reports.
- **PDF reports:** Generate a report containing video analysis results.
- **Model evaluation:** Review classification metrics and model performance.

## 3. Activities Recognized

The current model is trained to recognize six activity classes from the KTH human action dataset.

| Activity | Description |
|---|---|
| Boxing | Repeated boxing movements |
| Handclapping | Clapping hands |
| Handwaving | Waving a hand |
| Jogging | Jogging movement |
| Running | Running movement |
| Walking | Walking movement |

The model recognizes patterns from these classes. It does not identify every possible human activity.

## 4. How the System Works

The project follows a video-to-prediction pipeline.

1. **Video input:** The user uploads a video through the dashboard.
2. **Video processing:** OpenCV reads the video and processes its frames.
3. **Pose estimation:** MediaPipe detects human body landmarks in each frame.
4. **Feature extraction:** The system converts the detected pose into numerical features.
5. **Sequence preparation:** Consecutive frame features are grouped into time-based sequences.
6. **Activity classification:** The trained LSTM model analyzes each sequence and predicts an activity.
7. **Result generation:** Predictions are summarized and displayed in the dashboard.

### System workflow

```text
       Input Video
            |
            v
     Video Processing
            |
            v
      Pose Estimation
       (MediaPipe)
            |
            v
     Feature Extraction
            |
            v
    Sequence Preparation
            |
            v
       LSTM Model
            |
            v
    Activity Prediction
            |
            v
   Dashboard and Reports
```

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video reading and frame processing |
| MediaPipe | Human pose estimation |
| NumPy | Numerical operations |
| TensorFlow / Keras | Deep learning model development |
| LSTM | Recognition of movement patterns over time |
| Streamlit | Interactive web application |
| Plotly | Interactive charts |
| Pandas | Data handling |
| ReportLab | PDF report generation |
| FastAPI | API backend |
| Git and GitHub | Version control and project hosting |

## 6. Dataset

The current trained activity recognition model uses the **KTH Human Action Dataset**.

The dataset contains videos of people performing six activities:

- Boxing
- Handclapping
- Handwaving
- Jogging
- Running
- Walking

### Dataset preparation

The project includes a dataset preparation and feature extraction pipeline.

The KTH processing work completed so far includes:

| Item | Result |
|---|---:|
| Video recordings processed | 2,390 |
| Successfully extracted pose feature samples | 2,385 |
| Feature samples already available and reused | 5 |
| Features per frame | 283 |
| Frames per sequence | 30 |
| Sequence stride | 15 frames |
| Generated sequences | 11,733 |

The sequence preparation process creates overlapping windows from each recording. Dataset splits are kept separate to reduce the risk of the same recording appearing in both training and testing.

**Dataset note:** The current reported model results are based on KTH. They should not be interpreted as evidence that the model has already been trained on multiple independent datasets or recording sites.

## 7. Feature Extraction

The pose feature extractor produces **283 numerical features per frame**.

These features include:

| Feature group | Number of features |
|---|---:|
| Normalized landmark coordinates | 99 |
| Landmark confidence values | 66 |
| Joint angles | 10 |
| Body distances | 8 |
| Pose-presence indicator | 1 |
| Temporal velocity features | 99 |
| **Total** | **283** |

These values describe body position and movement. The LSTM model uses sequences of these features rather than analyzing each frame independently.

## 8. Deep Learning Model

The activity recognition system uses a Long Short-Term Memory (LSTM) neural network.

LSTM is suitable for sequential data because it can learn patterns that develop over time.

### Model architecture

The baseline model contains:

1. Input layer
2. Masking layer
3. LSTM layer with 64 units
4. Dropout layer with a rate of 0.3
5. Dense output layer with softmax activation

The model uses:

- Adam optimizer
- Sparse categorical cross-entropy loss
- Accuracy as a training metric
- Early stopping based on validation loss

### Training configuration

| Parameter | Value |
|---|---:|
| Input sequence length | 30 frames |
| Features per frame | 283 |
| Input shape | 30 Ã— 283 |
| Number of activity classes | 6 |
| LSTM units in baseline architecture | 64 |
| Baseline dropout | 0.3 |

The trained model and its metadata are stored separately so that the prediction pipeline can load the model with the correct input dimensions and class order.

## 9. Model Performance

The tuned LSTM model achieved **86.11% test accuracy** on the held-out KTH test data.

The earlier baseline model achieved **84.85% test accuracy**.

| Model | Test accuracy |
|---|---:|
| Baseline LSTM | 84.85% |
| Tuned LSTM | 86.11% |

### Tuned model: class-wise F1-score

| Activity | F1-score |
|---|---:|
| Boxing | 0.9478 |
| Handclapping | 0.9601 |
| Handwaving | 0.9589 |
| Jogging | 0.6409 |
| Running | 0.3261 |
| Walking | 0.8184 |

### Understanding these results

Overall accuracy does not tell the whole story.

The model performs well on boxing, handclapping, and handwaving. However, running has a considerably lower F1-score, which indicates that the model has difficulty distinguishing this activity reliably.

The model should therefore be evaluated further before being used in real-world applications.

These results describe performance on the prepared KTH test set. They do not guarantee similar performance on videos recorded in different environments.

## 10. Web Application

The project includes an interactive Streamlit dashboard.

The dashboard provides the following sections:

| Dashboard section | Purpose |
|---|---|
| Overview | Introduction and project summary |
| Video Analysis | Upload a video and view activity predictions |
| Model Performance | Explore model evaluation results |
| Reports & About | Generate reports and view project information |

### Try the deployed application

[Open Human Activity Health Monitor](https://human-activity-health-monitor-4hlskw3fmysc2yywqjqpbe.streamlit.app/)

## 11. Installation and Local Setup

Follow these steps to run the project on your computer.

### Step 1: Clone the repository

```powershell
git clone https://github.com/AKASHRAUT108/human-activity-health-monitor.git
```

### Step 2: Open the project directory

```powershell
cd human-activity-health-monitor
```

### Step 3: Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

### Step 4: Activate the virtual environment

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

### Step 5: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```powershell
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

### Required model files

The application needs the trained activity model and the MediaPipe pose model.

Expected paths:

```text
models/
â”œâ”€â”€ pose_landmarker_lite.task
â””â”€â”€ kth_tuned_conf03/
    â”œâ”€â”€ activity_lstm.keras
    â””â”€â”€ activity_lstm_metadata.json
```

Make sure these files are available before running video inference.

## 12. Project Structure

The repository is organized into modules so that different parts of the system can be maintained independently.

```text
human-activity-health-monitor/
â”‚
â”œâ”€â”€ api/
â”‚   â”œâ”€â”€ main.py
â”‚   â””â”€â”€ inference_service.py
â”‚
â”œâ”€â”€ app.py
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py
â”‚   â”œâ”€â”€ dataset_preparation/
â”‚   â”œâ”€â”€ feature_extraction/
â”‚   â”œâ”€â”€ pose_estimation/
â”‚   â”œâ”€â”€ video_processing/
â”‚   â””â”€â”€ activity_recognition/
â”‚
â”œâ”€â”€ scripts/
â”‚
â”œâ”€â”€ tests/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ pose_landmarker_lite.task
â”‚   â””â”€â”€ kth_tuned_conf03/
â”‚
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ PROJECT_OVERVIEW.md
â”‚   â”œâ”€â”€ feature_engineering.md
â”‚   â”œâ”€â”€ kth_dataset_preparation.md
â”‚   â”œâ”€â”€ lstm_activity_recognition.md
â”‚   â””â”€â”€ video_feature_extraction.md
â”‚
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ packages.txt
â””â”€â”€ README.md
```

Some folders contain additional modules and files that are not shown in this simplified view.

## 13. Testing

The project includes automated unit tests for important components.

Run the test suite using:

```powershell
python -m unittest discover -s tests -v
```

The tests help verify individual components such as data preparation, pose processing, feature extraction, sequence creation, and activity recognition.

Passing unit tests verifies software behavior under the tested conditions. It does not prove that the model will be accurate on every real-world video.

## 14. Current Limitations

The current system has several limitations:

- It recognizes only the six activity classes included in the current model.
- Prediction quality depends on video quality and pose detection.
- Different camera angles, lighting conditions, clothing, and body occlusion may affect results.
- Similar movements can be difficult for the model to distinguish.
- Running currently has lower classification performance than the other activity classes.
- The current reported evaluation uses KTH data and does not establish generalization to multiple independent datasets or sites.
- The system does not diagnose diseases or measure a person's medical health condition.

## 15. Future Improvements

Possible future improvements include:

- Training and evaluating on additional human activity datasets.
- Using data from multiple recording sites and environments.
- Improving recognition of visually similar activities such as jogging and running.
- Evaluating performance on videos recorded outside the training datasets.
- Improving robustness to camera angle, lighting, and partial body visibility.
- Adding more movement-related analysis and health indicators with appropriate validation.
- Expanding the dashboard and reporting features.

## 16. Documentation

More detailed technical documentation is available in the `docs/` directory:

- `PROJECT_OVERVIEW.md` â€” Overall project description and system design.
- `kth_dataset_preparation.md` â€” KTH dataset preparation.
- `video_feature_extraction.md` â€” Video-based feature extraction.
- `feature_engineering.md` â€” Pose and movement feature details.
- `lstm_activity_recognition.md` â€” LSTM model design and activity recognition.

## 17. Author and Repository

**GitHub Repository:**
https://github.com/AKASHRAUT108/human-activity-health-monitor

---

If you find this project useful, feel free to explore the repository and try the deployed application.

**Human Activity Health Monitor - Recognizing human movement through computer vision and deep learning.**

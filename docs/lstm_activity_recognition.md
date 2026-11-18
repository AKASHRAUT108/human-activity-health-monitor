
# Baseline Activity Recognition using LSTM

## 1. Objective

The baseline activity recognition system classifies human activities
using temporal sequences of pose-based features extracted from video.

The initial activity classes are:

- Boxing
- Handclapping
- Handwaving
- Jogging
- Running
- Walking

These classes correspond to the six activity categories in the KTH
human action dataset.

## 2. Input Features

The Stage 5 feature extractor produces 283 features per frame.

These include:

- 99 normalized landmark coordinates
- 66 landmark confidence values
- 10 joint angles
- 8 body distances
- 1 pose-presence indicator
- 99 temporal velocity features

Each recording is represented as a chronological sequence of
frame-level feature vectors.

## 3. Sequence Preparation

The SequenceBuilder converts each video into overlapping windows.

Default configuration:

| Parameter | Value |
|---|---:|
| Sequence length | 30 frames |
| Sequence stride | 15 frames |
| Features per frame | 283 |
| Sequence shape | 30 x 283 |

Sequences are generated independently for each recording.

The builder preserves subject, site, and split metadata.

## 4. Label Encoding

ActivityLabelEncoder converts activity names into integer indices.

The initial class order is:

| Class index | Activity |
|---:|---|
| 0 | Boxing |
| 1 | Handclapping |
| 2 | Handwaving |
| 3 | Jogging |
| 4 | Running |
| 5 | Walking |

## 5. LSTM Architecture

The baseline network contains:

1. Input layer
2. Masking layer
3. LSTM layer with 64 units
4. Dropout layer with rate 0.3
5. Dense softmax classification layer

The model uses:

- Adam optimizer
- Sparse categorical cross-entropy loss
- Accuracy as a training metric
- Early stopping based on validation loss

## 6. Training

The trainer accepts separate training and validation arrays.

It saves:

- `activity_lstm.keras`
- `activity_lstm_metadata.json`

The metadata stores the class order and input dimensions needed
for later inference.

## 7. Prediction

ActivityPredictor loads the trained model and predicts activities
from one sequence or a batch of sequences.

Each prediction includes:

- Predicted activity
- Confidence
- Class probabilities

Confidence is the model's estimated class probability. It is not
a guarantee that the prediction is correct.

## 8. Dataset and Evaluation Requirements

The current unit tests use controlled test data to verify the
software pipeline. They do not establish real-world accuracy.

Before reporting model performance:

1. Obtain and prepare the real video datasets.
2. Extract pose features from actual videos.
3. Use subject-aware or site-aware splits where reliable metadata exists.
4. Keep training, validation, and test recordings separate.
5. Evaluate on the held-out test set.
6. Report accuracy, precision, recall, F1-score, and a confusion matrix.

Do not use overlapping windows from the same recording across
training and testing.

For datasets without reliable subject identities, preserve official
splits where available and document the limitations.

## 9. Limitations

The model recognizes activity patterns represented in its training data.

It may perform poorly on unseen camera angles, environments,
people, clothing, occlusion, or activities.

The predictions are movement classifications, not medical diagnoses.
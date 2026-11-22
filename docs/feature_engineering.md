
# Stage 5 - Feature Engineering

## Purpose

Convert MediaPipe Pose Landmarker output into numerical features suitable
for machine-learning and deep-learning models.

## Input

One detected person's pose contains:

- 33 image landmarks.
- 33 world landmarks when available.
- XYZ coordinates.
- Visibility and presence confidence values.

## Extracted features

### 1. Body-centered coordinates

- XYZ coordinates for all 33 landmarks.
- Coordinates centered around the hip midpoint when available.
- Coordinates scaled by shoulder width when available.
- World coordinates are preferred.
- Image coordinates are used when world coordinates are unavailable.

### 2. Landmark confidence

- Visibility confidence for each landmark.
- Presence confidence for each landmark.
- Low-confidence landmarks are excluded from geometry calculations.

### 3. Joint angles

- Left and right elbow angles.
- Left and right shoulder angles.
- Left and right hip angles.
- Left and right knee angles.
- Left and right ankle angles.

Angles are calculated in 3D and normalized to the range [0, 1].

### 4. Body distances

- Shoulder width.
- Hip width.
- Left and right arm reach.
- Ankle width.
- Torso length.
- Left and right lower-leg length.

Distances use the normalized body coordinate system.

### 5. Temporal velocity

For a sequence belonging to one tracked person:

- XYZ coordinate changes per second for all 33 landmarks.
- Velocity is calculated using consecutive frames and video FPS.
- The first frame has zero velocity.
- A missing pose produces zero features for that frame.
- Velocity is zero when a landmark is not valid in both consecutive frames.

## Output

`extract_frame(pose)` returns a dictionary containing:

- `feature_vector`: one-dimensional NumPy float32 array.
- `feature_names`: names corresponding to the vector positions.
- `pose_present`: whether a usable pose was detected.

`extract_multiple(poses)` extracts features for all people detected in one frame.

`extract_sequence(pose_sequence, fps)` returns a two-dimensional NumPy float32 array:

`(number_of_frames, number_of_features)`

The sequence method expects poses belonging to one person in chronological order.
Person tracking and persistent person IDs will be implemented separately.

## Important limitations

- Pose landmarks are estimates, not exact anatomical measurements.
- Camera angle, occlusion, clothing, and lighting can affect accuracy.
- Normalized distances are geometric features, not clinical measurements.
- These features do not diagnose medical conditions.
- A missing pose is represented by zero values; downstream models should
  use the pose-present and confidence features to recognize missing data.

## Testing

Run:

`python -m unittest discover -s tests -p "test_feature_extractor.py" -v`

Then run the complete test suite:

`python -m unittest discover -s tests -v`

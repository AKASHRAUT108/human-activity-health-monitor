
"""Build the baseline LSTM activity recognition model."""

from __future__ import annotations

import tensorflow as tf

def build_lstm_model(
    sequence_length: int = 30,
    feature_dim: int = 283,
    num_classes: int = 6,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build and compile a baseline LSTM classifier."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    if feature_dim <= 0:
        raise ValueError("feature_dim must be greater than zero.")

    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")

    if lstm_units <= 0:
        raise ValueError("lstm_units must be greater than zero.")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="pose_feature_sequence",
    )

    x = tf.keras.layers.Masking(
        mask_value=0.0,
        name="missing_frame_masking",
    )(inputs)

    x = tf.keras.layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        name="activity_lstm",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=num_classes,
        activation="softmax",
        name="activity_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="human_activity_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
"""Keras model definitions."""

from __future__ import annotations

import keras
from keras import layers


def build_classifier(n_features: int, n_classes: int, dropout: float = 0.3) -> keras.Model:
    """Feed-forward classifier: two hidden layers with batch norm and dropout."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in (128, 64):
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="intrusion_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_autoencoder(n_features: int, bottleneck: int = 8) -> keras.Model:
    """Symmetric undercomplete autoencoder; the bottleneck forces compression."""
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    code = layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)
    x = layers.Dense(32, activation="relu")(code)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_features, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="anomaly_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def reconstruction_error(model: keras.Model, X, batch_size: int = 2048):
    """Per-row mean squared reconstruction error, used as the anomaly score."""
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return ((X - recon) ** 2).mean(axis=1)

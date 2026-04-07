from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=warnings, 2=errors, 3=fatal


def build_simple_cnn(
    input_shape: Tuple[int, int, int], 
    num_classes: int,
    learning_rate: float = 0.001,
    freeze_backbone: bool = False,
    intensity: str = "medium"
) -> tf.keras.Model:
    """Improved CNN with better feature extraction for distinguishing similar classes.
    
    Enhanced architecture with multiple conv blocks per stage for better feature learning.
    This is especially important for similar classes (e.g., different big cat species).
    
    Args:
        input_shape: (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        freeze_backbone: If True, freeze convolutional layers (for transfer learning)
        intensity: Training intensity level (less/medium/rigorous) to adjust architecture
    """
    # Adjust architecture complexity based on intensity
    # Make clear distinctions: LESS (simple) < MEDIUM (moderate) < RIGOROUS (complex)
    if intensity == "less":
        # Simple architecture - fewer layers, smaller capacity
        filters = [32, 64]  # Only 2 conv blocks
        dense_units = 32  # Smaller dense layer
        dropout_rate = 0.3
    elif intensity == "rigorous":
        # Maximum capacity - deepest network for best performance
        filters = [64, 128, 256, 512, 256, 128]  # 6 conv blocks
        dense_units = 512  # Much larger dense layer
        dropout_rate = 0.4
    else:  # medium
        # Moderate architecture - balanced between less and rigorous
        filters = [32, 64, 128, 256]  # 4 conv blocks
        dense_units = 128  # Medium dense layer
        dropout_rate = 0.35
    
    # Build model (CPU only)
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))
    
    # Build deeper convolutional blocks with multiple conv layers per stage
    # This helps extract more nuanced features for similar classes
    for i, num_filters in enumerate(filters):
        # First conv in block
        model.add(layers.Conv2D(num_filters, 3, padding="same", name=f"conv{i+1}_a"))
        model.add(layers.BatchNormalization(name=f"bn{i+1}_a"))
        model.add(layers.Activation("relu", name=f"relu{i+1}_a"))
        
        # Second conv in block (deeper feature extraction)
        # LESS: Only single conv per block (simpler)
        # MEDIUM/RIGOROUS: Double conv per block (deeper)
        if intensity != "less":
            model.add(layers.Conv2D(num_filters, 3, padding="same", name=f"conv{i+1}_b"))
            model.add(layers.BatchNormalization(name=f"bn{i+1}_b"))
            model.add(layers.Activation("relu", name=f"relu{i+1}_b"))
        
        # RIGOROUS: Add third conv layer for even deeper feature extraction
        if intensity == "rigorous" and i >= 2:  # Only for deeper layers
            model.add(layers.Conv2D(num_filters, 3, padding="same", name=f"conv{i+1}_c"))
            model.add(layers.BatchNormalization(name=f"bn{i+1}_c"))
            model.add(layers.Activation("relu", name=f"relu{i+1}_c"))
        
        model.add(layers.MaxPooling2D(2, name=f"pool{i+1}"))
        # Add dropout after pooling to prevent overfitting
        if i < len(filters) - 1:
            model.add(layers.Dropout(0.15, name=f"dropout_conv{i+1}"))
    
    model.add(layers.GlobalAveragePooling2D(name="global_pool"))
    model.add(layers.Dropout(dropout_rate, name="dropout_fc1"))
    
    # Dense layers: Clear distinction by intensity
    if intensity == "rigorous":
        # RIGOROUS: Multiple dense layers for complex feature learning
        model.add(layers.Dense(dense_units, activation="relu", name="dense1"))
        model.add(layers.BatchNormalization(name="bn_fc1"))
        model.add(layers.Dropout(dropout_rate * 0.6, name="dropout_fc1b"))
        model.add(layers.Dense(dense_units // 2, activation="relu", name="dense2"))
        model.add(layers.BatchNormalization(name="bn_fc2"))
        model.add(layers.Dropout(dropout_rate * 0.5, name="dropout_fc2"))
        model.add(layers.Dense(dense_units // 4, activation="relu", name="dense3"))
        model.add(layers.BatchNormalization(name="bn_fc3"))
        model.add(layers.Dropout(dropout_rate * 0.4, name="dropout_fc3"))
    elif intensity == "medium":
        # MEDIUM: Two dense layers
        model.add(layers.Dense(dense_units, activation="relu", name="dense1"))
        model.add(layers.BatchNormalization(name="bn_fc1"))
        model.add(layers.Dropout(dropout_rate * 0.5, name="dropout_fc1b"))
        model.add(layers.Dense(dense_units // 2, activation="relu", name="dense2"))
        model.add(layers.BatchNormalization(name="bn_fc2"))
        model.add(layers.Dropout(dropout_rate * 0.5, name="dropout_fc2"))
    else:  # less
        # LESS: Single dense layer (simplest)
        model.add(layers.Dense(dense_units, activation="relu", name="dense1"))
        model.add(layers.BatchNormalization(name="bn_fc"))
        model.add(layers.Dropout(dropout_rate * 0.5, name="dropout_fc2"))
    
    # Output layer must use float32 for numerical stability with softmax
    output_layer = layers.Dense(num_classes, activation="softmax", name="output", dtype='float32')
    model.add(output_layer)
    
    # Freeze backbone if requested (for LESS intensity)
    if freeze_backbone:
        for layer in model.layers:
            if layer.name.startswith("conv") or (layer.name.startswith("bn") and "fc" not in layer.name):
                layer.trainable = False
    
    # Compile model with optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


import tensorflow as tf
from tensorflow.keras import layers, models


def build_classifier(input_dim, n_classes, filters=64, dropout=0.3):
    """
    Build 1D CNN classifier for sign language recognition.
    
    Args:
        input_dim: Input dimension (latent space size from encoder)
        n_classes: Number of output classes
        filters: Number of filters in first conv layer
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),

        layers.Conv1D(filters=filters, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),

        layers.Conv1D(filters=filters * 2, kernel_size=3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(dropout),

        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),

        layers.Dense(n_classes, activation='softmax')
    ])

    return model


def compile_classifier(model, learning_rate=1e-3):
    """
    Compile classifier with sparse categorical crossentropy.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

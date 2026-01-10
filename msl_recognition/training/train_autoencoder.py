"""
Train autoencoder for dimensionality reduction.
"""
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from msl_recognition.modeling.autoencoder import build_autoencoder, compile_autoencoder
from msl_recognition.features import normalize_landmarks


def load_data(csv_path):
    """
    Load and prepare data for autoencoder training.
    
    Returns:
        X_train, X_val: Training and validation data (both X and y are the same for autoencoder)
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    y = df.iloc[:, 0].values
    X = normalize_landmarks(X)
    return train_test_split(X, X, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Load data
    print("Loading augmented landmarks...")
    X_train, X_val, _, _ = load_data('data/interim/augmented_landmarks.csv')
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build and compile autoencoder
    print("\nBuilding autoencoder...")
    autoencoder, encoder = build_autoencoder()
    autoencoder = compile_autoencoder(autoencoder)
    
    print(autoencoder.summary())
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train autoencoder
    print("\nTraining autoencoder...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('models/encoder.h5', save_best_only=True)
        ]
    )
    
    print("\n✅ Autoencoder training complete. Encoder saved to 'models/encoder.h5'")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

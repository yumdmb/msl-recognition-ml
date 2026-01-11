"""
Train autoencoder for dimensionality reduction on combined MSL dataset.
Reduces 63 landmark features to 32 encoded features.
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
        X_train, X_val: Training and validation data
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    X = normalize_landmarks(X)
    return train_test_split(X, X, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Configuration
    input_file = 'data/interim/combined_augmented.csv'
    output_model = 'models/combined_encoder.h5'
    
    print("=" * 60)
    print("MSL Combined Autoencoder Training")
    print("=" * 60 + "\n")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Run augment_landmarks_combined.py first!")
        exit(1)
    
    # Load data
    print(f"📂 Loading data from {input_file}...")
    X_train, X_val, _, _ = load_data(input_file)
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    
    # Build and compile autoencoder
    print("\n🔧 Building autoencoder (63 → 32 dimensions)...")
    autoencoder, encoder = build_autoencoder()
    autoencoder = compile_autoencoder(autoencoder)
    
    print(autoencoder.summary())
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train autoencoder
    print("\n🚀 Training autoencoder...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                output_model,
                save_best_only=True,
                monitor='val_loss'
            )
        ],
        verbose=1
    )
    
    # Save encoder (just the encoder part for inference)
    encoder.save(output_model)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ AUTOENCODER TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: {output_model}")
    print(f"Final validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Best epoch: {history.history['val_loss'].index(min(history.history['val_loss'])) + 1}")

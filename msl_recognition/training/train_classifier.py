"""
Train classifier with PSO hyperparameter optimization.
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

from msl_recognition.modeling.classifier import build_classifier, compile_classifier
from msl_recognition.modeling.pso_optimizer import PSOptimizer
from msl_recognition.features import normalize_landmarks


def load_data(csv_path):
    """
    Load and prepare data for classifier training.
    
    Returns:
        X_train, X_val, y_train, y_val: Training and validation data
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    y = pd.factorize(df.iloc[:, 0])[0]  # Convert labels to integers
    X = normalize_landmarks(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Load and encode data
    print("Loading augmented landmarks...")
    X_train, X_val, y_train, y_val = load_data('data/interim/augmented_landmarks.csv')
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Load trained encoder model
    print("\nLoading trained encoder...")
    encoder = tf.keras.models.load_model('models/encoder.h5', compile=False)
    X_train_enc = encoder.predict(X_train, verbose=0)
    X_val_enc = encoder.predict(X_val, verbose=0)
    print(f"Encoded shape: {X_train_enc.shape}")

    # Reshape for CNN input (add channel dimension)
    X_train_enc = X_train_enc[..., np.newaxis]  # shape: (batch, input_dim, 1)
    X_val_enc = X_val_enc[..., np.newaxis]

    # PSO Optimization
    print("\nRunning PSO hyperparameter optimization...")
    print("This may take several minutes...")
    optimizer = PSOptimizer(
        X_train_enc, y_train,
        X_val_enc, y_val,
        input_dim=X_train_enc.shape[1],
        n_classes=len(np.unique(y_train))
    )
    best_params = optimizer.optimize(n_particles=10, iterations=10)
    print(f"\n✅ PSO Best Parameters: {best_params}")

    # Train final model using best hyperparameters
    print("\nTraining final classifier with best hyperparameters...")
    classifier = build_classifier(
        input_dim=X_train_enc.shape[1],
        n_classes=len(np.unique(y_train)),
        filters=best_params['filters'],
        dropout=best_params['dropout']
    )
    classifier = compile_classifier(classifier, learning_rate=best_params['learning_rate'])

    print(classifier.summary())

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    classifier.fit(
        X_train_enc, y_train,
        validation_data=(X_val_enc, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('models/classifier.h5', save_best_only=True, monitor='val_accuracy')
        ]
    )

    print("\n✅ Classifier training complete. Model saved as 'models/classifier.h5'")
    
    # Evaluate final model
    val_loss, val_acc = classifier.evaluate(X_val_enc, y_val, verbose=0)
    print(f"Final validation accuracy: {val_acc*100:.2f}%")

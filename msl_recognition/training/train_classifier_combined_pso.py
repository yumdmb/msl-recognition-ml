"""
Train CNN classifier for combined MSL dataset (44 classes) with PSO optimization.
Uses Particle Swarm Optimization to find optimal hyperparameters.
Classes: 26 Alphabet (A-Z) + 11 Numbers (0-10) + 7 Words

Note: This takes approximately 2-3 hours on CPU.
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

from msl_recognition.modeling.classifier import build_classifier, compile_classifier
from msl_recognition.modeling.pso_optimizer import PSOptimizer
from msl_recognition.features import normalize_landmarks


def load_data(csv_path):
    """
    Load and prepare data for classifier training.
    
    Returns:
        X_train, X_val, y_train, y_val: Training and validation data
        label_map: Dictionary mapping class names to indices
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype('float32')
    
    # Convert labels to integers and create mapping
    labels, unique_labels = pd.factorize(df.iloc[:, 0], sort=True)
    label_map = {name: idx for idx, name in enumerate(unique_labels)}
    
    X = normalize_landmarks(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return X_train, X_val, y_train, y_val, label_map


if __name__ == "__main__":
    # Configuration
    input_file = 'data/interim/combined_augmented.csv'
    encoder_model = 'models/combined_encoder.h5'
    output_classifier = 'models/combined_classifier_pso.h5'
    output_labels = 'models/combined_labels.json'
    
    print("=" * 60)
    print("MSL Combined Classifier Training (WITH PSO)")
    print("Classes: Alphabet (A-Z) + Numbers (0-10) + Words (7)")
    print("⚠️  This will take approximately 2-3 hours on CPU!")
    print("=" * 60 + "\n")
    
    # Check if input files exist
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Run augment_landmarks_combined.py first!")
        exit(1)
    
    if not os.path.exists(encoder_model):
        print(f"❌ Error: Encoder model not found: {encoder_model}")
        print("   Run train_autoencoder_combined.py first!")
        exit(1)
    
    # Load data
    print(f"📂 Loading data from {input_file}...")
    X_train, X_val, y_train, y_val, label_map = load_data(input_file)
    n_classes = len(label_map)
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Number of classes: {n_classes}")
    
    # Print class distribution
    print("\n📊 Class distribution:")
    alphabet_count = sum(1 for k in label_map if len(k) == 1 and k.isalpha())
    number_count = sum(1 for k in label_map if k.startswith('NUM_'))
    word_count = sum(1 for k in label_map if len(k) > 1 and not k.startswith('NUM_'))
    print(f"   - Alphabet: {alphabet_count} classes")
    print(f"   - Numbers:  {number_count} classes")
    print(f"   - Words:    {word_count} classes")
    
    # Load trained encoder
    print(f"\n🔧 Loading encoder from {encoder_model}...")
    encoder = tf.keras.models.load_model(encoder_model, compile=False)
    X_train_enc = encoder.predict(X_train, verbose=0)
    X_val_enc = encoder.predict(X_val, verbose=0)
    print(f"   Encoded shape: {X_train_enc.shape}")
    
    # Reshape for CNN input (add channel dimension)
    X_train_enc = X_train_enc[..., np.newaxis]
    X_val_enc = X_val_enc[..., np.newaxis]
    
    # PSO Optimization
    print("\n" + "=" * 60)
    print("🔍 Running PSO Hyperparameter Optimization...")
    print("=" * 60)
    print("Parameters being optimized:")
    print("   - Filters: 32 - 256")
    print("   - Dropout: 0.1 - 0.5")
    print("   - Learning Rate: 1e-4 - 1e-2")
    print("\nThis will take several hours. Please be patient...")
    print("=" * 60 + "\n")
    
    optimizer = PSOptimizer(
        X_train_enc, y_train,
        X_val_enc, y_val,
        input_dim=X_train_enc.shape[1],
        n_classes=n_classes
    )
    
    # Run PSO with 10 particles and 10 iterations
    # Each iteration trains 10 models with 20 epochs each
    best_params = optimizer.optimize(n_particles=10, iterations=10)
    
    print("\n" + "=" * 60)
    print("✅ PSO Optimization Complete!")
    print("=" * 60)
    print(f"Best Hyperparameters Found:")
    print(f"   - Filters: {best_params['filters']}")
    print(f"   - Dropout: {best_params['dropout']:.3f}")
    print(f"   - Learning Rate: {best_params['learning_rate']:.6f}")
    print("=" * 60 + "\n")
    
    # Build final classifier with optimized hyperparameters
    print("🔧 Building final classifier with optimized hyperparameters...")
    classifier = build_classifier(
        input_dim=X_train_enc.shape[1],
        n_classes=n_classes,
        filters=best_params['filters'],
        dropout=best_params['dropout']
    )
    classifier = compile_classifier(classifier, learning_rate=best_params['learning_rate'])
    
    print(classifier.summary())
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save label mapping
    with open(output_labels, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"\n💾 Label mapping saved: {output_labels}")
    
    # Train final classifier
    print("\n🚀 Training final classifier with optimized hyperparameters...")
    history = classifier.fit(
        X_train_enc, y_train,
        validation_data=(X_val_enc, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                output_classifier,
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ],
        verbose=1
    )
    
    # Evaluate final model
    val_loss, val_acc = classifier.evaluate(X_val_enc, y_val, verbose=0)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ CLASSIFIER TRAINING COMPLETE (PSO OPTIMIZED)")
    print("=" * 60)
    print(f"Model saved: {output_classifier}")
    print(f"Labels saved: {output_labels}")
    print(f"\nOptimized Hyperparameters:")
    print(f"   - Filters: {best_params['filters']}")
    print(f"   - Dropout: {best_params['dropout']:.3f}")
    print(f"   - Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"\nResults:")
    print(f"   Final validation accuracy: {val_acc*100:.2f}%")
    print(f"   Final validation loss: {val_loss:.4f}")
    
    # Print best results
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
    
    # Save best hyperparameters to file for reference
    params_file = 'models/combined_best_params.json'
    with open(params_file, 'w') as f:
        json.dump({
            **best_params,
            'best_val_accuracy': float(best_val_acc),
            'final_val_accuracy': float(val_acc)
        }, f, indent=2)
    print(f"\n💾 Best hyperparameters saved: {params_file}")
    
    # Print some class predictions for verification
    print("\n📋 Sample predictions:")
    sample_preds = classifier.predict(X_val_enc[:5], verbose=0)
    idx_to_label = {v: k for k, v in label_map.items()}
    for i, (pred, actual) in enumerate(zip(sample_preds, y_val[:5])):
        pred_label = idx_to_label[np.argmax(pred)]
        actual_label = idx_to_label[actual]
        conf = np.max(pred) * 100
        status = "✓" if pred_label == actual_label else "✗"
        print(f"   {status} Predicted: {pred_label} ({conf:.1f}%) | Actual: {actual_label}")

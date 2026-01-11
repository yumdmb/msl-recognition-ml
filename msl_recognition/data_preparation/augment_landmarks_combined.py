"""
Augment combined MSL landmark data with noise and rotation transformations.
Increases dataset size by ~8x for better model training.
"""
import numpy as np
import pandas as pd
import os


def augment_data(df, noise_std=0.01, rotations=5):
    """
    Augment landmark data to increase dataset size.
    
    Args:
        df: DataFrame with landmarks
        noise_std: Standard deviation for Gaussian noise
        rotations: Number of rotation variants
    
    Returns:
        Augmented DataFrame
    """
    augmented = []
    
    print(f"Augmenting {len(df)} samples...")
    
    for idx, row in df.iterrows():
        landmarks = row.values[1:].reshape(-1, 3)
        class_label = row['class']
        
        # Original
        augmented.append([class_label, *landmarks.flatten()])
        
        # Noise augmentation (2 variants)
        for _ in range(2):
            noisy = landmarks + np.random.normal(0, noise_std, landmarks.shape)
            augmented.append([class_label, *noisy.flatten()])
        
        # Rotation augmentation (rotate around z-axis)
        for angle in np.linspace(-15, 15, rotations):
            rad = np.radians(angle)
            rot_matrix = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad), np.cos(rad), 0],
                [0, 0, 1]
            ])
            rotated = np.dot(landmarks, rot_matrix)
            augmented.append([class_label, *rotated.flatten()])
        
        # Progress indicator every 1000 samples
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    return pd.DataFrame(augmented, columns=df.columns)


if __name__ == "__main__":
    # Configuration
    input_file = 'data/interim/combined_landmarks.csv'
    output_file = 'data/interim/combined_augmented.csv'
    
    print("=" * 60)
    print("MSL Combined Data Augmentation")
    print("=" * 60 + "\n")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Run extract_landmarks_combined.py first!")
        exit(1)
    
    # Load extracted landmarks
    print(f"📂 Loading landmarks from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Original dataset: {len(df)} samples")
    print(f"   Classes: {df['class'].nunique()}")
    
    # Augment data
    print("\n🔄 Augmenting data (noise + rotation)...")
    augmented_df = augment_data(df)
    
    # Save augmented data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    augmented_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Original samples:  {len(df):,}")
    print(f"Augmented samples: {len(augmented_df):,} ({len(augmented_df)/len(df):.1f}x increase)")
    print(f"Output file: {output_file}")
    
    # Sample distribution
    print("\nSamples per class (after augmentation):")
    class_counts = augmented_df['class'].value_counts()
    print(f"  Min: {class_counts.min():,} samples")
    print(f"  Max: {class_counts.max():,} samples")
    print(f"  Avg: {class_counts.mean():,.0f} samples")

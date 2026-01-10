"""
Augment landmark data with noise and rotation transformations.
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
    
    for _, row in df.iterrows():
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
    
    return pd.DataFrame(augmented, columns=df.columns)


if __name__ == "__main__":
    # Load extracted landmarks
    input_file = 'data/interim/landmarks.csv'
    output_file = 'data/interim/augmented_landmarks.csv'
    
    print(f"Loading landmarks from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Original dataset: {len(df)} samples")
    
    # Augment data
    print("Augmenting data...")
    augmented_df = augment_data(df)
    
    # Save augmented data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    augmented_df.to_csv(output_file, index=False)
    
    print(f"✅ Augmented dataset: {len(augmented_df)} samples ({len(augmented_df)/len(df):.1f}x increase)")
    print(f"Saved to {output_file}")

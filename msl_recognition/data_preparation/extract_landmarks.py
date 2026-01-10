"""
Extract MediaPipe hand landmarks from MSL images and save to CSV.
"""
import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm


def extract_landmarks(image_path):
    """
    Extract hand landmarks from a single image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        List of 63 landmark values (21 landmarks × 3 coordinates) or None
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    results = mp_hands.process(image)
    mp_hands.close()
    
    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    return None


def process_dataset(dataset_path, output_file):
    """
    Process entire dataset and extract landmarks to CSV.
    
    Args:
        dataset_path: Path to dataset folder (e.g., 'data/raw/Dataset_MSL/Alphabet_MSL')
        output_file: Output CSV path (e.g., 'data/interim/landmarks.csv')
    """
    data = []
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    class_folders.sort()  # Ensure consistent ordering
    
    print(f"Processing {len(class_folders)} classes...")
    
    for class_name in tqdm(class_folders, desc="Extracting landmarks"):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks:
                data.append([class_name, *landmarks])
    
    # Create DataFrame with proper column names (21 landmarks × 3 coords = 63 + 1 class = 64)
    columns = ['class'] + [f'lm_{i}_{ax}' for i in range(21) for ax in ['x', 'y', 'z']]
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(df)} samples to {output_file}")
    print(f"Classes found: {sorted(df['class'].unique())}")


if __name__ == "__main__":
    process_dataset(
        dataset_path='data/raw/Dataset_MSL/Alphabet_MSL',
        output_file='data/interim/landmarks.csv'
    )

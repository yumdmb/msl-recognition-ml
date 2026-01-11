"""
Extract MediaPipe hand landmarks from all MSL datasets (Alphabet, Number, SingleWord).
Creates a combined dataset with 44 classes for unified model training.
"""
import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm


# Dataset configuration
DATASETS = {
    'alphabet': {
        'path': 'data/raw/Dataset_MSL/Alphabet_MSL',
        'prefix': ''  # A, B, C, ... Z (no prefix needed)
    },
    'number': {
        'path': 'data/raw/Dataset_MSL/Number_MSL',
        'prefix': 'NUM_'  # NUM_0, NUM_1, ... NUM_10 (to distinguish from letters)
    },
    'word': {
        'path': 'data/raw/Dataset_MSL/SingleWord_MSL',
        'prefix': ''  # AWAK, MAAF, etc. (already unique names)
    }
}


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


def process_single_dataset(dataset_path, prefix=''):
    """
    Process a single dataset folder and extract landmarks.
    
    Args:
        dataset_path: Path to dataset folder
        prefix: Prefix to add to class names (for disambiguation)
    
    Returns:
        List of [class_name, landmarks...] entries
    """
    data = []
    
    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset not found: {dataset_path}")
        return data
    
    class_folders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
    class_folders.sort()
    
    for class_name in tqdm(class_folders, desc=f"Processing {os.path.basename(dataset_path)}"):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Apply prefix to class name
        full_class_name = f"{prefix}{class_name}"
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks:
                data.append([full_class_name, *landmarks])
    
    return data


def process_all_datasets(output_file='data/interim/combined_landmarks.csv'):
    """
    Process all MSL datasets and create a combined landmarks CSV.
    
    Args:
        output_file: Output CSV path
    """
    all_data = []
    
    print("=" * 60)
    print("MSL Combined Landmark Extraction")
    print("Datasets: Alphabet (A-Z), Number (0-10), SingleWord (7 words)")
    print("=" * 60 + "\n")
    
    # Process each dataset
    for dataset_name, config in DATASETS.items():
        print(f"\n📁 Processing {dataset_name.upper()} dataset...")
        data = process_single_dataset(config['path'], config['prefix'])
        all_data.extend(data)
        print(f"   ✓ Extracted {len(data)} samples")
    
    # Create DataFrame
    columns = ['class'] + [f'lm_{i}_{ax}' for i in range(21) for ax in ['x', 'y', 'z']]
    df = pd.DataFrame(all_data, columns=columns)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Total classes: {df['class'].nunique()}")
    print(f"Output file: {output_file}")
    print("\nClasses by category:")
    
    # Group classes by type
    alphabet_classes = [c for c in df['class'].unique() if len(c) == 1 and c.isalpha()]
    number_classes = [c for c in df['class'].unique() if c.startswith('NUM_')]
    word_classes = [c for c in df['class'].unique() if len(c) > 1 and not c.startswith('NUM_')]
    
    print(f"  - Alphabet: {len(alphabet_classes)} classes ({', '.join(sorted(alphabet_classes)[:5])}...)")
    print(f"  - Number:   {len(number_classes)} classes ({', '.join(sorted(number_classes)[:5])}...)")
    print(f"  - Word:     {len(word_classes)} classes ({', '.join(sorted(word_classes))})")
    
    return df


if __name__ == "__main__":
    process_all_datasets()

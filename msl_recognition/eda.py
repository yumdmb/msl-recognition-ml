"""
Exploratory Data Analysis for Dataset_MSL
Counts images in each class folder and prints distribution.
"""
import os
from pathlib import Path

def count_images(data_dir: Path):
    class_counts = {}
    for subset in ['Alphabet_MSL', 'Number_MSL', 'SingleWord_MSL']:
        subset_dir = data_dir / subset
        if subset_dir.exists():
            for cls in sorted(subset_dir.iterdir()):
                if cls.is_dir():
                    count = len(list(cls.glob('*.*')))
                    class_counts[f"{subset}/{cls.name}"] = count
    return class_counts

if __name__ == '__main__':
    DATA_DIR = Path('Dataset_MSL')
    if not DATA_DIR.exists():
        print(f"Directory {DATA_DIR} not found. Please check the path.")
    else:
        counts = count_images(DATA_DIR)
        print("Class Distribution:")
        for cls, cnt in counts.items():
            print(f"{cls}: {cnt}")
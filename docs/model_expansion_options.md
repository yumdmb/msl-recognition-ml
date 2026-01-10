# MSL Model Expansion Options

This document outlines options for expanding the MSL recognition model to include Numbers and Words in addition to the existing Alphabet model.

## Current Status

✅ **Completed**: Alphabet recognition (A-Z, 26 classes)

- `models/encoder.h5` - Autoencoder for dimensionality reduction
- `models/classifier.h5` - CNN classifier (98.52% accuracy)

## Datasets Available

| Dataset  | Path                                  | Classes                                           |
| -------- | ------------------------------------- | ------------------------------------------------- |
| Alphabet | `data/raw/Dataset_MSL/Alphabet_MSL`   | A-Z (26)                                          |
| Numbers  | `data/raw/Dataset_MSL/Number_MSL`     | 0-10 (11)                                         |
| Words    | `data/raw/Dataset_MSL/SingleWord_MSL` | AWAK, MAAF, MAKAN, MINUM, SALAH, SAYA, TOLONG (7) |

---

## Option 1: Separate Models (Recommended)

Train **separate encoder + classifier** for each category.

### Output Structure

```
models/
├── alphabet_encoder.h5
├── alphabet_classifier.h5
├── number_encoder.h5
├── number_classifier.h5
├── word_encoder.h5
└── word_classifier.h5
```

### Pros

- Each model optimized for its category
- Can update one category without affecting others
- Better accuracy per category

### Cons

- Need to select which model to use at prediction time
- More complex inference logic

### Implementation

1. Run full pipeline for Numbers dataset
2. Run full pipeline for Words dataset
3. Update `realtime_predict.py` to support model selection

---

## Option 2: Reuse Alphabet Encoder

Reuse the existing `encoder.h5` and train **new classifiers only**.

### Output Structure

```
models/
├── encoder.h5              (shared, alphabet-trained)
├── alphabet_classifier.h5
├── number_classifier.h5
└── word_classifier.h5
```

### Pros

- Faster training (skip autoencoder step ~5 min per category)
- Smaller total model size

### Cons

- Encoder may not generalize well to different hand shapes
- Potentially lower accuracy for numbers/words

### Implementation

1. Extract landmarks for Numbers/Words
2. Augment data
3. Skip autoencoder training, use existing encoder
4. Train classifiers only

---

## Option 3: Combined Model

Train one model with **all classes combined** (44 total classes).

### Output Structure

```
models/
├── combined_encoder.h5
└── combined_classifier.h5
```

### Pros

- Single model for all predictions
- Simpler inference logic

### Cons

- Requires full retraining (~8+ hours)
- More complex class management
- If one category needs update, must retrain everything

### Implementation

1. Modify `extract_landmarks.py` to process all three datasets
2. Run full pipeline with combined dataset
3. Update `realtime_predict.py` with 44-class labels

---

## Recommendation

**Option 1 (Separate Models)** is recommended for:

- Modularity and maintainability
- Independent updates per category
- Best accuracy per category

Use a prediction wrapper that selects the appropriate model based on user context or a pre-classification step.

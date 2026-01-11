# MSL Recognition - Malaysian Sign Language Recognition

Real-time Malaysian Sign Language (MSL) recognition using MediaPipe hand tracking and deep learning (autoencoder + CNN classifier).

## Project Overview

This project trains ML models to recognize MSL signs from hand landmark data extracted via MediaPipe.

**Available Models:**

| Model        | Classes   | Description                                   |
| ------------ | --------- | --------------------------------------------- |
| Alphabet     | 26 (A-Z)  | Individual letters                            |
| Number       | 11 (0-10) | Numbers                                       |
| Words        | 7         | AWAK, MAAF, MAKAN, MINUM, SALAH, SAYA, TOLONG |
| **Combined** | **44**    | **All classes in one model (recommended)**    |

## Project Structure

```
msl-recognition-ml/
├── data/
│   ├── raw/                    # Original MSL dataset
│   │   └── Dataset_MSL/
│   │       ├── Alphabet_MSL/   # A-Z folders
│   │       ├── Number_MSL/     # 0-10 folders
│   │       └── SingleWord_MSL/ # Common words
│   ├── interim/                # Processed landmarks
│   │   ├── landmarks.csv
│   │   └── augmented_landmarks.csv
│   └── processed/              # Final datasets
├── models/                     # Trained models
│   ├── encoder.h5             # Autoencoder (alphabet only)
│   ├── classifier.h5          # Classifier (alphabet only)
│   ├── combined_encoder.h5    # Combined autoencoder (44 classes)
│   ├── combined_classifier.h5 # Combined classifier (44 classes)
│   └── combined_labels.json   # Label mapping for combined model
├── msl_recognition/           # Source code
│   ├── data_preparation/
│   │   ├── extract_landmarks.py          # Alphabet only
│   │   ├── extract_landmarks_combined.py # All datasets
│   │   ├── augment_landmarks.py          # Alphabet only
│   │   └── augment_landmarks_combined.py # All datasets
│   ├── training/
│   │   ├── train_autoencoder.py           # Alphabet only
│   │   ├── train_autoencoder_combined.py  # Combined
│   │   ├── train_classifier.py            # With PSO
│   │   ├── train_classifier_simple.py     # No PSO
│   │   └── train_classifier_combined.py   # Combined (no PSO)
│   ├── modeling/
│   │   ├── autoencoder.py
│   │   ├── classifier.py
│   │   └── pso_optimizer.py
│   └── features.py            # Landmark normalization
├── docs/                      # Documentation
├── realtime_predict.py        # Webcam (alphabet, PSO)
├── realtime_predict_combined.py # Webcam (combined, 44 classes)
├── main.py                    # FastAPI (alphabet)
├── main_combined.py           # FastAPI (combined, 44 classes)
├── Dockerfile                 # Docker deployment
└── docker-compose.yml
```

## Training Workflow

### Prerequisites

```bash
# Install dependencies
pip install mediapipe opencv-python pandas numpy scikit-learn tqdm tensorflow pyswarms
```

### Complete Training Pipeline

#### **Step 1: Extract Landmarks** (~15 minutes)

Extract MediaPipe hand landmarks from images:

```bash
python msl_recognition/data_preparation/extract_landmarks.py
```

**Output:** `data/interim/landmarks.csv` (8,760 samples for Alphabet)

---

#### **Step 2: Augment Data** (~1 minute)

Apply noise and rotation augmentation (8x increase):

```bash
python msl_recognition/data_preparation/augment_landmarks.py
```

**Output:** `data/interim/augmented_landmarks.csv` (70,080 samples)

---

#### **Step 3: Train Autoencoder** (~5-10 minutes)

Train autoencoder for dimensionality reduction (63 → 32):

```bash
python msl_recognition/training/train_autoencoder.py
```

**Output:** `models/encoder.h5`

---

#### **Step 4: Train Classifier**

**Option A - Simple (Recommended for quick iterations):**

```bash
python msl_recognition/training/train_classifier_simple.py
```

- **Time:** ~5-10 minutes
- **Uses:** Default hyperparameters (filters=64, dropout=0.3, lr=0.001)
- **Expected accuracy:** 95-98%
- **Output:** `models/classifier_no_pso.h5`

**Option B - With PSO Optimization (Best accuracy):**

```bash
python msl_recognition/training/train_classifier.py
```

- **Time:** ~2-3 hours (CPU)
- **Uses:** PSO to find optimal hyperparameters
- **Expected accuracy:** 98-99%
- **Output:** `models/classifier.h5`

---

## Combined Model Training (Recommended)

Train a unified model for all 44 classes (Alphabet + Numbers + Words). **This is the recommended approach** for production use as it auto-detects the sign type.

### Step 1: Extract All Landmarks (~20-30 minutes)

```bash
python msl_recognition/data_preparation/extract_landmarks_combined.py
```

**Output:** `data/interim/combined_landmarks.csv`

---

### Step 2: Augment Combined Data (~2-3 minutes)

```bash
python msl_recognition/data_preparation/augment_landmarks_combined.py
```

**Output:** `data/interim/combined_augmented.csv` (~8x increase)

---

### Step 3: Train Combined Autoencoder (~10-15 minutes)

```bash
python msl_recognition/training/train_autoencoder_combined.py
```

**Output:** `models/combined_encoder.h5`

---

### Step 4: Train Combined Classifier (~10-20 minutes)

```bash
python msl_recognition/training/train_classifier_combined.py
```

**Outputs:**

- `models/combined_classifier.h5`
- `models/combined_labels.json` (label mapping)

---

## Real-time Recognition

### Webcam Demo

**Combined model (44 classes - Recommended):**

```bash
python realtime_predict_combined.py
```

Features color-coded predictions:

- 🟢 Green = Alphabet (A-Z)
- 🟠 Orange = Number (0-10)
- 🟣 Magenta = Word (AWAK, MAAF, etc.)

**Alphabet-only model:**

```bash
python realtime_predict.py
```

Press `q` to quit.

### API Server

**Combined model (44 classes - Recommended):**

```bash
uvicorn main_combined:app --reload --host 0.0.0.0 --port 8000
```

**Alphabet-only model:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints:**

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict-image/` - Upload image for prediction
- `WS /ws/recognize` - WebSocket for real-time streaming

**Swagger docs:** http://localhost:8000/docs

---

## Docker Deployment

### Build and Run

```bash
# Build image
docker compose build

# Run container
docker compose up

# Run in background
docker compose up -d
```

**API accessible at:** http://localhost:8000

### Stop Container

```bash
docker compose down
```

---

## Model Expansion

To train models for Numbers and Words, see [docs/model_expansion_options.md](docs/model_expansion_options.md).

**Quick guide for separate models:**

1. Modify `extract_landmarks.py` dataset_path:

   - For Numbers: `'data/raw/Dataset_MSL/Number_MSL'`
   - For Words: `'data/raw/Dataset_MSL/SingleWord_MSL'`

2. Run full pipeline (Steps 1-4)

3. Rename output models:
   - `models/number_encoder.h5`, `models/number_classifier.h5`
   - `models/word_encoder.h5`, `models/word_classifier.h5`

---

## Training Times (CPU - Intel Core i7)

| Step                      | Time               | Output                  |
| ------------------------- | ------------------ | ----------------------- |
| Extract Landmarks         | ~15 min            | landmarks.csv           |
| Augment Data              | ~1 min             | augmented_landmarks.csv |
| Train Autoencoder         | ~5-10 min          | encoder.h5              |
| Train Classifier (Simple) | ~5-10 min          | classifier.h5           |
| Train Classifier (PSO)    | ~2-3 hours         | classifier.h5           |
| **Total (Simple)**        | **~30-45 min**     | ✅ Ready for deployment |
| **Total (PSO)**           | **~2.5-3.5 hours** | ✅ Best accuracy        |

---

## Results

**Alphabet Model (A-Z):**

- Training samples: 56,064
- Validation samples: 14,016
- **Validation accuracy: 98.52%**
- Model size: encoder.h5 (477 KB) + classifier.h5 (3.1 MB)

---

## API Usage Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Predict from image
curl -X POST http://localhost:8000/predict-image/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Python

```python
import requests

# Upload image
url = "http://localhost:8000/predict-image/"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
# {"success": true, "label": "A", "confidence": 0.98, "message": "Detected: A"}
```

### JavaScript (Frontend)

```javascript
// WebSocket for real-time streaming
const ws = new WebSocket("ws://localhost:8000/ws/recognize");

ws.onopen = () => {
  // Send base64 encoded JPEG frame
  ws.send(base64ImageData);
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.label, data.confidence);
};
```

---

## Development Notes

### Dataset Structure

Each class folder should contain images (`.jpg`, `.png`, `.jpeg`):

```
Dataset_MSL/Alphabet_MSL/
├── A/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── B/
└── ...
```

### Troubleshooting

**Issue: ModuleNotFoundError**

```bash
# Install missing dependencies
pip install -r requirements.txt
```

**Issue: Low accuracy (<90%)**

- Check dataset quality and size
- Try PSO optimization (`train_classifier.py`)
- Increase augmentation in `augment_landmarks.py`

**Issue: Docker build fails**

- Ensure models exist in `models/` directory
- Check `.dockerignore` doesn't exclude required files

---

## License

This project is part of a Final Year Project (FYP) for Malaysian Sign Language recognition.

## Credits

Dataset: Dataset_MSL (Malaysian Sign Language)  
Framework: MediaPipe, TensorFlow, FastAPI

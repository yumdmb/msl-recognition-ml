"""
Real-time MSL (Malaysian Sign Language) Recognition using PSO-optimized Combined Model.
Supports all 44 classes: Alphabet (A-Z), Numbers (0-10), Words (7).
Uses the PSO-optimized classifier for best accuracy.
"""
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from msl_recognition.features import normalize_landmarks


class MSLCombinedRecognizerPSO:
    """Malaysian Sign Language Recognizer for PSO-optimized combined model (44 classes)."""
    
    def __init__(self, model_dir="models"):
        self.MODEL_DIR = model_dir
        self.CONFIDENCE_THRESHOLD = 0.7
        
        # Load models and labels
        self.encoder, self.classifier, self.labels = self._load_models()
        
        # Create reverse mapping (index to label)
        self.idx_to_label = {v: k for k, v in self.labels.items()}
        
        # Initialize MediaPipe
        self.hands = self._init_mediapipe()
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def _load_models(self):
        """Load trained encoder, PSO-optimized classifier, and label mapping"""
        try:
            encoder_path = os.path.join(self.MODEL_DIR, "combined_encoder.h5")
            classifier_path = os.path.join(self.MODEL_DIR, "combined_classifier_pso.h5")
            labels_path = os.path.join(self.MODEL_DIR, "combined_labels.json")

            encoder = tf.keras.models.load_model(encoder_path, compile=False)
            classifier = tf.keras.models.load_model(classifier_path, compile=False)
            
            with open(labels_path, 'r') as f:
                labels = json.load(f)

            print("[INFO] PSO-optimized combined models loaded successfully.")
            print(f"[INFO] Total classes: {len(labels)}")
            
            # Try to load best params info
            params_path = os.path.join(self.MODEL_DIR, "combined_best_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                print(f"[INFO] Model accuracy: {params.get('best_val_accuracy', 0)*100:.2f}%")
            
            return encoder, classifier, labels
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            print("[HINT] Run the combined PSO training pipeline first:")
            print("  1. python msl_recognition/training/train_autoencoder_combined.py")
            print("  2. python msl_recognition/training/train_classifier_combined_pso.py")
            sys.exit(1)

    def _init_mediapipe(self):
        """Initialize MediaPipe Hands solution"""
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _get_sign_type(self, label):
        """Determine the type of sign (alphabet, number, or word)"""
        if len(label) == 1 and label.isalpha():
            return "ALPHABET"
        elif label.startswith("NUM_"):
            return "NUMBER"
        else:
            return "WORD"

    def _format_label(self, label):
        """Format label for display (remove NUM_ prefix for numbers)"""
        if label.startswith("NUM_"):
            return label[4:]  # Remove "NUM_" prefix
        return label

    def process_frame(self, frame):
        """Extract landmarks and predict from a frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Extract landmarks from the first detected hand
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark
            ]).flatten()
            landmarks = normalize_landmarks(landmarks.reshape(1, -1))

            # Encode with autoencoder
            encoded = self.encoder.predict(landmarks, verbose=0)
            
            # Reshape for CNN input (add channel dimension)
            encoded = encoded[..., np.newaxis]
            
            # Predict class
            preds = self.classifier.predict(encoded, verbose=0)[0]

            return results.multi_hand_landmarks[0], preds
        return None, None

    def predict_from_image(self, frame):
        """
        Public method for single-frame prediction (for API integration).
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            tuple: (predicted_label, confidence, sign_type) or (None, 0.0, None)
        """
        landmarks, preds = self.process_frame(frame)
        if landmarks is not None and preds is not None:
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx]
            if confidence > self.CONFIDENCE_THRESHOLD:
                label = self.idx_to_label[pred_idx]
                sign_type = self._get_sign_type(label)
                display_label = self._format_label(label)
                return display_label, float(confidence), sign_type
        return None, 0.0, None

    def draw_hand_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on frame"""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    def run(self):
        """Run real-time recognition with webcam"""
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("MSL Recognition (PSO Combined)", cv2.WINDOW_NORMAL)
        
        print("\n" + "="*60)
        print("MSL (Malaysian Sign Language) Recognition")
        print("Combined Model (PSO Optimized): Alphabet + Numbers + Words")
        print("="*60)
        print("Press 'q' to quit")
        print("="*60 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror image for more intuitive interaction
            frame = cv2.flip(frame, 1)

            landmarks, preds = self.process_frame(frame)

            if landmarks and preds is not None:
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx]

                if confidence > self.CONFIDENCE_THRESHOLD:
                    raw_label = self.idx_to_label[pred_idx]
                    sign_type = self._get_sign_type(raw_label)
                    display_label = self._format_label(raw_label)
                    
                    # Color based on sign type
                    colors = {
                        "ALPHABET": (0, 255, 0),   # Green
                        "NUMBER": (255, 165, 0),   # Orange
                        "WORD": (255, 0, 255)      # Magenta
                    }
                    color = colors.get(sign_type, (255, 255, 255))
                    
                    # Draw prediction text
                    label_text = f"{display_label} ({confidence:.2f})"
                    cv2.putText(frame, label_text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    
                    # Draw sign type
                    cv2.putText(frame, f"[{sign_type}]", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Draw hand landmarks (disabled - uncomment to show)
                    # self.draw_hand_landmarks(frame, landmarks)

            # Show instructions
            cv2.putText(frame, "PSO Combined | Press 'q' to quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show frame
            cv2.imshow("MSL Recognition (PSO Combined)", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n[INFO] MSL Recognition (PSO) stopped.")


if __name__ == "__main__":
    recognizer = MSLCombinedRecognizerPSO()
    recognizer.run()

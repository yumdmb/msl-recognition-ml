"""
Real-time MSL (Malaysian Sign Language) Recognition using MediaPipe and trained models
"""
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from msl_recognition.features import normalize_landmarks


class MSLRecognizer:
    """Malaysian Sign Language Recognizer using MediaPipe and trained CNN classifier."""
    
    def __init__(self, model_dir="models"):
        # Constants
        self.CLASS_LABELS = [chr(i) for i in range(65, 91)]  # A-Z
        self.CONFIDENCE_THRESHOLD = 0.7
        self.MODEL_DIR = model_dir

        # Load models
        self.encoder, self.classifier = self._load_models()

        # Initialize MediaPipe
        self.hands = self._init_mediapipe()
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def _load_models(self):
        """Load trained encoder and classifier models"""
        try:
            encoder_path = os.path.join(self.MODEL_DIR, "encoder.h5")
            classifier_path = os.path.join(self.MODEL_DIR, "classifier.h5")

            encoder = tf.keras.models.load_model(encoder_path, compile=False)
            classifier = tf.keras.models.load_model(classifier_path, compile=False)

            print("[INFO] Models loaded successfully.")
            return encoder, classifier
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
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
            tuple: (predicted_letter, confidence) or (None, 0.0)
        """
        landmarks, preds = self.process_frame(frame)
        if landmarks is not None and preds is not None:
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx]
            if confidence > self.CONFIDENCE_THRESHOLD:
                return self.CLASS_LABELS[pred_idx], float(confidence)
        return None, 0.0

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
        cv2.namedWindow("MSL Recognition", cv2.WINDOW_NORMAL)
        
        print("\n" + "="*50)
        print("MSL (Malaysian Sign Language) Recognition")
        print("="*50)
        print("Press 'q' to quit")
        print("="*50 + "\n")

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
                    label = f"{self.CLASS_LABELS[pred_idx]} ({confidence:.2f})"
                    
                    # Draw prediction text
                    cv2.putText(frame, label, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Draw hand landmarks
                    self.draw_hand_landmarks(frame, landmarks)

            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show frame
            cv2.imshow("MSL Recognition", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\n[INFO] MSL Recognition stopped.")


if __name__ == "__main__":
    recognizer = MSLRecognizer()
    recognizer.run()

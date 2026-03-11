import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load a pre-trained deepfake detection model (e.g., trained on DFDC dataset)
# For this example, we assume you have a model file named 'deepfake_model.h5'
try:
    model = load_model('deepfake_model.h5')
except Exception as e:
    print("Model file not found. Please train or download a model first.")

def detect_deepfake(image_path):
    # 1. Preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0                  # Normalize pixels
    img = np.expand_dims(img, axis=0)

    # 2. Predict using the CNN
    prediction = model.predict(img)[0][0]
    
    # 3. Output result
    label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = prediction if label == "FAKE" else 1 - prediction
    return label, confidence
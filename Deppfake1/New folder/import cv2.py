import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import sys

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# NOTE: For a real system, you must download pre-trained weights from a 
# trusted source (e.g., FaceForensics++ or Deepfake Detection Challenge).
# This code uses a placeholder architecture. 
# To get real weights, visit: https://github.com/serengil/deepfakes-detection
# and download the 'xception.h5' file.

MODEL_PATH = 'xception.h5'  # Path to your pre-trained deepfake weights
INPUT_SIZE = 299            # Xception model input size
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# -----------------------------------------------------------------------------
# MODULE 1: FACE DETECTION & PREPROCESSING
# -----------------------------------------------------------------------------
class FacePreprocessor:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    def detect_and_crop(self, image_path):
        """
        Detects the largest face in the image and crops it.
        Returns: Cropped face image (numpy array) or None.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No face detected. Skipping AI analysis.")
            return None

        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add some padding
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad)
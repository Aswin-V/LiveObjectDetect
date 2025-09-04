
import cv2
import ultralytics
import tensorflow as tf
import numpy as np
import streamlit as st
import PIL
from ultralytics import YOLO

def run_tests():
    """
    Runs a series of tests to verify the environment and dependencies.
    """
    print("--- Running Environment and Dependency Tests ---")

    # Test 1: Check OpenCV
    try:
        print(f"OpenCV version: {cv2.__version__}")
        # Create a dummy image to verify basic functionality
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
        if gray_image.shape == (100, 100):
            print("OpenCV functionality test: PASSED")
        else:
            print("OpenCV functionality test: FAILED")
    except Exception as e:
        print(f"OpenCV test failed: {e}")

    # Test 2: Check Ultralytics
    try:
        print(f"Ultralytics version: {ultralytics.__version__}")
        # Check for a key attribute to ensure the library is functional
        if hasattr(ultralytics, 'YOLO'):
            print("Ultralytics functionality test: PASSED")
        else:
            print("Ultralytics functionality test: FAILED")
    except Exception as e:
        print(f"Ultralytics test failed: {e}")

    # Test 3: Check TensorFlow
    try:
        print(f"TensorFlow version: {tf.__version__}")
        # Perform a simple tensor operation
        a = tf.constant(1)
        b = tf.constant(2)
        if tf.add(a, b).numpy() == 3:
            print("TensorFlow functionality test: PASSED")
        else:
            print("TensorFlow functionality test: FAILED")
    except Exception as e:
        print(f"TensorFlow test failed: {e}")

    # Test 4: Check other major libraries
    try:
        print(f"Pillow version: {PIL.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Streamlit version: {st.__version__}")
        print("Other major libraries are installed.")
    except Exception as e:
        print(f"Failed to check other libraries: {e}")

    print("\n--- All tests completed ---")

def check_yolo_model(model_name):
    """
    Attempts to download and initialize a YOLO model from Ultralytics.

    Args:
        model_name (str): The name of the model to check (e.g., 'yolov8n.pt').

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    print("-" * 30)
    print(f"Attempting to load model: {model_name}")
    try:
        # This line will automatically download the model if it's not cached
        model = YOLO(model_name)
        print(f"✅ Success! '{model_name}' loaded correctly.")
        print(f"Model Type: {type(model)}")
        return True
    except Exception as e:
        # A failure here often means the model is not recognized by your
        # installed version of the ultralytics package.
        print(f"❌ Failed to load '{model_name}'.")
        print(f"   Error: {e}")
        print("\n   This likely means your 'ultralytics' package is outdated and does not include this model.")
        print("   To fix this, run the following command in your terminal:")
        print("   pip install --upgrade ultralytics")
        return False
    finally:
        print("-" * 30)

if __name__ == "__main__":
    run_tests()

    print("Running Ultralytics YOLO model check...")

    # --- Check for YOLOv8 ---
    # This should almost always succeed if ultralytics is installed.
    check_yolo_model('yolov8n.pt')

    print("\n" + "="*40 + "\n")

    # --- Check for YOLOv11 ---
    # This will fail if the ultralytics package is not up-to-date.
    check_yolo_model('yolov10n.pt')

# 🤖 AI Video Analysis with Python & Streamlit

This is an interactive web application built with Streamlit that performs real-time AI analysis on images, videos, and live webcam feeds. It integrates multiple state-of-the-art AI models to provide a comprehensive analysis tool.

## ✨ Features

- **Multiple Input Sources:**
  - Upload an image file (`.jpg`, `.png`).
  - Upload a video file (`.mp4`, `.mov`, `.avi`).
  - Use a live webcam feed.
- **Multiple Analysis Models:**
  - **Gemini:** A powerful, cloud-based multimodal model from Google for general object, emotion, and activity detection.
  - **YOLO (v8, v9, v10, v11, v12):** State-of-the-art, real-time models that run locally for high performance. Supports multiple tasks:
    - **Detect:** Object Detection
    - **Segment:** Instance Segmentation
    - **Classify:** Image Classification
    - **Pose:** Pose Estimation
    - **OBB:** Oriented Bounding Box Detection
  - **DeepFace:** A facial attribute analysis model that detects faces and analyzes age, gender, and ethnicity.
- **Interactive UI:**
  - Sidebar for easy configuration of models and parameters.
  - Real-time display of analysis results, including bounding boxes and JSON data.
  - Secure API key management using a `.env` file.

## 🛠️ Installation

The project includes a convenient runner script that sets up a virtual environment and installs all necessary dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd LiveObjectDetect
    ```

2.  **Run the setup script:**
    This will create a virtual environment in a `.venv` directory, install the required packages from `requirements.txt`, and start the Streamlit application.
    ```bash
    python runner.py
    ```

3.  **(Optional) Manual Setup:**
    If you prefer to manage your own environment:
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt

    # Run the app
    streamlit run app.py
    ```

## 🚀 Usage

The application can be run in several modes, providing different user interfaces.

### Streamlit Web App (Default)

This is the default mode, providing a rich, interactive user interface in your browser.

```bash
python runner.py
```

Once running, open the provided URL in your web browser.

1.  **Select a Model:** Use the dropdown in the sidebar to choose between Gemini, YOLO, and DeepFace.
2.  **Configure the Model:**
    - For **Gemini**, enter your API key. You can get one from Google AI Studio. For convenience, you can also create a `.env` file in the project root and add `GEMINI_API_KEY="YOUR_API_KEY"`.
    - For **YOLO**, select the **Task** (Detect, Segment, Classify, Pose, OBB), **Version**, and **Model Size**. The appropriate model weights will be downloaded automatically on the first run. Adjust the confidence threshold to control detection sensitivity.
    - For **DeepFace**, no special configuration is required.
3.  **Select an Input Source:** Choose to upload an image, a video, or use the live webcam feed.
4.  **Control Processing:** Use the "Start", "Pause", and "Stop" buttons to control the analysis.

### Tkinter Desktop App

A native desktop application built with Python's standard Tkinter GUI toolkit. It offers a more traditional desktop experience compared to the web app and provides full control over all model parameters through the UI.

```bash
python runner.py tkinter
```

The Tkinter app also accepts the same command-line arguments as the standalone app for pre-configuring the model and video source (e.g., `python runner.py tkinter -v /path/to/video.mp4`).

### Standalone OpenCV App

This mode runs a high-performance window using OpenCV, showing the live feed and the processed feed side-by-side. It is ideal for performance testing as it has minimal UI overhead, with controls available via mouse clicks and keyboard shortcuts.

```bash
# Example with YOLOv8n (default) on webcam
python runner.py standalone

# Example with a larger YOLOv10 model on a video file
python runner.py standalone -m YOLO --yolo-version v10 --yolo-size l -v /path/to/your/video.mp4

# Example with YOLO-Pose on webcam using the --yolo-task flag
python runner.py standalone -m YOLO --yolo-task pose --yolo-version v8 --yolo-size s

# Example with DeepFace
python runner.py standalone -m DeepFace

# Example with Gemini (API key must be set in .env or via --api-key)
python runner.py standalone -m Gemini
```

**Controls:**
- Press `p` to pause/resume processing.
- Press `q` to quit the application.

### Environment Test Utility

A utility to check if all dependencies (OpenCV, TensorFlow, Ultralytics, etc.) are installed correctly and to test the download and loading of YOLO models. This is useful for troubleshooting setup issues.

```bash
python runner.py test
```

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. Please ensure you also comply with the licenses of the models and libraries used (e.g., YOLO, DeepFace, Streamlit).

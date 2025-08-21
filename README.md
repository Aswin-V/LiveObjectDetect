# 🤖 AI Video Analysis with Python & Streamlit

This is an interactive web application built with Streamlit that performs real-time AI analysis on images, videos, and live webcam feeds. It integrates multiple state-of-the-art AI models to provide a comprehensive analysis tool.

## ✨ Features

- **Multiple Input Sources:**
  - Upload an image file (`.jpg`, `.png`).
  - Upload a video file (`.mp4`, `.mov`, `.avi`).
  - Use a live webcam feed.
- **Multiple Analysis Models:**
  - **Gemini:** A powerful, cloud-based multimodal model from Google for general object, emotion, and activity detection.
  - **YOLO (v8, v9, v10):** State-of-the-art, real-time object detection models that run locally for high performance.
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

Once the application is running, open the provided URL in your web browser.

1.  **Select a Model:** Use the dropdown in the sidebar to choose between Gemini, YOLO, and DeepFace.
2.  **Configure the Model:**
    - For **Gemini**, enter your API key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey). For convenience, you can also create a `.env` file in the project root and add `GEMINI_API_KEY="YOUR_API_KEY"`.
    - For **YOLO**, select the version and model size. The model weights will be downloaded automatically on the first run. Adjust the confidence threshold to control detection sensitivity.
    - **DeepFace** requires no special configuration.
3.  **Select an Input Source:** Choose to upload an image, a video, or use your live webcam.
4.  **View Results:** The application will process the input and display the annotated video/image along with the raw JSON analysis results.

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. Please ensure you also comply with the licenses of the models and libraries used (e.g., YOLO, DeepFace, Streamlit).

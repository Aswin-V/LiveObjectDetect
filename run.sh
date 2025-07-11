#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

VENV_DIR="env"
PYTHON_CMD="python3"

echo "--- Setting up AI Video Analysis Environment ---"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one at './$VENV_DIR/'..."
    $PYTHON_CMD -m venv $VENV_DIR
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing/updating requirements from requirements.txt..."
pip install -r requirements.txt

echo "--- Starting Streamlit Application ---"
streamlit run app.py
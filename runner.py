import os
import sys
import subprocess

VENV_DIR = "env"
REQUIREMENTS_FILE = "requirements.txt"

def get_executable(name):
    """Gets the platform-specific executable path within the venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", f"{name}.exe")
    return os.path.join(VENV_DIR, "bin", name)

def main():
    """
    Sets up the environment and runs the Streamlit application.
    """
    print("--- Setting up AI Video Analysis Environment (Python Runner) ---")

    # 1. Check for and create the virtual environment if it doesn't exist.
    if not os.path.isdir(VENV_DIR):
        print(f"Virtual environment not found. Creating one at './{VENV_DIR}/'...")
        # Use the current python interpreter to create the venv
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")

    # Get paths to executables inside the virtual environment
    python_exe = get_executable("python")
    pip_exe = get_executable("pip")
    streamlit_exe = get_executable("streamlit")

    # 2. Install/update requirements using the venv's pip.
    print(f"Installing/updating requirements from {REQUIREMENTS_FILE}...")
    subprocess.run([pip_exe, "install", "-r", REQUIREMENTS_FILE], check=True)

    # 3. Start the Streamlit application.
    print("\n--- Starting Streamlit Application ---")
    print(f"Running: {streamlit_exe} run app.py")
    subprocess.run([streamlit_exe, "run", "app.py"])

if __name__ == "__main__":
    main()
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
    streamlit_exe = get_executable("streamlit")

    # Ensure pip is installed/upgraded in the virtual environment.
    # This makes the script resilient to corrupted venvs that might be missing pip.
    print("Ensuring 'pip' is available in the virtual environment...")
    ensurepip_result = subprocess.run(
        [python_exe, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True
    )

    if ensurepip_result.returncode != 0:
        # Check for a common error on Linux where the venv package is not installed.
        if "No module named ensurepip" in ensurepip_result.stderr:
            print("\n--- FATAL SETUP ERROR ---", file=sys.stderr)
            print("The Python virtual environment is missing essential components.", file=sys.stderr)
            print("This is common on Linux systems where 'venv' support is a separate package.", file=sys.stderr)
            print("\nTo fix this, please install the venv package for your Python version.", file=sys.stderr)
            print("Example for Debian/Ubuntu: sudo apt install python3.12-venv", file=sys.stderr)
            print("Example for Fedora/RHEL:   sudo dnf install python3-devel", file=sys.stderr)
            print("\nAfter installation, please delete the 'env' directory and run this script again.", file=sys.stderr)
            sys.exit(1)
        else:
            # It failed for some other reason, show the generic error and exit.
            print(f"Error during 'ensurepip':\n{ensurepip_result.stderr}", file=sys.stderr)
            raise subprocess.CalledProcessError(ensurepip_result.returncode, ensurepip_result.args, stderr=ensurepip_result.stderr)
    # 2. Install/update requirements using the venv's pip.
    # Using "python -m pip" is more robust than calling the pip executable directly.
    if os.path.exists(REQUIREMENTS_FILE):
        print(f"Installing/updating requirements from {REQUIREMENTS_FILE}...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", REQUIREMENTS_FILE], check=True)
    else:
        print(f"Warning: '{REQUIREMENTS_FILE}' not found. Skipping dependency installation.")

    print("\n--- Starting Streamlit Application ---")
    print(f"Running: {streamlit_exe} run app.py")
    subprocess.run([streamlit_exe, "run", "app.py"])

if __name__ == "__main__":
    main()
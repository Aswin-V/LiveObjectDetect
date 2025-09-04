"""
A utility script to set up the environment and run the AI Video Analysis application.

This script automates the following steps:
1.  Ensures a Python virtual environment exists and creates it if necessary.
2.  Installs or updates the required packages from the requirements.txt file.
3.  Checks for the latest version of the 'ultralytics' package and upgrades it if a new version is available.
4.  Runs the selected application (Streamlit, Tkinter, or Standalone).

Usage:
    python runner.py [app_type]

Arguments:
    app_type (str, optional): The type of application to run.
                              Choices: 'streamlit', 'standalone', 'tkinter', 'test'.
                              Defaults to 'streamlit'.
"""
import os
import sys
import subprocess
import argparse
import re

VENV_DIR = ".venv"
REQUIREMENTS_FILE = "requirements.txt"

def get_executable(name):
    """Gets the platform-specific executable path within the venv."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", f"{name}.exe")
    return os.path.join(VENV_DIR, "bin", name)

def get_package_version(python_exe, package_name):
    """Gets the installed version of a package."""
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "show", package_name],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        match = re.search(r"Version: ([\d\.]+[\w\.]*)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None

def get_latest_package_version(python_exe, package_name):
    """Gets the latest version of a package from PyPI."""
    try:
        # Using 'pip index versions' is a reliable way to get the latest version
        result = subprocess.run(
            [python_exe, "-m", "pip", "index", "versions", package_name],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        # The output is typically: `package_name (X.Y.Z)`
        match = re.search(rf"{re.escape(package_name)} \(([\d\.]+[\w\.]*)\)", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Warning: Could not check for the latest version of {package_name}.")
        return None
    return None

def main():
    """
    Sets up the environment and runs the selected application.
    """
    parser = argparse.ArgumentParser(description="Run the AI Video Analysis application.")
    parser.add_argument(
        "app_type", 
        type=str, 
        default="streamlit", 
        nargs='?', # makes the argument optional
        choices=["streamlit", "standalone", "tkinter", "test"],
        help="The type of application to run: 'streamlit' (web), 'standalone' (OpenCV), or 'tkinter' (Desktop)."
    )
    # The standalone app has its own arguments, so we need to allow them
    args, unknown = parser.parse_known_args()

    print(f"--- Setting up AI Video Analysis Environment (Mode: {args.app_type}) ---")
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
    # Ensure pip is installed/upgraded in the virtual environment.
    # This makes the script resilient to corrupted venvs that might be missing pip.
    print("Ensuring 'pip' is available in the virtual environment...")
    ensurepip_result = subprocess.run(
        [python_exe, "-m", "ensurepip", "--upgrade"],
        capture_output=True, text=True, encoding='utf-8'
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
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([python_exe, "-m", "pip", "install", "-r", REQUIREMENTS_FILE], check=True)
    else:
        print(f"Warning: '{REQUIREMENTS_FILE}' not found. Skipping dependency installation.")

    # --- Check and upgrade ultralytics ---
    print("\n--- Checking for ultralytics update ---")
    package_to_check = "ultralytics"
    installed_version = get_package_version(python_exe, package_to_check)
    latest_version = get_latest_package_version(python_exe, package_to_check)

    if not installed_version:
        print(f"{package_to_check} is not installed. It will be installed based on requirements.txt.")
    elif not latest_version:
        print(f"Could not check for the latest version of {package_to_check}. Skipping update check.")
    elif installed_version != latest_version:
        print(f"New version of {package_to_check} available: {latest_version} (installed: {installed_version}).")
        print(f"Upgrading {package_to_check}...")
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", package_to_check], check=True)
        print(f"Successfully upgraded {package_to_check} to {latest_version}.")
        print("Note: To pin this version for future installations, please manually update requirements.txt.")
    else:
        print(f"{package_to_check} is up-to-date (version {installed_version}).")

    if args.app_type == "streamlit":
        print("\n--- Starting Streamlit Application ---")
        streamlit_exe = get_executable("streamlit")
        command = [streamlit_exe, "run", "app.py"]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
    elif args.app_type == "standalone":
        print("\n--- Starting Standalone Application ---")
        # Pass through any unknown arguments to the standalone app
        command = [python_exe, "standalone_app.py"] + unknown
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
    elif args.app_type == "tkinter":
        print("\n--- Starting Tkinter Desktop Application ---")
        command = [python_exe, "tkinter_app.py"] + unknown
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
    elif args.app_type == "test":
        print("\n--- Running Test Application ---")
        command = [python_exe, "test_app.py"] + unknown
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)

if __name__ == "__main__":
    main()
#!/bin/bash
# scripts/launch_app.sh
# Script to launch the Streamlit user interface.

# Default values
DEFAULT_PYTHON_EXEC="python3" # or "python" depending on your environment
DEFAULT_STREAMLIT_APP_PATH="src/interface/streamlit_app.py"
VENV_DIR="venv" # Common virtual environment directory name

# --- Helper Functions ---
check_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

activate_venv() {
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Activating Python virtual environment from $VENV_DIR..."
        source "$VENV_DIR/bin/activate"
        # Update PYTHON_EXEC if venv python is different
        if check_command_exists "$VENV_DIR/bin/python"; then
            PYTHON_EXEC="$VENV_DIR/bin/python"
        elif check_command_exists "$VENV_DIR/bin/python3"; then
            PYTHON_EXEC="$VENV_DIR/bin/python3"
        fi
        echo "Using Python from venv: $PYTHON_EXEC"
    else
        echo "Virtual environment '$VENV_DIR' not found or activate script missing."
        echo "Proceeding with system Python: $PYTHON_EXEC"
    fi
}


# --- Main Script Logic ---

# Determine project root (assuming this script is in project_root/scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

echo "Project Root: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Failed to change directory to $PROJECT_ROOT"; exit 1; }


# 1. Determine Python executable
PYTHON_EXEC="$DEFAULT_PYTHON_EXEC"
if ! check_command_exists "$PYTHON_EXEC"; then
    # Try 'python' if 'python3' not found
    if check_command_exists "python"; then
        PYTHON_EXEC="python"
    else
        echo "Error: Neither 'python3' nor 'python' command found in PATH."
        echo "Please ensure Python is installed and accessible."
        exit 1
    fi
fi
echo "Using Python executable: $PYTHON_EXEC"
"$PYTHON_EXEC" --version # Print version for diagnostics


# 2. Activate virtual environment if present
# (Optional, comment out if not standard for your project or if users manage venv externally)
# activate_venv


# 3. Check if Streamlit is installed (basic check)
# This check might be too simple if streamlit is installed but not for the current PYTHON_EXEC
echo "Checking for Streamlit..."
if ! "$PYTHON_EXEC" -m streamlit --version >/dev/null 2>&1; then
    echo "Error: Streamlit does not seem to be installed for $PYTHON_EXEC."
    echo "Please install Streamlit: pip install streamlit (or using your project's requirements.txt)"
    echo "If using a virtual environment, ensure it's activated and Streamlit is installed there."
    exit 1
fi
echo "Streamlit found."
"$PYTHON_EXEC" -m streamlit --version


# 4. Check if the Streamlit app file exists
STREAMLIT_APP_FILE="$PROJECT_ROOT/$DEFAULT_STREAMLIT_APP_PATH"
if [ ! -f "$STREAMLIT_APP_FILE" ]; then
    echo "Error: Streamlit application file not found at $STREAMLIT_APP_FILE"
    exit 1
fi
echo "Streamlit app file found: $STREAMLIT_APP_FILE"


# 5. Launch Streamlit application
echo ""
echo "Launching Streamlit application: $STREAMLIT_APP_FILE"
echo "Access the application in your web browser (URL will be shown below)."
echo "Press Ctrl+C in this terminal to stop the application."
echo ""

# Command to run Streamlit.
# Pass any additional Streamlit arguments after "$STREAMLIT_APP_FILE" if needed.
# Example: --server.port 8502
"$PYTHON_EXEC" -m streamlit run "$STREAMLIT_APP_FILE" "$@"
# "$@" passes any arguments given to launch_app.sh directly to the streamlit run command.

# Exit code of streamlit will be the exit code of this script
exit $?

# --- For CLI entry point in setup.py ---
# This part is if you want `python -m cultural_artifact_explorer.scripts.launch_app` to work
# or an installed script `cae-launch-app` to call a Python main function.
# The shell script itself is typically run directly: `./scripts/launch_app.sh`

# def main_cli():
#     """
#     Python function that could be called by a console script entry point.
#     This would essentially replicate the subprocess call from streamlit_app.py's main_cli
#     or directly call Streamlit's Python API if preferred (though CLI is common).
#     """
#     import subprocess
#     import sys
#     import os
#     print("Python CLI: Launching Streamlit application via subprocess...")
#     script_path = os.path.join(os.path.dirname(__file__), "..", "src", "interface", "streamlit_app.py")
#     script_path = os.path.abspath(script_path) # Get absolute path
#     if not os.path.exists(script_path):
#         print(f"Error: Streamlit app Python file not found at {script_path}")
#         sys.exit(1)
#     try:
#         # Determine project root to run streamlit from there, making relative paths in app work
#         project_root_for_py = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         # Command: streamlit run src/interface/streamlit_app.py
#         # We need to ensure streamlit runs with project_root as CWD or that app_path is relative to it.
#         # Simplest is to ensure app_path is correct for streamlit.
#         # Here, script_path is absolute.
#         subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=True, cwd=project_root_for_py)
#     except FileNotFoundError:
#         print("Error: streamlit command not found (from Python). Make sure Streamlit is installed.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running Streamlit application (from Python): {e}")

# if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "main_cli":
#     # This allows `python scripts/launch_app.sh main_cli` (though not typical for .sh)
#     # Or more realistically, if this file was launch_app.py and used as console_script.
#     main_cli()
# elif __name__ == "__main__":
#      print("This is a shell script. To run the Streamlit app, execute it directly:")
#      print("  ./scripts/launch_app.sh")
#      print("Or, if you made it executable: `chmod +x scripts/launch_app.sh`")
#      print("Alternatively, run Streamlit directly from the project root:")
#      print("  python -m streamlit run src/interface/streamlit_app.py")

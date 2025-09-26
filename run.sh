#!/bin/bash

# This script sets up the environment and runs the gaze tracking application.
# It prioritizes using Conda to create a Python 3.9 environment.
# If Conda is not found, it will fall back to using venv.

# --- Configuration ---
PYTHON_VERSION="3.9"
CONDA_ENV_NAME="gaze_tracker_env"
VENV_DIR="venv"
SCRIPT_NAME="gaze_tracker.py" # Make sure this matches your Python file name

# --- Functions ---
function print_info {
    echo "INFO: $1"
}

function print_warning {
    echo "WARNING: $1"
}

function print_error {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Main Execution ---

# 1. Check if the main script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    print_error "The main script '$SCRIPT_NAME' was not found. Please rename your file or update the SCRIPT_NAME variable in this script."
fi

# 2. Check for Conda and set up environment
if command -v conda &> /dev/null; then
    print_info "Conda detected. Setting up environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION."

    # Check if the environment already exists
    if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
        print_info "Conda environment '$CONDA_ENV_NAME' not found. Creating it now..."
        conda create --name $CONDA_ENV_NAME python=$PYTHON_VERSION -y
        if [ $? -ne 0 ]; then
            print_error "Failed to create Conda environment. Please check your Conda installation."
        fi
    fi

    # Activate the Conda environment
    print_info "Activating Conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    
    # Install dependencies
    print_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies within Conda environment."
    fi

    # Run the application
    print_info "Starting the Gaze Tracker application..."
    print_info "Press 'q' in the application window to quit."
    python $SCRIPT_NAME

    # Deactivate
    print_info "Application finished."
    conda deactivate

else
    # --- Fallback to venv if Conda is not found ---
    print_warning "Conda not found. Falling back to use 'venv'."
    print_warning "Please ensure your default 'python3' is version 3.9 for best results."
    PYTHON_EXECUTABLE="python3"

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating Python virtual environment in './$VENV_DIR'..."
        $PYTHON_EXECUTABLE -m venv $VENV_DIR
        if [ $? -ne 0 ]; then
            print_error "Failed to create virtual environment."
        fi
    fi

    # Activate the virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    # Install dependencies
    print_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies."
    fi

    # Run the application
    print_info "Starting the Gaze Tracker application..."
    print_info "Press 'q' in the application window to quit."
    $PYTHON_EXECUTABLE $SCRIPT_NAME

    # Deactivate
    print_info "Application finished."
    deactivate
fi
#!/bin/bash

# This script sets up a Python virtual environment and install required packages.
# To run this script, execute it in the terminal with:
# source startup.sh

requirements_file="aero3760_requirements.txt"
env_name="aero3760_venv"

# Check if the requirements.txt file exists
if [ ! -f $requirements_file ]; then
    echo "$requirements_file not found. Please ensure it exists in the current directory."
    echo "Current directory: $(pwd)"
    return 1
fi
    python3 -m venv $env_name
    source $env_name/bin/activate

    # Install the required packages. Suppress out messages for already satisfied requirements.
    pip install --upgrade pip
    pip install -r $requirements_file 2>&1 | grep -v "Requirement already satisfied"

    echo 
    echo "venv is ready and activated."
    echo "To exit the virtual environment, use 'deactivate'."
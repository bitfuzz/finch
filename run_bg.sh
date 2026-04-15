#!/bin/bash
set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Remove CRLF issues if executed in windows git bash
if [ -n "$VENV_DIR" ]; then VENV_DIR=$(echo "$VENV_DIR" | tr -d '\r'); fi

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found. Please run setup.sh first."
    exit 1
fi

echo "[Finch] Starting application in the background..."

# Check if Windows venv or Unix venv
if [ -d "venv/Scripts" ]; then
    PYTHON_CMD="venv/Scripts/python"
else
    PYTHON_CMD="venv/bin/python"
fi

# Run the python script using the venv's python executable
nohup "$PYTHON_CMD" main.py > finch_app.log 2>&1 &

PID=$!

# Save the PID to a file for easy stopping later
echo $PID > finch_app.pid

echo "[Finch] Application started successfully with PID: $PID"
echo "[Finch] Logs are being written to finch_app.log"
echo "[Finch] To stop the application, you can use: kill \$(cat finch_app.pid)"

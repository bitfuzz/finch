#!/bin/bash
set -e

echo "[Finch] Setting up environment..."

# Goto script directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[Finch] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install requirements
echo "[Finch] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download models
echo "[Finch] Downloading models..."
chmod +x bin/download_models.sh
./bin/download_models.sh

echo ""
echo "[Finch] Setup complete! Run locally with:"
echo "        source venv/bin/activate"
echo "        python main.py"

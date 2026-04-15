#!/bin/bash
set -e

echo "[Finch] Setting up environment..."

# Goto script directory
cd "$(dirname "$0")"

# Handle macOS dependencies
if [ "$(uname)" = "Darwin" ]; then
    if command -v brew &> /dev/null; then
        echo "[Finch] Ensuring portaudio is installed (required for PyAudio)..."
        brew install portaudio || true
        # Fix missing headers for PyAudio installation on Apple Silicon
        export CFLAGS="-I$(brew --prefix)/include"
        export LDFLAGS="-L$(brew --prefix)/lib"
    else
        echo "[Finch] WARNING: Homebrew not found. Please install portaudio manually."
    fi
fi

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

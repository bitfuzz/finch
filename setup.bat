@echo off
echo [Finch] Setting up environment...

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo [Finch] Creating virtual environment...
    python -m venv venv
)

:: Activate and install requirements
echo [Finch] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

:: Download models
echo [Finch] Downloading models...
powershell -ExecutionPolicy Bypass -File bin\download_models.ps1

echo.
echo [Finch] Setup complete! Run locally with:
echo         venv\Scripts\activate
echo         python main.py

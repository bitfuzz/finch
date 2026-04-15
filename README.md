# Finch Transcriber

A lightweight, cross-platform terminal-based transcription application for macOS and Windows. 
Finch uses a hybrid model approach:
- **Zipformer** for real-time dictation with live visual feedback.
- **Parakeet v3** for high-accuracy batch transcription of meeting audio.

## Prerequisites
- Python 3.10+
- `ffmpeg` (optional but recommended for certain audio handling)

## Quick Setup
You can set up the virtual environment, install dependencies, and download the necessary models using the automated setup script.

**Platform Independent Setup (requires `make`):**
```bash
make setup
```

**Windows:**
```cmd
setup.bat
```

**macOS / Linux:**
```bash
bash setup.sh
```

## Usage
Start the Finch daemon:
```bash
make run
```
Or manually:
```bash
python main.py
```

### Controls
Once running, Finch monitors in the background:
- **Dictation**: Hold <kbd>Ctrl</kbd> + <kbd>Space</kbd>
- **Meeting Mode**: Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>R</kbd>

## Configuration
See `dictation_rules.json` to configure filler words, auto-capitalization targets, API integration, and model locations.

## Architecture
- `main.py`: Entry point and hotkey listener.
- `transcriber.py`: Dual-model ASR engine logic.
- `audio_capture.py` / `system_audio.py`: Cross-platform loopback audio capture logic.
- `dictation_ui.py`: Text injection and UI overlays.

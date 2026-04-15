#!/usr/bin/env bash
# bin/download_models.sh
# Downloads sherpa-onnx Zipformer (streaming) and Parakeet v3 (offline) models.
set -e

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
ASR_BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
PUNCT_BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# ---------------------------------------------------------------------------
# 1. Streaming model – Zipformer (English, INT8)
# ---------------------------------------------------------------------------
STREAMING_DIR="sherpa-onnx-streaming-zipformer-en-2023-06-26"
if [ ! -d "$STREAMING_DIR" ]; then
  echo "Downloading Zipformer streaming model …"
  curl -LO "${ASR_BASE_URL}/${STREAMING_DIR}.tar.bz2"
  tar -xf "${STREAMING_DIR}.tar.bz2"
  rm "${STREAMING_DIR}.tar.bz2"
  echo "Zipformer → $MODELS_DIR/$STREAMING_DIR"
else
  echo "Zipformer already present – skipping."
fi

# ---------------------------------------------------------------------------
# 2. Offline model – Parakeet TDT 0.6B v3 (English, INT8)
# ---------------------------------------------------------------------------
OFFLINE_DIR="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
if [ ! -d "$OFFLINE_DIR" ]; then
  echo "Downloading Parakeet v3 offline model (≈640 MB) …"
  curl -LO "${ASR_BASE_URL}/${OFFLINE_DIR}.tar.bz2"
  tar -xf "${OFFLINE_DIR}.tar.bz2"
  rm "${OFFLINE_DIR}.tar.bz2"
  echo "Parakeet v3 → $MODELS_DIR/$OFFLINE_DIR"
else
  echo "Parakeet v3 already present – skipping."
fi

# ---------------------------------------------------------------------------
# 3. Silero VAD - INT8
# ---------------------------------------------------------------------------
VAD_FILE="silero_vad.int8.onnx"
if [ ! -f "$VAD_FILE" ]; then
  echo "Downloading Silero VAD model ..."
  curl -L -C - -o "$VAD_FILE" "${ASR_BASE_URL}/${VAD_FILE}"
  echo "Silero VAD -> $MODELS_DIR/$VAD_FILE"
else
  echo "Silero VAD already present - skipping."
fi

# ---------------------------------------------------------------------------
# 4. English punctuation/casing model - INT8
# ---------------------------------------------------------------------------
PUNCT_DIR="sherpa-onnx-online-punct-en-2024-08-06"
if [ ! -d "$PUNCT_DIR" ]; then
  echo "Downloading punctuation model ..."
  curl -L -C - -o "${PUNCT_DIR}.tar.bz2" "${PUNCT_BASE_URL}/${PUNCT_DIR}.tar.bz2"
  tar -xf "${PUNCT_DIR}.tar.bz2"
  rm "${PUNCT_DIR}.tar.bz2"
  echo "Punctuation model -> $MODELS_DIR/$PUNCT_DIR"
else
  echo "Punctuation model already present - skipping."
fi

# ---------------------------------------------------------------------------
# 5. Whisper tiny language-ID model - INT8
# ---------------------------------------------------------------------------
LANG_ID_DIR="sherpa-onnx-whisper-tiny"
if [ ! -d "$LANG_ID_DIR" ]; then
  echo "Downloading language-ID model ..."
  curl -L -C - -o "${LANG_ID_DIR}.tar.bz2" "${ASR_BASE_URL}/${LANG_ID_DIR}.tar.bz2"
  tar -xf "${LANG_ID_DIR}.tar.bz2"
  rm "${LANG_ID_DIR}.tar.bz2"
  echo "Language-ID model -> $MODELS_DIR/$LANG_ID_DIR"
else
  echo "Language-ID model already present - skipping."
fi

echo ""
echo "All models ready in $MODELS_DIR"

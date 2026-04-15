# bin/download_models.ps1
# Downloads Zipformer (streaming) and Parakeet v3 (offline) models using curl.exe

$ModelsDir = Join-Path (Split-Path $PSScriptRoot -Parent) "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Push-Location $ModelsDir

$BaseUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
$PunctuationBaseUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models"

# ---------------------------------------------------------------------------
# 1. Zipformer streaming (English, INT8)
# ---------------------------------------------------------------------------
$StreamingDir = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
if (-not (Test-Path $StreamingDir)) {
    Write-Host "Downloading Zipformer streaming model ..."
    curl.exe -L -C - -o "$StreamingDir.tar.bz2" "$BaseUrl/$StreamingDir.tar.bz2"
    if ($LASTEXITCODE -ne 0) { Write-Error "Zipformer download failed."; exit 1 }
    tar -xf "$StreamingDir.tar.bz2"
    Remove-Item "$StreamingDir.tar.bz2"
    Write-Host "Zipformer ready -> $ModelsDir\$StreamingDir"
} else {
    Write-Host "Zipformer already present - skipping."
}

# ---------------------------------------------------------------------------
# 2. Parakeet TDT v3 offline (English, INT8, ~640 MB)
# ---------------------------------------------------------------------------
$OfflineDir = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
if (-not (Test-Path $OfflineDir)) {
    Write-Host "Downloading Parakeet v3 model (~640 MB) - resume supported ..."
    curl.exe -L -C - -o "$OfflineDir.tar.bz2" "$BaseUrl/$OfflineDir.tar.bz2"
    if ($LASTEXITCODE -ne 0) { Write-Error "Parakeet download failed."; exit 1 }
    tar -xf "$OfflineDir.tar.bz2"
    Remove-Item "$OfflineDir.tar.bz2"
    Write-Host "Parakeet v3 ready -> $ModelsDir\$OfflineDir"
} else {
    Write-Host "Parakeet v3 already present - skipping."
}

# ---------------------------------------------------------------------------
# 3. Silero VAD (INT8, ~208 KB)
# ---------------------------------------------------------------------------
$VadFile = "silero_vad.int8.onnx"
if (-not (Test-Path $VadFile)) {
    Write-Host "Downloading Silero VAD model ..."
    curl.exe -L -C - -o $VadFile "$BaseUrl/$VadFile"
    if ($LASTEXITCODE -ne 0) { Write-Error "Silero VAD download failed."; exit 1 }
    Write-Host "Silero VAD ready -> $ModelsDir\$VadFile"
} else {
    Write-Host "Silero VAD already present - skipping."
}

# ---------------------------------------------------------------------------
# 4. English punctuation/casing model (INT8, ~7 MB)
# ---------------------------------------------------------------------------
$PunctuationDir = "sherpa-onnx-online-punct-en-2024-08-06"
if (-not (Test-Path $PunctuationDir)) {
    Write-Host "Downloading punctuation model ..."
    curl.exe -L -C - -o "$PunctuationDir.tar.bz2" "$PunctuationBaseUrl/$PunctuationDir.tar.bz2"
    if ($LASTEXITCODE -ne 0) { Write-Error "Punctuation download failed."; exit 1 }
    tar -xf "$PunctuationDir.tar.bz2"
    Remove-Item "$PunctuationDir.tar.bz2"
    Write-Host "Punctuation model ready -> $ModelsDir\$PunctuationDir"
} else {
    Write-Host "Punctuation model already present - skipping."
}

# ---------------------------------------------------------------------------
# 5. Whisper tiny language-ID model (INT8, ~98 MB)
# ---------------------------------------------------------------------------
$LanguageIdDir = "sherpa-onnx-whisper-tiny"
if (-not (Test-Path $LanguageIdDir)) {
    Write-Host "Downloading language-ID model ..."
    curl.exe -L -C - -o "$LanguageIdDir.tar.bz2" "$BaseUrl/$LanguageIdDir.tar.bz2"
    if ($LASTEXITCODE -ne 0) { Write-Error "Language-ID download failed."; exit 1 }
    tar -xf "$LanguageIdDir.tar.bz2"
    Remove-Item "$LanguageIdDir.tar.bz2"
    Write-Host "Language-ID model ready -> $ModelsDir\$LanguageIdDir"
} else {
    Write-Host "Language-ID model already present - skipping."
}

Pop-Location
Write-Host ""
Write-Host "All models ready in $ModelsDir"

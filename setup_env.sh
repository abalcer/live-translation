#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh - bootstrap the live-translation environment on macOS
# (Apple Silicon, tested for M4 Pro).
#
# What it does:
#   1. Verifies/installs Homebrew.
#   2. Installs BlackHole 2ch + BlackHole 16ch virtual audio drivers.
#   3. Installs/starts Ollama and pulls the translation model.
#   4. Creates a Python virtualenv and installs requirements.txt.
#   5. Warms up (downloads) the mlx-whisper STT model and Silero VAD.
#
# Usage:
#   ./setup_env.sh
#   OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M ./setup_env.sh   # override model
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$1"; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$1"; exit 1; }

# ---- 0. sanity: Apple Silicon ----------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  die "This script targets macOS. Aborting."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  warn "Expected Apple Silicon (arm64); found $(uname -m)."
  warn "MLX-based STT/TTS (Metal) requires Apple Silicon - continuing anyway."
fi

# ---- 1. Homebrew ------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  log "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  log "Homebrew found: $(brew --version | head -n1)"
fi

# ---- 2. BlackHole virtual audio drivers -------------------------------------
log "Installing BlackHole 2ch and BlackHole 16ch (virtual audio drivers)..."
brew install --cask blackhole-2ch  || warn "blackhole-2ch install skipped/failed (may already be installed)."
brew install --cask blackhole-16ch || warn "blackhole-16ch install skipped/failed (may already be installed)."

cat <<'EOF'

  -------------------------------------------------------------------------
  MANUAL STEP REQUIRED (macOS Audio MIDI Setup):

    1. Open "Audio MIDI Setup" (Spotlight -> "Audio MIDI Setup").
    2. Click "+" -> "Create Multi-Output Device".
       Check both your real output (e.g. "MacBook Pro Speakers") and
       "BlackHole 2ch". This lets Meet's incoming audio also reach
       BlackHole 2ch, which this app captures as `meet_input`.
    3. While on a call, set macOS System Sound Output to that Multi-Output
       Device.
    4. In Google Meet's audio settings, set Microphone = "BlackHole 16ch".
       This app writes the translated + cloned voice there as `meet_output`.

  See the routing diagram at the top of config.yaml for the full picture.
  -------------------------------------------------------------------------

EOF

# ---- 3. Ollama ----------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  log "Installing Ollama..."
  brew install ollama
else
  log "Ollama found: $(ollama --version)"
fi

log "Starting Ollama service if not already running..."
if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
  (brew services start ollama >/dev/null 2>&1) || (nohup ollama serve >/tmp/ollama.log 2>&1 &)
  sleep 3
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
log "Pulling Ollama model: ${OLLAMA_MODEL} (this can take a while on first run)..."
ollama pull "${OLLAMA_MODEL}"

# ---- 4. Python virtualenv ------------------------------------------------------
log "Installing portaudio (required to build the PyAudio wheel)..."
brew install portaudio || warn "portaudio already installed or failed - check manually."

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  warn "${PYTHON_BIN} not found, installing via Homebrew..."
  brew install python@3.11
  PYTHON_BIN="python3.11"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtualenv at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

log "Installing Python dependencies from requirements.txt..."
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---- 5. Warm up / pre-download models -------------------------------------------
mkdir -p "${PROJECT_DIR}/reference" "${PROJECT_DIR}/models" "${PROJECT_DIR}/logs"

log "Pre-downloading mlx-whisper STT model (first run compiles + caches weights)..."
python3 - "${PROJECT_DIR}/config.yaml" <<'PY'
import sys
import numpy as np
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
model = cfg["stt"]["model"]
print(f"Warming up {model} ...")

import mlx_whisper
mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=model)
print("STT model ready.")
PY

log "Pre-downloading Silero VAD model..."
python3 - <<'PY'
from silero_vad import load_silero_vad
load_silero_vad()
print("Silero VAD ready.")
PY

REF_WAV="${PROJECT_DIR}/reference/reference_speaker.wav"
if [[ ! -f "${REF_WAV}" ]]; then
  warn "No reference voice sample found at reference/reference_speaker.wav"
  warn "Record ~10-20s of clean, single-speaker speech (WAV) and place it there"
  warn "before running main.py. Example using ffmpeg + the built-in mic:"
  warn "  ffmpeg -f avfoundation -i \":0\" -t 15 -ac 1 -ar 24000 \"${REF_WAV}\""
fi

log "Setup complete."
log "Activate the environment with:  source .venv/bin/activate"
log "Then run:                        python main.py"

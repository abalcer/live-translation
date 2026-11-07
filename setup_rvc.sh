#!/usr/bin/env bash
# ==============================================================================
# setup_rvc.sh - install the optional RVC voice-conversion backend (Applio).
#
# Applio pins its own torch/numpy/onnxruntime versions, which conflict with the
# main .venv, so it lives in its own checkout + venv under vendor/applio and
# runs as a separate worker process (src/rvc_worker.py), spawned by
# src/voice_converter.py when config.yaml -> tts.voice_conversion.enabled.
#
# What it does:
#   1. Clones Applio at a pinned commit into vendor/applio (skipped if present).
#   2. Creates vendor/applio/.venv and installs Applio's inference deps
#      (requirements-engine.txt - no Gradio web UI).
#   3. Downloads the contentvec embedder and rmvpe pitch predictor (~600 MB).
#
# Your trained model (models/<name>.pth + .index) is not downloaded - train it
# in Applio on the PC and copy both files into models/.
#
# Usage:
#   ./setup_rvc.sh
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLIO_DIR="${PROJECT_DIR}/vendor/applio"
APPLIO_REPO="https://github.com/IAHispano/Applio.git"
APPLIO_COMMIT="6d5d9511b7e719b1f613362bc767e841ad72fa72"   # tested with src/rvc_worker.py
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

log() { printf '\033[1;32m[rvc]\033[0m %s\n' "$1"; }
die() { printf '\033[1;31m[rvc]\033[0m %s\n' "$1"; exit 1; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} not found (brew install python@3.11)."

# ---- 1. Applio checkout -------------------------------------------------------
if [[ ! -d "${APPLIO_DIR}/rvc" ]]; then
  log "Cloning Applio @ ${APPLIO_COMMIT:0:7} into vendor/applio..."
  mkdir -p "${APPLIO_DIR}"
  git -C "${APPLIO_DIR}" init -q
  git -C "${APPLIO_DIR}" remote add origin "${APPLIO_REPO}"
  git -C "${APPLIO_DIR}" fetch -q --depth 1 origin "${APPLIO_COMMIT}"
  git -C "${APPLIO_DIR}" checkout -q FETCH_HEAD
else
  log "vendor/applio already present - skipping clone."
fi

# ---- 2. Applio venv -------------------------------------------------------------
if [[ ! -x "${APPLIO_DIR}/.venv/bin/python" ]]; then
  log "Creating vendor/applio/.venv..."
  "${PYTHON_BIN}" -m venv "${APPLIO_DIR}/.venv"
fi
log "Installing Applio inference dependencies (a few minutes on first run)..."
"${APPLIO_DIR}/.venv/bin/python" -m pip install -q --upgrade pip
"${APPLIO_DIR}/.venv/bin/python" -m pip install -q -r "${APPLIO_DIR}/requirements-engine.txt"

# ---- 3. Embedder + pitch predictor weights ------------------------------------
if [[ -f "${APPLIO_DIR}/rvc/models/predictors/rmvpe.pt" && \
      -f "${APPLIO_DIR}/rvc/models/embedders/contentvec/pytorch_model.bin" ]]; then
  log "contentvec + rmvpe already downloaded."
else
  log "Downloading contentvec + rmvpe..."
  (cd "${APPLIO_DIR}" && .venv/bin/python core.py prerequisites \
      --no-pretraineds-hifigan --models --no-exe)
fi

log "Done. Enable it in config.yaml -> tts.voice_conversion.enabled: true"

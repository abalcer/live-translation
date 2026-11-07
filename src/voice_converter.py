"""
Optional RVC voice conversion stage, applied to the TTS output before
playback (config.yaml `tts.voice_conversion`):

    TTS (e.g. Kokoro, native English pronunciation) -> RVC (your timbre) -> AudioPlayer

RVC does not synthesize speech - it keeps the words, pronunciation and
intonation of its input and swaps only the timbre for the one learned from
your recordings (models/<name>.pth + .index, trained in Applio). Because it
is language-agnostic, the model can be trained on Russian/Ukrainian speech
and still voice English TTS output.

Applio pins torch/numpy versions that conflict with the main .venv, so the
model runs in a child process under Applio's own venv (see setup_rvc.sh,
src/rvc_worker.py and src/rvc_protocol.py). Audio crosses the pipe as raw
float32, which costs a few milliseconds per phrase.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from src import rvc_protocol as proto
from src.utils import resample

logger = logging.getLogger("live-translation.vc")

_WORKER_SCRIPT = Path(__file__).resolve().parent / "rvc_worker.py"
# config.yaml keys forwarded to the worker as --flags; unset keys fall back
# to the worker's own defaults.
_WORKER_OPTIONS = ("pitch", "index_rate", "protect", "volume_envelope", "f0_method", "embedder")


class VoiceConverter:
    def __init__(self, cfg):
        self.vc_cfg = cfg.tts.voice_conversion
        self.sample_rate = cfg.tts.sample_rate
        self._proc: Optional[subprocess.Popen] = None
        self._model_rate = 0
        # The TTS executor already serializes calls; the lock just keeps the
        # request/response pairs intact if that ever changes.
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg) -> Optional["VoiceConverter"]:
        """Return a converter if tts.voice_conversion.enabled, else None."""
        vc_cfg = cfg.tts.get("voice_conversion") or {}
        return cls(cfg) if vc_cfg.get("enabled") else None

    # ---- lifecycle ------------------------------------------------------
    def load(self) -> None:
        """Start the worker and block until it has loaded and warmed up (~15 s)."""
        applio_dir = Path(self.vc_cfg.applio_dir).resolve()
        python = Path(self.vc_cfg.get("python") or applio_dir / ".venv" / "bin" / "python")
        model = Path(self.vc_cfg.model).resolve()
        index = Path(self.vc_cfg.index).resolve() if self.vc_cfg.get("index") else None

        if not (applio_dir / "rvc").is_dir() or not python.exists():
            raise RuntimeError(
                f"tts.voice_conversion is enabled but Applio is not installed in {applio_dir}. "
                "Run ./setup_rvc.sh first."
            )
        if not model.exists():
            raise FileNotFoundError(f"RVC model not found: {model}")
        if index is not None and not index.exists():
            logger.warning("RVC index not found (%s) - converting without it.", index)
            index = None

        cmd = [str(python), str(_WORKER_SCRIPT), "--model", str(model)]
        if index is not None:
            cmd += ["--index", str(index)]
        for key in _WORKER_OPTIONS:
            if self.vc_cfg.get(key) is not None:
                cmd += [f"--{key.replace('_', '-')}", str(self.vc_cfg[key])]

        logger.info("Starting RVC worker (model=%s, index_rate=%s)...",
                    model.name, self.vc_cfg.get("index_rate", "default"))
        # stderr is inherited so Applio's messages and tracebacks show up in
        # the console; cwd must be the Applio checkout (relative weight paths).
        self._proc = subprocess.Popen(
            cmd, cwd=applio_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        try:
            self._model_rate = proto.read_uint(self._proc.stdout)
        except EOFError:
            code = self._proc.wait()
            self._proc = None
            raise RuntimeError(f"RVC worker exited during startup (code {code}); see the log above.")
        logger.info("RVC ready on worker pid %d (model rate %d Hz).", self._proc.pid, self._model_rate)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()  # EOF -> worker exits its loop
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # ---- conversion -----------------------------------------------------
    def convert(self, audio: np.ndarray) -> np.ndarray:
        """Convert mono float32 audio at tts.sample_rate; returns the same rate."""
        if not audio.size:
            return audio
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("RVC worker is not running.")

        with self._lock:
            proto.write_audio(self._proc.stdin, resample(audio, self.sample_rate, proto.INPUT_SR))
            out = proto.read_audio(self._proc.stdout)
        if out is None:
            raise RuntimeError("RVC conversion failed; see the worker traceback above.")
        return resample(out, self._model_rate, self.sample_rate)

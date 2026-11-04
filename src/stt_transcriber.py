"""
mlx-whisper based speech-to-text, tuned for short utterances on Apple Silicon
(Metal backend via MLX). Stateless per call by design - it is meant to run
inside a thread-pool executor so the asyncio pipeline never blocks on it.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("live-translation.stt")


class Transcriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_repo = cfg.stt.model
        self._mlx_whisper = None

    def load(self) -> None:
        """Import + warm up the model once at startup (first call compiles/loads weights)."""
        import mlx_whisper
        self._mlx_whisper = mlx_whisper
        dummy = np.zeros(16000, dtype=np.float32)  # 1s of silence
        self.transcribe(dummy)
        logger.info("mlx-whisper ready: %s", self.model_repo)

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        """Returns (text, detected_language_code)."""
        if self._mlx_whisper is None:
            raise RuntimeError("Transcriber.load() must be called before transcribe()")

        kwargs = dict(
            path_or_hf_repo=self.model_repo,
            temperature=self.cfg.stt.temperature,
            condition_on_previous_text=self.cfg.stt.condition_on_previous_text,
            fp16=True,
            verbose=False,
        )
        if not self.cfg.stt.language_auto_detect:
            kwargs["language"] = self.cfg.stt.languages[0]

        result = self._mlx_whisper.transcribe(np.asarray(audio, dtype=np.float32), **kwargs)
        text = (result.get("text") or "").strip()
        lang = result.get("language") or self.cfg.stt.languages[0]
        return text, lang

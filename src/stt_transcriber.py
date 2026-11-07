"""
mlx-whisper based speech-to-text, tuned for short utterances on Apple Silicon
(Metal backend via MLX). Stateless per call by design - it is meant to run
inside a thread-pool executor so the asyncio pipeline never blocks on it.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("live-translation.stt")

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


class Transcriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_repo = cfg.stt.model
        self._mlx_whisper = None
        self.accept_languages = set(cfg.stt.get("accept_languages") or [])
        # "phrase *" drops anything starting with the phrase (subtitle credits
        # followed by a name); plain entries drop only an exact match.
        phrases = cfg.stt.get("drop_phrases") or []
        self.drop_exact = {_normalize(p) for p in phrases if not p.rstrip().endswith("*")}
        self.drop_prefixes = [_normalize(p.rstrip()[:-1]) for p in phrases if p.rstrip().endswith("*")]

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

    def reject_reason(self, text: str, lang: str) -> Optional[str]:
        """
        Why a transcript should not be translated, or None to keep it.

        whisper-large-v3-turbo hallucinates on silence/noise ("Thank you.",
        "Продолжение следует...", subtitle credits) with no_speech_prob ~0 and
        a normal avg_logprob, so its own confidence scores cannot catch it.
        Noise is almost always detected as "en", hence the language filter.
        """
        norm = _normalize(text)
        if not norm:
            return "no words"
        if self.accept_languages and lang not in self.accept_languages:
            return f"language {lang!r} not in stt.accept_languages"
        if norm in self.drop_exact:
            return "known Whisper hallucination"
        for prefix in self.drop_prefixes:
            if norm == prefix or norm.startswith(prefix + " "):
                return f"known Whisper hallucination ({prefix!r} ...)"
        return None

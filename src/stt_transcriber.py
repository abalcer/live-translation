"""
Whisper speech-to-text, tuned for short utterances. Two backends, selected by
config.yaml `stt.engine` (see create_transcriber()):

  - "mlx_whisper"    : mlx-whisper on Apple Silicon (Metal via MLX).
  - "faster_whisper" : faster-whisper (CTranslate2) on an NVIDIA GPU - used by
                       inference_server.py on the PC.

Stateless per call by design - it is meant to run inside a thread-pool
executor so the asyncio pipeline never blocks on it.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("live-translation.stt")

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def create_transcriber(cfg) -> "Transcriber":
    """Transcriber for cfg.stt.engine ("remote" is handled by src.remote_client)."""
    engine = cfg.stt.get("engine", "mlx_whisper")
    if engine == "mlx_whisper":
        return Transcriber(cfg)
    if engine == "faster_whisper":
        return FasterWhisperTranscriber(cfg)
    raise ValueError(f"Unknown stt.engine in config.yaml: {engine!r}")


class Transcriber:
    """mlx-whisper backend; also holds the language retry and drop filters."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_repo = cfg.stt.model
        self._mlx_whisper = None
        self.accept_languages = set(cfg.stt.get("accept_languages") or [])
        self.min_accept_prob = float(cfg.stt.get("min_accept_prob", 0.1))
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
        self._run(dummy, None)
        self._language_probs(dummy)
        logger.info("mlx-whisper ready: %s", self.model_repo)

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        """Returns (text, language_code)."""
        if not self._loaded():
            raise RuntimeError("Transcriber.load() must be called before transcribe()")

        audio = np.asarray(audio, dtype=np.float32)
        forced = None if self.cfg.stt.language_auto_detect else self.cfg.stt.languages[0]
        text, lang = self._run(audio, forced)
        if forced is None and self.accept_languages and lang not in self.accept_languages:
            # Speech played through a speaker or with a bad mic is sometimes
            # detected as another language ("Магазин не работал" -> cs, a quiet
            # phrase -> en with p(ru) = 0.18) and would be dropped. Retry it in
            # the most likely accepted language unless that is implausible
            # (English echo: p(ru) = 0.00). Noise that slips through mostly
            # becomes a known hallucination and is dropped by reject_reason().
            probs = self._language_probs(audio)
            best = max(self.accept_languages, key=lambda code: probs.get(code, 0.0))
            p_best = probs.get(best, 0.0)
            if p_best >= self.min_accept_prob:
                logger.info("re-transcribing as %s: detected %s, p(%s)=%.2f", best, lang, best, p_best)
                text, lang = self._run(audio, best)
            else:
                logger.debug("detected %s, best accepted %s p=%.2f < %.2f",
                             lang, best, p_best, self.min_accept_prob)
        return text, lang

    def _loaded(self) -> bool:
        return self._mlx_whisper is not None

    def _run(self, audio: np.ndarray, language: Optional[str]) -> Tuple[str, str]:
        kwargs = dict(
            path_or_hf_repo=self.model_repo,
            temperature=self.cfg.stt.temperature,
            condition_on_previous_text=self.cfg.stt.condition_on_previous_text,
            fp16=True,
            verbose=None,  # None = no "Detected language" line / progress bar
        )
        if language:
            kwargs["language"] = language
        result = self._mlx_whisper.transcribe(audio, **kwargs)
        text = (result.get("text") or "").strip()
        return text, result.get("language") or language or self.cfg.stt.languages[0]

    def _language_probs(self, audio: np.ndarray) -> Dict[str, float]:
        """Whisper's language distribution for the first 30 s (one encoder pass, ~0.4 s)."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self.model_repo, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(segment)
        return probs

    def reject_reason(self, text: str, lang: str) -> Optional[str]:
        """
        Why a transcript should not be translated, or None to keep it.

        whisper-large-v3-turbo hallucinates on silence/noise ("Thank you.",
        "Продолжение следует...", subtitle credits) with no_speech_prob ~0 and
        a normal avg_logprob, so its own confidence scores cannot catch it.
        Noise is almost always detected as "en", hence the language filter
        (transcribe() has already retried plausible ru/uk speech).
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


class FasterWhisperTranscriber(Transcriber):
    """
    faster-whisper (CTranslate2) backend for an NVIDIA GPU. stt.model is a
    faster-whisper model name ("large-v3-turbo") or a CTranslate2 repo/path;
    stt.device / stt.compute_type pick the GPU and precision.

    A detection pass already returns the whole language distribution, so the
    retry in transcribe() reuses it instead of running the encoder again.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None
        self._last_probs: Dict[str, float] = {}

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device = self.cfg.stt.get("device", "cuda")
        compute_type = self.cfg.stt.get("compute_type", "float16")
        self._model = WhisperModel(self.model_repo, device=device, compute_type=compute_type)
        self._run(np.zeros(16000, dtype=np.float32), None)  # 1s of silence: CUDA kernels, cuDNN
        logger.info("faster-whisper ready: %s on %s (%s)", self.model_repo, device, compute_type)

    def _loaded(self) -> bool:
        return self._model is not None

    def _run(self, audio: np.ndarray, language: Optional[str]) -> Tuple[str, str]:
        segments, info = self._model.transcribe(
            audio,
            language=language,
            temperature=self.cfg.stt.temperature,
            condition_on_previous_text=self.cfg.stt.condition_on_previous_text,
            beam_size=1,                # greedy, like mlx-whisper's default
            without_timestamps=True,
            vad_filter=False,           # utterances are already cut by Silero VAD
        )
        text = "".join(seg.text for seg in segments).strip()
        if language is None:
            self._last_probs = dict(info.all_language_probs or [])
        return text, info.language or language or self.cfg.stt.languages[0]

    def _language_probs(self, audio: np.ndarray) -> Dict[str, float]:
        # Called by transcribe() right after an auto-detect _run() on the same audio.
        return self._last_probs

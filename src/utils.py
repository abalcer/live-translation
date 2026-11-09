"""
Shared utilities: YAML config loading (with dotted attribute access), console
logging setup, per-utterance latency tracking, and audio resampling.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

import yaml


class DotDict(dict):
    """A dict that also allows attribute-style access, recursively.

    Kept intentionally simple (no schema/validation) - config.yaml is the
    single source of truth and is small enough to eyeball.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: Union[str, Path] = "config.yaml") -> DotDict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers so pipeline logs stay readable.
    for noisy in ("urllib3", "httpx", "httpx2", "huggingface_hub", "numba", "TTS", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("live-translation")


class LatencyTracker:
    """
    Tracks wall-clock timestamps for the stages a single utterance passes
    through and renders a compact one-line latency report, e.g.:

        [outgoing_mic_to_meet#17] VAD  612ms | STT  171ms | LLM   94ms | TTS  308ms | TOTAL  573ms | QUEUE  0.0s | "Hi, how are you?"

    Timeline recorded:
        created -> vad -> stt -> llm -> tts [-> vc]

    - "vc" is only stamped when RVC voice conversion is enabled
      (tts.voice_conversion); the VC column and TOTAL then include it.

    - "created" is stamped when speech onset is first detected by the VAD.
    - "vad"     is stamped when the VAD confirms end-of-speech (utterance ready).
      So VAD_ms reflects the *inherent* detection delay (utterance length +
      trailing-silence hangover needed to confirm you stopped talking) - not
      a bug, it's genuinely part of what the user waits for.
    - TOTAL_ms is measured from "vad" (utterance ready) to "tts" (audio ready
      to play), i.e. the pure system processing latency after you stop talking.
    """

    def __init__(self, route: str, utterance_id: int):
        self.route = route
        self.utterance_id = utterance_id
        self._t: Dict[str, float] = {"created": time.perf_counter()}

    def mark(self, stage: str) -> None:
        self._t[stage] = time.perf_counter()

    def mark_at(self, stage: str, t: float) -> None:
        """Stamp a stage at an earlier perf_counter() time (stages timed on a remote server)."""
        self._t[stage] = t

    def _delta_ms(self, a: str, b: str) -> float:
        if a not in self._t or b not in self._t:
            return float("nan")
        return (self._t[b] - self._t[a]) * 1000.0

    def report(self, extra: str = "") -> str:
        vad_ms = self._delta_ms("created", "vad")
        stt_ms = self._delta_ms("vad", "stt")
        llm_ms = self._delta_ms("stt", "llm")
        tts_ms = self._delta_ms("llm", "tts")
        last = "vc" if "vc" in self._t else "tts"
        total_ms = self._delta_ms("vad", last)
        parts = [
            f"VAD {vad_ms:6.0f}ms",
            f"STT {stt_ms:6.0f}ms",
            f"LLM {llm_ms:6.0f}ms",
            f"TTS {tts_ms:6.0f}ms",
        ]
        if last == "vc":
            parts.append(f"VC {self._delta_ms('tts', 'vc'):6.0f}ms")
        parts.append(f"TOTAL {total_ms:6.0f}ms")
        line = f"[{self.route}#{self.utterance_id}] " + " | ".join(parts)
        if extra:
            line += f" | {extra}"
        return line


def resample(audio, orig_sr: int, target_sr: int):
    """Rational-ratio resampling via scipy - good quality, cheap enough for short utterances."""
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly

    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    out = resample_poly(audio, up, down)
    return out.astype(np.float32, copy=False)

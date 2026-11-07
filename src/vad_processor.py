"""
Streaming Silero VAD wrapper: consumes fixed-size raw audio blocks and emits
complete speech utterances (onset -> confirmed silence) ready for STT.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np
import torch

from src.utils import LatencyTracker

logger = logging.getLogger("live-translation.vad")


def speech_level_db(audio: np.ndarray, sampling_rate: int) -> float:
    """
    Loudness of the speech part of a segment, in dBFS: the 90th percentile of
    30 ms frame RMS, so the VAD's silence padding does not drag it down.
    """
    frame = max(1, int(0.03 * sampling_rate))
    n = len(audio) // frame
    if n == 0:
        rms = np.sqrt(np.mean(np.square(audio))) if len(audio) else 0.0
    else:
        frames = audio[: n * frame].reshape(n, frame)
        rms = np.percentile(np.sqrt(np.mean(np.square(frames), axis=1)), 90)
    return float(20 * np.log10(rms + 1e-10))


@dataclass
class Utterance:
    route: str
    audio: np.ndarray          # float32 mono @ vad.sampling_rate
    utterance_id: int
    tracker: LatencyTracker
    t_started: float = field(default_factory=time.perf_counter)


class VADProcessor:
    """
    Wraps Silero VAD's streaming VADIterator. Feed it raw capture blocks of
    ANY size via `process_block`; internally re-chunks to the VAD's required
    window size and accumulates samples between speech-start / speech-end
    events into complete utterances, which are pushed to `utterance_queue`.
    """

    def __init__(self, route: str, cfg, utterance_queue: "asyncio.Queue[Utterance]",
                 loop: asyncio.AbstractEventLoop):
        self.route = route
        self.cfg = cfg
        self.utterance_queue = utterance_queue
        self.loop = loop
        self.sampling_rate = cfg.vad.sampling_rate
        self.window = cfg.vad.window_size_samples
        self.max_speech_samples = int(cfg.vad.max_speech_duration_s * self.sampling_rate)
        # Gates against Silero firing on clicks, breaths and background noise:
        # whisper-large-v3-turbo "hears" e.g. "Thank you." in near-silence and
        # reports no_speech_prob ~0, so this has to be stopped before STT.
        self.min_speech_samples = int(cfg.vad.get("min_speech_duration_ms", 0) * self.sampling_rate / 1000)
        self.min_rms_db = cfg.vad.get("min_rms_db")
        # Silero fires a little after the real onset - noticeably so for speech
        # played through a speaker, where the first word ("Сигнальный звонок",
        # "К правому") was lost. Keep the last pre_roll_ms of non-speech audio
        # and prepend it on "start". It only holds audio received after the
        # previous utterance ended, so nothing is transcribed twice.
        pre_roll_ms = cfg.vad.get("pre_roll_ms", 0)
        self._pre_roll: Deque[np.ndarray] = deque(
            maxlen=max(1, int(pre_roll_ms * self.sampling_rate / 1000 / self.window)) if pre_roll_ms else 0
        )

        self._model = None
        self._iterator = None
        self._pending = np.zeros(0, dtype=np.float32)  # leftover < window samples
        self._speech_buf: List[np.ndarray] = []
        self._in_speech = False
        self._speech_start_t: Optional[float] = None
        self._utterance_counter = 0

    def load(self) -> None:
        from silero_vad import VADIterator, load_silero_vad
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            sampling_rate=self.sampling_rate,
            threshold=self.cfg.vad.threshold,
            min_silence_duration_ms=self.cfg.vad.min_silence_duration_ms,
            speech_pad_ms=self.cfg.vad.speech_pad_ms,
        )
        logger.info("[%s] Silero VAD loaded", self.route)

    def _reject_reason(self, audio: np.ndarray) -> Optional[str]:
        """Why a VAD segment is not worth transcribing, or None to keep it."""
        if len(audio) < self.min_speech_samples:
            return (f"too short ({len(audio) * 1000 // self.sampling_rate} ms "
                    f"< {self.cfg.vad.min_speech_duration_ms} ms)")
        if self.min_rms_db is not None:
            level = speech_level_db(audio, self.sampling_rate)
            if level < self.min_rms_db:
                return f"too quiet ({level:.0f} dB < {self.min_rms_db} dB)"
        return None

    def _emit(self) -> None:
        if not self._speech_buf:
            return
        audio = np.concatenate(self._speech_buf)
        self._speech_buf = []
        self._pre_roll.clear()
        reason = self._reject_reason(audio)
        if reason:
            logger.info("[%s] dropped VAD segment: %s", self.route, reason)
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[%s] VAD segment kept: %d ms, %.0f dB", self.route,
                         len(audio) * 1000 // self.sampling_rate, speech_level_db(audio, self.sampling_rate))
        self._utterance_counter += 1

        tracker = LatencyTracker(self.route, self._utterance_counter)
        if self._speech_start_t is not None:
            tracker._t["created"] = self._speech_start_t  # anchor to true speech onset
        tracker.mark("vad")  # utterance confirmed complete

        utt = Utterance(route=self.route, audio=audio, utterance_id=self._utterance_counter, tracker=tracker)
        try:
            self.utterance_queue.put_nowait(utt)
        except asyncio.QueueFull:
            logger.warning("[%s] utterance queue full, dropping utterance #%d", self.route, utt.utterance_id)

    def process_block(self, block: np.ndarray) -> None:
        """Called from the asyncio consumer loop with raw float32 mono samples."""
        data = np.concatenate([self._pending, block])
        n_full = (len(data) // self.window) * self.window
        usable, self._pending = data[:n_full], data[n_full:]

        for start in range(0, n_full, self.window):
            chunk = usable[start:start + self.window]
            tensor = torch.from_numpy(chunk)
            event = self._iterator(tensor, return_seconds=False)

            if event is not None and "start" in event:
                self._in_speech = True
                self._speech_start_t = time.perf_counter()
                self._speech_buf = [*self._pre_roll, chunk]
                self._pre_roll.clear()
            elif event is not None and "end" in event:
                if self._in_speech:
                    self._speech_buf.append(chunk)
                self._in_speech = False
                self._emit()
            elif self._in_speech:
                self._speech_buf.append(chunk)
            else:
                self._pre_roll.append(chunk)

            if self._in_speech and sum(len(c) for c in self._speech_buf) >= self.max_speech_samples:
                logger.debug("[%s] max speech duration reached, force-cutting utterance", self.route)
                self._in_speech = False
                self._iterator.reset_states()
                self._emit()

    async def run(self, frame_queue: "asyncio.Queue[np.ndarray]") -> None:
        self.load()
        while True:
            block = await frame_queue.get()
            self.process_block(block)

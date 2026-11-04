"""
Streaming Silero VAD wrapper: consumes fixed-size raw audio blocks and emits
complete speech utterances (onset -> confirmed silence) ready for STT.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from src.utils import LatencyTracker

logger = logging.getLogger("live-translation.vad")


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

    def _emit(self) -> None:
        if not self._speech_buf:
            return
        audio = np.concatenate(self._speech_buf)
        self._speech_buf = []
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
                self._speech_buf = [chunk]
            elif event is not None and "end" in event:
                if self._in_speech:
                    self._speech_buf.append(chunk)
                self._in_speech = False
                self._emit()
            elif self._in_speech:
                self._speech_buf.append(chunk)

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

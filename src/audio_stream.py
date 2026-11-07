"""
Audio I/O layer: device discovery plus threaded capture/playback around
sounddevice (PortAudio), bridging blocking audio-thread callbacks into the
asyncio pipeline via a playback queue / thread-safe queue hand-off.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Deque, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("live-translation.audio")


def list_devices() -> None:
    print(sd.query_devices())


def find_device_index(name_substr: str, kind: str = "input") -> int:
    """
    Resolve a (partial, case-insensitive) device name to a PortAudio index.
    `kind` is "input" or "output" and gates on the relevant channel count.
    """
    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        i for i, d in enumerate(devices)
        if name_substr.lower() in d["name"].lower() and d[channel_key] > 0
    ]
    if not matches:
        raise RuntimeError(
            f"No {kind} device matching '{name_substr}' found.\n"
            f"Available devices:\n{sd.query_devices()}"
        )
    return matches[0]


class PlaybackQueue:
    """Thread-safe, unbounded FIFO of float32 mono chunks for TTS playback.

    The pipeline appends whole synthesized phrases; the realtime audio
    callback reads fixed-size blocks across chunk boundaries. Reads past the
    end return silence (zero-padding) instead of blocking or raising, which is
    exactly what you want from a realtime audio callback.

    Unbounded on purpose: the previous 8 s ring buffer cut the start off any
    translation longer than that and overwrote unplayed audio once the
    backlog exceeded it. Phrases must play in full and in order.
    """

    def __init__(self):
        self._chunks: Deque[np.ndarray] = deque()
        self._offset = 0      # samples of _chunks[0] already played
        self._queued = 0      # samples not yet played
        self._lock = threading.Lock()

    @property
    def queued_samples(self) -> int:
        with self._lock:
            return self._queued

    def write(self, data: np.ndarray) -> None:
        if not len(data):
            return
        with self._lock:
            self._chunks.append(np.array(data, dtype=np.float32, copy=True))
            self._queued += len(data)

    def clear(self) -> None:
        """Drop all buffered audio (used to cut off playback on a barge-in)."""
        with self._lock:
            self._chunks.clear()
            self._offset = 0
            self._queued = 0

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                chunk = self._chunks[0]
                take = min(n - filled, len(chunk) - self._offset)
                out[filled:filled + take] = chunk[self._offset:self._offset + take]
                filled += take
                self._offset += take
                if self._offset == len(chunk):
                    self._chunks.popleft()
                    self._offset = 0
            self._queued -= filled
        return out  # zero-padded (silence) on underrun


class AudioCapture:
    """
    Opens a PortAudio InputStream on `device_name` and pushes fixed-size
    float32 mono blocks onto an asyncio.Queue for the VAD stage to consume.

    The PortAudio callback runs on its own realtime thread, so hand-off to
    the asyncio queue goes through `loop.call_soon_threadsafe`.
    """

    def __init__(self, device_name: str, sample_rate: int, blocksize: int,
                 loop: asyncio.AbstractEventLoop, out_queue: "asyncio.Queue[np.ndarray]",
                 channels: int = 1):
        self.device_index = find_device_index(device_name, "input")
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self._loop = loop
        self._queue = out_queue
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("capture status: %s", status)
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, mono)
        except RuntimeError:
            pass  # event loop already closing

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info("capture started: device #%d @ %dHz (block=%d)",
                    self.device_index, self.sample_rate, self.blocksize)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class AudioPlayer:
    """
    Opens a PortAudio OutputStream on `device_name` and streams from a
    PlaybackQueue so synthesized TTS chunks can be pushed in without blocking
    (or being blocked by) realtime playback.
    """

    def __init__(self, device_name: str, sample_rate: int, blocksize: int,
                 channels: int = 1):
        self.device_index = find_device_index(device_name, "output")
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.queue = PlaybackQueue()
        self._stream: Optional[sd.OutputStream] = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            logger.debug("playback status: %s", status)
        chunk = self.queue.read(frames)
        if self.channels == 1:
            outdata[:, 0] = chunk
        else:
            for c in range(self.channels):
                outdata[:, c] = chunk

    def start(self) -> None:
        self._stream = sd.OutputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info("playback started: device #%d @ %dHz (block=%d)",
                    self.device_index, self.sample_rate, self.blocksize)

    def play(self, audio: np.ndarray) -> None:
        """Enqueue synthesized audio (float32 mono, already at self.sample_rate)."""
        self.queue.write(audio)

    @property
    def queued_seconds(self) -> float:
        """Audio still waiting to be played (what a new phrase would wait behind)."""
        return self.queue.queued_samples / self.sample_rate

    def clear(self) -> None:
        """Discard everything still queued for playback (barge-in / interruption)."""
        self.queue.clear()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

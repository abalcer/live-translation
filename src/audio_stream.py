"""
Audio I/O layer: device discovery plus threaded capture/playback around
sounddevice (PortAudio), bridging blocking audio-thread callbacks into the
asyncio pipeline via a ring buffer / thread-safe queue hand-off.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

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


class RingBuffer:
    """Thread-safe float32 mono ring buffer used to smooth TTS playback.

    The audio-thread callback reads from it at a fixed cadence; the pipeline
    writes synthesized chunks into it whenever they become ready. Reads past
    the write position return silence (zero-padding) instead of blocking or
    raising, which is exactly what you want from a realtime audio callback.
    """

    def __init__(self, capacity_samples: int):
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write = 0
        self._read = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        with self._lock:
            n = len(data)
            if n > self._capacity:
                data = data[-self._capacity:]
                n = len(data)
            end = self._write + n
            if end <= self._capacity:
                self._buf[self._write:end] = data
            else:
                first = self._capacity - self._write
                self._buf[self._write:] = data[:first]
                self._buf[: end - self._capacity] = data[first:]
            self._write = end % self._capacity
            self._filled = min(self._capacity, self._filled + n)

    def clear(self) -> None:
        """Drop all buffered audio (used to cut off playback on a barge-in)."""
        with self._lock:
            self._write = 0
            self._read = 0
            self._filled = 0

    def read(self, n: int) -> np.ndarray:
        with self._lock:
            avail = min(n, self._filled)
            out = np.zeros(n, dtype=np.float32)
            if avail > 0:
                end = self._read + avail
                if end <= self._capacity:
                    out[:avail] = self._buf[self._read:end]
                else:
                    first = self._capacity - self._read
                    out[:first] = self._buf[self._read:]
                    out[first:avail] = self._buf[: end - self._capacity]
                self._read = end % self._capacity
                self._filled -= avail
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
    RingBuffer so synthesized TTS chunks can be pushed in without blocking
    (or being blocked by) realtime playback.
    """

    def __init__(self, device_name: str, sample_rate: int, blocksize: int,
                 channels: int = 1, buffer_seconds: float = 8.0):
        self.device_index = find_device_index(device_name, "output")
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.ring = RingBuffer(int(sample_rate * buffer_seconds))
        self._stream: Optional[sd.OutputStream] = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            logger.debug("playback status: %s", status)
        chunk = self.ring.read(frames)
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
        self.ring.write(np.asarray(audio, dtype=np.float32))

    def clear(self) -> None:
        """Discard everything still queued for playback (barge-in / interruption)."""
        self.ring.clear()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

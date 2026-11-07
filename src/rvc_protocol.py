"""
Wire protocol between src/voice_converter.py (main .venv) and
src/rvc_worker.py (Applio's venv). Depends only on the stdlib and numpy,
because the worker imports it too (as a top-level module from its own
directory, while the main process imports it as `src.rvc_protocol`).

All integers are little-endian uint32, audio is mono float32:
    parent -> worker : n, then n samples at INPUT_SR
    worker -> parent : n, then n samples at the model rate,
                       or n == ERROR if the conversion failed
    worker startup   : a single uint32 = model sample rate ("ready")
    EOF on the worker's stdin shuts it down.
"""
from __future__ import annotations

import struct
from typing import BinaryIO, Optional

import numpy as np

ERROR = 0xFFFFFFFF
INPUT_SR = 16000  # contentvec and rmvpe run at 16 kHz


def read_exact(stream: BinaryIO, n: int) -> bytes:
    """Read exactly n bytes; raise EOFError if the other side closed the pipe."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf.extend(chunk)
    return bytes(buf)


def read_uint(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def write_uint(stream: BinaryIO, value: int) -> None:
    stream.write(struct.pack("<I", value))
    stream.flush()


def write_audio(stream: BinaryIO, audio: np.ndarray) -> None:
    data = np.ascontiguousarray(audio, dtype="<f4")
    stream.write(struct.pack("<I", data.size) + data.tobytes())
    stream.flush()


def read_audio(stream: BinaryIO) -> Optional[np.ndarray]:
    """Return the next audio block, or None if the sender reported ERROR."""
    n = read_uint(stream)
    if n == ERROR:
        return None
    return np.frombuffer(read_exact(stream, n * 4), dtype="<f4")

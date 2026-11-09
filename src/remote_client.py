"""
Client side of inference_server.py: STT and TTS (+ RVC) run on the PC's GPU
while this machine only captures, cuts utterances (VAD) and plays the result.
Enabled per stage by config.yaml `remote.stt` / `remote.tts`.

Wire format (HTTP/1.1 keep-alive, one request per utterance):

    GET  /health -> JSON {"stt": ..., "tts": ..., "vc": bool, "sample_rate": int}
    POST /stt    body: mono float32 LE samples at 16 kHz
                 -> JSON {"text", "lang", "server_ms"}
    POST /tts    body: JSON {"text", "lang", "speed_factor",
                             "voice_reference", "voice_reference_text"}
                 -> mono float32 LE samples at tts.sample_rate;
                    headers X-TTS-Ms, X-VC-Ms (only if RVC ran)
    Errors: non-2xx status with a JSON {"error": "..."} body.

A phrase of a few seconds is ~100-200 KB each way, so on a wired LAN the
transfer adds only a few milliseconds per stage.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np
import requests

from src.stt_transcriber import Transcriber

logger = logging.getLogger("live-translation.remote")


class RemoteError(RuntimeError):
    pass


class RemoteSession:
    """Shared HTTP session to inference_server.py (one per process)."""

    def __init__(self, cfg):
        rcfg = cfg.remote
        self.url = rcfg.url.rstrip("/")
        self.timeout = float(rcfg.get("timeout_s", 30))
        # requests.Session is not guaranteed thread-safe; STT and TTS run in
        # different executor threads, so each thread gets its own connection.
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
        return s

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        try:
            resp = self._session().request(method, self.url + path, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise RemoteError(f"inference server {self.url} unreachable: {exc}") from exc
        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            raise RemoteError(f"{method} {path} -> HTTP {resp.status_code}: {detail}")
        return resp

    def health(self) -> dict:
        return self.request("GET", "/health").json()


class RemoteTranscriber(Transcriber):
    """
    Drop-in Transcriber whose transcribe() runs on the server (which also does
    the ru/uk retry); reject_reason() still filters locally with this
    machine's config.
    """

    def __init__(self, cfg, session: RemoteSession):
        super().__init__(cfg)
        self.session = session
        self._ready = False

    def load(self) -> None:
        info = self.session.health()
        if not info.get("stt"):
            raise RemoteError(f"inference server {self.session.url} has no STT loaded")
        self._ready = True
        logger.info("Remote STT ready: %s at %s", info["stt"], self.session.url)

    def _loaded(self) -> bool:
        return self._ready

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        body = np.ascontiguousarray(audio, dtype="<f4").tobytes()
        t0 = time.perf_counter()
        data = self.session.request(
            "POST", "/stt", data=body, headers={"Content-Type": "application/octet-stream"},
        ).json()
        logger.debug("remote STT: server %.0f ms, round trip %.0f ms",
                     data.get("server_ms", float("nan")), (time.perf_counter() - t0) * 1000)
        return data.get("text", ""), data.get("lang", "")


class RemoteVoice:
    """TTS + optional RVC on the server, returned as one audio block."""

    def __init__(self, cfg, session: RemoteSession):
        self.sample_rate = cfg.tts.sample_rate
        self.session = session

    def load(self) -> None:
        info = self.session.health()
        if not info.get("tts"):
            raise RemoteError(f"inference server {self.session.url} has no TTS loaded")
        if info.get("sample_rate") != self.sample_rate:
            raise RemoteError(
                f"tts.sample_rate mismatch: server {info.get('sample_rate')} Hz, "
                f"local {self.sample_rate} Hz (AudioPlayer is opened at the local rate)"
            )
        logger.info("Remote TTS ready: %s%s at %s", info["tts"],
                    " + RVC" if info.get("vc") else "", self.session.url)

    def synthesize(self, text: str, lang: str, reference_wav: str, reference_text: str = "",
                   speed_factor: float = 1.0) -> Tuple[np.ndarray, Optional[float]]:
        """
        Returns (audio at tts.sample_rate, seconds the server spent in RVC or
        None if RVC did not run) - the caller uses the latter to split the
        TTS and VC columns of the latency log.
        """
        t0 = time.perf_counter()
        resp = self.session.request("POST", "/tts", json={
            "text": text,
            "lang": lang,
            "speed_factor": speed_factor,
            "voice_reference": reference_wav,
            "voice_reference_text": reference_text,
        })
        audio = np.frombuffer(resp.content, dtype="<f4").astype(np.float32)
        vc_ms = resp.headers.get("X-VC-Ms")
        logger.debug("remote TTS: server tts %s ms, vc %s ms, round trip %.0f ms",
                     resp.headers.get("X-TTS-Ms"), vc_ms, (time.perf_counter() - t0) * 1000)
        return audio, (float(vc_ms) / 1000.0 if vc_ms is not None else None)

"""
Client side of inference_server.py: the models run on the PC's GPU while this
machine only captures, cuts utterances (VAD) and plays the result.

  - config.yaml `remote.pipeline`: STT -> translation -> TTS -> RVC in a single
    request per utterance (RemotePipeline).
  - `remote.stt` / `remote.tts`: single stages (RemoteTranscriber,
    RemoteVoice); translation then goes from this machine to Ollama.

Wire format (HTTP/1.1 keep-alive, one request per utterance):

    GET  /health -> JSON {"stt": ..., "tts": ..., "translate": ..., "vc": bool,
                          "sample_rate": int}
    POST /translate?speak=1&speed=1.0&voice_reference=...&voice_reference_text=...
                 body: mono float32 LE samples at 16 kHz
                 -> uint32 LE n, n bytes of UTF-8 JSON
                      {"text", "lang", "dropped" (reason or null),
                       "translation", "tgt_lang",
                       "stt_ms", "llm_ms", "tts_ms", "vc_ms" (null if no RVC)},
                    then mono float32 LE samples at tts.sample_rate
                    (none if dropped, empty translation or speak=0)
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

import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
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


@dataclass
class PipelineResult:
    """One utterance through POST /translate."""
    text: str
    lang: str
    dropped: Optional[str]       # why STT output was not translated, or None
    translation: str
    tgt_lang: str
    stt_ms: float
    llm_ms: float
    tts_ms: float
    vc_ms: Optional[float]       # None if RVC did not run
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class RemotePipeline:
    """STT -> translation -> TTS -> RVC on the server in one request per utterance."""

    def __init__(self, cfg, session: RemoteSession):
        self.sample_rate = cfg.tts.sample_rate
        self.session = session

    def load(self, speaking: bool) -> None:
        info = self.session.health()
        if not info.get("stt") or not info.get("translate"):
            raise RemoteError(f"inference server {self.session.url} has no STT/translation loaded")
        if speaking and not info.get("tts"):
            raise RemoteError(f"inference server {self.session.url} has no TTS loaded")
        if speaking and info.get("sample_rate") != self.sample_rate:
            raise RemoteError(
                f"tts.sample_rate mismatch: server {info.get('sample_rate')} Hz, "
                f"local {self.sample_rate} Hz (AudioPlayer is opened at the local rate)"
            )
        logger.info("Remote pipeline ready at %s: STT %s -> %s -> %s%s", self.session.url,
                    info["stt"], info["translate"], info.get("tts") or "text only",
                    " + RVC" if info.get("vc") else "")

    def process(self, audio: np.ndarray, speak: bool, speed_factor: float = 1.0,
                voice_reference: str = "", voice_reference_text: str = "") -> PipelineResult:
        params = {"speak": "1" if speak else "0", "speed": f"{speed_factor:.3f}"}
        if voice_reference:
            params["voice_reference"] = voice_reference
        if voice_reference_text:
            params["voice_reference_text"] = voice_reference_text
        resp = self.session.request(
            "POST", "/translate", params=params,
            data=np.ascontiguousarray(audio, dtype="<f4").tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        body = resp.content
        (n,) = struct.unpack_from("<I", body, 0)
        meta = json.loads(body[4:4 + n].decode("utf-8"))
        out = np.frombuffer(body[4 + n:], dtype="<f4").astype(np.float32)
        return PipelineResult(
            text=meta.get("text", ""), lang=meta.get("lang", ""), dropped=meta.get("dropped"),
            translation=meta.get("translation", ""), tgt_lang=meta.get("tgt_lang", ""),
            stt_ms=float(meta.get("stt_ms") or 0.0), llm_ms=float(meta.get("llm_ms") or 0.0),
            tts_ms=float(meta.get("tts_ms") or 0.0),
            vc_ms=float(meta["vc_ms"]) if meta.get("vc_ms") is not None else None,
            audio=out,
        )

"""
GPU inference server for main.py: runs STT and TTS (+ optional RVC) on the
PC (NVIDIA / CUDA) so the Mac only captures audio, runs the VAD and plays
the translation into BlackHole. Translation already runs on the PC (Ollama).

    Mac: mic -> VAD -> utterance --LAN--> PC: STT
    Mac: ------------- text ----> Ollama (PC) -> translation
    Mac: BlackHole <- player <--LAN-- PC: TTS -> RVC

Reads the same config.yaml as main.py, with `server.overrides` deep-merged
over it (CUDA device, faster-whisper model, Applio location on the PC).
The wire format is documented in src/remote_client.py.

Run on the PC:
    .venv-server\\Scripts\\activate
    python inference_server.py              # all stages from config.yaml
    python inference_server.py --no-tts     # STT only
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from src.stt_transcriber import create_transcriber
from src.tts_cloner import VoiceCloner
from src.utils import DotDict, load_config, setup_logging
from src.voice_converter import VoiceConverter

logger = logging.getLogger("live-translation.server")

_ROOT = Path(__file__).resolve().parent
_WARMUP_TEXT = "Testing, one, two, three."
_MAX_BODY = 64 * 1024 * 1024  # 64 MB: far above any utterance (15 s @ 16 kHz = 960 KB)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def server_config(path: str = "config.yaml") -> DotDict:
    raw = dict(load_config(path))
    overrides = (raw.get("server") or {}).get("overrides") or {}
    return DotDict(deep_merge(raw, overrides))


class Models:
    """Loaded models plus one lock per GPU stage (each model serves one call at a time)."""

    def __init__(self, cfg, stt: bool, tts: bool):
        self.cfg = cfg
        self.transcriber = create_transcriber(cfg) if stt else None
        self.voice_cloner = None
        self.voice_converter = None
        if tts:
            # F5-TTS auto-captions its reference clip with the local STT, if loaded.
            self.voice_cloner = VoiceCloner(
                cfg, transcribe_fn=self.transcriber.transcribe if self.transcriber else None)
            self.voice_converter = VoiceConverter.from_config(cfg)
        self.stt_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.reference_dir = (_ROOT / cfg.paths.reference_dir).resolve()

    def load(self) -> None:
        if self.transcriber is not None:
            self.transcriber.load()
        if self.voice_cloner is not None:
            self.voice_cloner.load()
            speech = self._warmup_tts()
            if self.voice_converter is not None:
                self.voice_converter.load()
                if speech is not None and speech.size:
                    # The worker's own warm-up uses a test tone; the first real
                    # speech still took ~2.5 s (vs ~0.2 s after) on CUDA.
                    self.voice_converter.convert(speech)

    def _warmup_tts(self) -> Optional[np.ndarray]:
        """Pay the first-call cost (CUDA kernels, reference preprocessing) before serving."""
        route = next((r for r in self.cfg.routes if r.get("speak", True)), {})
        ref = route.get("voice_reference", "")
        try:
            return self.voice_cloner.synthesize(_WARMUP_TEXT, "en", self.reference_path(ref),
                                                route.get("voice_reference_text", ""))
        except FileNotFoundError as exc:
            logger.warning("TTS warm-up skipped: %s", exc)
            return None

    def close(self) -> None:
        if self.voice_converter is not None:
            self.voice_converter.close()

    def health(self) -> dict:
        return {
            "stt": f"{self.cfg.stt.engine}:{self.cfg.stt.model}" if self.transcriber else None,
            "tts": self.cfg.tts.engine if self.voice_cloner else None,
            "vc": self.voice_converter is not None,
            "sample_rate": self.cfg.tts.sample_rate,
        }

    def reference_path(self, requested: str) -> str:
        """Resolve a client-supplied reference path; only files under paths.reference_dir."""
        if not requested:
            return ""
        path = (_ROOT / requested).resolve()
        if not path.is_relative_to(self.reference_dir):
            raise ValueError(f"voice_reference must be inside {self.cfg.paths.reference_dir}/")
        return str(path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: one TCP connection per client thread
    models: Models  # set on the class before serving

    # ---- plumbing ----------------------------------------------------------
    def log_message(self, fmt, *args):  # route http.server's access log to DEBUG
        logger.debug("%s %s", self.address_string(), fmt % args)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > _MAX_BODY:
            raise ValueError(f"request body too large ({n} bytes)")
        return self.rfile.read(n) if n else b""

    def _send(self, status: int, body: bytes, content_type: str, headers: Optional[dict] = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

    # ---- routes --------------------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            self._json(200, self.models.health())
        else:
            self._json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/stt":
                self._stt(body)
            elif self.path == "/tts":
                self._tts(json.loads(body or b"{}"))
            else:
                self._json(404, {"error": f"unknown path {self.path}"})
        except (ValueError, FileNotFoundError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("%s failed", self.path)
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _stt(self, body: bytes) -> None:
        m = self.models
        if m.transcriber is None:
            raise ValueError("STT is disabled on this server")
        audio = np.frombuffer(body, dtype="<f4").astype(np.float32)
        t0 = time.perf_counter()
        with m.stt_lock:
            text, lang = m.transcriber.transcribe(audio)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("STT %5.0f ms | %4.1fs audio | %s: %s", ms, audio.size / 16000, lang, text)
        self._json(200, {"text": text, "lang": lang, "server_ms": round(ms, 1)})

    def _tts(self, req: dict) -> None:
        m = self.models
        if m.voice_cloner is None:
            raise ValueError("TTS is disabled on this server")
        text = str(req.get("text", ""))
        ref = m.reference_path(str(req.get("voice_reference", "")))
        headers = {}
        with m.tts_lock:
            t0 = time.perf_counter()
            audio = m.voice_cloner.synthesize(
                text, str(req.get("lang", "en")), ref, str(req.get("voice_reference_text", "")),
                float(req.get("speed_factor", 1.0)),
            )
            t1 = time.perf_counter()
            headers["X-TTS-Ms"] = f"{(t1 - t0) * 1000:.1f}"
            vc_note = ""
            if m.voice_converter is not None and audio.size:
                try:
                    audio = m.voice_converter.convert(audio)
                    headers["X-VC-Ms"] = f"{(time.perf_counter() - t1) * 1000:.1f}"
                    vc_note = f" | VC {float(headers['X-VC-Ms']):4.0f} ms"
                except Exception:
                    # Same policy as main.py: plain TTS voice beats silence.
                    logger.exception("RVC failed - returning unconverted TTS")
        logger.info("TTS %5.0f ms%s | %4.1fs audio | %s", float(headers["X-TTS-Ms"]), vc_note,
                    audio.size / m.cfg.tts.sample_rate, text)
        data = np.ascontiguousarray(audio, dtype="<f4").tobytes()
        self._send(200, data, "application/octet-stream", headers)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--host", help="override server.host")
    p.add_argument("--port", type=int, help="override server.port")
    p.add_argument("--no-stt", action="store_true", help="do not load STT")
    p.add_argument("--no-tts", action="store_true", help="do not load TTS / RVC")
    args = p.parse_args()

    cfg = server_config(args.config)
    setup_logging(cfg.pipeline.log_level)
    scfg = cfg.get("server") or {}
    host = args.host or scfg.get("host", "0.0.0.0")
    port = args.port or int(scfg.get("port", 8765))

    models = Models(cfg, stt=not args.no_stt, tts=not args.no_tts)
    logger.info("Loading models (first run downloads weights)...")
    models.load()
    Handler.models = models

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    logger.info("Inference server listening on http://%s:%d  %s", host, port, models.health())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        models.close()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()

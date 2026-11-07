"""
Live bidirectional RU<->EN speech translation with real-time voice cloning,
routed through BlackHole virtual audio devices into Google Meet.

Pipeline per route:

    capture -> VAD -> STT (mlx-whisper) -> LLM translate (Ollama)
            -> TTS (voice clone) -> [RVC voice conversion] -> playback

The RVC stage is optional (config.yaml tts.voice_conversion.enabled) and runs
in the same TTS worker thread, right after synthesis.

Stages run as independent asyncio tasks connected by asyncio.Queue, so while
utterance N is being synthesized, utterance N+1 is already being
transcribed - true pipeline execution instead of one long blocking chain.

Run:
    source .venv/bin/activate
    python main.py
"""
from __future__ import annotations

import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

from src.audio_stream import AudioCapture, AudioPlayer
from src.stt_transcriber import Transcriber
from src.translator import Translator
from src.tts_cloner import VoiceCloner
from src.utils import LatencyTracker, load_config, setup_logging
from src.vad_processor import Utterance, VADProcessor
from src.voice_converter import VoiceConverter

logger = logging.getLogger("live-translation.main")


@dataclass
class SttResult:
    route: str
    utterance_id: int
    text: str
    lang: str
    tracker: LatencyTracker


@dataclass
class TranslateResult:
    route: str
    utterance_id: int
    text: str
    lang: str
    tracker: LatencyTracker


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.routes: List[dict] = [r for r in cfg.routes if r.get("enabled", True)]
        if not self.routes:
            raise RuntimeError("No enabled routes in config.yaml -> routes")

        qsize = cfg.pipeline.max_queue_size
        self.frame_queues: Dict[str, asyncio.Queue] = {}
        self.utterance_queue: "asyncio.Queue[Utterance]" = asyncio.Queue(maxsize=qsize)
        self.stt_out_queue: "asyncio.Queue[SttResult]" = asyncio.Queue(maxsize=qsize)
        self.translate_out_queue: "asyncio.Queue[TranslateResult]" = asyncio.Queue(maxsize=qsize)

        self.captures: Dict[str, AudioCapture] = {}
        self.players: Dict[str, AudioPlayer] = {}
        self.vads: Dict[str, VADProcessor] = {}
        self._route_cfg = {r["name"]: r for r in self.routes}
        self._any_speaking = any(r.get("speak", True) for r in self.routes)

        self.transcriber = Transcriber(cfg)
        self.translator = Translator(cfg)
        self.voice_cloner = VoiceCloner(cfg, transcribe_fn=self.transcriber.transcribe)
        self.voice_converter = VoiceConverter.from_config(cfg)

        self._stt_executor = ThreadPoolExecutor(
            max_workers=cfg.pipeline.worker_threads.stt, thread_name_prefix="stt"
        )
        self._tts_executor = ThreadPoolExecutor(
            max_workers=cfg.pipeline.worker_threads.tts, thread_name_prefix="tts"
        )
        self._tasks: List[asyncio.Task] = []

    # ---- setup ------------------------------------------------------------
    def load_models(self) -> None:
        logger.info("Loading models (first run can take a while - downloading weights)...")
        self.transcriber.load()
        if self._any_speaking:
            self.voice_cloner.load()
            self._warmup_tts()
            if self.voice_converter is not None:
                # The worker warms itself up (rmvpe, FAISS index, Metal kernels)
                # before reporting ready.
                self.voice_converter.load()
        else:
            logger.info("All routes are text-only (speak: false) - skipping TTS model load.")
        logger.info("All models ready.")

    def _warmup_tts(self) -> None:
        """
        F5-TTS/XTTS lazily preprocess+cache each reference wav on its first
        `synthesize()` call, which otherwise makes the very first live
        utterance far slower than every subsequent one (observed ~14s vs
        ~5s). Pay that cost once here instead of on the first real phrase.
        """
        seen: set = set()
        for route in self.routes:
            if not route.get("speak", True):
                continue
            ref = route["voice_reference"]
            if ref in seen:
                continue
            seen.add(ref)
            logger.info("Warming up TTS voice reference: %s", ref)
            try:
                self.voice_cloner.synthesize(
                    "Проверка связи.", "ru", ref, route.get("voice_reference_text", "")
                )
            except Exception:
                logger.exception("TTS warm-up failed for %s", ref)

    def build_audio(self, loop: asyncio.AbstractEventLoop) -> None:
        devices = self.cfg.audio.devices
        for route in self.routes:
            name = route["name"]
            frame_q: asyncio.Queue = asyncio.Queue(maxsize=self.cfg.pipeline.max_queue_size)
            self.frame_queues[name] = frame_q

            self.captures[name] = AudioCapture(
                device_name=devices[route["input_device"]],
                sample_rate=self.cfg.audio.sample_rate,
                blocksize=self.cfg.audio.blocksize,
                loop=loop,
                out_queue=frame_q,
            )
            if route.get("speak", True):
                self.players[name] = AudioPlayer(
                    device_name=devices[route["output_device"]],
                    sample_rate=self.cfg.tts.sample_rate,
                    blocksize=self.cfg.audio.blocksize,
                )
            self.vads[name] = VADProcessor(
                route=name, cfg=self.cfg, utterance_queue=self.utterance_queue, loop=loop,
            )

    def start_audio(self) -> None:
        for cap in self.captures.values():
            cap.start()
        for player in self.players.values():
            player.start()

    def stop_audio(self) -> None:
        for cap in self.captures.values():
            cap.stop()
        for player in self.players.values():
            player.stop()

    # ---- pipeline stages ----------------------------------------------------
    async def _vad_stage(self, route: str) -> None:
        await self.vads[route].run(self.frame_queues[route])

    async def _stt_stage(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            utt = await self.utterance_queue.get()
            try:
                text, lang = await loop.run_in_executor(
                    self._stt_executor, self.transcriber.transcribe, utt.audio
                )
            except Exception:
                logger.exception("[%s] STT failed on utterance #%d", utt.route, utt.utterance_id)
                continue
            utt.tracker.mark("stt")
            if not text.strip():
                continue
            reason = self.transcriber.reject_reason(text, lang)
            if reason:
                logger.info("[%s] dropped STT(%s) %r: %s", utt.route, lang, text, reason)
                continue
            logger.info("[%s] STT(%s): %s", utt.route, lang, text)
            await self.stt_out_queue.put(
                SttResult(route=utt.route, utterance_id=utt.utterance_id, text=text, lang=lang, tracker=utt.tracker)
            )

    async def _translate_stage(self) -> None:
        while True:
            item = await self.stt_out_queue.get()
            translated, tgt_lang = await self.translator.translate(item.text, item.lang)
            item.tracker.mark("llm")
            if not translated.strip():
                continue
            logger.info("[%s] LLM(%s): %s", item.route, tgt_lang, translated)
            if not self._route_cfg[item.route].get("speak", True):
                # text-only route: the console line above is the deliverable,
                # no TTS / playback. Close the timeline here (TTS 0ms).
                item.tracker.mark("tts")
                logger.info(item.tracker.report(extra=f'"{translated}"'))
                continue
            await self.translate_out_queue.put(
                TranslateResult(route=item.route, utterance_id=item.utterance_id,
                                 text=translated, lang=tgt_lang, tracker=item.tracker)
            )

    async def _tts_stage(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self.translate_out_queue.get()
            rcfg = self._route_cfg[item.route]
            try:
                audio = await loop.run_in_executor(
                    self._tts_executor,
                    self.voice_cloner.synthesize,
                    item.text, item.lang, rcfg["voice_reference"], rcfg.get("voice_reference_text", ""),
                )
            except Exception:
                logger.exception("[%s] TTS failed on utterance #%d", item.route, item.utterance_id)
                continue
            item.tracker.mark("tts")
            if self.voice_converter is not None and audio.size:
                try:
                    audio = await loop.run_in_executor(
                        self._tts_executor, self.voice_converter.convert, audio
                    )
                except Exception:
                    # Better to be heard in the plain TTS voice than not at all.
                    logger.exception("[%s] RVC failed on utterance #%d - playing unconverted TTS",
                                     item.route, item.utterance_id)
                item.tracker.mark("vc")
            if audio.size:
                self.players[item.route].play(audio)
            logger.info(item.tracker.report(extra=f'"{item.text}"'))

    # ---- run --------------------------------------------------------------
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self.build_audio(loop)
        self.load_models()
        self.start_audio()

        for route in self.routes:
            self._tasks.append(asyncio.create_task(self._vad_stage(route["name"]), name=f"vad:{route['name']}"))
        self._tasks.append(asyncio.create_task(self._stt_stage(), name="stt"))
        self._tasks.append(asyncio.create_task(self._translate_stage(), name="translate"))
        self._tasks.append(asyncio.create_task(self._tts_stage(), name="tts"))

        logger.info("Pipeline running. Active routes: %s", ", ".join(r["name"] for r in self.routes))
        await asyncio.gather(*self._tasks)

    async def shutdown(self) -> None:
        logger.info("Shutting down...")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.stop_audio()
        if self.voice_converter is not None:
            self.voice_converter.close()
        self._stt_executor.shutdown(wait=False)
        self._tts_executor.shutdown(wait=False)


async def amain() -> None:
    cfg = load_config("config.yaml")
    setup_logging(cfg.pipeline.log_level)
    pipeline = Pipeline(cfg)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    run_task = asyncio.create_task(pipeline.run())
    await stop_event.wait()
    run_task.cancel()
    await pipeline.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass

"""
TTS wrapper. Supports four interchangeable backends, selected via
config.yaml `tts.engine`:

  - "f5_tts"  : F5-TTS - fast, MLX/PyTorch, excellent zero-shot cloning.
  - "xtts_v2" : Coqui XTTSv2 - PyTorch/MPS, broader language coverage.
  - "say"     : macOS built-in Speech Synthesis (`say` CLI) - no cloning, no
                reference audio, near-instant (~0.5-1s), but the classic
                "Compact" system voices sound mechanical/robotic.
  - "kokoro"  : Kokoro-82M - small open-weight neural TTS, no cloning, no
                reference audio, ~54 built-in natural-sounding voices.
                Latency after warm-up is comparable to "say" (~0.7-1s) but
                the voice is genuinely neural, not formant-synthesized.

f5_tts/xtts_v2 take a short reference WAV of the target speaker and
synthesize new text in that speaker's timbre - this is what turns
"translated text" into "the speaker's own voice, speaking the other
language" for the Meet call. "say" and "kokoro" just speak the translation
in a fixed system/model voice instead - no cloning, no reference file.
"""
from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# CPython refuses to start if PYTHONHASHSEED is outside this range - any child
# process (multiprocessing resource tracker, PortAudio helpers, ...) would then
# die at interpreter preinit with "Fatal Python error: config_init_hash_seed".
_MAX_HASH_SEED = 2**32 - 1

logger = logging.getLogger("live-translation.tts")

XTTS_LANG_MAP = {"en": "en", "ru": "ru"}


class VoiceCloner:
    def __init__(self, cfg, transcribe_fn: Optional[Callable[[np.ndarray], Tuple[str, str]]] = None):
        self.cfg = cfg
        self.engine_name = cfg.tts.engine
        self.device = cfg.tts.device
        self.sample_rate = cfg.tts.sample_rate
        # Injected, not imported directly, to avoid a hard dependency cycle
        # with src.stt_transcriber - used only to auto-caption reference wavs.
        self._transcribe_fn = transcribe_fn
        self._engine = None
        self._ref_text_cache: Dict[str, str] = {}

    # ---- lifecycle ------------------------------------------------------
    def load(self) -> None:
        if self.engine_name == "f5_tts":
            self._load_f5()
        elif self.engine_name == "xtts_v2":
            self._load_xtts()
        elif self.engine_name == "say":
            self._load_say()
        elif self.engine_name == "kokoro":
            self._load_kokoro()
        else:
            raise ValueError(f"Unknown tts.engine in config.yaml: {self.engine_name!r}")

    def _load_f5(self) -> None:
        from f5_tts.api import F5TTS
        self._engine = F5TTS(device=self.device)
        logger.info("F5-TTS loaded on %s", self.device)

    def _load_xtts(self) -> None:
        from TTS.api import TTS
        self._engine = TTS(self.cfg.tts.xtts.model_name).to(self.device)
        logger.info("XTTSv2 loaded on %s", self.device)

    def _load_say(self) -> None:
        if shutil.which("say") is None:
            raise RuntimeError("tts.engine is 'say' but the macOS `say` command was not found.")
        logger.info("Using macOS `say` (voice=%s)", self.cfg.tts.say.voice)

    def _load_kokoro(self) -> None:
        from kokoro import KPipeline
        self._engine = KPipeline(lang_code=self.cfg.tts.kokoro.lang_code, device=self.device)
        logger.info("Kokoro-82M loaded on %s (voice=%s)", self.device, self.cfg.tts.kokoro.voice)

    # ---- reference handling ----------------------------------------------
    def _reference_text_for(self, ref_path: str, configured_text: str) -> str:
        """
        F5-TTS conditions on a transcript of the reference audio. If none is
        configured, look for a cached sidecar `<ref>.txt`; otherwise
        auto-transcribe once (via the injected STT function) and cache it.
        """
        if configured_text:
            return configured_text
        if ref_path in self._ref_text_cache:
            return self._ref_text_cache[ref_path]

        sidecar = Path(ref_path).with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
        elif self._transcribe_fn is not None:
            import soundfile as sf

            from src.utils import resample

            audio, sr = sf.read(ref_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = resample(audio, sr, 16000)
            text, _ = self._transcribe_fn(audio)
            sidecar.write_text(text, encoding="utf-8")
            logger.info("Auto-transcribed reference '%s' -> cached at %s", ref_path, sidecar)
        else:
            logger.warning("No reference text and no transcriber available for %s", ref_path)
            text = ""
        self._ref_text_cache[ref_path] = text
        return text

    # ---- synthesis --------------------------------------------------------
    def synthesize(self, text: str, lang: str, reference_wav: str, reference_text: str = "") -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        if self.engine_name == "say":
            return self._synthesize_say(text)
        if self.engine_name == "kokoro":
            return self._synthesize_kokoro(text)

        if not Path(reference_wav).exists():
            raise FileNotFoundError(
                f"Voice reference sample not found: {reference_wav}\n"
                "Record ~10-20s of clean speech and place it there (see setup_env.sh)."
            )

        if self.engine_name == "f5_tts":
            return self._synthesize_f5(text, reference_wav, reference_text)
        return self._synthesize_xtts(text, lang, reference_wav)

    def _synthesize_say(self, text: str) -> np.ndarray:
        import soundfile as sf

        say_cfg = self.cfg.tts.say
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            subprocess.run(
                [
                    "say",
                    "-v", say_cfg.voice,
                    "-r", str(say_cfg.rate),
                    "-o", out_path,
                    "--file-format=WAVE",
                    f"--data-format=LEI16@{self.sample_rate}",
                    text,
                ],
                check=True,
                capture_output=True,
            )
            wav, sr = sf.read(out_path, dtype="float32")
        finally:
            os.remove(out_path)

        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            from src.utils import resample
            wav = resample(wav, sr, self.sample_rate)
        return wav.astype(np.float32, copy=False)

    def _synthesize_kokoro(self, text: str) -> np.ndarray:
        kokoro_cfg = self.cfg.tts.kokoro
        parts = [
            chunk.audio.numpy()
            for chunk in self._engine(text, voice=kokoro_cfg.voice, speed=kokoro_cfg.speed)
        ]
        audio = parts[0] if len(parts) == 1 else np.concatenate(parts)
        audio = audio.astype(np.float32, copy=False)

        kokoro_native_rate = 24000  # Kokoro's fixed native output rate
        if self.sample_rate != kokoro_native_rate:
            from src.utils import resample
            audio = resample(audio, kokoro_native_rate, self.sample_rate)
        return audio

    def _synthesize_f5(self, text: str, reference_wav: str, reference_text: str) -> np.ndarray:
        from src.utils import resample

        ref_text = self._reference_text_for(reference_wav, reference_text)
        # Pass an explicit in-range seed: F5-TTS otherwise picks
        # random.randint(0, sys.maxsize) and writes it verbatim into
        # os.environ["PYTHONHASHSEED"] (see f5_tts.model.utils.seed_everything),
        # which on 64-bit is ~2**63 and corrupts every subprocess spawned
        # afterwards.
        seed = self.cfg.tts.get("seed")
        if seed is None:
            seed = random.randint(0, _MAX_HASH_SEED)
        wav, sr, _ = self._engine.infer(
            ref_file=reference_wav,
            ref_text=ref_text,
            gen_text=text,
            speed=self.cfg.tts.speed,
            nfe_step=self.cfg.tts.nfe_step,
            cross_fade_duration=self.cfg.tts.cross_fade_duration_s,
            seed=int(seed) & _MAX_HASH_SEED,
        )
        # Belt and braces in case a future F5-TTS changes how it seeds.
        os.environ["PYTHONHASHSEED"] = str(int(seed) & _MAX_HASH_SEED)
        wav = np.asarray(wav, dtype=np.float32)
        if sr != self.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
        return wav

    def _synthesize_xtts(self, text: str, lang: str, reference_wav: str) -> np.ndarray:
        from src.utils import resample

        wav = self._engine.tts(
            text=text,
            speaker_wav=reference_wav,
            language=XTTS_LANG_MAP.get(lang, "en"),
        )
        wav = np.asarray(wav, dtype=np.float32)
        native_sr = getattr(getattr(self._engine, "synthesizer", None), "output_sample_rate", self.sample_rate)
        if native_sr != self.sample_rate:
            wav = resample(wav, native_sr, self.sample_rate)
        return wav

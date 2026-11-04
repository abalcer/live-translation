"""
Синхронний переклад через Gemini Live API — хмарна альтернатива main.py.

    [мікрофон] ──16 кГц PCM──▶ Gemini Live ──24 кГц PCM──▶ ресемплінг ──▶ [BlackHole]

Параметри беруться з config.yaml → gemini:, прапорці командного рядка мають пріоритет.

Запуск:
    python gemini_translate.py
    python gemini_translate.py --list-devices
    python gemini_translate.py --target de --input-device "MacBook"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from dataclasses import dataclass

import numpy as np
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.utils import load_config

SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000    # Live API приймає 16 кГц PCM
GEMINI_RATE = 24000   # Live API завжди повертає 24 кГц PCM

DEFAULTS = {
    "model": "gemini-3.5-live-translate-preview",
    "input_device": "",
    "output_device": "BlackHole",
    "chunk_size": 1024,
    "target_language_code": "en",
    "echo_target_language": False,
    "show_transcripts": True,
}


@dataclass
class Settings:
    model: str
    input_device: str
    output_device: str
    chunk_size: int
    target_language_code: str
    echo_target_language: bool
    show_transcripts: bool


class LinearResampler:
    """Потоковий ресемплер int16 mono PCM (лінійна інтерполяція, заміна audioop.ratecv).

    Зберігає останній семпл і дробову позицію між чанками, щоб на стиках не було клацань.
    """

    def __init__(self, rate_in: int, rate_out: int):
        self.step = rate_in / rate_out
        self.pos = 0.0      # позиція наступного вихідного семпла відносно self.prev
        self.prev = None    # останній вхідний семпл попереднього чанка

    def process(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if x.size == 0:
            return b""
        if self.prev is not None:
            x = np.concatenate(([self.prev], x))

        last = x.size - 1
        n = int((last - self.pos) // self.step) + 1 if last >= self.pos else 0
        t = self.pos + np.arange(n) * self.step
        y = np.interp(t, np.arange(x.size), x)

        self.pos = self.pos + n * self.step - last
        self.prev = x[-1]
        return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


# ---- налаштування ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Синхронний переклад через Gemini Live API.")
    ap.add_argument("--config", default="config.yaml", help="шлях до config.yaml")
    ap.add_argument("--list-devices", action="store_true", help="показати аудіопристрої й вийти")
    ap.add_argument("--model", help="модель Gemini Live")
    ap.add_argument("--target", help="мова перекладу, BCP-47 (en, de, uk, ...)")
    ap.add_argument("--input-device", help="підрядок назви мікрофона ('' = системний за замовчуванням)")
    ap.add_argument("--output-device", help="підрядок назви вихідного пристрою")
    return ap.parse_args()


def load_settings(args: argparse.Namespace) -> Settings:
    try:
        section = load_config(args.config).get("gemini") or {}
    except FileNotFoundError:
        section = {}
    values = {**DEFAULTS, **{k: v for k, v in section.items() if k in DEFAULTS}}

    overrides = {
        "model": args.model,
        "target_language_code": args.target,
        "input_device": args.input_device,
        "output_device": args.output_device,
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)


# ---- аудіопристрої ----------------------------------------------------------

def list_devices(pa: pyaudio.PyAudio) -> None:
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        print(f"{i:3d}  in={d['maxInputChannels']:<2d} out={d['maxOutputChannels']:<2d} "
              f"{int(d['defaultSampleRate']):>6d} Hz  {d['name']}")


def find_device(pa: pyaudio.PyAudio, name_substr: str, kind: str) -> int:
    """Індекс пристрою за підрядком назви (без урахування регістру).

    Порожній рядок означає системний пристрій за замовчуванням.
    """
    if not name_substr:
        info = (pa.get_default_input_device_info() if kind == "input"
                else pa.get_default_output_device_info())
        return int(info["index"])

    channel_key = "maxInputChannels" if kind == "input" else "maxOutputChannels"
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if name_substr.lower() in d["name"].lower() and d[channel_key] > 0:
            return i
    raise RuntimeError(
        f"Не знайдено {'вхідний' if kind == 'input' else 'вихідний'} пристрій "
        f"'{name_substr}'. Перелік: python gemini_translate.py --list-devices"
    )


# ---- Gemini Live -------------------------------------------------------------

def build_live_config(s: Settings) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=s.target_language_code,
            echo_target_language=s.echo_target_language,
        ),
    )


def log_events(response, server_content) -> None:
    """Друкує службові події сесії та транскрипти."""
    if response.setup_complete:
        print("[gemini] setup_complete")
    if response.voice_activity_detection_signal:
        print(f"[vad] {response.voice_activity_detection_signal.vad_signal_type}")
    if response.voice_activity:
        print(f"[vad] {response.voice_activity.voice_activity_type}")

    if not server_content:
        return
    if server_content.interrupted:
        print("[gemini] interrupted")
    if server_content.turn_complete:
        print(f"[gemini] turn_complete (reason={server_content.turn_complete_reason})")
    if server_content.generation_complete:
        print("[gemini] generation_complete")
    if server_content.waiting_for_input:
        print("[gemini] waiting_for_input — сесія чекає наступної репліки")
    if server_content.input_transcription and server_content.input_transcription.text:
        print(f"[gemini] почуто: {server_content.input_transcription.text!r}")
    if server_content.output_transcription and server_content.output_transcription.text:
        print(f"[gemini] переклад: {server_content.output_transcription.text!r}")


async def stream_mic_to_gemini(session, mic_stream, chunk_size: int) -> None:
    """Зчитує звук з мікрофона та надсилає в Gemini."""
    loop = asyncio.get_running_loop()
    mime_type = f"audio/pcm;rate={INPUT_RATE}"
    while True:
        data = await loop.run_in_executor(None, mic_stream.read, chunk_size, False)
        await session.send_realtime_input(audio=types.Blob(data=data, mime_type=mime_type))


async def stream_gemini_to_output(session, out_stream, out_rate: int, verbose: bool) -> None:
    """Приймає аудіо від Gemini, ресемплить під частоту пристрою та відтворює."""
    loop = asyncio.get_running_loop()
    resampler = LinearResampler(GEMINI_RATE, out_rate) if out_rate != GEMINI_RATE else None

    async for response in session.receive():
        server_content = response.server_content
        if verbose:
            log_events(response, server_content)
        if not (server_content and server_content.model_turn):
            continue

        for part in server_content.model_turn.parts:
            if part.inline_data and part.inline_data.data:
                audio = part.inline_data.data
                if resampler:
                    audio = resampler.process(audio)
                # Блокуючий виклик — виносимо в executor, щоб не заморожувати event loop
                await loop.run_in_executor(None, out_stream.write, audio)


async def run(pa: pyaudio.PyAudio, s: Settings) -> None:
    mic_index = find_device(pa, s.input_device, "input")
    out_index = find_device(pa, s.output_device, "output")

    # Вихідний пристрій (напр. BlackHole) працює на одній системній частоті, спільній для
    # всіх застосунків, що до нього підключені (напр. Google Meet). Якщо писати туди аудіо
    # на іншій частоті без ресемплінгу, звук або зникає, або відтворюється
    # прискорено/уповільнено. Тож беремо реальну частоту пристрою.
    out_rate = int(pa.get_device_info_by_index(out_index)["defaultSampleRate"])

    print(f"[+] Вхід:  {pa.get_device_info_by_index(mic_index)['name']} (index {mic_index})")
    print(f"[+] Вихід: {pa.get_device_info_by_index(out_index)['name']} (index {out_index}, {out_rate} Hz)")

    mic_stream = pa.open(format=SAMPLE_FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True,
                         input_device_index=mic_index, frames_per_buffer=s.chunk_size)
    out_stream = pa.open(format=SAMPLE_FORMAT, channels=CHANNELS, rate=out_rate, output=True,
                         output_device_index=out_index, frames_per_buffer=s.chunk_size)

    client = genai.Client()  # GEMINI_API_KEY з оточення / .env
    try:
        print(f"[*] Підключення до Gemini Live (model={s.model}, target={s.target_language_code})...")
        async with client.aio.live.connect(model=s.model, config=build_live_config(s)) as session:
            print("[✔] З'єднання встановлено! Говоріть у мікрофон...")
            await asyncio.gather(
                stream_mic_to_gemini(session, mic_stream, s.chunk_size),
                stream_gemini_to_output(session, out_stream, out_rate, s.show_transcripts),
            )
    except Exception:
        print("[!] Помилка під час роботи сесії:")
        traceback.print_exc()
        raise
    finally:
        for stream in (mic_stream, out_stream):
            stream.stop_stream()
            stream.close()


def main() -> int:
    load_dotenv()
    args = parse_args()
    pa = pyaudio.PyAudio()
    try:
        if args.list_devices:
            list_devices(pa)
            return 0
        asyncio.run(run(pa, load_settings(args)))
    except KeyboardInterrupt:
        print("\n[*] Завершення роботи...")
    except RuntimeError as e:
        print(f"[!] {e}")
        return 1
    finally:
        pa.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())

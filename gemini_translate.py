import asyncio
import os
import sys
import traceback
import numpy as np
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # GEMINI_API_KEY з файлу .env

# --- НАЛАШТУВАННЯ АУДІО ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_IN = 16000          # Вхідний потік з мікрофона (16kHz PCM)
GEMINI_OUTPUT_RATE = 24000  # Фіксована частота аудіо, яку повертає Gemini Live API
CHUNK_SIZE = 1024

p = pyaudio.PyAudio()

class LinearResampler:
    """Потоковий ресемплер int16 mono PCM (лінійна інтерполяція, заміна audioop.ratecv).

    Зберігає останній семпл і дробову позицію між чанками, щоб на стиках не було клацань.
    """

    def __init__(self, rate_in, rate_out):
        self.step = rate_in / rate_out
        self.pos = 0.0      # позиція наступного вихідного семпла відносно self.prev
        self.prev = None    # останній вхідний семпл попереднього чанка

    def process(self, pcm):
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

def get_device_indices():
    """Пошук мікрофона та віртуального кабелю BlackHole"""
    mic_idx = None
    blackhole_idx = None
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev.get('name', '')
        
        if 'BlackHole' in name and dev.get('maxOutputChannels', 0) > 0:
            blackhole_idx = i
        elif dev.get('maxInputChannels', 0) > 0 and mic_idx is None:
            mic_idx = i
            
    return mic_idx, blackhole_idx

mic_index, blackhole_index = get_device_indices()

if blackhole_index is None:
    print("[!] ПОМИЛКА: BlackHole не знайдено! Перевірте Audio MIDI Setup.")
    sys.exit(1)

print(f"[+] Мікрофон (Вхід): Index {mic_index}")
print(f"[+] BlackHole (Вихід): Index {blackhole_index}")

# BlackHole (як і будь-який реальний пристрій) працює на одній системній частоті
# дискретизації, спільній для всіх застосунків, що до нього підключені (напр. Google
# Meet). Якщо писати туди аудіо на іншій частоті без ресемплінгу, звук або зникає,
# або відтворюється прискорено/уповільнено. Тож беремо реальну частоту пристрою.
BLACKHOLE_RATE = int(p.get_device_info_by_index(blackhole_index)['defaultSampleRate'])
print(f"[+] BlackHole частота дискретизації: {BLACKHOLE_RATE} Hz")

# --- GEMINI LIVE API ---
client = genai.Client()
MODEL_ID = "gemini-3.5-live-translate-preview"

# Конфігурація перекладу (target_language_code: 'en', 'de', 'es' тощо)
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    translation_config=types.TranslationConfig(
        target_language_code="en",
        echo_target_language=False
    )
)

async def stream_mic_to_gemini(session, mic_stream):
    """Зчитує звук з мікрофона та надсилає в Gemini"""
    loop = asyncio.get_running_loop()

    while True:
        data = await loop.run_in_executor(None, mic_stream.read, CHUNK_SIZE, False)

        await session.send_realtime_input(
            audio=types.Blob(
                data=data,
                mime_type=f"audio/pcm;rate={RATE_IN}"
            )
        )

async def stream_gemini_to_blackhole(session, blackhole_stream):
    """Приймає аудіо від Gemini, ресемплить під частоту BlackHole та виводить у нього"""
    loop = asyncio.get_running_loop()
    resampler = LinearResampler(GEMINI_OUTPUT_RATE, BLACKHOLE_RATE)
    async for response in session.receive():
        server_content = response.server_content

        if response.setup_complete:
            print("[gemini] setup_complete")

        if response.voice_activity_detection_signal:
            print(f"[vad] {response.voice_activity_detection_signal.vad_signal_type}")
        if response.voice_activity:
            print(f"[vad] {response.voice_activity.voice_activity_type}")

        if server_content:
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

            if server_content.model_turn:
                for part in server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        audio = part.inline_data.data
                        if BLACKHOLE_RATE != GEMINI_OUTPUT_RATE:
                            audio = resampler.process(audio)
                        # Блокуючий виклик — виносимо в executor, щоб не заморожувати event loop
                        await loop.run_in_executor(None, blackhole_stream.write, audio)

async def main():
    mic_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE_IN,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK_SIZE
    )

    blackhole_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=BLACKHOLE_RATE,
        output=True,
        output_device_index=blackhole_index,
        frames_per_buffer=CHUNK_SIZE
    )

    try:
        print(f"[*] Підключення до Gemini Live Translate API (model={MODEL_ID})...")
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("[✔] З'єднання встановлено! Говоріть у мікрофон...")

            await asyncio.gather(
                stream_mic_to_gemini(session, mic_stream),
                stream_gemini_to_blackhole(session, blackhole_stream)
            )
    except Exception:
        print("[!] Помилка під час роботи сесії:")
        traceback.print_exc()
        raise
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        blackhole_stream.stop_stream()
        blackhole_stream.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Завершення роботи...")
    finally:
        p.terminate()
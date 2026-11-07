# CLAUDE.md

Цей файл — інструкції для Claude Code щодо роботи з репозиторієм.

## Про проєкт

Синхронний переклад мовлення для відеодзвінків на macOS (Apple Silicon): RU/UA → EN голосом користувача.
Звук потрапляє в Google Meet через віртуальний пристрій BlackHole 16ch, який обрано мікрофоном у Meet.

Дві незалежні реалізації:
- **`main.py`** — локальний конвеєр: Silero VAD → mlx-whisper → Ollama → TTS із клонуванням голосу.
- **`gemini_translate.py`** — хмарна альтернатива: один потік Gemini Live API (`gemini-3.5-live-translate-preview`). Читає `config.yaml → gemini:` (через `src.utils.load_config`), прапорці CLI мають пріоритет. Аудіо через PyAudio (не sounddevice), з власним потоковим `LinearResampler` 24 кГц → частота вихідного пристрою. Виводить у BlackHole 2ch, тоді як `main.py` — у BlackHole 16ch.

## Команди

```bash
./setup_env.sh                          # повне встановлення (Homebrew, BlackHole, Ollama, venv)
source .venv/bin/activate
python main.py                          # локальний конвеєр
python gemini_translate.py              # Gemini Live (потрібен GEMINI_API_KEY у .env)
python gemini_translate.py --list-devices   # аудіопристрої; --target/--model/--input-device/--output-device перекривають config.yaml
python tts_test.py "текст"              # тест TTS/клонування без мікрофона, VAD і STT
python tts_test.py --translate "текст"  # спершу переклад через Ollama, потім TTS
python tts_test.py --no-vc "text"       # те саме без RVC (для порівняння)
./setup_rvc.sh                          # встановити Applio для RVC у vendor/applio (опційно)
python bench_translate.py model1 model2 # затримка й якість перекладу кількох моделей Ollama на однакових фразах
python3 -c "import sounddevice as sd; print(sd.query_devices())"   # назви аудіопристроїв
```

Автотестів і лінтера немає. Зміни перевіряй через `tts_test.py` або запуском `main.py`.
Для `main.py` потрібен запущений Ollama (`ollama serve`) зі скачаною моделлю з `translation.model`.

## Архітектура (`main.py`)

Кожен маршрут із `config.yaml → routes` — окремий конвеєр з асинхронних задач, з'єднаних через `asyncio.Queue`:

```
AudioCapture → VADProcessor → [Utterance] → STT → [SttResult] → Translate → [TranslateResult] → TTS → [RVC] → AudioPlayer
```

- Блокуючі виклики моделей виконуються в `ThreadPoolExecutor` (`pipeline.worker_threads`).
  Для STT і TTS тут має бути **1**: один Metal/MPS-контекст не можна використовувати паралельно.
- `src/audio_stream.py` — захоплення й відтворення через sounddevice з кільцевим буфером. Пристрої шукаються за підрядком назви, без урахування регістру.
- `src/vad_processor.py` — Silero VAD; вікно 512 семплів при 16 кГц змінювати не можна.
- `src/stt_transcriber.py` — mlx-whisper, повертає `(text, lang)`.
- `src/translator.py` — HTTP-запит до Ollama `/api/generate`; напрям перекладу береться з `translation.direction_map`.
- `src/tts_cloner.py` — `VoiceCloner` з рушіями `f5_tts` | `xtts_v2` (клонують голос) і `say` | `kokoro` (без клонування).
  Для F5-TTS потрібен транскрипт зразка: якщо `voice_reference_text` порожній, він береться з `reference/<ім'я>.txt`, а якщо файлу немає — генерується через STT і зберігається туди ж.
- `src/voice_converter.py` — опційний етап RVC (`tts.voice_conversion.enabled`): замінює тембр виходу будь-якого рушія TTS на голос користувача.
  Applio має несумісні з `.venv` залежності, тому модель працює в окремому процесі `src/rvc_worker.py` під `vendor/applio/.venv` (cwd = `vendor/applio`), обмін — сирі float32 через stdin/stdout за протоколом із `src/rvc_protocol.py`.
  `rvc_worker.py` не можна імпортувати з основного конвеєра: він залежить лише від stdlib, numpy, `rvc_protocol` і Applio. Тому `rvc_protocol.py` теж має залежати лише від stdlib і numpy. Applio закріплено на коміті в `setup_rvc.sh`; воркер використовує його внутрішні API (`Pipeline.pipeline`, кешування FAISS), тож оновлення Applio треба перевіряти.
- `src/utils.py` — `load_config` (повертає `DotDict` з доступом через атрибути), логування, `LatencyTracker` (затримка на кожному етапі), `resample`.
- Під час старту `main.py` «прогріває» кожен голосовий зразок, щоб перша фраза не чекала на завантаження.

## Конфігурація

Усі параметри зберігаються в `config.yaml`, коментарі в ньому англійською.
Робоча частота конвеєра — 16 кГц; TTS видає 24 кГц, і `AudioPlayer` перетворює звук на частоту пристрою.
Назви пристроїв залежать від мови macOS (наприклад, «Мікрофон MacBook Pro»).

## Правила

- **Ніколи не комітити** `.env`, `reference/` (зразок голосу користувача — біометричні дані), `logs/`, `models/` (зокрема модель RVC з голосом користувача), `vendor/`, `*.wav`. Усе це вже є в `.gitignore`.
- Документація для користувача (`README.md`) пишеться українською, коментарі в коді — як у файлі, який редагуєш: у `src/`, `main.py` і `config.yaml` англійською, у `gemini_translate.py` українською.
- Додаючи новий TTS-рушій, оновлюй `VoiceCloner.load()` і `synthesize()`, коментар `tts.engine` у `config.yaml` і таблицю в README.
- Код орієнтований лише на macOS/Apple Silicon (MLX, MPS, CoreAudio, BlackHole). Сумісність з іншими платформами не потрібна.

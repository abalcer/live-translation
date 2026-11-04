# Синхронний переклад через Gemini Live API

`gemini_live_translate.py` — окремий скрипт (альтернатива локальному конвеєру
`main.py`). Один двонапрямлений потік до Google Gemini Live робить весь
ланцюжок: **почув → переклав → озвучив**.

```
[BlackHole 2ch]  ──16kHz PCM──▶  Gemini Live  ──24kHz PCM──▶  [BlackHole 16ch]
```

> ⚠️ Live API — платний рівень. Проєкт має мати ненульовий баланс кредитів,
> інакше сесія одразу падає з `1011 — prepayment credits are depleted`
> (керування: <https://ai.studio/projects>). Скрипт це розпізнає й зупиняється
> з поясненням замість нескінченних перепідключень.

## Два режими (за назвою моделі `gemini.model`)

| модель | режим | керування |
|---|---|---|
| `gemini-3.5-live-translate-preview` | спеціалізований потоковий переклад, найнижча затримка | лише `target_language_code`; голос і промпт ігноруються |
| `gemini-3.1-flash-live-preview`, `gemini-2.5-flash-native-audio-*` | діалогова модель у ролі синхроніста | системний промпт + `voice` |

## Налаштування

```bash
pip install -r requirements.txt          # додано google-genai
export GEMINI_API_KEY=...                 # ключ або ефемерний токен (AQ....)
```

Ефемерні токени (`AQ.…`) розпізнаються автоматично → `v1alpha`.

Маршрутизація звуку (macOS):

1. Джерело, яке треба перекласти, спрямуй у **BlackHole 2ch**
   (напр. «Пристрій із багатьма виходами» = навушники + BlackHole 2ch,
   виставлений як системний вихід, або вихід конкретного застосунку).
2. **BlackHole 16ch** вибери як мікрофон у Meet / Zoom — або слухай його напряму.

## Запуск

```bash
python gemini_live_translate.py                       # параметри з config.yaml → gemini:
python gemini_live_translate.py --list-devices
python gemini_live_translate.py --source англійською --target українською --voice Charon
python gemini_live_translate.py --model gemini-2.5-flash-native-audio-preview-09-2025
```

Ctrl+C — зупинка. Сесія автоматично перепідключається (ліміт з'єднання ~10–15 хв)
з відновленням контексту через `session_resumption`.

## Параметри — `config.yaml` → секція `gemini:`

| ключ | призначення |
|---|---|
| `model` | див. таблицю режимів вище |
| `input_device` / `output_device` | підрядок назви пристрою (регістр не важливий) |
| `output_channels` | `1` — лише канал 0; `2` — дублювати в L/R |
| `target_language_code` | BCP-47 (`uk-UA`); у режимі translate береться коротка форма `uk` |
| `echo_target_language` | translate: озвучувати й мовлення, що вже цільовою мовою |
| `voice` | prompt-режим: `Puck`, `Charon`, `Kore`, `Fenrir`, `Aoede`, … |
| `source_language` / `target_language` | prompt-режим: орудний відмінок для промпту |
| `show_transcripts` | друкувати в консоль, що чути / що озвучено |
| `system_instruction` | prompt-режим: повне перевизначення інструкції |

У prompt-режимі синхронність забезпечує автоматичний VAD: щойно з'являється нова
репліка мовця, модель перериває власне озвучення (`server_content.interrupted`),
а скрипт скидає буфер відтворення — переклад не «накопичує відставання».
Режим translate робить це вбудовано.

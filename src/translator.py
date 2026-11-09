"""
Async client for a local Ollama server that performs strict, instant
RU<->EN translation with no meta-commentary - built for a simultaneous
interpretation pipeline where every extra word is added latency.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Tuple

import requests

logger = logging.getLogger("live-translation.translator")


SYSTEM_PROMPTS = {
    "ru_to_en": (
        "You are a professional simultaneous interpreter working live on a "
        "video call. Translate the user's Russian text into natural, fluent "
        "English.\n"
        "STRICT RULES:\n"
        "- Output ONLY the translation. No quotes, no notes, no explanations, "
        "no 'Translation:' prefix, no meta-commentary of any kind.\n"
        "- Preserve the speaker's tone, register and intent exactly.\n"
        "- Never add information that is not in the source.\n"
        "- If the input is a sentence fragment, translate it as a fragment - "
        "do not complete it into a full sentence.\n"
        "- Keep names, numbers and technical terms unchanged.\n"
        "- If the input is empty, noise, or a non-lexical filler, output an "
        "empty string and nothing else."
    ),
    "en_to_ua": (
        "Ти — професійний синхронний перекладач на відеодзвінку. "
        "Переклади англійський текст користувача природною, вільною "
        "українською мовою.\n"
        "СУВОРІ ПРАВИЛА:\n"
        "- Виводь ЛИШЕ переклад. Жодних лапок, пояснень, приміток на кшталт "
        "«Переклад:», жодного метатексту.\n"
        "- Зберігай тон, регістр і намір мовця точно.\n"
        "- Не додавай інформації, якої немає в оригіналі.\n"
        "- Якщо на вході незавершений фрагмент, перекладай як фрагмент, не "
        "добудовуй речення.\n"
        "- Імена, числа й терміни залишай як є.\n"
        "- Якщо на вході порожньо, шум або нечленороздільний звук, виведи "
        "порожній рядок і більше нічого."
    ),
    "en_to_ru": (
        "Ты — профессиональный синхронный переводчик на видеозвонке. "
        "Переведи английский текст пользователя на естественный, беглый "
        "русский язык.\n"
        "СТРОГИЕ ПРАВИЛА:\n"
        "- Выводи ТОЛЬКО перевод. Никаких кавычек, пояснений, пометок вроде "
        "«Перевод:», никакого метатекста.\n"
        "- Сохраняй тон, регистр и смысл высказывания говорящего.\n"
        "- Не добавляй информацию, которой нет в оригинале.\n"
        "- Если на входе — незаконченный фрагмент, переводи как фрагмент, не "
        "достраивай предложение.\n"
        "- Имена, числа и термины оставляй как есть.\n"
        "- Если на входе пусто, шум или нечленораздельный звук, выведи "
        "пустую строку и ничего больше."
    ),
}

# translation.style "concise": the spoken English must keep pace with the
# speaker, so the interpreter compresses instead of rendering every word.
# Measured with translategemma:12b + Kokoro on the same phrases: -16% audio on
# conversational speech with fillers, -8% on dense narration; mistral-nemo
# compressed more but turned telegraphic ("Strict gray suit like uniform")
# and dropped content.
CONCISE_PROMPTS = {
    "ru_to_en": (
        "You are a professional simultaneous interpreter working live on a "
        "video call. Your translation is spoken aloud by a TTS voice and must "
        "keep pace with the speaker, so translate the user's Russian or "
        "Ukrainian text into SHORT, natural spoken English.\n"
        "STRICT RULES:\n"
        "- Output ONLY the translation. No quotes, no notes, no explanations, "
        "no 'Translation:' prefix, no meta-commentary of any kind.\n"
        "- Keep the full meaning: every fact, name, number, negation and the "
        "speaker's intent and tone.\n"
        "- Compress like a real simultaneous interpreter: drop filler words, "
        "hedges and repetitions; prefer short, common words and simple "
        "sentence structure; merge or shorten clauses when the meaning "
        "survives.\n"
        "- The English must not be longer than the source text.\n"
        "- Never add information that is not in the source.\n"
        "- If the input is a sentence fragment, translate it as a fragment.\n"
        "- Keep names, numbers and technical terms unchanged.\n"
        "- If the input is empty, noise, or a non-lexical filler, output an "
        "empty string and nothing else."
    ),
}


def system_prompt(key: str, style: str = "faithful") -> str:
    """System prompt for a direction key; "concise" falls back to faithful where undefined."""
    if style == "concise" and key in CONCISE_PROMPTS:
        return CONCISE_PROMPTS[key]
    return SYSTEM_PROMPTS[key]


DIRECTION_KEY = {
    ("ru", "en"): "ru_to_en",
    ("ua", "en"): "ru_to_en",   # Whisper reports Ukrainian as "ua"/"uk";
    ("uk", "en"): "ru_to_en",   # the ru->en prompt handles it well enough
    ("en", "ru"): "en_to_ru",
    ("en", "ua"): "en_to_ua",
    ("en", "uk"): "en_to_ua",
}

# Reasoning models (Qwen3.5, DeepSeek-R1, ...) may still wrap their answer in a
# <think>...</think> block even when asked not to - strip it before use.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class Translator:
    """Thin executor-backed wrapper around Ollama's POST /api/generate."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.endpoint = cfg.translation.endpoint
        self.model = cfg.translation.model
        self.timeout = cfg.translation.timeout_s
        self.direction_map = dict(cfg.translation.direction_map)
        self.style = cfg.translation.get("style", "faithful")
        self._executor = ThreadPoolExecutor(
            max_workers=cfg.pipeline.worker_threads.translation,
            thread_name_prefix="translate",
        )
        # Recent (source, translation) pairs, kept only for terminology
        # consistency hints - never re-translated themselves.
        self._context: Deque[Tuple[str, str]] = deque(maxlen=6)

    def _target_lang(self, src_lang: str) -> str:
        return self.direction_map.get(src_lang, "en" if src_lang != "en" else "ru")

    def _build_prompt(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        key = DIRECTION_KEY.get((src_lang, tgt_lang), "ru_to_en")
        system = system_prompt(key, self.style)

        context_block = ""
        if self._context:
            recent = list(self._context)[-3:]
            lines = "\n".join(f"- {s} -> {t}" for s, t in recent)
            context_block = (
                "Recent conversation context (for terminology consistency "
                f"only, do not re-translate it):\n{lines}\n\n"
            )
        prompt = f"{context_block}Text to translate:\n{text}"
        return system, prompt

    def _call_ollama(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            # Disable chain-of-thought on reasoning models (Qwen3.5 etc.):
            # otherwise it burns the num_predict budget and blows the latency
            # budget on every utterance. No-op on non-thinking models.
            "think": False,
            "keep_alive": self.cfg.translation.keep_alive,
            "options": {
                "temperature": self.cfg.translation.temperature,
                "num_predict": self.cfg.translation.num_predict,
                "num_ctx": self.cfg.translation.get("num_ctx", 4096),
            },
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    async def translate(self, text: str, src_lang: str) -> Tuple[str, str]:
        """Returns (translated_text, target_lang). Empty text in -> empty text out."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.translate_sync, text, src_lang)

    def translate_sync(self, text: str, src_lang: str) -> Tuple[str, str]:
        """Blocking translate() for callers that already run in a worker thread (inference_server.py)."""
        text = text.strip()
        tgt_lang = self._target_lang(src_lang)
        if not text:
            return "", tgt_lang

        system, prompt = self._build_prompt(text, src_lang, tgt_lang)
        try:
            translated = self._call_ollama(system, prompt)
        except requests.RequestException as exc:
            logger.error("Ollama request failed (%s): %s", self.endpoint, exc)
            return "", tgt_lang

        translated = _THINK_BLOCK.sub("", translated).strip().strip('"').strip("«»").strip()
        if translated:
            self._context.append((text, translated))
        return translated, tgt_lang

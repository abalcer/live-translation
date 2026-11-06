"""
Translation latency/quality benchmark for Ollama models.

Sends the same fixed set of phrases to each model, using exactly the system
prompt and generation options of the live pipeline (src.translator), and
reports per-model latency taken from Ollama's own timing fields:

    load    - model load time (only non-zero on a cold model)
    prompt  - prompt processing (prefill)
    gen     - token generation
    total   - full server-side request time (what the pipeline waits for)

Each model is warmed up first, so the timed runs reflect steady-state latency
(the pipeline keeps the model resident via keep_alive). Translations are
printed side by side so quality can be compared too.

No audio, VAD, STT or TTS is involved.

Usage:
    source .venv/bin/activate

    # models from the command line (default: translation.model from config.yaml)
    python bench_translate.py qwen2.5:7b-instruct gemma3:12b qwen3:14b

    # own phrases (one per line), 3 timed runs per phrase
    python bench_translate.py -f phrases.txt -n 3 gemma3:12b

    # translations only for the summary table, no per-phrase output
    python bench_translate.py -q qwen2.5:7b-instruct gemma3:12b
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import requests

from src.translator import _THINK_BLOCK, DIRECTION_KEY, SYSTEM_PROMPTS
from src.utils import load_config

# Mix of short/long, RU/UA, fragments, numbers and terms - typical call speech.
DEFAULT_PHRASES = [
    ("ru", "Добрый день, всех слышно?"),
    ("ru", "Давайте обсудим план на следующий квартал."),
    ("ru", "Я думаю, что нам нужно перенести релиз на две недели, потому что тестирование ещё не закончено."),
    ("ru", "ну вот, и потом"),
    ("ua", "Добрий день, мене чути?"),
    ("ua", "Бюджет проєкту становить сто двадцять тисяч доларів."),
    ("ua", "Можете поділитися екраном, будь ласка?"),
    ("ua", "Ми інтегрували Kubernetes і тепер деплой займає близько п'яти хвилин."),
]

NS = 1e-9  # Ollama reports durations in nanoseconds


def call(cfg, model: str, system: str, prompt: str) -> dict:
    """One /api/generate request with the same payload as Translator._call_ollama."""
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": cfg.translation.keep_alive,
        "options": {
            "temperature": cfg.translation.temperature,
            "num_predict": cfg.translation.num_predict,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(cfg.translation.endpoint, json=payload, timeout=cfg.translation.timeout_s * 5)
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    text = _THINK_BLOCK.sub("", data.get("response") or "").strip().strip('"').strip("«»").strip()
    return {
        "text": text,
        "wall": wall,
        "load": data.get("load_duration", 0) * NS,
        "prompt": data.get("prompt_eval_duration", 0) * NS,
        "gen": data.get("eval_duration", 0) * NS,
        "total": data.get("total_duration", 0) * NS,
        "tokens": data.get("eval_count", 0),
    }


def build(cfg, lang: str, text: str):
    tgt = dict(cfg.translation.direction_map).get(lang, "en")
    key = DIRECTION_KEY.get((lang, tgt), "ru_to_en")
    return SYSTEM_PROMPTS[key], f"Text to translate:\n{text}"


def load_phrases(path: str | None, default_lang: str):
    if not path:
        return DEFAULT_PHRASES
    with open(path, encoding="utf-8") as fh:
        return [(default_lang, ln.strip()) for ln in fh if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Ollama translation models.")
    ap.add_argument("models", nargs="*", help="Ollama model names (default: translation.model)")
    ap.add_argument("-f", "--file", help="phrases file, one phrase per line")
    ap.add_argument("--src-lang", default="ru", help="source language for --file phrases (default: ru)")
    ap.add_argument("-n", "--runs", type=int, default=2, help="timed runs per phrase (default: 2)")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print the summary table")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or [cfg.translation.model]
    phrases = load_phrases(args.file, args.src_lang)

    summary = []
    for model in models:
        print(f"\n=== {model} ===")
        try:
            # Warm-up: loads the model into memory; its load time is reported separately.
            sys_p, prompt = build(cfg, *phrases[0])
            warm = call(cfg, model, sys_p, prompt)
        except requests.RequestException as exc:
            print(f"  skipped: {exc}  (is it pulled? `ollama pull {model}`)")
            continue
        print(f"  cold load: {warm['load']:.2f}s  (first request total {warm['wall']:.2f}s)")

        rows = []
        for lang, text in phrases:
            sys_p, prompt = build(cfg, lang, text)
            runs = [call(cfg, model, sys_p, prompt) for _ in range(args.runs)]
            rows.extend(runs)
            if not args.quiet:
                med = statistics.median(r["total"] for r in runs)
                print(f"  [{med:5.2f}s] {text}\n           -> {runs[0]['text']}")

        totals = [r["total"] for r in rows]
        gen_tps = sum(r["tokens"] for r in rows) / max(sum(r["gen"] for r in rows), 1e-9)
        summary.append({
            "model": model,
            "load": warm["load"],
            "median": statistics.median(totals),
            "p90": sorted(totals)[int(0.9 * (len(totals) - 1))],
            "max": max(totals),
            "prompt": statistics.median(r["prompt"] for r in rows),
            "tps": gen_tps,
        })

    if not summary:
        return 1

    print("\n" + "-" * 78)
    print(f"{'model':<28}{'load':>7}{'median':>8}{'p90':>7}{'max':>7}{'prefill':>9}{'tok/s':>8}")
    for s in sorted(summary, key=lambda s: s["median"]):
        print(f"{s['model']:<28}{s['load']:>6.1f}s{s['median']:>7.2f}s{s['p90']:>6.2f}s"
              f"{s['max']:>6.2f}s{s['prompt']:>8.2f}s{s['tps']:>8.0f}")
    print("(median/p90/max = server-side time per phrase on a warm model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

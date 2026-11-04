"""
Standalone test harness for the voice-cloning TTS model.

Feeds text straight into src.tts_cloner.VoiceCloner (same engine/config the
live pipeline uses). By default the result is played through the speakers AND
written to a WAV file; use --no-save to play straight to the speaker without
touching disk, or --no-play to only write the file. Optionally the input text
is first translated to English via Ollama, exactly like the live pipeline does.

No audio capture, VAD or STT is involved.

Usage:
    source .venv/bin/activate

    # inline text
    python tts_test.py "Привет, это проверка синтеза речи."

    # from a text file (whole file synthesized as one utterance)
    python tts_test.py -f script.txt

    # from stdin
    echo "Текст для озвучки" | python tts_test.py

    # translate the input to English first, then synthesize it
    python tts_test.py --translate "Привіт, як справи?"

    # line-by-line, with per-line latency, custom voice + output path
    python tts_test.py -f script.txt --per-line \
        --ref reference/reference_speaker.wav --out out.wav

Common options:
    --translate        translate the input text to English (via Ollama) before
                       synthesizing; the English text is what gets spoken
    --src-lang CODE    source language of the input for --translate (default: ru)
    --ref PATH         voice reference WAV (default: first route's voice_reference)
    --ref-text TEXT    transcript of the reference clip (F5-TTS); auto-detected
                       from <ref>.txt sidecar or Whisper if omitted
    --lang CODE        language hint, used by XTTSv2 (default: ru, or en with --translate)
    --engine NAME      f5_tts | xtts_v2  (default: config tts.engine)
    --nfe-step N       F5-TTS denoising steps override (quality/speed knob)
    --speed X          speaking rate override
    --per-line         synthesize each non-empty line separately
    --out PATH         output WAV path (default: logs/tts_test_<ts>.wav)
    --no-save          play straight to the speaker, do not write a WAV file
    --no-play          do not play the result through the speakers (write file only)
    --device NAME      playback output device (substring match;
                       default: config audio.devices.monitor_output)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.tts_cloner import VoiceCloner
from src.utils import load_config, setup_logging


def read_input_text(args) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").strip()
    if args.text:
        return " ".join(args.text).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    sys.exit("No text given. Pass it as an argument, with -f FILE, or on stdin.")


def resolve_reference(cfg, override: str | None) -> str:
    if override:
        return override
    for route in cfg.routes:
        ref = route.get("voice_reference")
        if ref:
            return ref
    sys.exit("No voice_reference found in config.yaml routes; pass --ref PATH.")


def translate_to_english(cfg, text: str, src_lang: str) -> str:
    """Run the text through the same Ollama translator the live pipeline uses."""
    from src.translator import Translator

    translator = Translator(cfg)

    async def _run() -> str:
        out, _ = await translator.translate(text, src_lang)
        return out

    translated = asyncio.run(_run())
    if not translated:
        sys.exit(
            "Translation returned empty (is Ollama running and the model pulled?). "
            f"endpoint={cfg.translation.endpoint} model={cfg.translation.model}"
        )
    return translated


def build_cloner(cfg, args) -> VoiceCloner:
    if args.engine:
        cfg.tts.engine = args.engine
    if args.nfe_step is not None:
        cfg.tts.nfe_step = args.nfe_step
    if args.speed is not None:
        cfg.tts.speed = args.speed

    transcribe_fn = None
    ref_txt_sidecar = Path(resolve_reference(cfg, args.ref)).with_suffix(".txt")
    needs_whisper = (
        cfg.tts.engine == "f5_tts"
        and not args.ref_text
        and not ref_txt_sidecar.exists()
    )
    if needs_whisper:
        # Only pay the Whisper import/load cost when we actually have to
        # caption the reference clip.
        from src.stt_transcriber import Transcriber

        transcriber = Transcriber(cfg)
        transcriber.load()
        transcribe_fn = transcriber.transcribe

    cloner = VoiceCloner(cfg, transcribe_fn=transcribe_fn)
    cloner.load()
    return cloner


def synthesize(cloner, text: str, lang: str, ref: str, ref_text: str,
               per_line: bool, sample_rate: int) -> np.ndarray:
    segments = [text]
    if per_line:
        segments = [ln.strip() for ln in text.splitlines() if ln.strip()]

    gap = np.zeros(int(0.25 * sample_rate), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for i, seg in enumerate(segments, 1):
        t0 = time.perf_counter()
        wav = cloner.synthesize(seg, lang, ref, ref_text)
        dt = time.perf_counter() - t0
        dur = len(wav) / sample_rate if len(wav) else 0.0
        rtf = dt / dur if dur else float("nan")
        print(f"  [{i}/{len(segments)}] {dt:5.2f}s synth -> {dur:5.2f}s audio "
              f"(RTF {rtf:.2f})  | {seg[:70]}")
        if len(wav):
            chunks.append(wav)
            if per_line and i < len(segments):
                chunks.append(gap)

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def play(wav: np.ndarray, sample_rate: int, cfg, device_override: str | None) -> None:
    import sounddevice as sd

    from src.audio_stream import find_device_index

    name = device_override or cfg.audio.devices.get("monitor_output")
    label = "system default"
    if name:
        try:
            sd.default.device = (None, find_device_index(name, "output"))
            label = name
        except RuntimeError as exc:
            print(f"(playback device '{name}' not found, using system default)\n{exc}")

    print(f"playing on: {label}")
    sd.play(wav, sample_rate)
    sd.wait()


def main() -> None:
    p = argparse.ArgumentParser(description="Test the voice-cloning TTS model on text input.")
    p.add_argument("text", nargs="*", help="Text to synthesize (or use -f / stdin).")
    p.add_argument("-f", "--file", help="Read text to synthesize from this file.")
    p.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml).")
    p.add_argument("--translate", action="store_true",
                   help="Translate the input text to English (via Ollama) before synthesizing.")
    p.add_argument("--src-lang", default="ru",
                   help="Source language of the input for --translate (default: ru).")
    p.add_argument("--ref", help="Voice reference WAV (default: first route's voice_reference).")
    p.add_argument("--ref-text", default="", help="Transcript of the reference clip (F5-TTS).")
    p.add_argument("--lang", help="Language hint for XTTSv2 (default: ru, or en with --translate).")
    p.add_argument("--engine", choices=["f5_tts", "xtts_v2"], help="Override tts.engine.")
    p.add_argument("--nfe-step", type=int, help="Override F5-TTS nfe_step.")
    p.add_argument("--speed", type=float, help="Override tts.speed.")
    p.add_argument("--per-line", action="store_true", help="Synthesize each line separately.")
    p.add_argument("--out", help="Output WAV path (default: logs/tts_test_<timestamp>.wav).")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write a WAV file; play straight to the speaker.")
    p.add_argument("--no-play", action="store_true", help="Do not play the result.")
    p.add_argument("--device", help="Playback output device (substring match).")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.pipeline.log_level)

    source_text = read_input_text(args)
    ref = resolve_reference(cfg, args.ref)
    if not Path(ref).exists():
        sys.exit(f"Voice reference not found: {ref}")

    text = source_text
    if args.translate:
        print(f"translating ({args.src_lang} -> en) via {cfg.translation.model}...")
        text = translate_to_english(cfg, source_text, args.src_lang)
        print(f"  src: {source_text}")
        print(f"  en : {text}\n")

    lang = args.lang or ("en" if args.translate else "ru")
    sample_rate = cfg.tts.sample_rate
    print(f"engine={cfg.tts.engine}  ref={ref}  lang={lang}  "
          f"nfe_step={cfg.tts.nfe_step}  speed={cfg.tts.speed}")
    print(f"text to speak: {len(text)} chars\n")

    cloner = build_cloner(cfg, args)

    t0 = time.perf_counter()
    wav = synthesize(cloner, text, lang, ref, args.ref_text, args.per_line, sample_rate)
    total = time.perf_counter() - t0

    if not len(wav):
        sys.exit("Synthesis produced no audio.")
    if args.no_save and args.no_play:
        sys.exit("--no-save and --no-play together leave nothing to do.")

    audio_dur = len(wav) / sample_rate
    if args.no_save:
        print(f"\ntotal: {total:.2f}s synth for {audio_dur:.2f}s audio  (not saved)")
    else:
        out_path = Path(args.out) if args.out else Path(
            cfg.paths.logs_dir) / f"tts_test_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), wav, sample_rate)
        print(f"\ntotal: {total:.2f}s synth for {audio_dur:.2f}s audio  ->  {out_path}")

    if not args.no_play:
        play(wav, sample_rate, cfg, args.device)


if __name__ == "__main__":
    main()

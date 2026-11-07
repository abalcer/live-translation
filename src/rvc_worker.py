"""
RVC voice-conversion worker. Runs as a child process under Applio's own venv
(vendor/applio/.venv/bin/python) with cwd = the Applio checkout, because
Applio pins torch/numpy versions that conflict with the main .venv and
resolves its embedder/predictor weights relative to the working directory.

Spawned and driven by src/voice_converter.py over stdin/stdout (see
src/rvc_protocol.py). Do not import this module from the main pipeline: it
must only depend on the stdlib, numpy, rvc_protocol and Applio.

Applio print()s progress messages, so the real stdout fd is kept only for
the protocol and fd 1 is redirected to stderr.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

import rvc_protocol as proto  # sibling module: this script's dir is on sys.path


class _CachedIndex:
    """FAISS index wrapper that memoizes the full reconstruct_n() matrix."""

    def __init__(self, index):
        self._index = index
        self.ntotal = index.ntotal
        self._all = None

    def reconstruct_n(self, start, n):
        if start == 0 and n == self.ntotal:
            if self._all is None:
                self._all = self._index.reconstruct_n(0, self.ntotal)
            return self._all
        return self._index.reconstruct_n(start, n)

    def search(self, *args, **kwargs):
        return self._index.search(*args, **kwargs)


class _CachingFaiss:
    """
    Stand-in for the `faiss` module inside rvc.infer.pipeline. Applio's
    Pipeline.pipeline() re-reads the .index file (hundreds of MB) and
    reconstructs every vector on each call; cache both so a phrase only pays
    for the actual nearest-neighbour search.
    """

    def __init__(self, faiss_module):
        self._faiss = faiss_module
        self._cache = {}

    def read_index(self, path, *args):
        if path not in self._cache:
            self._cache[path] = _CachedIndex(self._faiss.read_index(path, *args))
        return self._cache[path]

    def __getattr__(self, name):
        return getattr(self._faiss, name)


class Converter:
    """Applio model kept loaded between phrases, with MPS-friendly input lengths."""

    def __init__(self, args):
        sys.path.insert(0, os.getcwd())
        from rvc.infer import pipeline as applio_pipeline
        from rvc.infer.infer import VoiceConverter

        applio_pipeline.faiss = _CachingFaiss(applio_pipeline.faiss)

        self.args = args
        self.vc = VoiceConverter()
        self.vc.get_vc(args.model, 0)
        self.vc.load_hubert(args.embedder)
        self.sample_rate = int(self.vc.tgt_sr)
        self.index_path = args.index if args.index and os.path.exists(args.index) else ""
        if args.index and not self.index_path:
            print(f"[rvc_worker] index not found, converting without it: {args.index}")
        # MPS re-compiles its graphs for every new input length: a phrase of
        # an unseen length took ~1 s vs ~0.3 s for a repeated one. Inputs are
        # padded with silence to a multiple of this many samples.
        self.bucket = max(1, int(args.bucket_s * proto.INPUT_SR))

    def _run(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float64)
        peak = np.abs(audio).max() / 0.95 if audio.size else 0.0
        if peak > 1:
            audio /= peak
        out = self.vc.vc.pipeline(
            model=self.vc.hubert_model,
            net_g=self.vc.net_g,
            sid=0,
            audio=audio,
            pitch=self.args.pitch,
            f0_method=self.args.f0_method,
            file_index=self.index_path,
            index_rate=self.args.index_rate,
            pitch_guidance=self.vc.use_f0,
            volume_envelope=self.args.volume_envelope,
            version=self.vc.version,
            protect=self.args.protect,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
        )
        return np.asarray(out, dtype=np.float32)

    def convert(self, audio: np.ndarray) -> np.ndarray:
        n = audio.size
        if not n:
            return np.zeros(0, dtype=np.float32)
        padded = np.zeros(-(-n // self.bucket) * self.bucket, dtype=np.float64)
        padded[:n] = audio
        out = self._run(padded)
        return out[: int(round(n * self.sample_rate / proto.INPUT_SR))]

    def warm_up(self, max_s: float) -> None:
        """Load rmvpe + the FAISS index and compile every bucket up to max_s."""
        t = np.arange(int(max_s * proto.INPUT_SR), dtype=np.float64) / proto.INPUT_SR
        tone = 0.1 * np.sin(2 * np.pi * 120 * t)
        for size in range(self.bucket, tone.size + 1, self.bucket):
            self._run(tone[:size])


def parse_args() -> argparse.Namespace:
    # Defaults live here; src/voice_converter.py only passes what config.yaml sets.
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--index", default="")
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--index-rate", type=float, default=0.3)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--volume-envelope", type=float, default=1.0)
    p.add_argument("--f0-method", default="rmvpe")
    p.add_argument("--embedder", default="contentvec")
    p.add_argument("--bucket-s", type=float, default=1.0)
    p.add_argument("--warmup-max-s", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_in = sys.stdin.buffer

    converter = Converter(args)
    converter.warm_up(args.warmup_max_s)
    proto.write_uint(proto_out, converter.sample_rate)

    while True:
        try:
            audio = proto.read_audio(proto_in)
        except EOFError:
            return
        try:
            proto.write_audio(proto_out, converter.convert(audio))
        except Exception:
            traceback.print_exc()
            proto.write_uint(proto_out, proto.ERROR)


if __name__ == "__main__":
    main()

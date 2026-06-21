#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["SLACK_LOG_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tokenizers import Tokenizer

import train


def initialize_globals(cfg: train.PropagatorConfig) -> None:
    train.config = cfg
    train.tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    train.token_ids = train.ensure_special_tokens(train.tokenizer)
    train.text_vocab_size = train.tokenizer.get_vocab_size()
    (
        train.vocab_size,
        train.audio_token_start,
        train.audio_token_end,
        train.image_token_start,
        train.image_token_end,
    ) = train.compute_vocab_sizes(train.text_vocab_size)
    train.init_global_token_ids()


def sine_wave(seconds: float, sample_rate: int, frequency: float) -> np.ndarray:
    samples = int(round(seconds * sample_rate))
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    return (0.1 * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def main() -> None:
    backend = os.environ.get("AUDIO_BACKEND", "mimi").strip().lower()
    if backend == "mimi":
        frame_rate = 12.5
        codebook_size = 2048
    elif backend == "encodec":
        frame_rate = 75.0
        codebook_size = 1024
    else:
        raise ValueError(f"Unsupported AUDIO_BACKEND for smoke test: {backend}")
    cfg = train.PropagatorConfig(
        audio_backend=backend,
        audio_sample_rate=24_000,
        audio_frames_per_second=frame_rate,
        audio_codebook_size=codebook_size,
        max_audio_seconds=1.0,
        max_audio_tokens_per_row=8 * 75,
    )
    initialize_globals(cfg)

    short_seconds = 0.20
    long_seconds = 0.60
    short, long = train.encode_audio_batch_to_token_ids(
        [
            (sine_wave(short_seconds, cfg.audio_sample_rate, 220.0), cfg.audio_sample_rate),
            (sine_wave(long_seconds, cfg.audio_sample_rate, 440.0), cfg.audio_sample_rate),
        ]
    )

    assert abs(len(short) - round(short_seconds * cfg.audio_frames_per_second)) <= 1
    assert abs(len(long) - round(long_seconds * cfg.audio_frames_per_second)) <= 1
    assert len(short) < len(long), "batched codec padding leaked into the shorter sample"

    for frames in (short, long):
        assert frames
        for frame in frames:
            assert len(frame) == cfg.audio_codebooks
            for codebook, token_id in enumerate(frame):
                parsed = train.audio_code_from_token_id(token_id)
                assert parsed is not None
                assert parsed[0] == codebook
                assert 0 <= parsed[1] < cfg.audio_codebook_size

    decoded, sample_rate, error = train.decode_audio_token_ids_to_waveform(
        [token_id for frame in short for token_id in frame]
    )
    assert error is None
    assert sample_rate == cfg.audio_sample_rate
    assert decoded.size > 0
    assert np.isfinite(decoded).all()

    row = {
        "audio": {
            "array": sine_wave(short_seconds, cfg.audio_sample_rate, 330.0),
            "sampling_rate": cfg.audio_sample_rate,
        },
        "text": "the smoke test transcript",
    }
    for task in ("asr", "tts", "audio", "hybrid"):
        inputs, targets, weights, _ = train.tokenize_audio_asr(
            row,
            {"audio_task_mix": {task: 1.0}},
        )
        input_main = np.asarray(inputs, dtype=np.int32)[:, 0]
        target_main = np.asarray(targets, dtype=np.int32)[:, 0]
        assert np.any(np.asarray(weights) > 0.0)
        if task in {"asr", "audio", "hybrid"}:
            assert train.token_ids_audio_in in input_main
        if task == "asr":
            assert train.token_ids_text_out in target_main
        elif task == "tts":
            assert train.token_ids_audio_out in target_main
            assert train.token_ids_audio_in not in input_main
        elif task == "audio":
            assert train.token_ids_audio_out in target_main
        else:
            assert train.token_ids_hybrid_out in target_main
            assert train.token_ids_audio_out in target_main
    print(
        f"smoke_audio_preprocessing: PASS backend={backend} "
        f"short_frames={len(short)} long_frames={len(long)}"
    )


if __name__ == "__main__":
    main()

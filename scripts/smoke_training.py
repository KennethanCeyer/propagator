#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["SLACK_LOG_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tokenizers import Tokenizer

import train

warnings.filterwarnings("ignore", message="Some donated buffers were not usable")


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


def check_control_examples() -> None:
    rows = train.synthetic_control_rows(24, split_name="train")
    matching = [
        row
        for row in rows
        if row["output"][0]["content"] == "What is your name?"
    ]
    assert matching, "identity control example is missing"
    for row in matching:
        assert "Propagator" in row["output"][1]["content"]


def check_weighted_sampler() -> None:
    stream_ids = np.asarray(
        [
            1_000_000_000,
            1_000_000_000,
            1_000_000_001,
            2_000_000_000,
            2_000_000_001,
            3_000_000_000,
            3_000_000_001,
        ],
        dtype=np.int64,
    )
    sampler = train.StatefulChunkSampler(
        stream_ids,
        batch_size=8,
        seed=7,
        source_weights=[0.5, 0.25, 0.25],
    )
    assert sampler.source_lane_counts == {0: 4, 1: 2, 2: 2}
    indices, reset = sampler.next_indices()
    sampled_sources = stream_ids[indices] // 1_000_000_000 - 1
    counts = {int(source): int(np.sum(sampled_sources == source)) for source in np.unique(sampled_sources)}
    assert counts == {0: 4, 1: 2, 2: 2}
    assert bool(np.all(reset))


def check_plain_text_continuation() -> None:
    text = "The first sentence establishes context. " * 12 + "The final sentence should be predicted."
    inputs, targets, weights, _ = train.tokenize_plain_text({"text": text})
    input_main = [frame[0] for frame in inputs]
    target_main = [frame[0] for frame in targets]
    assert train.token_ids_text_in in input_main
    assert train.token_ids_text_out in target_main
    assert any(weight > 0.0 for weight in weights)


def check_tiny_multimodal_forward(cfg: train.PropagatorConfig) -> None:
    model = train.PropagatorModel(cfg, train.vocab_size, nnx.Rngs(cfg.seed))
    inputs = np.zeros((2, 4, 8), dtype=np.int32)
    targets = np.zeros((2, 4, 8), dtype=np.int32)
    weights = np.zeros((2, 4), dtype=np.float32)

    inputs[:, 0, 0] = train.token_ids_session
    targets[:, 0, 0] = train.token_ids_listen
    weights[:, 0] = 1.0

    inputs[:, 1, 0] = train.token_ids_model
    targets[:, 1, 0] = train.token_ids_audio_out
    weights[:, 1] = 1.0

    for codebook in range(cfg.audio_codebooks):
        token = train.audio_token_id(codebook, codebook % cfg.audio_codebook_size)
        inputs[:, 2, codebook] = token
        targets[:, 2, codebook] = token
    weights[:, 2] = 1.0

    inputs[:, 3, 0] = train.token_ids_audio_end
    targets[:, 3, 0] = train.token_ids_model_end
    weights[:, 3] = 1.0

    memories = model.initial_memories(2)
    total_loss, ce_loss, _, metrics = model.forward_with_memories(
        jnp.asarray(inputs),
        jnp.asarray(targets),
        jnp.asarray(weights),
        memories,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    total = float(jax.device_get(total_loss))
    ce = float(jax.device_get(ce_loss))
    assert math.isfinite(total) and math.isfinite(ce)
    assert total < 30.0, f"audio auxiliary loss is still over-scaled: {total}"
    assert len(metrics) == train.VALIDATION_METRIC_SIZE

    q0_candidates = jnp.asarray(
        train.build_audio_codebook_candidate_token_ids(0, allow_stop=False),
        dtype=jnp.int32,
    )
    start = jnp.asarray([train.token_ids_audio_out], dtype=jnp.int32)
    q0_logits, hidden, next_memories = train.runtime_audio_frame_step(
        model,
        start,
        model.initial_memories(1),
        jnp.ones((1,), dtype=jnp.bool_),
        q0_candidates,
    )
    q0 = q0_candidates[jnp.argmax(q0_logits, axis=-1)]
    teacher_frame = jnp.asarray(targets[:1, 2, :], dtype=jnp.int32).at[:, 0].set(q0)
    aux_logits = model.project_audio_aux_teacher(hidden, teacher_frame)
    assert q0_logits.shape == (1, cfg.audio_codebook_size)
    assert aux_logits.shape == (1, 7, cfg.audio_codebook_size)
    frame = [int(jax.device_get(q0[0]))]
    frame.extend(
        train.audio_token_id(codebook, int(code))
        for codebook, code in enumerate(np.asarray(jax.device_get(jnp.argmax(aux_logits, axis=-1)))[0], start=1)
    )
    assert len(frame) == cfg.audio_codebooks
    _, _, after_frame = train.runtime_audio_frame_step(
        model,
        jnp.asarray([frame], dtype=jnp.int32),
        next_memories,
        jnp.ones((1,), dtype=jnp.bool_),
        q0_candidates,
    )
    assert len(after_frame) == cfg.num_layers

    generated, _, stop_token, stop_reason = train.generate_audio_frames_after_prefix(
        model,
        cfg.seed,
        [
            train.token_ids_session,
            train.token_ids_user,
            *train.encode_text("Say this aloud: smoke test."),
            train.token_ids_user_end,
            train.token_ids_model,
            train.token_ids_audio_out,
        ],
        max_tokens=16,
    )
    assert stop_token is None and stop_reason == "max_tokens"
    assert len(generated) == 16
    for frame_start in range(0, len(generated), cfg.audio_codebooks):
        frame_tokens = generated[frame_start : frame_start + cfg.audio_codebooks]
        assert [train.audio_code_from_token_id(token)[0] for token in frame_tokens] == list(range(cfg.audio_codebooks))

    optimizer = nnx.Optimizer(model, train.build_optimizer(total_steps=20), wrt=nnx.Param)
    train_loss, final_memories = train.train_step_stateful(
        model,
        optimizer,
        jnp.asarray(inputs.copy()),
        jnp.asarray(targets.copy()),
        jnp.asarray(weights.copy()),
        model.initial_memories(2),
        jnp.ones((2,), dtype=jnp.bool_),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    assert math.isfinite(float(jax.device_get(train_loss)))
    assert len(final_memories) == cfg.num_layers


def main() -> None:
    cfg = train.PropagatorConfig(
        hidden_size=32,
        num_layers=1,
        memory_key_size=8,
        memory_value_size=16,
        mlp_multiplier=2,
        train_unroll_len=4,
        batch_size=2,
        precision="float32",
        audio_codebook_size=16,
        eval_use_candidate_head=False,
    )
    initialize_globals(cfg)
    assert len(train.parse_eval_text_cases()) >= 8
    assert cfg.eval_use_candidate_head is False
    check_control_examples()
    check_weighted_sampler()
    check_plain_text_continuation()
    check_tiny_multimodal_forward(cfg)
    print("smoke_training: PASS")


if __name__ == "__main__":
    main()

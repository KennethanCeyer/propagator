#!/usr/bin/env python3
"""Lightweight Propagator regression audits.

This script is intentionally small and deterministic. It does not train a model.

It checks:
- sample_05_format_following protocol construction through the real train.py
  tokenizer/protocol builder.
- local posttrain repetition counts.
- cached source modality balance summaries, when cache metadata exists.

The protocol checks fail on errors. Dataset/cache imbalance checks are warnings
by default because the current repository is known to be imbalanced; pass
--strict-data to make those warnings fail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "data" / "regression" / "sample_05_format_following.jsonl"
DEFAULT_TOKENIZER = REPO_ROOT / "outputs" / "propagator-multimodal_1b" / "tokenizer.with_protocol_tokens.json"
DEFAULT_POSTTRAIN = REPO_ROOT / "data" / "datasets" / "propagator_posttrain_cleaned.jsonl"
DEFAULT_TRAIN_META = Path("/mnt/disks/propagator-cache/cache/propagator_train_fe55d3fdb5.meta.json")
DEFAULT_VAL_META = Path("/mnt/disks/propagator-cache/cache/propagator_val_2fc7bd25f3.meta.json")
DEFAULT_GENERATED_SAMPLE = REPO_ROOT / "outputs" / "propagator-multimodal_1b" / "step_300000" / "sample_05_format_following.txt"
DEFAULT_VAL_METRICS = REPO_ROOT / "outputs" / "propagator-multimodal_1b" / "step_300000" / "validation_metrics.json"
DEFAULT_RUN_CONFIG = REPO_ROOT / "outputs" / "propagator-multimodal_1b" / "run_config.json"
DEFAULT_MIX_FILE = REPO_ROOT / "data" / "mixes" / "propagator_dataset_mix.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{lineno}: expected object row")
            rows.append(row)
    return rows


def main_token(frame: Any) -> int:
    if isinstance(frame, list):
        if not frame:
            raise AssertionError("empty token frame")
        return int(frame[0])
    return int(frame)


def decode_ids(train: Any, ids: list[int]) -> str:
    if not ids:
        return ""
    return train.tokenizer.decode(ids, skip_special_tokens=True).strip()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def import_train_module() -> Any:
    os.environ.setdefault("SLACK_LOG_ENABLED", "0")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import train  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import train.py. Use the repository virtualenv, for example: "
            "./.venv/bin/python scripts/audit_prop_regressions.py"
        ) from exc
    return train


def init_train_protocol_globals(train: Any, tokenizer_path: Path) -> None:
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
    train.config = types.SimpleNamespace(
        listen_loss_weight=0.05,
        control_loss_weight=1.0,
        content_loss_weight=1.0,
        user_inner_loss_weight=0.05,
        allow_user_interrupts=False,
        synthesize_turn_silence=False,
        silence_end_tokens=0,
        enable_audio=False,
        audio_out_loss_weight=2.0,
        output_modality_loss_weight=2.0,
        image_recognition_only=True,
        image_input_resolution=160,
        image_max_input_resolution=192,
        image_patch_size=16,
        image_patch_vocab_size=1024,
        image_tokens_per_sample=64,
        tokenizer_path=str(tokenizer_path),
    )
    train.tokenizer = train.Tokenizer.from_file(str(tokenizer_path))
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


def token_name_map(train: Any) -> tuple[dict[str, int], dict[int, str]]:
    ids = dict(train.token_ids)
    rev = {int(v): str(k) for k, v in ids.items()}
    return ids, rev


def assert_protocol_row(train: Any, row: dict[str, Any], unroll_len: int) -> dict[str, Any]:
    row_id = str(row.get("id") or "<unnamed>")
    expected_text = str(row.get("expected_text") or row["output"][-1]["content"])
    checks = set(row.get("checks") or [])
    xs, ys, ws, meta = train.tokenize_duplex(row)
    ids, rev = token_name_map(train)
    x_main = [main_token(x) for x in xs]
    y_main = [main_token(y) for y in ys]

    def require(cond: bool, message: str) -> None:
        if not cond:
            raise AssertionError(f"{row_id}: {message}")

    require(len(xs) == len(ys) == len(ws), "input/target/weight lengths differ")
    require(len(xs) > 0, "empty tokenized stream")
    require(x_main[0] == ids["session"], "stream must begin with [SESSION]")
    require(y_main[0] == ids["listen"], "[SESSION] target must be [LISTEN]")
    require(x_main[1] == ids["user"], "second token must be [USER]")
    require(y_main[1] == ids["listen"], "[USER] target must be [LISTEN]")

    user_end_targets = [i for i, y in enumerate(y_main) if y == ids["user_end"]]
    model_end_targets = [i for i, y in enumerate(y_main) if y == ids["model_end"]]
    text_out_targets = [i for i, y in enumerate(y_main) if y == ids["text_out"]]

    require(len(user_end_targets) == 1, f"expected one [USER_END] target, got {len(user_end_targets)}")
    require(len(model_end_targets) == 1, f"expected one [MODEL_END] target, got {len(model_end_targets)}")
    require(len(text_out_targets) == 1, f"expected one [TEXT_OUTPUT] target, got {len(text_out_targets)}")

    user_end_idx = user_end_targets[0]
    model_idx = user_end_idx + 1
    text_out_idx = text_out_targets[0]
    model_end_idx = model_end_targets[0]

    require(all(y == ids["listen"] for y in y_main[1:user_end_idx]), "user phase must target [LISTEN]")
    require(x_main[model_idx] == ids["user_end"], "[USER_END] input must follow final user token")
    require(y_main[model_idx] == ids["model"], "[USER_END] target must be [MODEL]")
    require(x_main[text_out_idx] == ids["model"], "[MODEL] input must target [TEXT_OUTPUT]")
    require(text_out_idx == model_idx + 1, "[MODEL] transition must immediately follow [USER_END]")
    require(model_end_idx > text_out_idx, "[MODEL_END] target must follow [TEXT_OUTPUT]")
    require(float(ws[user_end_idx]) > 0.0, "final user token -> [USER_END] must be supervised")
    require(float(ws[model_idx]) > 0.0, "[USER_END] -> [MODEL] must be supervised")
    require(float(ws[text_out_idx]) > 0.0, "[MODEL] -> [TEXT_OUTPUT] must be supervised")
    require(float(ws[model_end_idx]) > 0.0, "answer -> [MODEL_END] must be supervised")

    control_ids = {
        ids["pad"],
        ids["session"],
        ids["user"],
        ids["model"],
        ids["listen"],
        ids["user_end"],
        ids["model_end"],
        ids["session_end"],
        ids["user_interrupt"],
        ids["audio_in"],
        ids.get("image_in", -1),
        ids["audio_out"],
        ids["silence"],
        ids["text_in"],
        ids["text_out"],
    }
    answer_ids: list[int] = []
    for y in y_main[text_out_idx:model_end_idx]:
        if y not in control_ids:
            answer_ids.append(y)
    decoded = decode_ids(train, answer_ids)
    require(normalize_text(decoded) == normalize_text(expected_text), f"decoded target answer {decoded!r} != {expected_text!r}")

    if "format_one_word" in checks:
        require(len(normalize_text(decoded).split()) == 1, f"expected one-word target, got {decoded!r}")
    if "chunk_boundary" in checks:
        require(model_idx >= unroll_len, f"expected response phase after chunk boundary {unroll_len}, got index {model_idx}")

    return {
        "id": row_id,
        "tokens": len(xs),
        "user_end_index": user_end_idx,
        "model_index": model_idx,
        "text_out_index": text_out_idx,
        "model_end_index": model_end_idx,
        "decoded_answer": decoded,
        "model_to_text_out_weight": float(ws[text_out_idx]),
        "meta": meta,
        "model_target_name": rev.get(y_main[model_idx], str(y_main[model_idx])),
        "text_out_target_name": rev.get(y_main[text_out_idx], str(y_main[text_out_idx])),
    }


def run_protocol_checks(args: argparse.Namespace) -> bool:
    train = import_train_module()
    init_train_protocol_globals(train, args.tokenizer)
    rows = load_jsonl(args.fixture)
    if not rows:
        raise SystemExit(f"No fixture rows found: {args.fixture}")
    print(f"[protocol] fixture={args.fixture}")
    ok = True
    for row in rows:
        try:
            result = assert_protocol_row(train, row, args.unroll_len)
        except AssertionError as exc:
            ok = False
            print(f"[protocol][FAIL] {exc}")
            continue
        print(
            "[protocol][OK] "
            f"id={result['id']} tokens={result['tokens']} "
            f"user_end={result['user_end_index']} model={result['model_index']} "
            f"text_out={result['text_out_index']} model_end={result['model_end_index']} "
            f"answer={result['decoded_answer']!r} "
            f"model_to_text_out_weight={result['model_to_text_out_weight']}"
        )
    return ok


def run_audio_alignment_check(args: argparse.Namespace) -> bool:
    train = import_train_module()
    init_train_protocol_globals(train, args.tokenizer)
    ids, _ = token_name_map(train)
    user_ids = [ids["text_in"], *train.encode_text("Say yes as speech.")]
    audio_frame_0 = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
    audio_frame_1 = [1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017]
    model_ids = [ids["audio_out"], audio_frame_0, audio_frame_1]
    xs, ys, ws, meta = train.tokenize_modal_exchange(user_ids, model_ids)
    x_main = [main_token(x) for x in xs]
    y_main = [main_token(y) for y in ys]

    def require(cond: bool, message: str) -> None:
        if not cond:
            raise AssertionError(f"audio_alignment: {message}")

    audio_out_targets = [i for i, y in enumerate(y_main) if y == ids["audio_out"]]
    model_end_targets = [i for i, y in enumerate(y_main) if y == ids["model_end"]]
    require(len(audio_out_targets) == 1, f"expected one [AUDIO_OUTPUT] target, got {len(audio_out_targets)}")
    require(len(model_end_targets) == 1, f"expected one [MODEL_END] target, got {len(model_end_targets)}")
    audio_out_idx = audio_out_targets[0]
    require(x_main[audio_out_idx] == ids["model"], "[MODEL] input must target [AUDIO_OUTPUT]")
    require(float(ws[audio_out_idx]) == 2.0, "[MODEL] -> [AUDIO_OUTPUT] should use audio_out_loss_weight=2.0")
    require(list(ys[audio_out_idx + 1]) == audio_frame_0, "first audio frame q0-q7 target was not preserved")
    require(list(ys[audio_out_idx + 2]) == audio_frame_1, "second audio frame q0-q7 target was not preserved")
    require(list(xs[audio_out_idx + 2]) == audio_frame_0, "teacher-forced previous audio frame input was not preserved")
    require(y_main[audio_out_idx + 3] == ids["model_end"], "last audio frame must target [MODEL_END]")
    print(
        "[audio][OK] "
        f"frames=2 audio_out_index={audio_out_idx} "
        f"audio_out_weight={float(ws[audio_out_idx])} "
        f"model_end_weight={float(ws[audio_out_idx + 3])} meta={meta}"
    )
    return True


def run_image_protocol_check(args: argparse.Namespace) -> bool:
    train = import_train_module()
    init_train_protocol_globals(train, args.tokenizer)
    ids, _ = token_name_map(train)
    if "image_in" not in ids:
        raise AssertionError("image_protocol: [IMAGE_INPUT] is not registered as a special token")

    user_ids = [
        ids["image_in"],
        *train.encode_text("image context: red mug on desk"),
        ids["text_in"],
        *train.encode_text("What object is visible?"),
    ]
    model_ids = [ids["text_out"], *train.encode_text("A red mug is visible.")]
    xs, ys, ws, meta = train.tokenize_modal_exchange(user_ids, model_ids)
    x_main = [main_token(x) for x in xs]
    y_main = [main_token(y) for y in ys]

    def require(cond: bool, message: str) -> None:
        if not cond:
            raise AssertionError(f"image_protocol: {message}")

    image_positions = [i for i, x in enumerate(x_main) if x == ids["image_in"]]
    require(len(image_positions) == 1, f"expected one [IMAGE_INPUT] input, got {len(image_positions)}")
    image_idx = image_positions[0]
    require(y_main[image_idx] == ids["listen"], "[IMAGE_INPUT] must be user-context input targeting [LISTEN]")
    require(float(ws[image_idx]) > 0.0, "[IMAGE_INPUT] listen transition must be supervised")
    require(ids["image_in"] not in y_main, "[IMAGE_INPUT] must not appear as an output target")
    require(ids["text_out"] in y_main, "image QA fixture must select [TEXT_OUTPUT]")
    require(ids["model_end"] in y_main, "image QA fixture must terminate with [MODEL_END]")

    pixels = np.zeros((24, 24, 3), dtype=np.uint8)
    pixels[..., 0] = 220
    pixels[:, 10:14, 1] = 180
    pixel_row = {"pixels": pixels, "question": "What color dominates?", "answer": "red"}
    img_xs, img_ys, img_ws, img_meta = train.tokenize_image_recognition(pixel_row, {})
    img_x_main = [main_token(x) for x in img_xs]
    visual_count = sum(1 for token_id in img_x_main if train.is_image_token_id(token_id))
    require(visual_count == train.config.image_tokens_per_sample, f"expected {train.config.image_tokens_per_sample} visual tokens, got {visual_count}")
    require(ids["image_in"] in img_x_main, "pixel image row did not include [IMAGE_INPUT]")
    require(ids["text_in"] in img_x_main, "pixel image row did not include [TEXT_INPUT] question prefix")
    try:
        train.tokenize_image_recognition(
            {"image_text": "A red mug is on a desk.", "question": "What object is visible?", "answer": "mug"},
            {},
        )
    except train.DataQualityError:
        pass
    else:
        raise AssertionError("image_protocol: caption-only image row should be rejected")
    print(
        "[image][OK] "
        f"image_index={image_idx} image_weight={float(ws[image_idx])} "
        f"text_out_targets={y_main.count(ids['text_out'])} "
        f"visual_tokens={visual_count} meta={meta} pixel_meta={img_meta}"
    )
    return True


def run_runtime_state_check() -> bool:
    train = import_train_module()
    import jax.numpy as jnp

    cfg = train.PropagatorConfig(
        hidden_size=16,
        num_layers=1,
        memory_key_size=8,
        memory_value_size=8,
        associative_groups=2,
        mlp_multiplier=2,
        use_swiglu=True,
        moe_num_experts=1,
        train_unroll_len=4,
        audio_codebook_size=16,
        precision="float32",
    )
    train.config = cfg
    model = train.PropagatorModel(cfg, 64, rngs=train.nnx.Rngs(0))

    def require(cond: bool, message: str) -> None:
        if not cond:
            raise AssertionError(f"runtime_state: {message}")

    memories = model.initial_memories(batch_size=2)
    require(len(memories) == 1, "tiny test should have one layer")
    require(tuple(memories[0].shape) == (2, 8, 8), f"unexpected memory shape {memories[0].shape}")
    require(float(jnp.max(jnp.abs(memories[0]))) == 0.0, "initial memories must be zero")

    stale = tuple(jnp.ones_like(m) for m in memories)
    reset = model.reset_memories(stale, jnp.array([1, 0], dtype=jnp.bool_))
    reset_sums = jnp.sum(reset[0], axis=(1, 2))
    require(float(reset_sums[0]) == 0.0, "reset_mask=True lane was not zeroed")
    require(float(reset_sums[1]) == 64.0, "reset_mask=False lane should preserve stale memory")

    _, stepped = model.step_hidden(
        jnp.array([3, 4], dtype=jnp.int32),
        memories,
        jnp.array([1, 0], dtype=jnp.bool_),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    lane_norms = jnp.linalg.norm(stepped[0].reshape(2, -1), axis=1)
    require(float(lane_norms[0]) > 0.0, "valid=True lane did not update memory")
    require(float(lane_norms[1]) == 0.0, "valid=False lane changed memory")

    single = model.initial_memories(batch_size=1)
    _, mem_a = model.step_hidden(
        jnp.array([3], dtype=jnp.int32),
        single,
        jnp.array([1], dtype=jnp.bool_),
        jnp.array([0], dtype=jnp.int32),
    )
    _, mem_ab = model.step_hidden(
        jnp.array([4], dtype=jnp.int32),
        mem_a,
        jnp.array([1], dtype=jnp.bool_),
        jnp.array([1], dtype=jnp.int32),
    )
    reset_before_b = model.reset_memories(mem_a, jnp.array([1], dtype=jnp.bool_))
    _, mem_b_after_reset = model.step_hidden(
        jnp.array([4], dtype=jnp.int32),
        reset_before_b,
        jnp.array([1], dtype=jnp.bool_),
        jnp.array([1], dtype=jnp.int32),
    )
    carry_delta = float(jnp.linalg.norm(mem_ab[0] - mem_b_after_reset[0]))
    require(carry_delta > 0.0, "carried memory and reset memory paths should differ")

    print(
        "[state-runtime][OK] "
        f"initial_shape={tuple(memories[0].shape)} "
        f"reset_sums={[float(x) for x in reset_sums]} "
        f"valid_lane_norms={[float(x) for x in lane_norms]} "
        f"carry_vs_reset_delta={carry_delta:.6f}"
    )
    return True


def audit_state_source(path: Path, strict: bool) -> bool:
    if not path.exists():
        print(f"[state][WARN] source missing: {path}")
        return not strict
    source = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "StatefulChunkSampler": "StatefulChunkSampler" in source,
        "reset_mask": "reset_mask" in source,
        "stop_gradient": "stop_gradient" in source,
        "forward_with_memories": "forward_with_memories" in source,
        "chunk_positions": "chunk_positions" in source,
    }
    print(f"[state] source={path} checks={checks}")
    warnings: list[str] = []
    for name, passed in checks.items():
        if not passed:
            warnings.append(f"missing source invariant: {name}")
    session_reset_markers = [
        'token_ids["session"]',
        "token_ids['session']",
        "token_ids_session",
    ]
    has_session_token_reference = any(marker in source for marker in session_reset_markers)
    has_session_hard_reset = "session" in source and "reset_mask" in source and "token_ids" in source and "reset" in source
    if has_session_token_reference and not has_session_hard_reset:
        warnings.append("[SESSION] appears to be protocol-only; memory reset must be provided by reset masks")
    if checks["reset_mask"]:
        print("[state][NOTE] recurrent memory reset/carry is controlled by reset masks and stream state; keep explicit leakage tests for runtime paths")
    for warning in warnings:
        print(f"[state][WARN] {warning}")
    return (not warnings) or (not strict)


def audit_posttrain(path: Path, strict: bool) -> bool:
    if not path.exists():
        print(f"[data][WARN] posttrain file missing: {path}")
        return not strict
    counters: Counter[str] = Counter()
    total = 0
    for row in load_jsonl(path):
        total += 1
        blob = json.dumps(row, ensure_ascii=False).lower()
        if "what is your name" in blob:
            counters["name_question"] += 1
        if "repeat only the code word" in blob:
            counters["repeat_codeword"] += 1
        if "json" in blob:
            counters["json"] += 1
        if "yaml" in blob:
            counters["yaml"] += 1
        if "markdown" in blob or "table" in blob:
            counters["markdown_or_table"] += 1
    print(f"[data] posttrain={path} rows={total} counts={dict(counters)}")
    warnings: list[str] = []
    if total:
        if counters["name_question"] / total > 0.02:
            warnings.append(f"name_question share is {counters['name_question'] / total:.1%}")
        if counters["repeat_codeword"] / total > 0.02:
            warnings.append(f"repeat_codeword share is {counters['repeat_codeword'] / total:.1%}")
    for warning in warnings:
        print(f"[data][WARN] {warning}")
    return (not warnings) or (not strict)


def audit_cache_meta(path: Path, label: str, strict: bool) -> bool:
    if not path.exists():
        print(f"[cache][WARN] {label} meta missing: {path}")
        return not strict
    data = json.loads(path.read_text(encoding="utf-8"))
    comps = data.get("components") or []
    total = 0
    by_mode: Counter[str] = Counter()
    for comp in comps:
        chunks = int(comp.get("chunks") or comp.get("num_chunks") or 0)
        total += chunks
        mode = str(comp.get("mode") or "unknown")
        by_mode[mode] += chunks
    print(f"[cache] {label}={path} total_chunks={total} by_mode={dict(by_mode)}")
    warnings: list[str] = []
    if total:
        plain_share = by_mode["plain_text"] / total
        if plain_share > 0.50:
            warnings.append(f"{label} plain_text share is {plain_share:.1%}")
        if by_mode["duplex_chat"] / total < 0.05:
            warnings.append(f"{label} duplex_chat share is {by_mode['duplex_chat'] / total:.1%}")
    for warning in warnings:
        print(f"[cache][WARN] {warning}")
    return (not warnings) or (not strict)


def audit_mix_file(path: Path, strict: bool) -> bool:
    if not path.exists():
        print(f"[mix][WARN] mix file missing: {path}")
        return not strict
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: expected JSON list")
    total_weight = 0.0
    by_mode: Counter[str] = Counter()
    local_files: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: every mix entry must be an object")
        weight = float(row.get("weight") or 0.0)
        total_weight += weight
        by_mode[str(row.get("mode") or "unknown")] += weight
        if row.get("data_files"):
            local_files[str(row["data_files"])] += weight
    by_mode_pct = {k: round(100.0 * v / total_weight, 2) for k, v in by_mode.items()} if total_weight else {}
    print(f"[mix] path={path} entries={len(rows)} total_weight={total_weight:.4f} by_mode_pct={by_mode_pct} local_files={dict(local_files)}")
    warnings: list[str] = []
    if total_weight <= 0:
        warnings.append("total mix weight is zero")
    plain_share = (by_mode["plain_text"] / total_weight) if total_weight else 0.0
    if plain_share > 0.25:
        warnings.append(f"plain_text configured share is high: {plain_share:.1%}")
    identity_weight = sum(v for k, v in local_files.items() if "identity" in k)
    if total_weight and identity_weight / total_weight > 0.03:
        warnings.append(f"identity configured share is high: {identity_weight / total_weight:.1%}")
    for warning in warnings:
        print(f"[mix][WARN] {warning}")
    return (not warnings) or (not strict)


def audit_local_mix_tokenization(args: argparse.Namespace, strict: bool) -> bool:
    train = import_train_module()
    init_train_protocol_globals(train, args.tokenizer)
    ids, _ = token_name_map(train)
    mix_path = args.mix_file
    if not mix_path.exists():
        print(f"[local-tokenize][WARN] mix file missing: {mix_path}")
        return not strict
    mix_rows = json.loads(mix_path.read_text(encoding="utf-8"))
    local_specs = [row for row in mix_rows if isinstance(row, dict) and row.get("data_files")]
    warnings: list[str] = []
    total_checked = 0
    print(f"[local-tokenize] mix={mix_path} local_specs={len(local_specs)} limit_per_file={args.local_tokenize_limit}")
    for spec in local_specs:
        rel = Path(str(spec["data_files"]))
        path = rel if rel.is_absolute() else REPO_ROOT / rel
        if not path.exists():
            warning = f"local data file missing: {path}"
            print(f"[local-tokenize][WARN] {warning}")
            warnings.append(warning)
            continue
        rows = load_jsonl(path)
        task_counts: Counter[str] = Counter()
        stats_total: Counter[str] = Counter()
        user_prompts: Counter[str] = Counter()
        checked = 0
        mode = str(spec.get("mode") or "duplex_chat")
        for row in rows[: max(0, int(args.local_tokenize_limit))]:
            checked += 1
            total_checked += 1
            task_counts[str(row.get("task") or "<none>")] += 1
            messages = row.get("output") or []
            if messages and not isinstance(messages, list):
                warning = f"{path}: row {checked} has non-list output"
                print(f"[local-tokenize][WARN] {warning}")
                warnings.append(warning)
                continue
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "")
                content = str(message.get("content") or "")
                if role == "user":
                    user_prompts[normalize_text(content)] += 1
                if role in {"assistant", "model", "gpt"}:
                    forbidden = [marker for marker in ("[AUDIO_INPUT]", "[AUDIO_OUTPUT]", "[IMAGE_INPUT]") if marker in content]
                    if forbidden:
                        warning = f"{path}: assistant text contains raw modality marker(s) {forbidden} in row id={row.get('id')}"
                        print(f"[local-tokenize][WARN] {warning}")
                        warnings.append(warning)
            try:
                xs, ys, ws, meta = train.tokenize_row_by_mode(row, mode, spec)
            except Exception as exc:
                warning = f"{path}: tokenize_row_by_mode({mode}) failed for row id={row.get('id')}: {type(exc).__name__}: {exc}"
                print(f"[local-tokenize][WARN] {warning}")
                warnings.append(warning)
                continue
            if not (len(xs) == len(ys) == len(ws) and len(xs) > 0):
                warning = f"{path}: invalid tokenized lengths for row id={row.get('id')}"
                print(f"[local-tokenize][WARN] {warning}")
                warnings.append(warning)
            for key, value in meta.items():
                stats_total[str(key)] += int(value)
            user_blob = " ".join(
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and str(message.get("role") or "") == "user"
            )
            if "[IMAGE_INPUT]" in user_blob or mode == "image_recognition":
                x_main = [main_token(x) for x in xs]
                if ids.get("image_in", -1) not in x_main:
                    warning = f"{path}: row id={row.get('id')} contains [IMAGE_INPUT] text but no image_in token in input"
                    print(f"[local-tokenize][WARN] {warning}")
                    warnings.append(warning)
        top_repeat = user_prompts.most_common(1)
        unique_prompts = len(user_prompts)
        print(
            "[local-tokenize][OK] "
            f"path={path} checked={checked} rows={len(rows)} unique_user_prompts={unique_prompts} "
            f"top_repeat={top_repeat[:1]} tasks={dict(task_counts)} stats={dict(stats_total)}"
        )
    if total_checked == 0:
        warning = "no local mix rows were tokenized"
        print(f"[local-tokenize][WARN] {warning}")
        warnings.append(warning)
    return (not warnings) or (not strict)


def audit_generated_sample(path: Path, strict: bool) -> bool:
    if not path.exists():
        print(f"[sample][WARN] generated sample missing: {path}")
        return not strict
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_model = False
    saw_text_out = False
    saw_model_end = False
    content_tokens: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "## model stream":
            in_model = True
            continue
        if in_model and stripped.startswith("## "):
            break
        if not in_model or "->" not in line:
            continue
        _, rhs = line.split("->", 1)
        token = rhs.strip()
        if token == "[TEXT_OUTPUT]":
            saw_text_out = True
            continue
        if not saw_text_out:
            continue
        if token == "[MODEL_END]":
            saw_model_end = True
            break
        if token.startswith("[") and token.endswith("]"):
            continue
        content_tokens.append(token)

    prefix = "".join(content_tokens[:16]).strip()
    compact_prefix = normalize_text(prefix.replace(" ", ""))
    warnings: list[str] = []
    if not saw_text_out:
        warnings.append("generated sample never selected [TEXT_OUTPUT]")
    if not saw_model_end:
        warnings.append("generated sample did not terminate with [MODEL_END]")
    if content_tokens and not compact_prefix.startswith("lima"):
        warnings.append(f"generated sample starts with {prefix!r}, expected one-word answer 'Lima'")
    if len(content_tokens) > 3:
        warnings.append(f"generated sample emitted {len(content_tokens)} content tokens, expected one word")
    joined = "".join(content_tokens).lower()
    if "propagator" in joined:
        warnings.append("generated sample contains identity intrusion 'Propagator'")
    if "sand" in joined:
        warnings.append("generated sample contains repetitive unrelated 'sand' continuation")

    print(
        f"[sample] path={path} text_out={saw_text_out} model_end={saw_model_end} "
        f"content_tokens={len(content_tokens)} prefix={prefix!r}"
    )
    for warning in warnings:
        print(f"[sample][WARN] {warning}")
    return (not warnings) or (not strict)


def audit_run_config(path: Path, strict: bool) -> bool:
    if not path.exists():
        print(f"[config][WARN] run config missing: {path}")
        return not strict
    data = json.loads(path.read_text(encoding="utf-8"))
    interesting = {
        "stateful_train": data.get("stateful_train"),
        "stateful_validation": data.get("stateful_validation"),
        "train_unroll_len": data.get("train_unroll_len"),
        "validation_batches": data.get("validation_batches"),
        "audio_backend": data.get("audio_backend"),
        "audio_sample_rate": data.get("audio_sample_rate"),
        "audio_codebooks": data.get("audio_codebooks"),
        "audio_codebook_size": data.get("audio_codebook_size"),
        "audio_frame_rate": data.get("audio_frame_rate"),
        "early_stopping_patience": data.get("early_stopping_patience"),
        "early_stopping_min_delta": data.get("early_stopping_min_delta"),
    }
    print(f"[config] path={path} {interesting}")
    warnings: list[str] = []
    if data.get("train_unroll_len", 0) < 64:
        warnings.append(f"run used short train_unroll_len={data.get('train_unroll_len')}")
    if data.get("stateful_train") is not True or data.get("stateful_validation") is not True:
        warnings.append("run was not stateful for both train and validation")
    for warning in warnings:
        print(f"[config][WARN] {warning}")
    return (not warnings) or (not strict)


def audit_validation_metrics(path: Path, strict: bool) -> bool:
    if not path.exists():
        print(f"[metrics][WARN] validation metrics missing: {path}")
        return not strict
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = [
        "decision_acc",
        "listen_acc",
        "user_end_acc",
        "model_end_acc",
        "text_token_acc",
        "text_task_acc",
        "asr_task_acc",
        "tts_task_acc",
        "duplex_task_acc",
        "audio_main_acc",
        "audio_aux_token_acc",
        "audio_all_codebook_frame_exact_acc",
    ]
    summary = {k: data.get(k) for k in keys if k in data}
    try:
        train = import_train_module()
        composite = train.validation_composite_score(data)
        data.update(composite)
        summary.update({k: composite[k] for k in sorted(composite)})
    except Exception as exc:
        print(f"[metrics][WARN] could not compute validation composite score: {type(exc).__name__}: {exc}")
    print(f"[metrics] path={path} {summary}")
    warnings: list[str] = []
    if str(data.get("duplex_task_acc")).lower() == "nan":
        warnings.append("duplex_task_acc is NaN")
    if float(data.get("tts_task_acc") or 0.0) < 0.45:
        warnings.append(f"tts_task_acc is low: {data.get('tts_task_acc')}")
    if float(data.get("text_task_acc") or 0.0) < 0.50:
        warnings.append(f"text_task_acc is low: {data.get('text_task_acc')}")
    if float(data.get("audio_all_codebook_frame_exact_acc") or 0.0) == 0.0:
        warnings.append("audio_all_codebook_frame_exact_acc is zero")
    for warning in warnings:
        print(f"[metrics][WARN] {warning}")
    return (not warnings) or (not strict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--posttrain", type=Path, default=DEFAULT_POSTTRAIN)
    parser.add_argument("--train-meta", type=Path, default=DEFAULT_TRAIN_META)
    parser.add_argument("--val-meta", type=Path, default=DEFAULT_VAL_META)
    parser.add_argument("--generated-sample", type=Path, default=DEFAULT_GENERATED_SAMPLE)
    parser.add_argument("--validation-metrics", type=Path, default=DEFAULT_VAL_METRICS)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--mix-file", type=Path, default=DEFAULT_MIX_FILE)
    parser.add_argument("--train-source", type=Path, default=REPO_ROOT / "train.py")
    parser.add_argument("--unroll-len", type=int, default=32)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument("--strict-sample", action="store_true")
    parser.add_argument("--strict-source", action="store_true")
    parser.add_argument("--skip-runtime-state", action="store_true")
    parser.add_argument("--local-tokenize-limit", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = run_protocol_checks(args)
    ok = run_audio_alignment_check(args) and ok
    ok = run_image_protocol_check(args) and ok
    if not args.skip_runtime_state:
        ok = run_runtime_state_check() and ok
    if not args.protocol_only:
        ok = audit_state_source(args.train_source, args.strict_source) and ok
        ok = audit_generated_sample(args.generated_sample, args.strict_sample) and ok
        ok = audit_run_config(args.run_config, args.strict_data) and ok
        ok = audit_validation_metrics(args.validation_metrics, args.strict_data) and ok
        ok = audit_posttrain(args.posttrain, args.strict_data) and ok
        ok = audit_mix_file(args.mix_file, args.strict_data) and ok
        ok = audit_local_mix_tokenization(args, args.strict_data) and ok
        ok = audit_cache_meta(args.train_meta, "train", args.strict_data) and ok
        ok = audit_cache_meta(args.val_meta, "val", args.strict_data) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# coding: utf-8

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from datasets import load_dataset
from dotenv import load_dotenv
from flax import nnx
from pydantic_settings import BaseSettings
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

load_dotenv()

TRAINING_RUN_NAME = "propagator-duplex"
DEFAULT_OUTPUT_ROOT = str(Path("outputs") / TRAINING_RUN_NAME)

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[SESSION]",
    "[USER]",
    "[MODEL]",
    "[LISTEN]",
    "[USER_END]",
    "[MODEL_END]",
    "[SESSION_END]",
    "[USER_INTERRUPT]",
]


class PropagatorConfig(BaseSettings):
    hidden_size: int = 512
    num_layers: int = 8
    memory_key_size: int = 128
    memory_value_size: int = 256
    mlp_multiplier: int = 4

    train_unroll_len: int = 512
    seq_len: int | None = None
    batch_size: int = 8

    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    epochs: int = 2
    max_train_steps: int | None = None
    seed: int = 42

    eval_every: int = 500
    checkpoint_every: int = 2000
    sample_gen_len: int = 256
    sample_chunks: str = '["Hello", "could you", "tell me", "what", "your name", "is?"]'
    temperature: float = 0.7
    top_k: int = 50

    write_rate: float = 0.1
    forget_rate: float = 0.02
    memory_l2: float = 1e-6

    user_inner_loss_weight: float = 0.01
    listen_loss_weight: float = 0.05
    control_loss_weight: float = 1.0
    interrupt_input_loss_weight: float = 0.0
    content_loss_weight: float = 1.0
    min_supervised_targets: int = 1

    output_root: str = DEFAULT_OUTPUT_ROOT
    dataset_name: str = "xinrongzhang2022/Duplex-UltraChat"
    dataset_mode: str = "duplex_chat"
    dataset_split: str = "train"
    validation_split: str = "train"
    validation_skip_rows: int | None = None

    max_train_chunks: int = 1_000_000
    max_val_chunks: int = 50_000
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    streaming: bool = True

    stateful_train: bool = True
    stateful_validation: bool = True
    validation_batches: int = 64

    tokenizer_path: str = "assets/tokenizer-byte-bpe-4096.json"
    tokenizer_vocab_size: int = 4096
    tokenizer_train_rows: int = 200_000
    tokenizer_min_frequency: int = 2
    force_train_tokenizer: bool = False
    require_byte_level_bpe: bool = True
    save_augmented_tokenizer: bool = True
    precision: str = "float16"

    inference_candidate_vocab_size: int = 8192
    eval_use_candidate_head: bool = True

    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0


config: PropagatorConfig
tokenizer: Tokenizer
vocab_size: int
token_ids: dict[str, int]
tokenizer_fingerprint: str
candidate_token_ids_host: np.ndarray

train_input_tokens: np.ndarray
train_target_tokens: np.ndarray
train_loss_weights: np.ndarray
train_stream_ids: np.ndarray
train_chunk_positions: np.ndarray

val_input_tokens: np.ndarray
val_target_tokens: np.ndarray
val_loss_weights: np.ndarray
val_stream_ids: np.ndarray
val_chunk_positions: np.ndarray

token_ids_pad: int
token_ids_unk: int
token_ids_session: int
token_ids_user: int
token_ids_model: int
token_ids_listen: int
token_ids_user_end: int
token_ids_model_end: int
token_ids_session_end: int
token_ids_user_interrupt: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Propagator on a chunk-decision duplex streaming protocol.")

    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--memory-key-size", type=int)
    parser.add_argument("--memory-value-size", type=int)
    parser.add_argument("--mlp-multiplier", type=int)

    parser.add_argument("--train-unroll-len", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--sample-gen-len", type=int)
    parser.add_argument("--sample-chunks", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)

    parser.add_argument("--write-rate", type=float)
    parser.add_argument("--forget-rate", type=float)
    parser.add_argument("--memory-l2", type=float)

    parser.add_argument("--user-inner-loss-weight", type=float)
    parser.add_argument("--listen-loss-weight", type=float)
    parser.add_argument("--control-loss-weight", type=float)
    parser.add_argument("--interrupt-input-loss-weight", type=float)
    parser.add_argument("--content-loss-weight", type=float)
    parser.add_argument("--min-supervised-targets", type=int)

    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-mode", type=str, choices=["instruction_chat", "duplex_chat"])
    parser.add_argument("--dataset-split", type=str)
    parser.add_argument("--validation-split", type=str)
    parser.add_argument("--validation-skip-rows", type=int)

    parser.add_argument("--max-train-chunks", type=int)
    parser.add_argument("--max-val-chunks", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")

    parser.add_argument("--stateful-train", action="store_true")
    parser.add_argument("--stateless-train", action="store_true")
    parser.add_argument("--stateful-validation", action="store_true")
    parser.add_argument("--stateless-validation", action="store_true")
    parser.add_argument("--validation-batches", type=int)

    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--tokenizer-vocab-size", type=int)
    parser.add_argument("--tokenizer-train-rows", type=int)
    parser.add_argument("--tokenizer-min-frequency", type=int)
    parser.add_argument("--force-train-tokenizer", action="store_true")
    parser.add_argument("--no-require-byte-level-bpe", action="store_true")
    parser.add_argument("--save-augmented-tokenizer", action="store_true")
    parser.add_argument("--no-save-augmented-tokenizer", action="store_true")
    parser.add_argument("--precision", type=str, choices=["float32", "float16"])

    parser.add_argument("--inference-candidate-vocab-size", type=int)
    parser.add_argument("--eval-use-candidate-head", action="store_true")
    parser.add_argument("--eval-use-full-head", action="store_true")

    parser.add_argument("--optimizer", type=str, choices=["adamw", "lion"])
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--label-smoothing", type=float)

    return parser.parse_args()


def build_config() -> PropagatorConfig:
    base_config = PropagatorConfig()
    cli_args = parse_args()
    raw_updates = vars(cli_args)
    updates = {key: value for key, value in raw_updates.items() if value is not None}

    if raw_updates.get("no_streaming"):
        updates["streaming"] = False
    if raw_updates.get("stateless_train"):
        updates["stateful_train"] = False
    if raw_updates.get("stateless_validation"):
        updates["stateful_validation"] = False
    if raw_updates.get("eval_use_full_head"):
        updates["eval_use_candidate_head"] = False
    if raw_updates.get("no_save_augmented_tokenizer"):
        updates["save_augmented_tokenizer"] = False
    if raw_updates.get("no_require_byte_level_bpe"):
        updates["require_byte_level_bpe"] = False

    for key in (
        "no_streaming",
        "stateless_train",
        "stateless_validation",
        "eval_use_full_head",
        "no_save_augmented_tokenizer",
        "no_require_byte_level_bpe",
    ):
        updates.pop(key, None)

    cfg = base_config.model_copy(update=updates)

    if cfg.seq_len is not None:
        cfg = cfg.model_copy(update={"train_unroll_len": cfg.seq_len})
    if cfg.max_train_rows is not None:
        cfg = cfg.model_copy(update={"max_train_chunks": cfg.max_train_rows})
    if cfg.max_val_rows is not None:
        cfg = cfg.model_copy(update={"max_val_chunks": cfg.max_val_rows})

    return cfg


def tokenizer_file_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


def json_contains_type(value, expected_type: str) -> bool:
    if isinstance(value, dict):
        if value.get("type") == expected_type:
            return True
        return any(json_contains_type(v, expected_type) for v in value.values())
    if isinstance(value, list):
        return any(json_contains_type(v, expected_type) for v in value)
    return False


def tokenizer_json_is_byte_level_bpe(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    model_ok = data.get("model", {}).get("type") == "BPE"
    pre_tokenizer_ok = json_contains_type(data.get("pre_tokenizer"), "ByteLevel")
    decoder_ok = json_contains_type(data.get("decoder"), "ByteLevel")
    return bool(model_ok and pre_tokenizer_ok and decoder_ok)


def iter_tokenizer_training_texts():
    ds = load_dataset(config.dataset_name, split=config.dataset_split, streaming=config.streaming)
    produced = 0
    rows = 0

    for row in tqdm(ds, desc="Training tokenizer", total=config.tokenizer_train_rows):
        if rows >= config.tokenizer_train_rows:
            break
        rows += 1

        try:
            if config.dataset_mode == "duplex_chat" and "output" in row:
                for role, content, is_idle in read_duplex_events(row):
                    if not is_idle and content.strip():
                        produced += 1
                        yield content
            elif "conversations" in row:
                for msg in row["conversations"]:
                    content = msg.get("value", "")
                    if content:
                        produced += 1
                        yield str(content)
            else:
                for key in ("text", "content", "prompt", "response"):
                    if key in row and row[key]:
                        produced += 1
                        yield str(row[key])
        except Exception as exc:
            if rows <= 10:
                print(f"Tokenizer training row skipped: {exc}")
            continue

    if produced == 0:
        raise RuntimeError("No text was yielded for tokenizer training")


def train_byte_level_bpe_tokenizer(path: Path) -> Tokenizer:
    print(
        f"Training local byte-level BPE tokenizer: path={path}, "
        f"vocab_size={config.tokenizer_vocab_size}, rows={config.tokenizer_train_rows}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        bpe_model = models.BPE(unk_token="[UNK]", byte_fallback=True)
    except TypeError:
        bpe_model = models.BPE(unk_token="[UNK]")

    tok = Tokenizer(bpe_model)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        min_frequency=config.tokenizer_min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_tokenizer_training_texts(), trainer=trainer)
    tok.save(str(path))

    if not tokenizer_json_is_byte_level_bpe(path):
        raise RuntimeError(f"Trained tokenizer is not recognized as byte-level BPE: {path}")

    return tok


def load_or_train_tokenizer() -> Tokenizer:
    path = Path(config.tokenizer_path)
    should_train = config.force_train_tokenizer or not path.exists()

    if path.exists() and config.require_byte_level_bpe and not tokenizer_json_is_byte_level_bpe(path):
        print(f"Existing tokenizer is not byte-level BPE, retraining: {path}")
        should_train = True

    if should_train:
        return train_byte_level_bpe_tokenizer(path)

    print(f"Loading tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def ensure_special_tokens(tokenizer_obj: Tokenizer) -> dict[str, int]:
    missing = [tok for tok in SPECIAL_TOKENS if tokenizer_obj.token_to_id(tok) is None]
    if missing:
        print(f"Adding missing special tokens to tokenizer: {missing}")
        tokenizer_obj.add_special_tokens(missing)
        Path(config.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer_obj.save(config.tokenizer_path)

    ids = {}
    for tok in SPECIAL_TOKENS:
        idx = tokenizer_obj.token_to_id(tok)
        if idx is None:
            raise ValueError(f"Failed to add special token to tokenizer: {tok}")
        ids[tok.strip("[]").lower()] = int(idx)
    return ids

def save_tokenizer_snapshot() -> None:
    if not config.save_augmented_tokenizer:
        return
    output_dir = Path(config.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.with_protocol_tokens.json"))


def canonical_role(role: str) -> str:
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    return role


def encode_text(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def control_token_ids() -> set[int]:
    return {
        token_ids["session"],
        token_ids["user"],
        token_ids["model"],
        token_ids["listen"],
        token_ids["user_end"],
        token_ids["model_end"],
        token_ids["session_end"],
        token_ids["user_interrupt"],
        token_ids["pad"],
        token_ids["unk"],
    }


def decision_token_ids() -> set[int]:
    return {
        token_ids["listen"],
        token_ids["user_end"],
        token_ids["user_interrupt"],
    }


def is_control_id(token_id: int) -> bool:
    return token_id in control_token_ids()


def default_loss_weight_for_target(target_id: int) -> float:
    if target_id == token_ids["pad"]:
        return 0.0
    if target_id == token_ids["listen"]:
        return float(config.listen_loss_weight)
    if is_control_id(target_id):
        return float(config.control_loss_weight)
    return float(config.content_loss_weight)


def pad_to_len(values: list[int], length: int, pad_value: int) -> list[int]:
    values = values[:length]
    return values + [pad_value] * (length - len(values))


def pad_weights(values: list[float], length: int) -> list[float]:
    values = values[:length]
    return values + [0.0] * (length - len(values))


def read_duplex_events(row: dict) -> list[tuple[str, str, bool]]:
    if "output" not in row:
        raise KeyError("Duplex row must contain an output field")

    events = []
    for event in row["output"]:
        role = canonical_role(str(event.get("role", "")))
        content = event.get("content", "")
        if content is None or role not in {"user", "assistant"}:
            continue

        content = str(content)
        is_idle = content == "<idle>"
        if not is_idle and not content.strip():
            continue

        events.append((role, content, is_idle))

    return events


def non_idle_events(row: dict) -> list[tuple[str, str]]:
    return [(role, content) for role, content, is_idle in read_duplex_events(row) if not is_idle]


def add_target_stats(stats: dict[str, int], target_id: int, weight: float) -> None:
    if weight <= 0.0 or target_id == token_ids["pad"]:
        stats["ignored"] += 1
        return
    if target_id == token_ids["listen"]:
        stats["listen"] += 1
    elif target_id == token_ids["user_end"]:
        stats["user_end"] += 1
        stats["control"] += 1
    elif target_id == token_ids["user_interrupt"]:
        stats["interrupt"] += 1
        stats["control"] += 1
    elif target_id == token_ids["model_end"]:
        stats["model_end"] += 1
        stats["control"] += 1
    elif is_control_id(target_id):
        stats["control"] += 1
    else:
        stats["content"] += 1


def remove_target_stats(stats: dict[str, int], target_id: int, weight: float) -> None:
    if weight <= 0.0 or target_id == token_ids["pad"]:
        stats["ignored"] -= 1
        return
    if target_id == token_ids["listen"]:
        stats["listen"] -= 1
    elif target_id == token_ids["user_end"]:
        stats["user_end"] -= 1
        stats["control"] -= 1
    elif target_id == token_ids["user_interrupt"]:
        stats["interrupt"] -= 1
        stats["control"] -= 1
    elif target_id == token_ids["model_end"]:
        stats["model_end"] -= 1
        stats["control"] -= 1
    elif is_control_id(target_id):
        stats["control"] -= 1
    else:
        stats["content"] -= 1


def tokenize_duplex(row: dict) -> tuple[list[int], list[int], list[float], dict[str, int]]:
    events = non_idle_events(row)

    in_ids: list[int] = []
    tr_ids: list[int] = []
    weights: list[float] = []
    stats = {
        "listen": 0,
        "user_end": 0,
        "model_end": 0,
        "interrupt": 0,
        "content": 0,
        "control": 0,
        "ignored": 0,
    }

    user_open = False
    model_open = False
    last_user_token_index: int | None = None
    pending_model_token_index: int | None = None

    def add(input_id: int, target_id: int, weight_override: float | None = None) -> int:
        w = default_loss_weight_for_target(target_id) if weight_override is None else float(weight_override)
        in_ids.append(int(input_id))
        tr_ids.append(int(target_id))
        weights.append(float(w))
        add_target_stats(stats, int(target_id), float(w))
        return len(in_ids) - 1

    def set_target(index: int, target_id: int, weight_override: float | None = None) -> None:
        old_target = tr_ids[index]
        old_weight = weights[index]
        remove_target_stats(stats, int(old_target), float(old_weight))

        new_weight = default_loss_weight_for_target(target_id) if weight_override is None else float(weight_override)
        tr_ids[index] = int(target_id)
        weights[index] = float(new_weight)
        add_target_stats(stats, int(target_id), float(new_weight))

    def start_user_if_needed() -> None:
        nonlocal user_open
        if not user_open:
            add(token_ids["user"], token_ids["listen"])
            user_open = True

    def add_user_chunk(tokens: list[int]) -> None:
        nonlocal last_user_token_index
        if not tokens:
            return

        for i, token_id in enumerate(tokens):
            is_last_token_in_chunk = i == len(tokens) - 1
            if is_last_token_in_chunk:
                last_user_token_index = add(token_id, token_ids["listen"])
            else:
                add(token_id, token_ids["listen"], config.user_inner_loss_weight)

    def close_user_to_model() -> None:
        nonlocal user_open, last_user_token_index
        if user_open and last_user_token_index is not None:
            set_target(last_user_token_index, token_ids["user_end"], config.control_loss_weight)
        add(token_ids["user_end"], token_ids["model"], config.control_loss_weight)
        user_open = False
        last_user_token_index = None

    def start_model_with_token(first_token: int) -> None:
        nonlocal model_open, pending_model_token_index
        add(token_ids["model"], first_token, config.content_loss_weight)
        pending_model_token_index = add(first_token, token_ids["pad"], 0.0)
        model_open = True

    def push_model_token(token_id: int) -> None:
        nonlocal pending_model_token_index
        if pending_model_token_index is None:
            pending_model_token_index = add(token_id, token_ids["pad"], 0.0)
            return
        set_target(pending_model_token_index, token_id, config.content_loss_weight)
        pending_model_token_index = add(token_id, token_ids["pad"], 0.0)

    def close_model_normally(next_target_after_model_end: int) -> None:
        nonlocal model_open, pending_model_token_index
        if not model_open:
            return
        if pending_model_token_index is not None:
            set_target(pending_model_token_index, token_ids["model_end"], config.control_loss_weight)
        add(token_ids["model_end"], next_target_after_model_end, config.control_loss_weight)
        model_open = False
        pending_model_token_index = None

    def inject_user_interrupt() -> None:
        nonlocal model_open, pending_model_token_index
        if not model_open:
            return

        if pending_model_token_index is not None:
            set_target(
                pending_model_token_index,
                token_ids["user_interrupt"],
                config.interrupt_input_loss_weight,
            )

        add(token_ids["user_interrupt"], token_ids["model_end"], config.control_loss_weight)
        add(token_ids["model_end"], token_ids["user"], config.control_loss_weight)

        model_open = False
        pending_model_token_index = None

    add(token_ids["session"], token_ids["listen"])

    for i, (role, content) in enumerate(events):
        next_role = events[i + 1][0] if i + 1 < len(events) else None
        tokens = encode_text(content)
        if not tokens:
            continue

        if role == "user":
            if model_open:
                inject_user_interrupt()
            start_user_if_needed()
            add_user_chunk(tokens)
            if next_role == "assistant":
                close_user_to_model()

        elif role == "assistant":
            if user_open:
                close_user_to_model()
            if not model_open:
                start_model_with_token(tokens[0])
                for token_id in tokens[1:]:
                    push_model_token(token_id)
            else:
                for token_id in tokens:
                    push_model_token(token_id)

            if next_role == "user":
                # The actual interrupt edge is injected when the following user event is consumed.
                # Keeping model_open here lets inject_user_interrupt() convert the edge into:
                # previous_model_token -> [USER_INTERRUPT] with optional zero loss
                # [USER_INTERRUPT] -> [MODEL_END]
                # [MODEL_END] -> [USER]
                pass
            elif next_role != "assistant":
                close_model_normally(token_ids["session_end"])

    if model_open:
        close_model_normally(token_ids["session_end"])

    if not in_ids or in_ids[-1] != token_ids["session_end"]:
        add(token_ids["session_end"], token_ids["pad"], 0.0)

    return in_ids, tr_ids, weights, stats


def tokenize_instruction_chat(row: dict) -> tuple[list[int], list[int], list[float], dict[str, int]]:
    if "conversations" not in row:
        raise KeyError("Instruction chat row must contain a conversations field")

    converted = []
    for msg in row["conversations"]:
        role = canonical_role(str(msg.get("from", "")))
        content = msg.get("value", "")
        if role in {"user", "assistant"} and content:
            converted.append({"role": role, "content": str(content)})

    return tokenize_duplex({"output": converted})


def cache_prefix(split_name: str, max_chunks: int, split_spec: str, skip_rows: int) -> Path:
    sig_str = "|".join(
        [
            "user_interrupt_stateful",
            ",".join(SPECIAL_TOKENS),
            config.dataset_name,
            config.dataset_mode,
            split_name,
            split_spec,
            str(skip_rows),
            str(config.train_unroll_len),
            str(vocab_size),
            tokenizer_fingerprint,
            str(config.tokenizer_vocab_size),
            str(max_chunks),
            str(config.user_inner_loss_weight),
            str(config.listen_loss_weight),
            str(config.control_loss_weight),
            str(config.interrupt_input_loss_weight),
            str(config.content_loss_weight),
            str(config.min_supervised_targets),
        ]
    )
    sig = hashlib.md5(sig_str.encode()).hexdigest()[:10]
    return Path("outputs/cache") / f"propagator_{split_name}_{sig}"


def chunk_tokenized_stream(
    in_ids: list[int],
    tr_ids: list[int],
    row_weights: list[float],
    unroll_len: int,
) -> list[tuple[list[int], list[int], list[float], dict[str, int]]]:
    chunks = []
    if not in_ids:
        return chunks

    for start in range(0, len(in_ids), unroll_len):
        chunk_in = in_ids[start : start + unroll_len]
        chunk_tr = tr_ids[start : start + unroll_len]
        chunk_w = row_weights[start : start + unroll_len]
        if not chunk_in:
            continue

        stats = {
            "listen": 0,
            "user_end": 0,
            "model_end": 0,
            "interrupt": 0,
            "content": 0,
            "control": 0,
            "ignored": 0,
        }
        supervised = 0
        for target_id, weight in zip(chunk_tr, chunk_w, strict=True):
            if weight > 0.0 and target_id != token_ids["pad"]:
                supervised += 1
            add_target_stats(stats, int(target_id), float(weight))

        if supervised < config.min_supervised_targets:
            continue

        chunks.append((chunk_in, chunk_tr, chunk_w, stats))

    return chunks


def tokenize_dataset_rows(
    dataset,
    split_name: str,
    cache_path: Path,
    max_chunks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unroll_len = config.train_unroll_len
    print(f"Tokenizing {split_name}: mode={config.dataset_mode}, max_chunks={max_chunks}, train_unroll_len={unroll_len}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    input_tokens = np.memmap(str(cache_path) + ".input.bin", dtype=np.int32, mode="w+", shape=(max_chunks, unroll_len))
    target_tokens = np.memmap(str(cache_path) + ".target.bin", dtype=np.int32, mode="w+", shape=(max_chunks, unroll_len))
    loss_weights = np.memmap(str(cache_path) + ".weight.bin", dtype=np.float32, mode="w+", shape=(max_chunks, unroll_len))
    stream_ids = np.memmap(str(cache_path) + ".stream_id.bin", dtype=np.int64, mode="w+", shape=(max_chunks,))
    chunk_positions = np.memmap(str(cache_path) + ".chunk_pos.bin", dtype=np.int32, mode="w+", shape=(max_chunks,))

    pad_id = token_ids["pad"]
    actual_count = 0
    source_rows = 0
    aggregate_stats = {
        "listen": 0,
        "user_end": 0,
        "model_end": 0,
        "interrupt": 0,
        "content": 0,
        "control": 0,
        "ignored": 0,
        "skipped_chunks": 0,
        "source_rows": 0,
        "errors": 0,
    }

    for row in tqdm(dataset, desc=f"Tokenizing {split_name}", total=max_chunks):
        if actual_count >= max_chunks:
            break

        stream_id = source_rows
        source_rows += 1

        try:
            if config.dataset_mode == "duplex_chat":
                in_ids, tr_ids, row_weights, _ = tokenize_duplex(row)
            else:
                in_ids, tr_ids, row_weights, _ = tokenize_instruction_chat(row)

            if len(in_ids) != len(tr_ids) or len(in_ids) != len(row_weights):
                aggregate_stats["errors"] += 1
                continue

            chunks = chunk_tokenized_stream(in_ids, tr_ids, row_weights, unroll_len)
            if not chunks:
                aggregate_stats["skipped_chunks"] += 1
                continue

            for chunk_pos, (chunk_in, chunk_tr, chunk_w, chunk_stats) in enumerate(chunks):
                if actual_count >= max_chunks:
                    break

                input_tokens[actual_count] = np.asarray(pad_to_len(chunk_in, unroll_len, pad_id), dtype=np.int32)
                target_tokens[actual_count] = np.asarray(pad_to_len(chunk_tr, unroll_len, pad_id), dtype=np.int32)
                loss_weights[actual_count] = np.asarray(pad_weights(chunk_w, unroll_len), dtype=np.float32)
                stream_ids[actual_count] = stream_id
                chunk_positions[actual_count] = chunk_pos

                for key in ("listen", "user_end", "model_end", "interrupt", "content", "control", "ignored"):
                    aggregate_stats[key] += int(chunk_stats[key])

                actual_count += 1

        except Exception as exc:
            aggregate_stats["errors"] += 1
            if aggregate_stats["errors"] <= 10:
                print(f"Tokenization error in {split_name}: {exc}")

    input_tokens.flush()
    target_tokens.flush()
    loss_weights.flush()
    stream_ids.flush()
    chunk_positions.flush()

    aggregate_stats["source_rows"] = source_rows

    meta = {
        "num_rows": actual_count,
        "num_chunks": actual_count,
        "source_rows": source_rows,
        "train_unroll_len": unroll_len,
        "stats": aggregate_stats,
        "dataset_name": config.dataset_name,
        "dataset_mode": config.dataset_mode,
        "protocol": "user_interrupt_stateful",
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(str(cache_path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Tokenized {actual_count} chunks from {source_rows} source rows for {split_name}")
    print(json.dumps(aggregate_stats, ensure_ascii=False, indent=2))

    return (
        input_tokens[:actual_count],
        target_tokens[:actual_count],
        loss_weights[:actual_count],
        stream_ids[:actual_count],
        chunk_positions[:actual_count],
    )


def load_cache_or_tokenize(
    split_name: str,
    max_chunks: int,
    ds_split: str,
    skip_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cp = cache_prefix(split_name, max_chunks, ds_split, skip_rows)
    meta_path = Path(str(cp) + ".meta.json")
    unroll_len = config.train_unroll_len

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_rows = int(meta.get("num_chunks", meta["num_rows"]))
        print(f"Loading cached {split_name} chunks: chunks={num_rows}, path={cp}")

        inputs = np.memmap(str(cp) + ".input.bin", dtype=np.int32, mode="r", shape=(num_rows, unroll_len))
        targets = np.memmap(str(cp) + ".target.bin", dtype=np.int32, mode="r", shape=(num_rows, unroll_len))
        weights = np.memmap(str(cp) + ".weight.bin", dtype=np.float32, mode="r", shape=(num_rows, unroll_len))
        stream_ids = np.memmap(str(cp) + ".stream_id.bin", dtype=np.int64, mode="r", shape=(num_rows,))
        chunk_positions = np.memmap(str(cp) + ".chunk_pos.bin", dtype=np.int32, mode="r", shape=(num_rows,))
        return inputs, targets, weights, stream_ids, chunk_positions

    ds = load_dataset(config.dataset_name, split=ds_split, streaming=config.streaming)

    if skip_rows > 0:
        if not config.streaming and hasattr(ds, "select"):
            end = min(skip_rows + max_chunks, len(ds))
            ds = ds.select(range(skip_rows, end))
        else:
            print(f"Skipping {skip_rows} source rows before tokenizing {split_name}")
            ds = ds.skip(skip_rows)

    return tokenize_dataset_rows(ds, split_name, cp, max_chunks)


def load_tokenizer_and_datasets() -> tuple[np.ndarray, ...]:
    global tokenizer, vocab_size, token_ids, tokenizer_fingerprint

    tokenizer = load_or_train_tokenizer()
    token_ids = ensure_special_tokens(tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    tokenizer_fingerprint = tokenizer_file_fingerprint(Path(config.tokenizer_path))
    print(f"Tokenizer fingerprint: {tokenizer_fingerprint}")
    save_tokenizer_snapshot()

    train_split = config.dataset_split
    val_split = config.validation_split

    if config.validation_skip_rows is not None:
        val_skip = config.validation_skip_rows
    elif train_split == val_split:
        val_skip = config.max_train_chunks
    else:
        val_skip = 0

    train_data = load_cache_or_tokenize("train", config.max_train_chunks, train_split, 0)
    val_data = load_cache_or_tokenize("val", config.max_val_chunks, val_split, val_skip)

    if len(train_data[0]) == 0:
        raise RuntimeError("No training chunks were tokenized")
    if len(val_data[0]) == 0:
        raise RuntimeError("No validation chunks were tokenized")

    return (*train_data, *val_data)


def build_candidate_token_ids(vocab_size_: int) -> np.ndarray:
    if config.inference_candidate_vocab_size <= 0 or config.inference_candidate_vocab_size >= vocab_size_:
        return np.arange(vocab_size_, dtype=np.int32)

    candidate_limit = max(config.inference_candidate_vocab_size, len(SPECIAL_TOKENS))
    ids = set(range(min(candidate_limit, vocab_size_)))
    for value in token_ids.values():
        ids.add(int(value))
    return np.asarray(sorted(i for i in ids if 0 <= i < vocab_size_), dtype=np.int32)


def rms_norm(x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    ms = jnp.mean(x_f32**2, axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(ms + 1e-6)).astype(x.dtype)


class PropagatorBlock(nnx.Module):
    def __init__(self, cfg: PropagatorConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        std = 1.0 / jnp.sqrt(cfg.hidden_size)

        self.read_key_proj = nnx.Linear(
            cfg.hidden_size,
            cfg.memory_key_size,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(std),
        )
        self.write_key_proj = nnx.Linear(
            cfg.hidden_size,
            cfg.memory_key_size,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(std),
        )
        self.write_value_proj = nnx.Linear(
            cfg.hidden_size,
            cfg.memory_value_size,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(std),
        )
        self.read_proj = nnx.Linear(
            cfg.memory_value_size,
            cfg.hidden_size,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(std),
        )

        self.write_gate = nnx.Linear(cfg.hidden_size, 1, rngs=rngs)
        self.forget_gate = nnx.Linear(cfg.hidden_size, 1, rngs=rngs)

        self.norm1 = nnx.RMSNorm(cfg.hidden_size, rngs=rngs)
        self.norm2 = nnx.RMSNorm(cfg.hidden_size, rngs=rngs)
        self.norm3 = nnx.RMSNorm(cfg.hidden_size, rngs=rngs)

        self.fc1 = nnx.Linear(cfg.hidden_size, cfg.mlp_multiplier * cfg.hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.mlp_multiplier * cfg.hidden_size, cfg.hidden_size, rngs=rngs)

        self.gamma1 = nnx.Param(jnp.ones((cfg.hidden_size,)) * 0.1)
        self.gamma2 = nnx.Param(jnp.ones((cfg.hidden_size,)) * 0.1)

    def __call__(self, x: jax.Array, memory: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = jnp.float16 if self.cfg.precision == "float16" else jnp.float32
        x = x.astype(dtype)
        memory_f32 = memory.astype(jnp.float32)

        scale = jax.lax.rsqrt(jnp.asarray(self.cfg.memory_key_size, dtype=jnp.float32))

        h = self.norm1(x).astype(dtype)
        read_key = (rms_norm(self.read_key_proj(h)) * scale).astype(jnp.float32)
        if read_key.ndim == 3:
            read_key = jnp.squeeze(read_key, 1)

        read_value = jnp.einsum("bkv,bk->bv", memory_f32, read_key).astype(dtype)
        x = x + (self.gamma1[...] * self.read_proj(read_value)).astype(dtype)

        mlp_in = self.norm2(x).astype(dtype)
        mlp_hidden = jax.nn.silu(self.fc1(mlp_in)).astype(dtype)
        x = x + (self.gamma2[...] * self.fc2(mlp_hidden)).astype(dtype)

        w = self.norm3(x).astype(dtype)

        write_key = (rms_norm(self.write_key_proj(w)) * scale).astype(jnp.float32)
        if write_key.ndim == 3:
            write_key = jnp.squeeze(write_key, 1)

        write_value = jnp.tanh(self.write_value_proj(w)).astype(jnp.float32)
        if write_value.ndim == 3:
            write_value = jnp.squeeze(write_value, 1)

        value_hat = jnp.einsum("bkv,bk->bv", memory_f32, write_key)
        err = jnp.clip(write_value - value_hat, -1.0, 1.0)

        eta = jax.nn.sigmoid(self.write_gate(w)).astype(jnp.float32) * self.cfg.write_rate
        if eta.ndim == 3:
            eta = jnp.squeeze(eta, 1)

        forget = jax.nn.sigmoid(self.forget_gate(w)).astype(jnp.float32) * self.cfg.forget_rate
        if forget.ndim == 3:
            forget = jnp.squeeze(forget, 1)

        update = jnp.einsum("bk,bv->bkv", write_key, err)
        new_memory = (1.0 - forget[:, :, None]) * memory_f32 + eta[:, :, None] * update
        new_memory = jnp.clip(new_memory, -10.0, 10.0)

        valid_f = valid.astype(jnp.float32)[:, None, None]
        final_memory = valid_f * new_memory + (1.0 - valid_f) * memory_f32

        return x.astype(jnp.float32), final_memory


class PropagatorModel(nnx.Module):
    def __init__(self, cfg: PropagatorConfig, vocab_size_: int, rngs: nnx.Rngs):
        self.cfg = cfg
        self.token_emb = nnx.Embed(vocab_size_, cfg.hidden_size, rngs=rngs)
        self.blocks = nnx.List([PropagatorBlock(cfg, rngs) for _ in range(cfg.num_layers)])
        self.norm = nnx.RMSNorm(cfg.hidden_size, rngs=rngs)

    def initial_memories(self, batch_size: int) -> tuple[jax.Array, ...]:
        return tuple(
            jnp.zeros((batch_size, self.cfg.memory_key_size, self.cfg.memory_value_size), dtype=jnp.float32)
            for _ in range(self.cfg.num_layers)
        )

    def reset_memories(
        self,
        memories: tuple[jax.Array, ...],
        reset_mask: jax.Array,
    ) -> tuple[jax.Array, ...]:
        zeros = self.initial_memories(reset_mask.shape[0])
        reset = reset_mask.astype(jnp.float32)[:, None, None]
        return tuple(reset * z + (1.0 - reset) * m for m, z in zip(memories, zeros, strict=True))

    def step_hidden(
        self,
        token_ids_: jax.Array,
        memories: tuple[jax.Array, ...],
        valid: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        x = self.token_emb(token_ids_)
        next_memories = []
        for block, memory in zip(self.blocks, memories, strict=True):
            x, next_memory = block(x, memory, valid)
            next_memories.append(next_memory)
        x = self.norm(x)
        return x.astype(jnp.float32), tuple(next_memories)

    def project_full(self, hidden: jax.Array) -> jax.Array:
        return (hidden @ self.token_emb.embedding[...].T).astype(jnp.float32)

    def project_candidates(self, hidden: jax.Array, candidate_ids: jax.Array) -> jax.Array:
        candidate_embeddings = self.token_emb.embedding[candidate_ids]
        return (hidden @ candidate_embeddings.T).astype(jnp.float32)

    def step(
        self,
        token_ids_: jax.Array,
        memories: tuple[jax.Array, ...],
        valid: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        hidden, next_memories = self.step_hidden(token_ids_, memories, valid)
        return self.project_full(hidden), next_memories

    def step_candidates(
        self,
        token_ids_: jax.Array,
        memories: tuple[jax.Array, ...],
        valid: jax.Array,
        candidate_ids: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        hidden, next_memories = self.step_hidden(token_ids_, memories, valid)
        return self.project_candidates(hidden, candidate_ids), next_memories

    def forward_with_memories(
        self,
        inputs: jax.Array,
        targets: jax.Array,
        loss_weights: jax.Array,
        init_memories: tuple[jax.Array, ...],
        reset_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        input_mask = inputs != token_ids_pad
        memories = self.reset_memories(init_memories, reset_mask)

        smooth = jnp.asarray(self.cfg.label_smoothing, dtype=jnp.float32)
        vocab_n = jnp.asarray(vocab_size, dtype=jnp.float32)

        @jax.checkpoint
        def scan_step(carry, step_inputs):
            step_in, step_target, step_weight, step_valid = step_inputs
            step_logits, next_carry = self.step(step_in, carry, step_valid)

            log_probs = jax.nn.log_softmax(step_logits, axis=-1)
            nll = -jnp.take_along_axis(log_probs, step_target[..., None], axis=-1).squeeze(-1)
            smooth_loss = -jnp.mean(log_probs, axis=-1)
            mixed_nll = (1.0 - smooth) * nll + smooth * smooth_loss

            weighted_nll = mixed_nll * step_weight

            pred = jnp.argmax(step_logits, axis=-1).astype(jnp.int32)
            supervised = step_weight > 0.0
            correct = jnp.logical_and(supervised, pred == step_target)

            decision_target = jnp.logical_or(step_target == token_ids_listen, step_target == token_ids_user_end)
            decision_target = jnp.logical_or(decision_target, step_target == token_ids_user_interrupt)
            decision_mask = jnp.logical_and(supervised, decision_target)

            listen_mask = jnp.logical_and(supervised, step_target == token_ids_listen)
            user_end_mask = jnp.logical_and(supervised, step_target == token_ids_user_end)
            interrupt_mask = jnp.logical_and(supervised, step_target == token_ids_user_interrupt)
            model_end_mask = jnp.logical_and(supervised, step_target == token_ids_model_end)

            metrics = (
                jnp.sum(jnp.logical_and(correct, decision_mask).astype(jnp.float32)),
                jnp.sum(decision_mask.astype(jnp.float32)),
                jnp.sum(jnp.logical_and(correct, listen_mask).astype(jnp.float32)),
                jnp.sum(listen_mask.astype(jnp.float32)),
                jnp.sum(jnp.logical_and(correct, user_end_mask).astype(jnp.float32)),
                jnp.sum(user_end_mask.astype(jnp.float32)),
                jnp.sum(jnp.logical_and(correct, interrupt_mask).astype(jnp.float32)),
                jnp.sum(interrupt_mask.astype(jnp.float32)),
                jnp.sum(jnp.logical_and(correct, model_end_mask).astype(jnp.float32)),
                jnp.sum(model_end_mask.astype(jnp.float32)),
            )

            return next_carry, (weighted_nll, step_weight, metrics)

        final_memories, (step_losses, step_weights, metrics_t) = jax.lax.scan(
            scan_step,
            memories,
            (inputs.T, targets.T, loss_weights.T, input_mask.T),
        )

        ce_loss = jnp.sum(step_losses) / jnp.maximum(1.0, jnp.sum(step_weights))
        reg_loss = self.cfg.memory_l2 * jnp.mean(jnp.asarray([jnp.mean(m**2) for m in final_memories]))

        metrics = tuple(jnp.sum(x, axis=0) for x in metrics_t)

        return ce_loss + reg_loss, ce_loss, tuple(jax.lax.stop_gradient(m) for m in final_memories), metrics

    def __call__(self, inputs: jax.Array, targets: jax.Array, loss_weights: jax.Array) -> tuple[jax.Array, jax.Array]:
        batch_size = inputs.shape[0]
        init_memories = self.initial_memories(batch_size)
        reset_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
        total_loss, ce_loss, _, _ = self.forward_with_memories(inputs, targets, loss_weights, init_memories, reset_mask)
        return total_loss, ce_loss


@nnx.jit
def train_step_stateless(
    model: PropagatorModel,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    def compute_loss(m):
        return m(inputs, targets, weights)

    grads, ce_loss = nnx.grad(compute_loss, has_aux=True)(model)
    optimizer.update(model, grads)
    return ce_loss


@nnx.jit
def train_step_stateful(
    model: PropagatorModel,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    memories: tuple[jax.Array, ...],
    reset_mask: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    def compute_loss(m):
        total_loss, ce_loss, final_memories, _ = m.forward_with_memories(
            inputs,
            targets,
            weights,
            memories,
            reset_mask,
        )
        return total_loss, (ce_loss, final_memories)

    grads, (ce_loss, final_memories) = nnx.grad(compute_loss, has_aux=True)(model)
    optimizer.update(model, grads)
    return ce_loss, final_memories


@nnx.jit
def validation_step_stateful(
    model: PropagatorModel,
    inputs: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    memories: tuple[jax.Array, ...],
    reset_mask: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    _, ce_loss, final_memories, metrics = model.forward_with_memories(
        inputs,
        targets,
        weights,
        memories,
        reset_mask,
    )
    return ce_loss, final_memories, metrics


@nnx.jit
def runtime_step_full(
    model: PropagatorModel,
    input_id: jax.Array,
    memories: tuple[jax.Array, ...],
    valid: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    return model.step(input_id, memories, valid)


@nnx.jit
def runtime_step_candidates(
    model: PropagatorModel,
    input_id: jax.Array,
    memories: tuple[jax.Array, ...],
    valid: jax.Array,
    candidate_ids: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    return model.step_candidates(input_id, memories, valid, candidate_ids)


@nnx.jit
def prefill_stream_candidates(
    model: PropagatorModel,
    input_ids: jax.Array,
    candidate_ids: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    batch_size, _ = input_ids.shape
    token_mask = input_ids != token_ids_pad
    memories = model.initial_memories(batch_size)

    def scan_step(carry, step_inputs):
        step_logits, next_carry = model.step_candidates(step_inputs[0], carry, step_inputs[1], candidate_ids)
        return next_carry, step_logits

    final_memories, logits_t = jax.lax.scan(scan_step, memories, (input_ids.T, token_mask.T))
    return logits_t[-1], final_memories


@nnx.jit
def prefill_stream_full(model: PropagatorModel, input_ids: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    batch_size, _ = input_ids.shape
    token_mask = input_ids != token_ids_pad
    memories = model.initial_memories(batch_size)

    def scan_step(carry, step_inputs):
        step_logits, next_carry = model.step(step_inputs[0], carry, step_inputs[1])
        return next_carry, step_logits

    final_memories, logits_t = jax.lax.scan(scan_step, memories, (input_ids.T, token_mask.T))
    return logits_t[-1], final_memories


def make_block_mask(candidate_ids: jax.Array, blocked_ids: list[int]) -> jax.Array:
    candidate_np = np.asarray(candidate_ids)
    blocked = set(int(x) for x in blocked_ids)
    mask = np.asarray([int(x) in blocked for x in candidate_np], dtype=np.bool_)
    return jnp.asarray(mask)


@nnx.jit
def sample_candidate_token_jit(
    logits: jax.Array,
    key: jax.Array,
    candidate_ids: jax.Array,
    blocked_mask: jax.Array,
    temperature: jax.Array,
) -> jax.Array:
    logits = logits / jnp.maximum(temperature, 1e-6)
    logits = jnp.where(blocked_mask[None, :], jnp.finfo(jnp.float32).min, logits)
    if config.top_k > 0:
        values, indices = jax.lax.top_k(logits, min(config.top_k, logits.shape[-1]))
        sampled = jax.random.categorical(key, values, axis=-1)
        local_ids = jnp.take_along_axis(indices, sampled[:, None], axis=-1).squeeze(-1)
        return candidate_ids[local_ids].astype(jnp.int32)
    local_ids = jax.random.categorical(key, logits, axis=-1)
    return candidate_ids[local_ids].astype(jnp.int32)


@nnx.jit
def generate_fixed_candidates_jit(
    model: PropagatorModel,
    start_logits: jax.Array,
    memories: tuple[jax.Array, ...],
    key: jax.Array,
    candidate_ids: jax.Array,
    blocked_mask: jax.Array,
    temperature: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    def scan_step(carry, _):
        logits, mem, rng_key, stopped = carry
        rng_key, subkey = jax.random.split(rng_key)
        sampled = sample_candidate_token_jit(logits, subkey, candidate_ids, blocked_mask, temperature)
        emitted = jnp.where(stopped, token_ids_pad, sampled)
        valid = jnp.logical_not(stopped)
        next_logits, next_mem = model.step_candidates(emitted, mem, valid, candidate_ids)
        next_stopped = jnp.logical_or(stopped, sampled == token_ids_model_end)
        next_stopped = jnp.logical_or(next_stopped, sampled == token_ids_session_end)
        return (next_logits, next_mem, rng_key, next_stopped), emitted

    batch_size = start_logits.shape[0]
    stopped0 = jnp.zeros((batch_size,), dtype=jnp.bool_)

    (_, final_memories, _, _), tokens_t = jax.lax.scan(
        scan_step,
        (start_logits, memories, key, stopped0),
        xs=None,
        length=config.sample_gen_len,
    )
    return jnp.swapaxes(tokens_t, 0, 1), final_memories


def model_blocked_ids_for_generation() -> list[int]:
    return [
        token_ids_pad,
        token_ids_unk,
        token_ids_session,
        token_ids_user,
        token_ids_model,
        token_ids_listen,
        token_ids_user_end,
        token_ids_user_interrupt,
    ]


def token_label(token_id: int) -> str:
    names = {
        token_ids_pad: "[PAD]",
        token_ids_unk: "[UNK]",
        token_ids_session: "[SESSION]",
        token_ids_user: "[USER]",
        token_ids_model: "[MODEL]",
        token_ids_listen: "[LISTEN]",
        token_ids_user_end: "[USER_END]",
        token_ids_model_end: "[MODEL_END]",
        token_ids_session_end: "[SESSION_END]",
        token_ids_user_interrupt: "[USER_INTERRUPT]",
    }
    if token_id in names:
        return names[token_id]
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return decoded if decoded else f"<tok:{token_id}>"


def parse_sample_chunks() -> list[str]:
    raw = config.sample_chunks.strip()
    if not raw:
        return ["Hello!"]
    try:
        value = json.loads(raw)
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            chunks = [str(x) for x in value if str(x)]
            return chunks or ["Hello!"]
    except json.JSONDecodeError:
        pass
    return [part for part in raw.split("|") if part] or ["Hello!"]


def step_runtime(
    model: PropagatorModel,
    token_id: int,
    memories: tuple[jax.Array, ...],
    use_candidate_head: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...], np.ndarray | None]:
    input_id = jnp.asarray([int(token_id)], dtype=jnp.int32)
    valid = jnp.ones_like(input_id, dtype=jnp.bool_)
    if use_candidate_head:
        candidate_ids = jnp.asarray(candidate_token_ids_host, dtype=jnp.int32)
        logits, memories = runtime_step_candidates(model, input_id, memories, valid, candidate_ids)
        return logits, memories, np.asarray(candidate_token_ids_host, dtype=np.int32)
    logits, memories = runtime_step_full(model, input_id, memories, valid)
    return logits, memories, None


def argmax_token_from_logits(logits: jax.Array, candidate_ids: np.ndarray | None = None) -> int:
    values = np.asarray(jax.device_get(logits[0]))
    if candidate_ids is None:
        return int(values.argmax())
    return int(candidate_ids[int(values.argmax())])


def sample_model_token_from_logits(
    logits: jax.Array,
    key: jax.Array,
    candidate_ids_np: np.ndarray | None,
    use_candidate_head: bool,
) -> int:
    if use_candidate_head:
        assert candidate_ids_np is not None
        candidate_ids = jnp.asarray(candidate_ids_np, dtype=jnp.int32)
        mask = make_block_mask(candidate_ids, model_blocked_ids_for_generation())
        token = sample_candidate_token_jit(
            logits,
            key,
            candidate_ids,
            mask,
            jnp.asarray(config.temperature, dtype=jnp.float32),
        )
        return int(jax.device_get(token[0]))

    blocked = jnp.asarray(model_blocked_ids_for_generation(), dtype=jnp.int32)
    scaled = logits / jnp.maximum(jnp.asarray(config.temperature, dtype=jnp.float32), 1e-6)
    scaled = scaled.at[:, blocked].set(jnp.finfo(jnp.float32).min)
    if config.top_k > 0:
        values, indices = jax.lax.top_k(scaled, min(config.top_k, scaled.shape[-1]))
        sampled = jax.random.categorical(key, values, axis=-1)
        token = jnp.take_along_axis(indices, sampled[:, None], axis=-1).squeeze(-1).astype(jnp.int32)
    else:
        token = jax.random.categorical(key, scaled, axis=-1).astype(jnp.int32)
    return int(jax.device_get(token[0]))


def user_mode_effective_decision(raw_token_id: int) -> int:
    if raw_token_id == token_ids_user_end:
        return token_ids_user_end
    return token_ids_listen


def generate_sample(
    model: PropagatorModel,
    seed: int,
    use_candidate_head: bool = True,
) -> str:
    chunks = parse_sample_chunks()
    key = jax.random.PRNGKey(seed)
    memories = model.initial_memories(1)
    lines: list[str] = []

    lines.append("# runtime loop sample")
    lines.append("")
    lines.append("## user stream")

    logits, memories, candidate_ids_np = step_runtime(model, token_ids_session, memories, use_candidate_head)
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    lines.append(f"[SESSION] -> {token_label(user_mode_effective_decision(raw))}  raw={token_label(raw)}")

    logits, memories, candidate_ids_np = step_runtime(model, token_ids_user, memories, use_candidate_head)
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    lines.append(f"[USER] -> {token_label(user_mode_effective_decision(raw))}  raw={token_label(raw)}")

    effective_decision = token_ids_listen
    for chunk in chunks:
        tokenized = encode_text(chunk)
        if not tokenized:
            lines.append(f"{json.dumps(chunk, ensure_ascii=False)} -> [LISTEN]  raw=<empty>")
            continue

        raw = token_ids_listen
        for token_id in tokenized:
            logits, memories, candidate_ids_np = step_runtime(model, token_id, memories, use_candidate_head)
            raw = argmax_token_from_logits(logits, candidate_ids_np)

        effective_decision = user_mode_effective_decision(raw)
        lines.append(f"{json.dumps(chunk, ensure_ascii=False)} -> {token_label(effective_decision)}  raw={token_label(raw)}")

        if effective_decision == token_ids_user_end:
            break

    if effective_decision != token_ids_user_end:
        lines.append("")
        lines.append("## model stream")
        lines.append("not started because runtime policy did not receive [USER_END].")
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append("## model stream")

    logits, memories, candidate_ids_np = step_runtime(model, token_ids_user_end, memories, use_candidate_head)
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    lines.append(f"[USER_END] -> {token_label(raw)}")

    if raw != token_ids_model:
        lines.append(f"stopped: expected [MODEL], got {token_label(raw)}")
        return "\n".join(lines) + "\n"

    current_input = token_ids_model
    logits, memories, candidate_ids_np = step_runtime(model, current_input, memories, use_candidate_head)

    for _ in range(config.sample_gen_len):
        key, subkey = jax.random.split(key)
        next_token = sample_model_token_from_logits(logits, subkey, candidate_ids_np, use_candidate_head)
        lines.append(f"{token_label(current_input)} -> {token_label(next_token)}")

        if next_token in {token_ids_model_end, token_ids_session_end, token_ids_listen}:
            break

        current_input = next_token
        logits, memories, candidate_ids_np = step_runtime(model, current_input, memories, use_candidate_head)

    return "\n".join(lines) + "\n"


@dataclass
class StatefulChunkSampler:
    stream_ids: np.ndarray
    batch_size: int
    seed: int

    def __post_init__(self):
        stream_ids_np = np.asarray(self.stream_ids)
        if len(stream_ids_np) == 0:
            raise ValueError("No chunks available for sampler")

        boundaries = np.flatnonzero(np.diff(stream_ids_np) != 0) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(stream_ids_np)]])
        self.stream_ranges = [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]

        self.rng = np.random.default_rng(self.seed)
        self.order = np.arange(len(self.stream_ranges), dtype=np.int64)
        self.rng.shuffle(self.order)
        self.order_pos = 0

        self.lane_pos = np.zeros((self.batch_size,), dtype=np.int64)
        self.lane_end = np.zeros((self.batch_size,), dtype=np.int64)
        self.lane_needs_reset = np.ones((self.batch_size,), dtype=np.bool_)

        for lane in range(self.batch_size):
            self._assign_stream(lane)

    def _next_stream_range(self) -> tuple[int, int]:
        if self.order_pos >= len(self.order):
            self.rng.shuffle(self.order)
            self.order_pos = 0
        stream_idx = int(self.order[self.order_pos])
        self.order_pos += 1
        return self.stream_ranges[stream_idx]

    def _assign_stream(self, lane: int) -> None:
        start, end = self._next_stream_range()
        self.lane_pos[lane] = start
        self.lane_end[lane] = end
        self.lane_needs_reset[lane] = True

    def next_indices(self) -> tuple[np.ndarray, np.ndarray]:
        indices = np.empty((self.batch_size,), dtype=np.int64)
        reset_mask = np.empty((self.batch_size,), dtype=np.bool_)

        for lane in range(self.batch_size):
            if self.lane_pos[lane] >= self.lane_end[lane]:
                self._assign_stream(lane)

            indices[lane] = self.lane_pos[lane]
            reset_mask[lane] = self.lane_needs_reset[lane]

            self.lane_pos[lane] += 1
            self.lane_needs_reset[lane] = False

        return indices, reset_mask


def get_batch_by_indices(
    inputs_arr: np.ndarray,
    targets_arr: np.ndarray,
    weights_arr: np.ndarray,
    indices: np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_inputs = jnp.asarray(np.asarray(inputs_arr[indices], dtype=np.int32))
    batch_targets = jnp.asarray(np.asarray(targets_arr[indices], dtype=np.int32))
    batch_weights = jnp.asarray(np.asarray(weights_arr[indices], dtype=np.float32))
    return batch_inputs, batch_targets, batch_weights


def get_random_batch(step_index: int, shuffled_indices: np.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    start_idx = (step_index * config.batch_size) % len(train_input_tokens)
    indices = shuffled_indices[start_idx : start_idx + config.batch_size]
    if len(indices) < config.batch_size:
        wrap = config.batch_size - len(indices)
        indices = np.concatenate([indices, shuffled_indices[:wrap]])

    return get_batch_by_indices(train_input_tokens, train_target_tokens, train_loss_weights, indices)


def get_validation_random_batch(idx: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    start = (idx * config.batch_size) % len(val_input_tokens)
    indices = (np.arange(config.batch_size) + start) % len(val_input_tokens)
    return get_batch_by_indices(val_input_tokens, val_target_tokens, val_loss_weights, indices)


def validation_metric_dict(metric_sums: np.ndarray) -> dict[str, float]:
    (
        decision_correct,
        decision_total,
        listen_correct,
        listen_total,
        user_end_correct,
        user_end_total,
        interrupt_correct,
        interrupt_total,
        model_end_correct,
        model_end_total,
    ) = [float(x) for x in metric_sums]

    def ratio(num: float, den: float) -> float:
        return num / den if den > 0 else float("nan")

    return {
        "decision_acc": ratio(decision_correct, decision_total),
        "listen_acc": ratio(listen_correct, listen_total),
        "user_end_acc": ratio(user_end_correct, user_end_total),
        "interrupt_acc": ratio(interrupt_correct, interrupt_total),
        "model_end_acc": ratio(model_end_correct, model_end_total),
        "decision_total": decision_total,
        "user_end_total": user_end_total,
        "interrupt_total": interrupt_total,
    }


def run_validation(model: PropagatorModel, step: int) -> tuple[float, dict[str, float]]:
    losses = []
    metric_sums = np.zeros((10,), dtype=np.float64)

    if config.stateful_validation:
        sampler = StatefulChunkSampler(val_stream_ids, config.batch_size, config.seed + 10_000 + step)
        memories = model.initial_memories(config.batch_size)

        for _ in range(config.validation_batches):
            indices, reset_mask_np = sampler.next_indices()
            batch_inputs, batch_targets, batch_weights = get_batch_by_indices(
                val_input_tokens,
                val_target_tokens,
                val_loss_weights,
                indices,
            )
            reset_mask = jnp.asarray(reset_mask_np, dtype=jnp.bool_)
            ce_loss, memories, metrics = validation_step_stateful(
                model,
                batch_inputs,
                batch_targets,
                batch_weights,
                memories,
                reset_mask,
            )
            losses.append(float(ce_loss))
            metric_sums += np.asarray([float(jax.device_get(x)) for x in metrics], dtype=np.float64)

    else:
        for i in range(config.validation_batches):
            batch_inputs, batch_targets, batch_weights = get_validation_random_batch(i)
            memories = model.initial_memories(config.batch_size)
            reset_mask = jnp.ones((config.batch_size,), dtype=jnp.bool_)
            ce_loss, _, metrics = validation_step_stateful(
                model,
                batch_inputs,
                batch_targets,
                batch_weights,
                memories,
                reset_mask,
            )
            losses.append(float(ce_loss))
            metric_sums += np.asarray([float(jax.device_get(x)) for x in metrics], dtype=np.float64)

    return float(np.mean(losses)), validation_metric_dict(metric_sums)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def save_metric_plot(steps: list[int], values: list[float], path: Path, title: str, step: int) -> None:
    if not steps:
        return
    plt.figure(figsize=(10, 4))
    v_arr = np.asarray(values, dtype=np.float32)
    plt.plot(steps, v_arr, alpha=0.3)
    if len(v_arr) > 1:
        window = max(5, len(v_arr) // 20)
        plt.plot(steps, rolling_mean(v_arr, window), linewidth=2)
    plt.title(f"{title} - Step {step}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


def shuffle_data_for_epoch(epoch: int) -> np.ndarray:
    rng = np.random.default_rng(config.seed + epoch)
    return rng.permutation(len(train_input_tokens)).astype(np.int64)


def build_optimizer(total_steps: int):
    warmup_steps = min(config.warmup_steps, total_steps // 2) if total_steps > 1 else 0
    decay_steps = max(1, total_steps - warmup_steps)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.05,
    )

    if config.optimizer == "lion":
        base_tx = optax.lion(lr_schedule, weight_decay=config.weight_decay)
    else:
        base_tx = optax.adamw(lr_schedule, weight_decay=config.weight_decay)

    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        base_tx,
    )


def save_checkpoint(checkpointer: ocp.StandardCheckpointer, model: PropagatorModel, output_dir: Path) -> None:
    print(f"\n[Checkpoint] Saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, state = nnx.split(model)
    checkpointer.save(os.path.abspath(output_dir / "checkpoint"), state, force=True)
    checkpointer.wait_until_finished()
    print("[Checkpoint] Done\n")


def init_global_token_ids() -> None:
    global token_ids_pad, token_ids_unk, token_ids_session, token_ids_user, token_ids_model
    global token_ids_listen, token_ids_user_end, token_ids_model_end, token_ids_session_end, token_ids_user_interrupt

    token_ids_pad = token_ids["pad"]
    token_ids_unk = token_ids["unk"]
    token_ids_session = token_ids["session"]
    token_ids_user = token_ids["user"]
    token_ids_model = token_ids["model"]
    token_ids_listen = token_ids["listen"]
    token_ids_user_end = token_ids["user_end"]
    token_ids_model_end = token_ids["model_end"]
    token_ids_session_end = token_ids["session_end"]
    token_ids_user_interrupt = token_ids["user_interrupt"]


def main() -> None:
    global config, train_input_tokens, train_target_tokens, train_loss_weights, train_stream_ids, train_chunk_positions
    global val_input_tokens, val_target_tokens, val_loss_weights, val_stream_ids, val_chunk_positions
    global candidate_token_ids_host

    config = build_config()

    loaded = load_tokenizer_and_datasets()
    (
        train_input_tokens,
        train_target_tokens,
        train_loss_weights,
        train_stream_ids,
        train_chunk_positions,
        val_input_tokens,
        val_target_tokens,
        val_loss_weights,
        val_stream_ids,
        val_chunk_positions,
    ) = loaded

    init_global_token_ids()
    candidate_token_ids_host = build_candidate_token_ids(vocab_size)

    print(f"Tokenizer path: {config.tokenizer_path}")
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Tokenizer fingerprint: {tokenizer_fingerprint}")
    print(f"Token ids: {json.dumps(token_ids, ensure_ascii=False, indent=2)}")
    print(f"Candidate inference head size: {len(candidate_token_ids_host)} / {vocab_size}")
    print(f"Stateful train: {config.stateful_train}, stateful validation: {config.stateful_validation}")

    steps_per_epoch = max(1, len(train_input_tokens) // config.batch_size)
    total_steps = min(config.epochs * steps_per_epoch, config.max_train_steps or 10**9)

    model = PropagatorModel(config, vocab_size, nnx.Rngs(config.seed))
    optimizer = nnx.Optimizer(model, build_optimizer(total_steps), wrt=nnx.Param)
    checkpointer = ocp.StandardCheckpointer()

    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    val_decision_accs: list[float] = []
    val_user_end_accs: list[float] = []

    shuffled = shuffle_data_for_epoch(0)

    if config.stateful_train:
        train_sampler = StatefulChunkSampler(train_stream_ids, config.batch_size, config.seed)
        carry_memories = model.initial_memories(config.batch_size)
    else:
        train_sampler = None
        carry_memories = None

    pbar = tqdm(range(total_steps), desc="Training")

    for step in pbar:
        if not config.stateful_train and step > 0 and step % steps_per_epoch == 0:
            shuffled = shuffle_data_for_epoch(step // steps_per_epoch)
            print(f"\n[Shuffle] Epoch {step // steps_per_epoch} started")

        if config.stateful_train:
            assert train_sampler is not None
            assert carry_memories is not None

            indices, reset_mask_np = train_sampler.next_indices()
            batch_inputs, batch_targets, batch_weights = get_batch_by_indices(
                train_input_tokens,
                train_target_tokens,
                train_loss_weights,
                indices,
            )
            reset_mask = jnp.asarray(reset_mask_np, dtype=jnp.bool_)
            ce_loss_val, carry_memories = train_step_stateful(
                model,
                optimizer,
                batch_inputs,
                batch_targets,
                batch_weights,
                carry_memories,
                reset_mask,
            )
        else:
            batch_inputs, batch_targets, batch_weights = get_random_batch(step, shuffled)
            ce_loss_val = train_step_stateless(model, optimizer, batch_inputs, batch_targets, batch_weights)

        train_losses.append(float(ce_loss_val))
        pbar.set_postfix({"loss": f"{train_losses[-1]:.4f}"})

        act_step = step + 1

        if act_step % config.eval_every == 0:
            v_loss, v_metrics = run_validation(model, act_step)

            val_steps.append(act_step)
            val_losses.append(v_loss)
            val_decision_accs.append(v_metrics["decision_acc"])
            val_user_end_accs.append(v_metrics["user_end_acc"])

            out_dir = Path(config.output_root) / f"step_{act_step}"
            out_dir.mkdir(parents=True, exist_ok=True)

            save_metric_plot(list(range(1, act_step + 1)), train_losses, out_dir / "train_loss.png", "Train weighted CE", act_step)
            save_metric_plot(val_steps, val_losses, out_dir / "val_loss.png", "Validation weighted CE", act_step)
            save_metric_plot(val_steps, val_decision_accs, out_dir / "val_decision_acc.png", "Validation decision accuracy", act_step)
            save_metric_plot(val_steps, val_user_end_accs, out_dir / "val_user_end_acc.png", "Validation user_end accuracy", act_step)

            sample = generate_sample(
                model,
                config.seed + act_step,
                use_candidate_head=config.eval_use_candidate_head,
            )

            (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
            (out_dir / "validation_metrics.json").write_text(
                json.dumps(v_metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(f"\n[Eval] CE={v_loss:.4f}")
            print(json.dumps(v_metrics, ensure_ascii=False, indent=2))
            print(f"[Sample] {sample[:500]}")

        if act_step % config.checkpoint_every == 0 or act_step == total_steps:
            save_checkpoint(checkpointer, model, Path(config.output_root) / f"step_{act_step}")


if __name__ == "__main__":
    main()

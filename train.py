#!/usr/bin/env python3
# coding: utf-8

import argparse
import atexit
import functools
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import socket
socket.setdefaulttimeout(120.0)
import urllib.request
import warnings
import wave
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import datasets as hf_datasets
from datasets import load_dataset
from dotenv import load_dotenv
from flax import nnx
from pydantic_settings import BaseSettings
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

# Fix Orbax 0.11.39 incompatibility with JAX 0.6.2+ where jax.sharding.set_mesh returns None
# instead of a context manager. Orbax expects a context manager.
import jax.sharding
_original_set_mesh = jax.sharding.set_mesh


class _MeshContextManager:
    def __init__(self, mesh: Any) -> None:
        self.mesh = mesh
        self.previous: Any | None = None

    def __enter__(self) -> Any:
        self.previous = _original_set_mesh(self.mesh)
        return self.mesh

    def __exit__(self, exc_type: type[Any] | None, exc: BaseException | None, tb: Any | None) -> bool:
        _ = _original_set_mesh(self.previous)
        return False


def _set_mesh_compat_patch(mesh):
    result = _original_set_mesh(mesh)
    if hasattr(result, "__enter__") and hasattr(result, "__exit__"):
        return result
    return _MeshContextManager(mesh)
jax.sharding.set_mesh = _set_mesh_compat_patch

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
if hasattr(hf_datasets, "disable_progress_bars"):
    hf_datasets.disable_progress_bars()

LOGGER = logging.getLogger("propagator")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SlackLogMirror(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.enabled = False
        self.webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        self.channel_id = os.environ.get("SLACK_CHANNEL_ID", "").strip()
        self.prefix = os.environ.get("SLACK_LOG_PREFIX", "propagator").strip()
        self.flush_seconds = max(1.0, float(os.environ.get("SLACK_LOG_FLUSH_SECONDS", "30.0")))
        self.buffer_lines = max(1, int(os.environ.get("SLACK_LOG_BUFFER_LINES", "5")))
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=max(1, int(os.environ.get("SLACK_LOG_QUEUE_SIZE", "256"))))
        self.thread: threading.Thread | None = None
        self.drop_notice_sent = False
        if self.webhook_url:
            self.enabled = True
        elif self.bot_token and self.channel_id:
            if self.bot_token.startswith("xapp-"):
                self._diagnostic(
                    "disabled: SLACK_BOT_TOKEN is an app-level xapp token; "
                    "chat.postMessage needs SLACK_WEBHOOK_URL or a Web API token such as xoxb-* with chat:write."
                )
            else:
                self.enabled = True
        elif self.bot_token and not self.channel_id:
            self._diagnostic("disabled: SLACK_BOT_TOKEN is set but SLACK_CHANNEL_ID is missing.")
        elif self.channel_id and not self.bot_token:
            self._diagnostic("disabled: SLACK_CHANNEL_ID is set but SLACK_BOT_TOKEN or SLACK_WEBHOOK_URL is missing.")
        else:
            self._diagnostic("disabled: set SLACK_WEBHOOK_URL or both SLACK_BOT_TOKEN and SLACK_CHANNEL_ID.")
        if self.enabled:
            transport = "webhook" if self.webhook_url else "chat.postMessage"
            self._diagnostic(f"enabled via {transport}.")
            self.thread = threading.Thread(target=self._run, name="slack-log-mirror", daemon=True)
            self.thread.start()

    def _diagnostic(self, message: str) -> None:
        LOGGER.warning("[Slack] %s", message)

    def enqueue(self, message: str) -> None:
        if not self.enabled:
            return
        text = message.strip()
        if not text:
            return
        if self.prefix:
            text = f"[{self.prefix}] {text}"
        if len(text) > 3500:
            text = text[:3497] + "..."
        try:
            self.queue.put_nowait(text)
        except queue.Full:
            if not self.drop_notice_sent:
                self.drop_notice_sent = True
                self._diagnostic("queue full; dropping Slack log lines until the sender catches up.")

    def emit(self, record: logging.LogRecord) -> None:
        self.enqueue(self.format(record))

    def close(self) -> None:
        if not self.enabled:
            super().close()
            return
        try:
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                self._diagnostic("queue full during shutdown; buffered Slack logs may be incomplete.")
            if self.thread is not None:
                self.thread.join(timeout=3.0)
        finally:
            super().close()

    def _run(self) -> None:
        buffer: list[str] = []
        last_flush = time.monotonic()

        def flush(force: bool = False) -> None:
            nonlocal last_flush
            if not buffer:
                return
            if not force and len(buffer) < self.buffer_lines and time.monotonic() - last_flush < self.flush_seconds:
                return
            body = "\n".join(buffer)
            buffer.clear()
            last_flush = time.monotonic()
            message = f"[{self.prefix}]\n```{body}```" if self.prefix else f"```{body}```"
            if len(message) > 3500:
                message = message[:3497] + "..."
            try:
                self._post(message)
            except (RuntimeError, TimeoutError, OSError, urllib.error.URLError) as exc:
                self._diagnostic(f"post failed: {type(exc).__name__}: {exc}")

        while True:
            try:
                message = self.queue.get(timeout=1.0)
            except queue.Empty:
                flush()
                continue
            if message is None:
                flush(force=True)
                break
            buffer.append(message)
            flush()

    def _post(self, message: str) -> None:
        if self.webhook_url:
            payload = {"text": message}
            request = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        else:
            payload = {"channel": self.channel_id, "text": message}
            request = urllib.request.Request(
                "https://slack.com/api/chat.postMessage",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                method="POST",
            )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        if not self.webhook_url:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid Slack response: {exc}") from exc
            if not parsed.get("ok"):
                error = parsed.get("error", "unknown_error")
                raise RuntimeError(f"Slack API error: {error}")


class UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging() -> None:
    is_main_process = multiprocessing.current_process().name == "MainProcess"
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(UtcFormatter("[%(asctime)sZ] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    LOGGER.addHandler(console_handler)

    if is_main_process and env_flag("SLACK_LOG_ENABLED", False):
        slack_handler = SlackLogMirror()
        if slack_handler.enabled:
            slack_handler.setFormatter(UtcFormatter("[%(asctime)sZ] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
            LOGGER.addHandler(slack_handler)


def log_info(*args: Any, **kwargs: Any) -> None:
    sep = str(kwargs.pop("sep", " "))
    end = str(kwargs.pop("end", "\n"))
    kwargs.pop("flush", None)
    if kwargs:
        raise TypeError(f"unsupported log_info keyword arguments: {', '.join(sorted(kwargs))}")
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end
    LOGGER.info("%s", message)


configure_logging()

TRAINING_RUN_NAME = "propagator-duplex"
DEFAULT_OUTPUT_ROOT = str(Path("outputs") / "propagator-multimodal")

# Global configuration and state
config: Any = None
tokenizer: Tokenizer | None = None
token_ids: dict[str, int] = {}
audio_token_id: Any = None
_RUN_LOCK_FD: int | None = None
_RUN_LOCK_PATH: Path | None = None
_RUN_LOCK_OWNER_PID: int | None = None
_ACTIVE_POOLS: list[Any] = []
train_control_input_tokens: np.ndarray | None = None
train_control_target_tokens: np.ndarray | None = None
train_control_loss_weights: np.ndarray | None = None
train_control_stream_ids: np.ndarray | None = None
train_control_chunk_positions: np.ndarray | None = None
train_control_chunk_task_ids: np.ndarray | None = None
val_control_input_tokens: np.ndarray | None = None
val_control_target_tokens: np.ndarray | None = None
val_control_loss_weights: np.ndarray | None = None
val_control_stream_ids: np.ndarray | None = None
val_control_chunk_positions: np.ndarray | None = None
val_control_chunk_task_ids: np.ndarray | None = None

VALIDATION_METRIC_SIZE = 44

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
    "[TEXT]",
    "[TEXT_IN]",
    "[TEXT_OUT]",
    "[AUDIO]",
    "[AUDIO_IN]",
    "[AUDIO_OUT]",
    "[AUDIO_END]",
    "[IMAGE]",
    "[IMAGE_IN]",
    "[HYBRID]",
    "[HYBRID_OUT]",
    "[SILENCE]",
]


def _tokenize_modal_input_prefix(token: str) -> list[int]:
    """Return standardized modal prefix tokens for user-side inputs."""
    if token == "audio":
        return [token_ids["audio"], token_ids["audio_in"]]
    if token == "image":
        return [token_ids["image"], token_ids["image_in"]]
    if token == "hybrid":
        return [token_ids["hybrid"], token_ids["audio"], token_ids["audio_in"]]
    return []

DEFAULT_DATASET_MIX = json.dumps(
    [
        {
            "name": "json",
            "data_files": "data/propagator_identity.jsonl",
            "split": "train",
            "validation_split": "train",
            "mode": "duplex_chat",
            "streaming": False,
            "repeat": 200,
            "weight": 0.05,
        },
        {
            "name": "KurtDu/EchoX-Dialogues-Plus",
            "split": "train",
            "validation_split": "train",
            "mode": "echox_s2s_dialogue",
            "echox_subsets": ["S2S-QA/AudioQA"],
            "target_modality": "hybrid",
            "max_wer": 0.25,
            "estimated_chunks": 6_500_000,
            "weight": 0.25,
        },
        {
            "name": "xinrongzhang2022/Duplex-UltraChat",
            "split": "train",
            "validation_split": "train",
            "mode": "duplex_chat",
            "weight": 0.15,
        },
        {
            "name": "databricks/databricks-dolly-15k",
            "split": "train",
            "validation_split": "train",
            "mode": "dolly_instruction",
            "weight": 0.08,
        },
        {
            "name": "distil-whisper/librispeech_asr",
            "split": "train.clean.100",
            "validation_split": "validation.clean",
            "mode": "audio_asr",
            "audio_task_mix": {"asr": 0.70, "tts": 0.30},
            "weight": 0.08,
        },
        {
            "name": "google/fleurs",
            "config": "en_us",
            "split": "train",
            "validation_split": "validation",
            "mode": "audio_asr",
            "transcript_key": "transcription",
            "audio_task_mix": {"asr": 0.80, "tts": 0.20},
            "weight": 0.08,
        },
        {
            "name": "PolyAI/minds14",
            "config": "en-US",
            "split": "train",
            "validation_split": "train",
            "mode": "audio_asr",
            "transcript_key": "english_transcription",
            "audio_task_mix": {"asr": 0.90, "tts": 0.10},
            "weight": 0.05,
        },
        {
            "name": "facebook/voxpopuli",
            "config": "en",
            "split": "train",
            "validation_split": "validation",
            "mode": "audio_asr",
            "transcript_key": "normalized_text",
            "audio_task_mix": {"asr": 0.80, "tts": 0.20},
            "weight": 0.05,
        },
        {
            "name": "edinburghcstr/ami",
            "config": "ihm",
            "split": "train",
            "validation_split": "validation",
            "mode": "audio_asr",
            "transcript_key": "text",
            "audio_task_mix": {"asr": 1.0},
            "weight": 0.06,
        },
        {
            "name": "blabble-io/libritts_r",
            "config": "clean",
            "split": "train.clean.360",
            "validation_split": "test.clean",
            "mode": "audio_asr",
            "transcript_key": "text_normalized",
            "audio_task_mix": {"asr": 0.10, "tts": 0.90},
            "weight": 0.15,
        },
    ],
    separators=(",", ":"),
)
DEFAULT_DATASET_MIX_PATH = Path(__file__).resolve().parent / "data" / "propagator_dataset_mix.json"
if DEFAULT_DATASET_MIX_PATH.exists():
    DEFAULT_DATASET_MIX = json.dumps(
        json.loads(DEFAULT_DATASET_MIX_PATH.read_text(encoding="utf-8")),
        separators=(",", ":"),
    )


class PropagatorConfig(BaseSettings):
    hidden_size: int = 1536
    num_layers: int = 24
    memory_key_size: int = 384
    memory_value_size: int = 768
    associative_groups: int = 4
    mlp_multiplier: int = 4
    use_swiglu: bool = True
    moe_num_experts: int = 1
    moe_top_k: int = 2
    rope_base: float = 10_000.0
    rope_position_scale: float = 16.0
    rope_max_position: int = 1_048_576

    train_unroll_len: int = 32
    seq_len: int | None = None
    batch_size: int = 512

    learning_rate: float = 3e-4
    warmup_steps: int = 5000
    epochs: int = 3
    max_steps: int = 0
    seed: int = 42

    eval_every: int = 5000
    checkpoint_every: int = 10_000
    train_log_every: int = 2000
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.01
    sample_gen_len: int = 256
    sample_chunks: str = '["Answer with exactly one lowercase word:", "is water wet?"]'
    eval_text_cases: str = json.dumps(
        [
            {"name": "identity_name", "chunks": ["What", "is your name?"]},
            {"name": "instruction_summary", "chunks": ["Give me", "a three-item checklist for preparing a guest room."]},
            {"name": "factual_qa", "chunks": ["What is the capital", "of France?"]},
            {"name": "reasoning", "chunks": ["A box has three red balls and two blue balls.", "How many balls are there?"]},
            {"name": "context_recall", "chunks": ["The code word is amber.", "Repeat only the code word."]},
            {"name": "format_following", "chunks": ["Answer with one word:", "is water wet?"]},
            {"name": "json_action", "chunks": ["Return JSON only with keys status and action.", "The pump is hot and vibrating."]},
            {"name": "extraction", "chunks": ["Extract device and location:", "Sensor A12 reports 71 C in bay 4."]},
            {"name": "image_recognition", "chunks": ["A red mug is on a desk in the image.", "What object is visible?"]},
            {"name": "architecture", "chunks": ["In one sentence,", "how does Propagator store context?"]},
            {"name": "turn_policy_silence", "chunks": ["I am still speaking", "[SILENCE]", "[SILENCE]"]},
        ],
        separators=(",", ":"),
    )
    eval_image_cases: str = json.dumps(
        [
            {"name": "red_mug", "image_text": "A red mug is on a desk.", "question": "What object is visible?"},
            {"name": "blue_car", "image_text": "A blue car is parked on the street.", "question": "What color is the car?"},
        ],
        separators=(",", ":"),
    )
    temperature: float = 0.7
    top_k: int = 50

    write_rate: float = 0.02
    forget_rate: float = 0.002
    memory_l2: float = 1e-6
    remat_scan_step: bool = True

    user_inner_loss_weight: float = 0.01
    listen_loss_weight: float = 0.05
    control_loss_weight: float = 1.0
    interrupt_input_loss_weight: float = 1.0
    content_loss_weight: float = 1.0
    min_supervised_targets: int = 1

    output_root: str = DEFAULT_OUTPUT_ROOT
    dataset_name: str = "xinrongzhang2022/Duplex-UltraChat"
    dataset_mode: str = "duplex_chat"
    dataset_mix: str = DEFAULT_DATASET_MIX
    dataset_split: str = "train"
    validation_split: str = "train"
    validation_skip_rows: int | None = None
    dataset_trust_remote_code: bool = True
    data_pack_count: int = 0
    data_pack_index: int = 0

    max_train_chunks: int = 0
    max_val_chunks: int = 0
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    streaming: bool = True
    cache_root: str = "outputs/cache"
    cache_flush_every: int = 4096
    cache_resume: bool = True
    cache_storage: str = "auto"
    cache_read_mode: str = "auto"
    cache_read_memory_fraction: float = 0.50
    echox_cache_raw_shards: bool = False
    echox_raw_cache_dir: str | None = None
    echox_raw_cache_min_free_gb: float = 96.0

    text_preprocessing_workers: int = 0
    audio_preprocessing_workers: int = 0
    text_preprocessing_chunk_size: int = 64
    audio_preprocessing_chunk_size: int = 2
    text_preprocessing_batch_rows: int = 0
    audio_preprocessing_batch_rows: int = 0
    tokenize_start_method: str = "auto"
    tokenize_imap_chunk_size: int = 0
    tokenize_maxtasks_per_child: int = 0

    stateful_train: bool = True
    stateful_validation: bool = True
    validation_batches: int = 16
    validation_control_batches: int = 0
    same_split_validation_stride: int = 10
    same_split_validation_offset: int = 0
    synthetic_control_train_examples: int = 2048
    synthetic_control_val_examples: int = 512
    synthetic_control_train_rate: float = 0.05
    synthetic_interrupt_fraction: float = 0.60

    tokenizer_path: str = "assets/tokenizer-byte-bpe-16000.json"
    tokenizer_vocab_size: int = 16_000
    tokenizer_train_rows: int = 0
    tokenizer_min_frequency: int = 2
    force_train_tokenizer: bool = False
    require_byte_level_bpe: bool = True
    save_augmented_tokenizer: bool = True
    precision: str = "bfloat16"

    enable_audio: bool = True
    audio_backend: str = "mimi"
    audio_sample_rate: int = 24_000
    audio_codebooks: int = 8
    audio_codebook_size: int = 2048
    mimi_repo: str = "kyutai/moshika-pytorch-bf16"
    mimi_filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi_cache_dir: str | None = None
    max_audio_seconds: float = 0.0
    max_audio_tokens_per_row: int = 0
    audio_task_mix: str = '{"asr":0.25,"tts":0.35,"audio":0.20,"hybrid":0.20}'
    tts_prompt_template: str = "Say this aloud: {text}"
    audio_eval_prompt: str = "Say this aloud: hello, I am listening."
    audio_eval_prompts: str = json.dumps(
        [
            "Say this aloud: the code word is amber.",
            "Read this number sequence aloud: two, seven, four.",
        ],
        separators=(",", ":"),
    )
    eval_audio_samples: int = 2
    eval_audio_every: int = 5_000
    audio_frames_per_second: float = 12.5
    eval_audio_seconds: float = 5.0
    eval_audio_tokens: int = 3000
    eval_audio_input_samples: int = 2
    eval_audio_input_text_tokens: int = 128
    eval_audio_input_audio_seconds: float = 5.0
    eval_audio_input_audio_tokens: int = 512
    audio_eval_normalize_rms: float = 0.06
    audio_eval_peak_limit: float = 0.95
    asr_eval_case_fold: bool = False
    audio_low_rms_threshold: float = 0.005
    audio_token_loss_weight: float = 1.0
    audio_codebook_loss_weight: float = 1.0
    audio_out_loss_weight: float = 2.0
    audio_end_loss_weight: float = 2.0
    output_modality_loss_weight: float = 2.0
    audio_min_generation_seconds: float = 1.0
    audio_min_generation_tokens: int = 256
    silence_short_tokens: int = 2
    silence_end_tokens: int = 4
    silence_token_loss_weight: float = 0.5
    synthesize_turn_silence: bool = False
    image_input_resolution: int = 160
    image_max_input_resolution: int = 192
    image_patch_size: int = 16
    image_patch_vocab_size: int = 1024
    image_tokens_per_sample: int = 64
    image_recognition_only: bool = True

    inference_candidate_vocab_size: int = 8192
    eval_use_candidate_head: bool = False
    eval_use_full_audio_head: bool = False

    optimizer: str = "adamw"
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0

    gcs_backup_dir: str | None = None
    gcs_sync_every: int = 10_000
    gcs_backup_keep: int = 5
    gcs_async_backup: bool = True
    local_eval_keep: int = 8
    local_checkpoint_keep: int = 1
    resume_checkpoint: bool = True

    enable_data_sharding: bool = True
    data_axis_name: str = "data"
    auto_batch_hbm_gb: float = 0.0
    auto_batch_memory_util: float = 0.78
    auto_batch_max_per_device: int = 16
    auto_batch_multiple_per_device: int = 8

    edge_vram_mb: int = 2048
    edge_vram_util_target: float = 0.70
    quantization_bits: int = 4
    write_edge_report: bool = True


config: PropagatorConfig
tokenizer: Tokenizer
vocab_size: int
text_vocab_size: int
audio_token_start: int
audio_token_end: int
image_token_start: int
image_token_end: int
token_ids: dict[str, int]
tokenizer_fingerprint: str
candidate_token_ids_host: np.ndarray
audio_candidate_token_ids_host: np.ndarray
_audio_codec: Any | None = None
_audio_codec_error: str | None = None
batch_sharding: NamedSharding | None = None
vector_sharding: NamedSharding | None = None
memory_sharding: NamedSharding | None = None
replicated_sharding: NamedSharding | None = None
data_mesh: Mesh | None = None

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
val_chunk_task_ids: np.ndarray

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
token_ids_audio_in: int
token_ids_audio_out: int
token_ids_audio_end: int
token_ids_silence: int
token_ids_text_in: int
token_ids_text_out: int
token_ids_hybrid_out: int
token_ids_image_in: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Propagator on a chunk-decision duplex streaming protocol.")

    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--memory-key-size", type=int)
    parser.add_argument("--memory-value-size", type=int)
    parser.add_argument("--associative-groups", type=int)
    parser.add_argument("--mlp-multiplier", type=int)
    parser.add_argument("--use-swiglu", action="store_true")
    parser.add_argument("--no-swiglu", action="store_true")
    parser.add_argument("--moe-num-experts", type=int)
    parser.add_argument("--moe-top-k", type=int)
    parser.add_argument("--rope-base", type=float)
    parser.add_argument("--rope-position-scale", type=float)
    parser.add_argument("--rope-max-position", type=int)

    parser.add_argument("--train-unroll-len", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--train-log-every", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--sample-gen-len", type=int)
    parser.add_argument("--sample-chunks", type=str)
    parser.add_argument("--eval-text-cases", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)

    parser.add_argument("--write-rate", type=float)
    parser.add_argument("--forget-rate", type=float)
    parser.add_argument("--memory-l2", type=float)
    parser.add_argument("--remat-scan-step", action="store_true")
    parser.add_argument("--no-remat-scan-step", action="store_true")

    parser.add_argument("--user-inner-loss-weight", type=float)
    parser.add_argument("--listen-loss-weight", type=float)
    parser.add_argument("--control-loss-weight", type=float)
    parser.add_argument("--interrupt-input-loss-weight", type=float)
    parser.add_argument("--content-loss-weight", type=float)
    parser.add_argument("--min-supervised-targets", type=int)
    parser.add_argument("--output-root", type=str)

    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--dataset-mode",
        type=str,
        choices=[
            "instruction_chat",
            "duplex_chat",
            "dolly_instruction",
            "plain_text",
            "audio_asr",
            "mimi_codes_asr",
            "mimi_codes_tts",
            "mimi_codes_speech_text",
            "image_recognition",
            "echox_s2s_dialogue",
            "speech_dialogue",
        ],
    )
    parser.add_argument("--dataset-mix", type=str)
    parser.add_argument("--dataset-split", type=str)
    parser.add_argument("--validation-split", type=str)
    parser.add_argument("--validation-skip-rows", type=int)
    parser.add_argument("--dataset-trust-remote-code", action="store_true")
    parser.add_argument("--no-dataset-trust-remote-code", action="store_true")
    parser.add_argument("--data-pack-count", type=int)
    parser.add_argument("--data-pack-index", type=int)

    parser.add_argument("--max-train-chunks", type=int)
    parser.add_argument("--max-val-chunks", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--cache-root", type=str)
    parser.add_argument("--cache-flush-every", type=int)
    parser.add_argument("--no-cache-resume", action="store_true")
    parser.add_argument("--cache-storage", type=str, choices=["auto", "disk", "memory"])
    parser.add_argument("--cache-read-mode", type=str, choices=["auto", "mmap", "memory"])
    parser.add_argument("--cache-read-memory-fraction", type=float)
    parser.add_argument("--echox-cache-raw-shards", action="store_true")
    parser.add_argument("--no-echox-cache-raw-shards", action="store_true")
    parser.add_argument("--echox-raw-cache-dir", type=str)
    parser.add_argument("--echox-raw-cache-min-free-gb", type=float)
    parser.add_argument("--text-preprocessing-workers", type=int)
    parser.add_argument("--audio-preprocessing-workers", type=int)
    parser.add_argument("--text-preprocessing-chunk-size", type=int)
    parser.add_argument("--audio-preprocessing-chunk-size", type=int)
    parser.add_argument("--text-preprocessing-batch-rows", type=int)
    parser.add_argument("--audio-preprocessing-batch-rows", type=int)
    parser.add_argument("--tokenize-start-method", type=str)
    parser.add_argument("--tokenize-imap-chunk-size", type=int)
    parser.add_argument("--tokenize-maxtasks-per-child", type=int)

    parser.add_argument("--stateful-train", action="store_true")
    parser.add_argument("--stateless-train", action="store_true")
    parser.add_argument("--stateful-validation", action="store_true")
    parser.add_argument("--stateless-validation", action="store_true")
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--validation-control-batches", type=int)
    parser.add_argument("--same-split-validation-stride", type=int)
    parser.add_argument("--same-split-validation-offset", type=int)
    parser.add_argument("--synthetic-control-train-examples", type=int)
    parser.add_argument("--synthetic-control-val-examples", type=int)
    parser.add_argument("--synthetic-control-train-rate", type=float)
    parser.add_argument("--synthetic-interrupt-fraction", type=float)

    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--tokenizer-vocab-size", type=int)
    parser.add_argument("--tokenizer-train-rows", type=int)
    parser.add_argument("--tokenizer-min-frequency", type=int)
    parser.add_argument("--force-train-tokenizer", action="store_true")
    parser.add_argument("--no-require-byte-level-bpe", action="store_true")
    parser.add_argument("--save-augmented-tokenizer", action="store_true")
    parser.add_argument("--no-save-augmented-tokenizer", action="store_true")
    parser.add_argument("--precision", type=str, choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--enable-audio", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--audio-backend", type=str, choices=["mimi", "encodec", "none"])
    parser.add_argument("--audio-sample-rate", type=int)
    parser.add_argument("--audio-codebooks", type=int)
    parser.add_argument("--audio-codebook-size", type=int)
    parser.add_argument("--mimi-repo", type=str)
    parser.add_argument("--mimi-filename", type=str)
    parser.add_argument("--mimi-cache-dir", type=str)
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument("--max-audio-tokens-per-row", type=int)
    parser.add_argument("--audio-task-mix", type=str)
    parser.add_argument("--tts-prompt-template", type=str)
    parser.add_argument("--audio-eval-prompt", type=str)
    parser.add_argument("--audio-eval-prompts", type=str)
    parser.add_argument("--eval-audio-samples", type=int)
    parser.add_argument("--eval-audio-every", type=int)
    parser.add_argument("--audio-frames-per-second", type=float)
    parser.add_argument("--eval-audio-seconds", type=float)
    parser.add_argument("--eval-audio-tokens", type=int)
    parser.add_argument("--eval-audio-input-samples", type=int)
    parser.add_argument("--eval-audio-input-text-tokens", type=int)
    parser.add_argument("--eval-audio-input-audio-seconds", type=float)
    parser.add_argument("--eval-audio-input-audio-tokens", type=int)
    parser.add_argument("--asr-eval-case-fold", action="store_true")
    parser.add_argument("--no-asr-eval-case-fold", action="store_true")
    parser.add_argument("--audio-eval-normalize-rms", type=float)
    parser.add_argument("--audio-eval-peak-limit", type=float)
    parser.add_argument("--audio-low-rms-threshold", type=float)
    parser.add_argument("--audio-token-loss-weight", type=float)
    parser.add_argument("--audio-codebook-loss-weight", type=float)
    parser.add_argument("--audio-out-loss-weight", type=float)
    parser.add_argument("--audio-end-loss-weight", type=float)
    parser.add_argument("--output-modality-loss-weight", type=float)
    parser.add_argument("--audio-min-generation-seconds", type=float)
    parser.add_argument("--audio-min-generation-tokens", type=int)
    parser.add_argument("--silence-short-tokens", type=int)
    parser.add_argument("--silence-end-tokens", type=int)
    parser.add_argument("--silence-token-loss-weight", type=float)
    parser.add_argument("--no-synthesize-turn-silence", action="store_true")
    parser.add_argument("--image-input-resolution", type=int)
    parser.add_argument("--image-max-input-resolution", type=int)
    parser.add_argument("--image-patch-size", type=int)
    parser.add_argument("--image-patch-vocab-size", type=int)
    parser.add_argument("--image-tokens-per-sample", type=int)
    parser.add_argument("--no-image-recognition-only", action="store_true")

    parser.add_argument("--inference-candidate-vocab-size", type=int)
    parser.add_argument("--eval-use-candidate-head", action="store_true")
    parser.add_argument("--eval-use-full-head", action="store_true")
    parser.add_argument("--eval-use-full-audio-head", action="store_true")

    parser.add_argument("--optimizer", type=str, choices=["adamw", "lion"])
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--label-smoothing", type=float)

    parser.add_argument("--gcs-backup-dir", type=str)
    parser.add_argument("--gcs-sync-every", type=int)
    parser.add_argument("--gcs-backup-keep", type=int)
    parser.add_argument("--sync-backup-blocking", action="store_true")
    parser.add_argument("--local-eval-keep", type=int)
    parser.add_argument("--local-checkpoint-keep", type=int)
    parser.add_argument("--no-checkpoint-resume", action="store_true")
    parser.add_argument("--enable-data-sharding", action="store_true")
    parser.add_argument("--no-data-sharding", action="store_true")
    parser.add_argument("--auto-batch-hbm-gb", type=float)
    parser.add_argument("--auto-batch-memory-util", type=float)
    parser.add_argument("--auto-batch-max-per-device", type=int)
    parser.add_argument("--auto-batch-multiple-per-device", type=int)

    parser.add_argument("--edge-vram-mb", type=int)
    parser.add_argument("--edge-vram-util-target", type=float)
    parser.add_argument("--quantization-bits", type=int)
    parser.add_argument("--no-write-edge-report", action="store_true")

    return parser.parse_args()


def build_config() -> PropagatorConfig:
    base_config = PropagatorConfig()
    cli_args = parse_args()
    raw_updates = vars(cli_args)
    updates = {
        key: value
        for key, value in raw_updates.items()
        if value is not None and not (isinstance(value, bool) and value is False)
    }

    if raw_updates.get("no_streaming"):
        updates["streaming"] = False
    if raw_updates.get("no_dataset_trust_remote_code"):
        updates["dataset_trust_remote_code"] = False
    if raw_updates.get("no_cache_resume"):
        updates["cache_resume"] = False
    if raw_updates.get("no_echox_cache_raw_shards"):
        updates["echox_cache_raw_shards"] = False
    if raw_updates.get("stateless_train"):
        updates["stateful_train"] = False
    if raw_updates.get("stateless_validation"):
        updates["stateful_validation"] = False
    if raw_updates.get("no_audio"):
        updates["enable_audio"] = False
    if raw_updates.get("no_synthesize_turn_silence"):
        updates["synthesize_turn_silence"] = False
    if raw_updates.get("no_image_recognition_only"):
        updates["image_recognition_only"] = False
    if raw_updates.get("no_remat_scan_step"):
        updates["remat_scan_step"] = False
    if raw_updates.get("no_swiglu"):
        updates["use_swiglu"] = False
    if raw_updates.get("eval_use_full_head"):
        updates["eval_use_candidate_head"] = False
    if raw_updates.get("eval_use_full_audio_head"):
        updates["eval_use_full_audio_head"] = True
    if raw_updates.get("sync_backup_blocking"):
        updates["gcs_async_backup"] = False
    if raw_updates.get("no_checkpoint_resume"):
        updates["resume_checkpoint"] = False
    if raw_updates.get("no_data_sharding"):
        updates["enable_data_sharding"] = False
    if raw_updates.get("no_save_augmented_tokenizer"):
        updates["save_augmented_tokenizer"] = False
    if raw_updates.get("no_require_byte_level_bpe"):
        updates["require_byte_level_bpe"] = False
    if raw_updates.get("no_write_edge_report"):
        updates["write_edge_report"] = False
    if raw_updates.get("asr_eval_case_fold"):
        updates["asr_eval_case_fold"] = True
    if raw_updates.get("no_asr_eval_case_fold"):
        updates["asr_eval_case_fold"] = False

    for key in (
        "no_streaming",
        "no_dataset_trust_remote_code",
        "no_cache_resume",
        "no_echox_cache_raw_shards",
        "stateless_train",
        "stateless_validation",
        "no_audio",
        "no_synthesize_turn_silence",
        "no_image_recognition_only",
        "no_remat_scan_step",
        "no_swiglu",
        "eval_use_full_head",
        "eval_use_full_audio_head",
        "sync_backup_blocking",
        "no_checkpoint_resume",
        "no_data_sharding",
        "no_save_augmented_tokenizer",
        "no_require_byte_level_bpe",
        "no_write_edge_report",
        "asr_eval_case_fold",
        "no_asr_eval_case_fold",
    ):
        updates.pop(key, None)

    cfg = base_config.model_copy(update=updates)

    backend_updates: dict[str, Any] = {}
    if cfg.audio_backend == "mimi":
        if raw_updates.get("audio_codebook_size") is None:
            backend_updates["audio_codebook_size"] = 2048
        if raw_updates.get("audio_frames_per_second") is None:
            backend_updates["audio_frames_per_second"] = 12.5
        if raw_updates.get("max_audio_tokens_per_row") is None:
            backend_updates["max_audio_tokens_per_row"] = 0
    elif cfg.audio_backend == "encodec":
        if raw_updates.get("audio_codebook_size") is None:
            backend_updates["audio_codebook_size"] = 1024
        if raw_updates.get("audio_frames_per_second") is None:
            backend_updates["audio_frames_per_second"] = 75.0
        if raw_updates.get("max_audio_tokens_per_row") is None:
            backend_updates["max_audio_tokens_per_row"] = 0
    if backend_updates:
        cfg = cfg.model_copy(update=backend_updates)
    if cfg.enable_audio and int(cfg.audio_codebooks) != 8:
        raise ValueError("The current frame model requires exactly 8 audio codebooks")
    if cfg.audio_backend == "mimi" and int(cfg.audio_codebook_size) != 2048:
        raise ValueError("Mimi requires --audio-codebook-size 2048")
    if cfg.audio_backend == "encodec" and int(cfg.audio_codebook_size) != 1024:
        raise ValueError("EnCodec 6 kbps requires --audio-codebook-size 1024")

    if cfg.seq_len is not None:
        cfg = cfg.model_copy(update={"train_unroll_len": cfg.seq_len})
    if cfg.max_train_rows is not None:
        cfg = cfg.model_copy(update={"max_train_chunks": cfg.max_train_rows})
    if cfg.max_val_rows is not None:
        cfg = cfg.model_copy(update={"max_val_chunks": cfg.max_val_rows})
    if cfg.gcs_backup_dir is None and Path("/gcs").exists():
        cfg = cfg.model_copy(update={"gcs_backup_dir": str(Path("/gcs") / "propagator-backups" / TRAINING_RUN_NAME)})

    groups = max(1, int(cfg.associative_groups))
    if int(cfg.memory_key_size) % groups != 0:
        raise ValueError("--memory-key-size must be divisible by --associative-groups")
    if int(cfg.moe_num_experts) < 1:
        raise ValueError("--moe-num-experts must be >= 1")
    if int(cfg.moe_top_k) < 1:
        raise ValueError("--moe-top-k must be >= 1")
    if int(cfg.image_input_resolution) <= 0:
        raise ValueError("--image-input-resolution must be positive")
    if int(cfg.image_max_input_resolution) < int(cfg.image_input_resolution):
        raise ValueError("--image-max-input-resolution must be >= --image-input-resolution")
    if int(cfg.image_patch_size) <= 0:
        raise ValueError("--image-patch-size must be positive")
    if int(cfg.image_patch_vocab_size) <= 0:
        raise ValueError("--image-patch-vocab-size must be positive")
    if int(cfg.image_tokens_per_sample) <= 0:
        raise ValueError("--image-tokens-per-sample must be positive")
    cfg = cfg.model_copy(update={"tokenize_start_method": str(cfg.tokenize_start_method).lower().strip() or "auto"})
    if str(cfg.tokenize_start_method) not in {"auto", "fork", "spawn", "forkserver"}:
        raise ValueError("--tokenize-start-method must be one of: auto, fork, spawn, forkserver")
    if int(cfg.data_pack_count) < 0:
        raise ValueError("--data-pack-count must be >= 0")
    if int(cfg.data_pack_count) == 1:
        cfg = cfg.model_copy(update={"data_pack_count": 0, "data_pack_index": 0})
    elif int(cfg.data_pack_count) > 1:
        cfg = cfg.model_copy(update={"data_pack_index": int(cfg.data_pack_index) % int(cfg.data_pack_count)})

    seconds_updates: dict[str, int] = {}
    if raw_updates.get("eval_audio_tokens") is None:
        seconds_updates["eval_audio_tokens"] = audio_seconds_to_token_budget(cfg.eval_audio_seconds, cfg)
    if raw_updates.get("eval_audio_input_audio_tokens") is None:
        seconds_updates["eval_audio_input_audio_tokens"] = audio_seconds_to_token_budget(
            cfg.eval_audio_input_audio_seconds,
            cfg,
        )
    if raw_updates.get("audio_min_generation_tokens") is None:
        seconds_updates["audio_min_generation_tokens"] = audio_seconds_to_token_budget(
            cfg.audio_min_generation_seconds,
            cfg,
        )
    if seconds_updates:
        cfg = cfg.model_copy(update=seconds_updates)

    return cfg


def audio_seconds_to_token_budget(seconds: float, cfg: PropagatorConfig) -> int:
    seconds_f = max(0.0, float(seconds))
    if seconds_f <= 0.0:
        return 0
    codebooks = max(1, int(cfg.audio_codebooks))
    frames = max(1, int(math.ceil(seconds_f * max(1.0, float(cfg.audio_frames_per_second)))))
    return frames * codebooks


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


def progress_bar(*args, **kwargs):
    kwargs.setdefault("disable", not (sys.stderr.isatty() or sys.stdout.isatty()))
    kwargs.setdefault("mininterval", 10.0)
    return tqdm(*args, **kwargs)


def parse_json_object(raw: str, fallback: dict[str, float]) -> dict[str, float]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    out: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or fallback


def parse_dataset_mix() -> list[dict[str, Any]]:
    raw = (config.dataset_mix or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("dataset_mix must be a JSON list")
            specs = [dict(item) for item in parsed if isinstance(item, dict)]
        except Exception as exc:
            raise ValueError(f"Invalid --dataset-mix JSON: {exc}") from exc
    else:
        specs = [
            {
                "name": config.dataset_name,
                "split": config.dataset_split,
                "validation_split": config.validation_split,
                "mode": config.dataset_mode,
                "weight": 1.0,
            }
        ]

    normalized = []
    for spec in specs:
        if not spec.get("name"):
            continue
        normalized.append(
            {
                "name": str(spec["name"]),
                "config": spec.get("config") or spec.get("subset"),
                "split": str(spec.get("split", config.dataset_split)),
                "validation_split": str(spec.get("validation_split", config.validation_split)),
                "mode": str(spec.get("mode", config.dataset_mode)),
                "weight": float(spec.get("weight", 1.0)),
                "text_key": spec.get("text_key"),
                "prompt_key": spec.get("prompt_key"),
                "response_key": spec.get("response_key"),
                "audio_key": spec.get("audio_key", "audio"),
                "transcript_key": spec.get("transcript_key"),
                "codes_key": spec.get("codes_key"),
                "audio_task_mix": spec.get("audio_task_mix"),
                "target_modality": spec.get("target_modality", "hybrid"),
                "max_wer": spec.get("max_wer", 0.25),
                "echox_subsets": spec.get("echox_subsets"),
                "max_shards": spec.get("max_shards"),
                "max_chunks": spec.get("max_chunks"),
                "estimated_chunks": spec.get("estimated_chunks"),
                "part_rows": spec.get("part_rows"),
                "part_chunks": spec.get("part_chunks"),
                "debug_max_rows": spec.get("debug_max_rows"),
                "data_files": spec.get("data_files"),
                "streaming": bool(spec.get("streaming", config.streaming)),
                "repeat": max(1, int(spec.get("repeat", 1))),
            }
        )

    if not normalized:
        raise ValueError("No valid dataset specs were provided")

    total = sum(max(0.0, spec["weight"]) for spec in normalized)
    if total <= 0.0:
        raise ValueError("At least one dataset_mix weight must be positive")
    for spec in normalized:
        spec["weight"] = max(0.0, spec["weight"]) / total
    return normalized


def split_for_dataset_spec(spec: dict[str, Any], split_name: str) -> str:
    if split_name == "val":
        return str(spec.get("validation_split") or config.validation_split)
    return str(spec.get("split") or config.dataset_split)


def dataset_partition_filter(row: Any, idx: int, stride: int, offset: int, keep_validation: bool) -> bool:
    is_validation_row = (int(idx) % int(stride)) == int(offset)
    return is_validation_row if keep_validation else not is_validation_row


def data_pack_filter(row: Any, idx: int, pack_count: int, pack_index: int) -> bool:
    return (int(idx) % int(pack_count)) == int(pack_index)


def active_data_pack() -> tuple[int, int] | None:
    pack_count = int(config.data_pack_count)
    if pack_count <= 1:
        return None
    pack_index = int(config.data_pack_index) % pack_count
    return pack_count, pack_index


def apply_data_pack_partition(dataset: Any, spec: dict[str, Any], split_name: str) -> Any:
    pack = active_data_pack()
    if pack is None:
        return dataset
    pack_count, pack_index = pack
    log_info(
        f"Applying staged data pack for {spec['name']} {split_name}: "
        f"idx % {pack_count} == {pack_index}"
    )
    if not config.streaming and hasattr(dataset, "select"):
        try:
            length = len(dataset)
            return dataset.select(range(pack_index, length, pack_count))
        except Exception:
            pass
    if hasattr(dataset, "filter"):
        return dataset.filter(
            data_pack_filter,
            with_indices=True,
            fn_kwargs={"pack_count": pack_count, "pack_index": pack_index},
        )
    return dataset


def apply_same_split_partition(dataset: Any, spec: dict[str, Any], split_name: str, split: str) -> Any:
    if config.validation_skip_rows is not None:
        return dataset
    train_split = str(spec.get("split") or config.dataset_split)
    validation_split = str(spec.get("validation_split") or config.validation_split)
    if train_split != validation_split:
        return dataset
    if split != train_split:
        return dataset
    stride = max(1, int(config.same_split_validation_stride))
    if stride <= 1:
        return dataset
    offset = int(config.same_split_validation_offset) % stride
    keep_validation = split_name == "val"
    label = "validation" if keep_validation else "training"
    log_info(
        f"Applying same-split {label} partition for {spec['name']}: "
        f"idx % {stride} {'==' if keep_validation else '!='} {offset}"
    )
    if not config.streaming and hasattr(dataset, "select"):
        length = len(dataset)
        if keep_validation:
            indices = range(offset, length, stride)
        else:
            indices = [idx for idx in range(length) if idx % stride != offset]
        return dataset.select(indices)
    if hasattr(dataset, "filter"):
        return dataset.filter(
            dataset_partition_filter,
            with_indices=True,
            fn_kwargs={"stride": stride, "offset": offset, "keep_validation": keep_validation},
        )
    return dataset


def dataset_fingerprint(specs: list[dict[str, Any]], split_name: str) -> str:
    serializable = [
        {
            "name": spec.get("name"),
            "config": spec.get("config"),
            "split": split_for_dataset_spec(spec, split_name),
            "mode": spec.get("mode"),
            "weight": spec.get("weight"),
            "audio_key": spec.get("audio_key"),
            "transcript_key": spec.get("transcript_key"),
            "codes_key": spec.get("codes_key"),
            "audio_task_mix": spec.get("audio_task_mix"),
            "target_modality": spec.get("target_modality"),
            "max_wer": spec.get("max_wer"),
            "echox_subsets": spec.get("echox_subsets"),
            "max_shards": spec.get("max_shards"),
            "max_chunks": spec.get("max_chunks"),
            "part_rows": spec.get("part_rows"),
            "part_chunks": spec.get("part_chunks"),
            "debug_max_rows": spec.get("debug_max_rows"),
            "echox_sharded_cache": True if spec.get("mode") in {"echox_s2s_dialogue", "speech_dialogue"} else None,
            "data_files": spec.get("data_files"),
            "repeat": spec.get("repeat", 1),
            "same_split_validation_stride": config.same_split_validation_stride,
            "same_split_validation_offset": config.same_split_validation_offset,
            "data_pack_count": config.data_pack_count,
            "data_pack_index": config.data_pack_index,
        }
        for spec in specs
    ]
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def source_dataset_fingerprint(spec: dict[str, Any], split_name: str) -> str:
    serializable = {
        "name": spec.get("name"),
        "config": spec.get("config"),
        "split": split_for_dataset_spec(spec, split_name),
        "mode": spec.get("mode"),
        "text_key": spec.get("text_key"),
        "prompt_key": spec.get("prompt_key"),
        "response_key": spec.get("response_key"),
        "audio_key": spec.get("audio_key"),
        "transcript_key": spec.get("transcript_key"),
        "codes_key": spec.get("codes_key"),
        "audio_task_mix": spec.get("audio_task_mix"),
        "target_modality": spec.get("target_modality"),
        "max_wer": spec.get("max_wer"),
        "echox_subsets": spec.get("echox_subsets"),
        "max_shards": spec.get("max_shards"),
        "max_chunks": spec.get("max_chunks"),
        "part_rows": spec.get("part_rows"),
        "part_chunks": spec.get("part_chunks"),
        "debug_max_rows": spec.get("debug_max_rows"),
        "echox_sharded_cache": True if spec.get("mode") in {"echox_s2s_dialogue", "speech_dialogue"} else None,
        "data_files": spec.get("data_files"),
        "repeat": spec.get("repeat", 1),
        "streaming": spec.get("streaming"),
        "same_split_validation_stride": config.same_split_validation_stride,
        "same_split_validation_offset": config.same_split_validation_offset,
        "data_pack_count": config.data_pack_count,
        "data_pack_index": config.data_pack_index,
    }
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True)


def source_cache_prefix(spec: dict[str, Any], split_name: str, source_idx: int) -> Path:
    preprocessing_protocol = "source_cache_staged_row_pack"
    sig_str = "|".join(
        [
            preprocessing_protocol,
            ",".join(SPECIAL_TOKENS),
            source_dataset_fingerprint(spec, split_name),
            split_name,
            str(config.train_unroll_len),
            str(vocab_size),
            str(text_vocab_size),
            str(audio_token_start),
            str(audio_token_end),
            str(image_token_start),
            str(image_token_end),
            tokenizer_fingerprint,
            str(config.tokenizer_vocab_size),
            str(config.user_inner_loss_weight),
            str(config.listen_loss_weight),
            str(config.control_loss_weight),
            str(config.interrupt_input_loss_weight),
            str(config.content_loss_weight),
            str(config.min_supervised_targets),
            str(config.audio_backend),
            str(config.audio_codebooks),
            str(config.audio_codebook_size),
            str(config.audio_frames_per_second),
            str(config.mimi_repo),
            str(config.mimi_filename),
            str(config.max_audio_seconds),
            str(config.max_audio_tokens_per_row),
            str(config.audio_task_mix),
            str(config.tts_prompt_template),
            str(config.audio_token_loss_weight),
            str(config.audio_codebook_loss_weight),
            str(config.audio_out_loss_weight),
            str(config.audio_end_loss_weight),
            str(config.output_modality_loss_weight),
            str(config.synthesize_turn_silence),
            str(config.silence_end_tokens),
        ]
    )
    sig = hashlib.md5(sig_str.encode()).hexdigest()[:10]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(spec.get("name", "source")))[:64].strip("._-") or "source"
    return cache_root_path() / f"source_{split_name}_{sig}_{safe_name}"


@dataclass
class IterableDatasetAdapter:
    factory: Any
    filters: tuple[tuple[Any, bool, dict[str, Any]], ...] = ()
    skip_count: int = 0

    def __iter__(self):
        yielded = 0
        for idx, row in enumerate(self.factory()):
            keep = True
            for fn, with_indices, fn_kwargs in self.filters:
                if with_indices:
                    keep = bool(fn(row, idx, **fn_kwargs))
                else:
                    keep = bool(fn(row, **fn_kwargs))
                if not keep:
                    break
            if not keep:
                continue
            if yielded < self.skip_count:
                yielded += 1
                continue
            yielded += 1
            yield row

    def filter(self, fn, with_indices: bool = False, fn_kwargs: dict[str, Any] | None = None):
        return IterableDatasetAdapter(
            self.factory,
            (*self.filters, (fn, bool(with_indices), dict(fn_kwargs or {}))),
            self.skip_count,
        )

    def skip(self, count: int):
        return IterableDatasetAdapter(self.factory, self.filters, self.skip_count + max(0, int(count)))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def echox_shard_urls(spec: dict[str, Any]) -> list[str]:
    data_files = as_list(spec.get("data_files"))
    if data_files:
        return [str(item) for item in data_files]

    from huggingface_hub import HfApi, hf_hub_url

    repo = str(spec.get("name") or "KurtDu/EchoX-Dialogues-Plus")
    subsets = as_list(spec.get("echox_subsets")) or ["S2S-QA/AudioQA"]
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    selected = [
        path
        for path in files
        if path.endswith(".tar.gz") and any(path.startswith(str(prefix).rstrip("/") + "/") for prefix in subsets)
    ]
    selected.sort()
    max_shards = spec.get("max_shards")
    if max_shards is not None:
        selected = selected[: max(1, int(max_shards))]
    return [hf_hub_url(repo, path, repo_type="dataset") for path in selected]


def _is_remote_ref(ref: str) -> bool:
    return ref.startswith("http://") or ref.startswith("https://")


def cache_root_path() -> Path:
    return Path(config.cache_root)


def _echox_raw_cache_path(shard_ref: str) -> Path:
    raw_dir = Path(config.echox_raw_cache_dir) if config.echox_raw_cache_dir else cache_root_path() / "echox_raw_shards"
    parsed = urllib.parse.urlparse(shard_ref)
    basename = Path(parsed.path).name or "shard.tar.gz"
    digest = hashlib.md5(shard_ref.encode("utf-8")).hexdigest()[:16]
    return raw_dir / f"{digest}.{basename}"


def stage_echox_shard_ref(
    shard_ref: str,
    progress_callback: Any | None = None,
    shard_index: int | None = None,
) -> str:
    if not _is_remote_ref(shard_ref) or not bool(config.echox_cache_raw_shards):
        return shard_ref

    cache_path = _echox_raw_cache_path(shard_ref)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    min_free_bytes = int(max(0.0, float(config.echox_raw_cache_min_free_gb)) * 1024**3)
    if min_free_bytes > 0 and free_disk_bytes(cache_path) < min_free_bytes:
        return shard_ref

    lock_path = Path(str(cache_path) + ".lock")
    lock_fd: int | None = None

    def progress_put(event: str, **payload: Any) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event, **payload)
        except Exception:
            pass

    try:
        lock_fd = acquire_pid_file_lock(lock_path, f"EchoX raw shard {shard_index}", poll_seconds=2.0, log_events=False)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            progress_put("raw_cached", raw_bytes=int(cache_path.stat().st_size))
            return str(cache_path)
        if min_free_bytes > 0 and free_disk_bytes(cache_path) < min_free_bytes:
            progress_put("raw_bypass", raw_bytes=0)
            return shard_ref

        tmp_path = Path(str(cache_path) + f".{os.getpid()}.partial")
        tmp_path.unlink(missing_ok=True)
        downloaded = 0
        last_progress = time.time()
        bypass_cache = False
        try:
            with urllib.request.urlopen(shard_ref, timeout=60) as response, tmp_path.open("wb") as out:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    downloaded += len(block)
                    if min_free_bytes > 0 and free_disk_bytes(cache_path) < min_free_bytes:
                        bypass_cache = True
                        break
                    now = time.time()
                    if now - last_progress >= 10.0:
                        progress_put("raw_download", raw_bytes=downloaded)
                        last_progress = now
        except OSError as exc:
            if getattr(exc, "errno", None) == 28:
                tmp_path.unlink(missing_ok=True)
                progress_put("raw_bypass", raw_bytes=downloaded)
                return shard_ref
            raise
        if bypass_cache:
            tmp_path.unlink(missing_ok=True)
            progress_put("raw_bypass", raw_bytes=downloaded)
            return shard_ref
        tmp_path.replace(cache_path)
        progress_put("raw_cached", raw_bytes=int(cache_path.stat().st_size))
        return str(cache_path)
    finally:
        for tmp_file in cache_path.parent.glob(f"{cache_path.name}.*.partial"):
            try:
                if f".{os.getpid()}." in tmp_file.name:
                    tmp_file.unlink(missing_ok=True)
            except Exception:
                pass
        release_pid_file_lock(lock_fd, lock_path)


def open_echox_tar(shard_ref: str):
    if _is_remote_ref(shard_ref):
        response = urllib.request.urlopen(shard_ref, timeout=60)
        return response, tarfile.open(fileobj=response, mode="r|gz")
    return None, tarfile.open(str(shard_ref), mode="r:gz")


def read_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf

        array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        return np.asarray(array, dtype=np.float32), int(sr)
    except Exception:
        return None


def iter_echox_tar_rows(spec: dict[str, Any]):
    max_wer = float(spec.get("max_wer", 0.25))
    pack = active_data_pack()
    for shard_url in echox_shard_urls(spec):
        if not spec.get("suppress_worker_logs"):
            log_info(f"[EchoX] streaming shard {shard_url}")
        staged_ref = stage_echox_shard_ref(str(shard_url))
        response = None
        tar = None
        try:
            response, tar = open_echox_tar(staged_ref)
            examples: list[dict[str, Any]] = []
            pending: dict[int, dict[str, bytes]] = {}
            audio_to_indices: dict[str, list[int]] = {}
            required_counts: dict[int, int] = {}

            for member in tar:
                if member.name == "data.json":
                    f = tar.extractfile(member)
                    if f is None:
                        raise DataQualityError(f"EchoX shard has unreadable data.json: {shard_url}")
                    examples = json.load(f)
                    for idx, example in enumerate(examples):
                        if pack is not None:
                            pack_count, pack_index = pack
                            if idx % pack_count != pack_index:
                                continue
                        paths = []
                        for turn in example.get("conversations", []):
                            path = turn.get("audio")
                            if path:
                                paths.append(str(path))
                        required_counts[idx] = len(set(paths))
                        if required_counts[idx] > 0:
                            pending[idx] = {}
                        for path in set(paths):
                            audio_to_indices.setdefault(path, []).append(idx)
                    continue

                if not examples or not member.isfile() or not member.name.endswith(".wav"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                audio_bytes = f.read()
                for idx in audio_to_indices.get(member.name, []):
                    pending[idx][member.name] = audio_bytes
                    if len(pending[idx]) < required_counts[idx]:
                        continue
                    example = dict(examples[idx])
                    conversations = []
                    usable = True
                    for turn in example.get("conversations", []):
                        turn_copy = dict(turn)
                        path = turn_copy.get("audio")
                        if path:
                            blob = pending[idx].get(str(path))
                            if blob is None:
                                usable = False
                                break
                            turn_copy["audio"] = {"bytes": blob, "path": str(path)}
                        if "wer" in turn_copy and turn_copy["wer"] is not None:
                            try:
                                if float(turn_copy["wer"]) > max_wer:
                                    usable = False
                                    break
                            except (TypeError, ValueError):
                                usable = False
                                break
                        conversations.append(turn_copy)
                    pending.pop(idx, None)
                    if usable:
                        example["conversations"] = conversations
                        example["source_shard"] = shard_url
                        yield example
        except (tarfile.ReadError, EOFError, DataQualityError) as e:
            if not spec.get("suppress_worker_logs"):
                log_info(f"[EchoX] Error reading shard {shard_url}: {e}. Skipping remainder.")
        except Exception as e:
            if not spec.get("suppress_worker_logs"):
                log_info(f"[EchoX] Unexpected error reading shard {shard_url}: {e}. Skipping remainder.")
        finally:
            if tar is not None:
                tar.close()
            if response is not None:
                response.close()


def load_dataset_from_spec(spec: dict[str, Any], split: str):
    mode = str(spec.get("mode", config.dataset_mode))
    if mode in {"echox_s2s_dialogue", "speech_dialogue"}:
        return IterableDatasetAdapter(lambda: iter_echox_tar_rows(spec))

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": bool(spec.get("streaming", config.streaming)),
    }
    if str(spec["name"]) not in {"json", "csv", "parquet", "text"}:
        kwargs["trust_remote_code"] = bool(config.dataset_trust_remote_code)
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]
    if spec.get("config"):
        return load_dataset(str(spec["name"]), str(spec["config"]), **kwargs)
    return load_dataset(str(spec["name"]), **kwargs)


def row_texts_for_tokenizer(row: dict, mode: str, spec: dict[str, Any]) -> list[str]:
    try:
        if mode == "duplex_chat" and "output" in row:
            return [content for _, content, is_idle in read_duplex_events(row) if not is_idle and content.strip()]
        if mode == "instruction_chat" and "conversations" in row:
            return [str(msg.get("value", "")) for msg in row["conversations"] if msg.get("value")]
        if mode == "dolly_instruction":
            parts = [row.get("instruction", ""), row.get("context", ""), row.get("response", "")]
            return [str(part) for part in parts if part]
        if mode == "audio_asr":
            transcript = extract_transcript(row, spec)
            return [transcript] if transcript else []
        if mode in {"echox_s2s_dialogue", "speech_dialogue"} and "conversations" in row:
            return [
                str(turn.get("value", ""))
                for turn in row.get("conversations", [])
                if turn.get("value")
            ]
        keys = [spec.get("text_key"), spec.get("prompt_key"), spec.get("response_key"), "text", "content", "prompt", "response"]
        return [str(row[key]) for key in keys if key and key in row and row[key]]
    except Exception:
        return []


def fallback_tokenizer_row_text(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    output = row.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for msg in output:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "")
            content = str(msg.get("content") or msg.get("value") or "")
            if role == "user" and content:
                parts.append(content)
        return "\n".join(parts)
    return str(row.get("question") or row.get("prompt") or row.get("image_text") or row.get("content") or "")


def iter_tokenizer_training_texts():
    def _is_local_data_source_spec(spec: dict[str, Any]) -> bool:
        data_files = spec.get("data_files")
        if not data_files:
            return False
        if isinstance(data_files, (list, tuple)):
            return all(isinstance(path, str) and "://" not in path for path in data_files)
        return isinstance(data_files, str) and "://" not in data_files

    produced = 0
    rows = 0
    specs = parse_dataset_mix()
    row_limit = int(config.tokenizer_train_rows)
    uncapped = row_limit <= 0
    tokenizer_specs = specs
    if uncapped:
        local_specs = [spec for spec in specs if _is_local_data_source_spec(spec)]
        if local_specs:
            tokenizer_specs = local_specs
    per_spec_rows = None if uncapped else max(1, math.ceil(row_limit / len(specs)))

    def log_spec_skip(name: str, split: str, exc: Exception) -> None:
        log_info(f"[Tokenizer] Skipping tokenizer source {name}:{split}: {exc}")

    for spec in tokenizer_specs:
        split = split_for_dataset_spec(spec, "train")
        try:
            ds = load_dataset_from_spec(spec, split)
        except Exception as exc:
            log_spec_skip(spec.get("name", "<unknown>"), split, exc)
            continue

        try:
            dataset_iter = safe_dataset_iter(ds, repeat_count=1, skip_rows=0, skip_log_label=f"Tokenizer:{spec.get('name')}")
            total = per_spec_rows if per_spec_rows is not None else None
            for row in progress_bar(dataset_iter, desc=f"Training tokenizer:{spec['name']}", total=total):
                if not uncapped and rows >= row_limit:
                    break
                rows += 1

                try:
                    texts = row_texts_for_tokenizer(row, str(spec.get("mode", config.dataset_mode)), spec)
                except Exception as exc:
                    log_info(f"[Tokenizer] Skipping malformed row from {spec.get('name')} split={split}: {exc}")
                    continue

                for text in texts:
                    if text:
                        produced += 1
                        yield text

                if per_spec_rows is not None and rows % per_spec_rows == 0:
                    break
        except Exception as exc:
            log_spec_skip(spec.get("name", "<unknown>"), split, exc)

    if produced == 0:
        fallback_files = [
            Path("data/propagator_instruction_balanced_seed.jsonl"),
            Path("data/propagator_image_recognition_seed.jsonl"),
            Path("data/propagator_identity.jsonl"),
            Path("data/propagator_posttrain_10k.jsonl"),
        ]
        for path in fallback_files:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        user_text = fallback_tokenizer_row_text(row)
                        if user_text:
                            produced += 1
                            yield user_text
                if produced > 0:
                    log_info(f"[Tokenizer] Fallback text seed loaded from {path}")
                    return
            except Exception as exc:
                log_info(f"[Tokenizer] Failed fallback load from {path}: {exc}")

    if produced == 0:
        raise RuntimeError("No text was yielded for tokenizer training")


def train_byte_level_bpe_tokenizer(path: Path) -> Tokenizer:
    log_info(
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
        log_info(f"Existing tokenizer is not byte-level BPE, retraining: {path}")
        should_train = True

    if should_train:
        return train_byte_level_bpe_tokenizer(path)

    log_info(f"Loading tokenizer: {path}")
    return Tokenizer.from_file(str(path))


def ensure_special_tokens(tokenizer_obj: Tokenizer) -> dict[str, int]:
    missing = [tok for tok in SPECIAL_TOKENS if tokenizer_obj.token_to_id(tok) is None]
    if missing:
        log_info(f"Adding missing special tokens to tokenizer: {missing}")
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


class DataQualityError(Exception):
    """Exception raised when a dataset row is malformed or missing required data."""
    pass


def extract_transcript(row: dict, spec: dict[str, Any] | None = None) -> str:
    keys = []
    if spec and spec.get("transcript_key"):
        keys.append(str(spec["transcript_key"]))
    keys.extend(["text", "sentence", "transcript", "transcription", "normalized_text", "text_normalized", "text_original", "target"])
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_audio_array(row: dict, spec: dict[str, Any] | None = None) -> tuple[np.ndarray, int] | None:
    keys = []
    if spec and spec.get("audio_key"):
        keys.append(str(spec["audio_key"]))
    keys.extend(["audio", "speech", "array"])

    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if isinstance(value, dict):
            if "array" in value and value["array"] is not None:
                sr = int(value.get("sampling_rate") or value.get("sample_rate") or config.audio_sample_rate)
                return np.asarray(value["array"], dtype=np.float32), sr
            if "bytes" in value and value["bytes"] is not None:
                return read_audio_bytes(bytes(value["bytes"]))
            if "path" in value and value["path"]:
                return read_audio_file(str(value["path"]))
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.asarray(value, dtype=np.float32), int(row.get("sampling_rate", config.audio_sample_rate))
    return None


def read_audio_file(path: str) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf

        array, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(array, dtype=np.float32), int(sr)
    except Exception:
        return None


def get_audio_codec() -> dict[str, Any] | None:
    global _audio_codec, _audio_codec_error
    if not config.enable_audio or config.audio_backend == "none":
        return None
    if _audio_codec is not None:
        return _audio_codec
    if _audio_codec_error is not None:
        return None

    if config.audio_backend == "mimi":
        try:
            import rustymimi
            from huggingface_hub import hf_hub_download

            model_path = os.environ.get("MIMI_MODEL_PATH", "").strip()
            if not model_path:
                cache_dir = config.mimi_cache_dir or os.environ.get("HF_HOME")
                model_path = hf_hub_download(
                    repo_id=config.mimi_repo,
                    filename=config.mimi_filename,
                    cache_dir=cache_dir,
                )
            model = rustymimi.Tokenizer(
                model_path,
                num_codebooks=int(config.audio_codebooks),
                dtype="f32",
            )
            _audio_codec = {"backend": "mimi", "model": model, "model_path": model_path}
            _audio_codec_error = None
            return _audio_codec
        except Exception as exc:
            _audio_codec_error = f"{type(exc).__name__}: {exc}"
            log_info(
                "[Audio] Mimi backend unavailable for requested --audio-backend=mimi; "
                "continuing without codec-based audio preprocessing. "
                f"Error: {_audio_codec_error}"
            )
            return None

    if config.audio_backend == "encodec":
        _audio_codec_error = "encodec backend is disabled in this TPU/JAX-focused preprocessing path"
        log_info(
            "[Audio] Encodec backend is intentionally disabled to avoid torch dependency. "
            "Use --audio-backend=mimi for preprocessing."
        )
        return None

    if config.audio_backend != "none":
        _audio_codec_error = f"unsupported audio backend: {config.audio_backend}"
        log_info(f"[Audio] Unknown audio backend requested. Audio will be disabled. Error: {_audio_codec_error}")
        return None

    return None


def audio_codes_to_token_frames(codes_np: np.ndarray) -> list[list[int]]:
    n_q = min(int(config.audio_codebooks), int(codes_np.shape[0]))
    frames: list[list[int]] = []
    for frame_idx in range(codes_np.shape[1]):
        frame = []
        for codebook_idx in range(n_q):
            code = int(codes_np[codebook_idx, frame_idx]) % config.audio_codebook_size
            frame.append(audio_token_id(codebook_idx, code))
        frame.extend([token_ids["pad"]] * (8 - len(frame)))
        frames.append(frame)
        if int(config.max_audio_tokens_per_row) > 0 and len(frames) * 8 >= int(config.max_audio_tokens_per_row):
            return frames
    return frames


def normalize_audio_array_for_codec(audio: tuple[np.ndarray, int]) -> tuple[np.ndarray, int] | None:
    array, sr = audio
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 2 and array.shape[0] > array.shape[1]:
        array = array.T
    array = np.asarray(array, dtype=np.float32)

    if float(config.max_audio_seconds) > 0.0:
        max_samples = int(float(config.max_audio_seconds) * sr)
        array = array[:, :max_samples]
    if array.size == 0:
        return None
    return array, int(sr)


def _resample_audio_linear(array: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio with simple linear interpolation (CPU-only).

    This keeps preprocessing away from torch/torchaudio dependencies.
    """
    if int(source_sr) <= 0 or int(target_sr) <= 0 or array.size == 0:
        return np.asarray(array, dtype=np.float32)
    if int(source_sr) == int(target_sr):
        return np.asarray(array, dtype=np.float32)

    source_len = int(array.shape[-1])
    target_len = max(1, int(round(source_len * float(target_sr) / float(source_sr))))
    if target_len == source_len:
        return np.asarray(array, dtype=np.float32)

    x_source = np.linspace(0.0, 1.0, source_len, endpoint=False, dtype=np.float64)
    x_target = np.linspace(0.0, 1.0, target_len, endpoint=False, dtype=np.float64)
    lower = np.floor(x_target * source_len).astype(np.int64)
    lower = np.clip(lower, 0, source_len - 1)
    upper = np.minimum(lower + 1, source_len - 1)
    alpha = x_target * source_len - lower.astype(np.float64)
    alpha = alpha.reshape(1, -1)
    out = (1.0 - alpha) * array[:, lower] + alpha * array[:, upper]
    return np.ascontiguousarray(out.astype(np.float32))


def encode_audio_batch_to_token_ids(audios: list[tuple[np.ndarray, int] | None]) -> list[list[list[int]]]:
    codec = get_audio_codec()
    if codec is None:
        return [[] for _ in audios]

    normalized = [normalize_audio_array_for_codec(audio) if audio is not None else None for audio in audios]
    valid_indices = [idx for idx, audio in enumerate(normalized) if audio is not None]
    if not valid_indices:
        return [[] for _ in audios]

    model = codec["model"]

    if codec["backend"] != "mimi":
        return [[] for _ in audios]

    model = codec["model"]
    wavs_np: list[np.ndarray] = []
    wav_lengths: list[int] = []
    for idx in valid_indices:
        array, sr = normalized[idx]
        if array.shape[0] > 1:
            array = array.mean(axis=0, keepdims=True)
        wav_np = _resample_audio_linear(array, int(sr), int(config.audio_sample_rate))
        if wav_np.size == 0:
            wav_np = np.zeros((1, 1), dtype=np.float32)
        wavs_np.append(np.ascontiguousarray(wav_np, dtype=np.float32))
        wav_lengths.append(int(wav_np.shape[-1]))

    max_len = max(wav_lengths)
    batch_np = np.zeros((len(wavs_np), 1, max_len), dtype=np.float32)
    for local_idx, wav_np in enumerate(wavs_np):
        batch_np[local_idx, :, : wav_np.shape[-1]] = wav_np

    model.reset()
    codes_np = np.asarray(model.encode(batch_np), dtype=np.int32)

    results: list[list[list[int]]] = [[] for _ in audios]
    total_frames = int(codes_np.shape[-1])
    for local_idx, original_idx in enumerate(valid_indices):
        expected_frames = int(
            math.ceil(wav_lengths[local_idx] * float(config.audio_frames_per_second) / float(config.audio_sample_rate))
        )
        expected_frames = max(1, min(total_frames, expected_frames))
        results[original_idx] = audio_codes_to_token_frames(codes_np[local_idx, :, :expected_frames])
    return results


def encode_audio_to_token_ids(row: dict, spec: dict[str, Any] | None = None) -> list[list[int]]:
    audio = extract_audio_array(row, spec)
    if audio is None:
        return []
    return encode_audio_batch_to_token_ids([audio])[0]



def decode_audio_token_ids_to_waveform(token_ids_: list[int]) -> tuple[np.ndarray, int, str | None]:
    codec = get_audio_codec()
    if codec is None:
        return np.zeros((config.audio_sample_rate,), dtype=np.float32), config.audio_sample_rate, _audio_codec_error

    if codec["backend"] != "mimi":
        return np.zeros((config.audio_sample_rate,), dtype=np.float32), config.audio_sample_rate, _audio_codec_error

    n_q = int(config.audio_codebooks)
    frames: list[list[int]] = []
    current = [0] * n_q
    seen = set()

    for token_id in token_ids_:
        parsed = audio_code_from_token_id(int(token_id))
        if parsed is None:
            if int(token_id) == token_ids_audio_end and seen:
                frames.append(current.copy())
            continue
        codebook_idx, code = parsed
        if 0 <= codebook_idx < n_q:
            current[codebook_idx] = int(code)
            seen.add(codebook_idx)
        if len(seen) == n_q:
            frames.append(current.copy())
            current = [0] * n_q
            seen = set()

    if seen:
        frames.append(current.copy())

    if not frames:
        return np.zeros((config.audio_sample_rate,), dtype=np.float32), config.audio_sample_rate, "no_audio_tokens"

    model = codec["model"]
    codes_np = np.ascontiguousarray(np.asarray(frames, dtype=np.uint32).T[None, :, :])
    model.reset()
    array = np.asarray(model.decode(codes_np), dtype=np.float32).squeeze(0).squeeze(0)
    return np.clip(array, -1.0, 1.0), config.audio_sample_rate, None


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(audio_i16.tobytes())


def audio_signal_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    audio_f = np.asarray(audio, dtype=np.float32)
    if audio_f.size == 0:
        return {"duration_seconds": 0.0, "rms": 0.0, "peak": 0.0}
    return {
        "duration_seconds": float(audio_f.shape[-1] / max(1, int(sample_rate))),
        "rms": float(np.sqrt(np.mean(np.square(audio_f)))),
        "peak": float(np.max(np.abs(audio_f))),
    }


def normalize_audio_for_eval(audio: np.ndarray) -> tuple[np.ndarray, float]:
    audio_f = np.asarray(audio, dtype=np.float32)
    target_rms = max(0.0, float(config.audio_eval_normalize_rms))
    peak_limit = max(0.0, float(config.audio_eval_peak_limit))
    if audio_f.size == 0 or target_rms <= 0.0 or peak_limit <= 0.0:
        return audio_f, 1.0

    rms = float(np.sqrt(np.mean(np.square(audio_f))))
    peak = float(np.max(np.abs(audio_f)))
    if rms <= 1e-8 or peak <= 1e-8:
        return audio_f, 1.0

    gain = min(target_rms / rms, peak_limit / peak)
    if not np.isfinite(gain) or gain <= 0.0:
        return audio_f, 1.0
    return np.clip(audio_f * gain, -peak_limit, peak_limit), float(gain)


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
        token_ids.get("text", -1),
        token_ids["audio_in"],
        token_ids.get("audio", -1),
        token_ids.get("image_in", -1),
        token_ids.get("image", -1),
        token_ids["audio_out"],
        token_ids["audio_end"],
        token_ids["silence"],
        token_ids["text_in"],
        token_ids["text_out"],
        token_ids.get("hybrid", -1),
        token_ids.get("hybrid_out", -1),
        token_ids["pad"],
        token_ids["unk"],
    }


def decision_token_ids() -> set[int]:
    return {
        token_ids["listen"],
        token_ids["user_end"],
        token_ids["user_interrupt"],
    }


def new_target_stats() -> dict[str, int]:
    return {
        "text_in": 0,
        "text_out": 0,
        "audio_in": 0,
        "audio": 0,
        "audio_out": 0,
        "audio_end": 0,
        "image_in": 0,
        "image": 0,
        "hybrid_out": 0,
        "listen": 0,
        "user_end": 0,
        "model_end": 0,
        "interrupt": 0,
        "content": 0,
        "control": 0,
        "ignored": 0,
    }


def ensure_target_stats(stats: dict[str, int]) -> None:
    for key, value in new_target_stats().items():
        stats.setdefault(key, value)


def is_control_id(token_id: int) -> bool:
    return token_id in control_token_ids()


def default_loss_weight_for_target(target_id: int) -> float:
    if target_id == token_ids["pad"]:
        return 0.0
    if target_id == token_ids["listen"]:
        return float(config.listen_loss_weight)
    if target_id == token_ids["audio_out"]:
        return float(config.audio_out_loss_weight)
    if target_id == token_ids["audio_end"]:
        return float(config.audio_end_loss_weight)
    if target_id in {token_ids["text_out"], token_ids.get("hybrid_out", -1)}:
        return float(config.output_modality_loss_weight)
    if is_audio_token_id(target_id):
        return float(config.audio_token_loss_weight)
    if is_control_id(target_id):
        return float(config.control_loss_weight)
    return float(config.content_loss_weight)


def pad_to_len(values, length: int, pad_value: int):
    values = values[:length]
    if not values:
        return values
    if isinstance(values[0], list):
        return values + [[pad_value] * len(values[0])] * (length - len(values))
    return values + [pad_value] * (length - len(values))


def pad_weights(values: list[float], length: int) -> list[float]:
    values = values[:length]
    return values + [0.0] * (length - len(values))


def read_duplex_events(row: dict) -> list[tuple[str, str, bool]]:
    raw_events = None
    if isinstance(row.get("output"), list):
        raw_events = row.get("output")
    elif isinstance(row.get("messages"), list):
        raw_events = row.get("messages")
    elif isinstance(row.get("conversations"), list):
        raw_events = row.get("conversations")
    if raw_events is None:
        raise DataQualityError("Duplex row has no supported conversation field")

    events = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        role = canonical_role(str(event.get("role", event.get("from", ""))))
        content = event.get("content", event.get("value", ""))
        if content is None or role not in {"user", "assistant"}:
            continue

        content = str(content)
        is_idle = content == "<idle>"
        if not is_idle and not content.strip():
            continue

        events.append((role, content, is_idle))

    if not events:
        raise DataQualityError("Duplex row produced no usable conversation events")
    return events


def non_idle_events(row: dict) -> list[tuple[str, str]]:
    return [(role, content) for role, content, is_idle in read_duplex_events(row) if not is_idle]


def add_target_stats(stats: dict[str, int], target_id: int, weight: float) -> None:
    ensure_target_stats(stats)
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
    elif target_id == token_ids["audio_out"]:
        stats["audio_out"] += 1
        stats["control"] += 1
    elif target_id == token_ids["audio_end"]:
        stats["audio_end"] += 1
        stats["control"] += 1
    elif target_id == token_ids["text_out"]:
        stats["text_out"] += 1
        stats["control"] += 1
    elif target_id == token_ids.get("hybrid_out", -1):
        stats["hybrid_out"] += 1
        stats["control"] += 1
    elif is_audio_token_id(target_id):
        stats["audio"] += 1
    elif is_control_id(target_id):
        stats["control"] += 1
    else:
        stats["content"] += 1


def add_input_stats(stats: dict[str, int], input_id: int) -> None:
    ensure_target_stats(stats)
    if input_id == token_ids.get("text_in", -1):
        stats["text_in"] += 1
    elif input_id == token_ids.get("audio_in", -1):
        stats["audio_in"] += 1
    elif input_id == token_ids.get("image_in", -1):
        stats["image_in"] += 1
    elif is_image_token_id(input_id):
        stats["image"] += 1


def remove_target_stats(stats: dict[str, int], target_id: int, weight: float) -> None:
    ensure_target_stats(stats)
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
    elif target_id == token_ids["audio_out"]:
        stats["audio_out"] -= 1
        stats["control"] -= 1
    elif target_id == token_ids["audio_end"]:
        stats["audio_end"] -= 1
        stats["control"] -= 1
    elif target_id == token_ids["text_out"]:
        stats["text_out"] -= 1
        stats["control"] -= 1
    elif target_id == token_ids.get("hybrid_out", -1):
        stats["hybrid_out"] -= 1
        stats["control"] -= 1
    elif is_audio_token_id(target_id):
        stats["audio"] -= 1
    elif is_control_id(target_id):
        stats["control"] -= 1
    else:
        stats["content"] -= 1


def tokenize_duplex(
    row: dict,
    *,
    allow_user_interrupts: bool | None = None,
) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    events = non_idle_events(row)
    if allow_user_interrupts is None:
        allow_user_interrupts = bool(row.get("allow_user_interrupts", False))

    in_ids: list[list[int]] = []
    tr_ids: list[list[int]] = []
    weights: list[float] = []
    stats = new_target_stats()

    user_open = False
    model_open = False
    last_user_token_index: int | None = None
    pending_model_token_index: int | None = None

    def add(input_id, target_id, weight_override: float | None = None) -> int:
        in_frame = input_id if isinstance(input_id, list) else [int(input_id)] + [token_ids["pad"]] * 7
        tr_frame = target_id if isinstance(target_id, list) else [int(target_id)] + [token_ids["pad"]] * 7
        add_input_stats(stats, int(in_frame[0]))
        target_main = tr_frame[0]
        w = default_loss_weight_for_target(target_main) if weight_override is None else float(weight_override)
        in_ids.append(in_frame)
        tr_ids.append(tr_frame)
        weights.append(float(w))
        add_target_stats(stats, int(target_main), float(w))
        return len(in_ids) - 1

    def set_target(index: int, target_id, weight_override: float | None = None) -> None:
        old_target = tr_ids[index][0]
        old_weight = weights[index]
        remove_target_stats(stats, int(old_target), float(old_weight))

        tr_frame = target_id if isinstance(target_id, list) else [int(target_id)] + [token_ids["pad"]] * 7
        target_main = tr_frame[0]
        new_weight = default_loss_weight_for_target(target_main) if weight_override is None else float(weight_override)
        tr_ids[index] = tr_frame
        weights[index] = float(new_weight)
        add_target_stats(stats, int(target_main), float(new_weight))

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
            if config.synthesize_turn_silence and config.silence_end_tokens > 0:
                set_target(last_user_token_index, token_ids["listen"], config.listen_loss_weight)
                for silence_idx in range(config.silence_end_tokens):
                    is_terminal_silence = silence_idx == config.silence_end_tokens - 1
                    add(
                        token_ids["silence"],
                        token_ids["user_end"] if is_terminal_silence else token_ids["listen"],
                        config.control_loss_weight if is_terminal_silence else config.silence_token_loss_weight,
                    )
            else:
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
            tokens = text_output_ids(tokens)
            if not model_open:
                start_model_with_token(tokens[0])
                for token_id in tokens[1:]:
                    push_model_token(token_id)
            else:
                for token_id in tokens:
                    push_model_token(token_id)

            if next_role == "user":
                if allow_user_interrupts:
                    # The actual interrupt edge is injected when the following user event is consumed.
                    # Keeping model_open here lets inject_user_interrupt() convert the edge into:
                    # previous_model_token -> [USER_INTERRUPT] with optional zero loss
                    # [USER_INTERRUPT] -> [MODEL_END]
                    # [MODEL_END] -> [USER]
                    pass
                else:
                    close_model_normally(token_ids["user"])
            elif next_role != "assistant":
                close_model_normally(token_ids["session_end"])

    if model_open:
        close_model_normally(token_ids["session_end"])

    if not in_ids or in_ids[-1][0] != token_ids["session_end"]:
        add(token_ids["session_end"], token_ids["pad"], 0.0)

    return in_ids, tr_ids, weights, stats


def tokenize_instruction_chat(row: dict) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if "conversations" not in row:
        raise KeyError("Instruction chat row must contain a conversations field")

    converted = []
    for msg in row["conversations"]:
        role = canonical_role(str(msg.get("from", "")))
        content = msg.get("value", "")
        if role in {"user", "assistant"} and content:
            converted.append({"role": role, "content": str(content)})

    return tokenize_duplex({"output": converted})


def tokenize_dolly_instruction(row: dict) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    instruction = str(row.get("instruction", "")).strip()
    context = str(row.get("context", "")).strip()
    response = str(row.get("response", "")).strip()
    if not instruction or not response:
        raise KeyError("Dolly row must contain instruction and response")
    user_text = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    return tokenize_duplex({"output": [{"role": "user", "content": user_text}, {"role": "assistant", "content": response}]})


def tokenize_plain_text(row: dict, spec: dict[str, Any] | None = None) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    keys = []
    if spec and spec.get("text_key"):
        keys.append(str(spec["text_key"]))
    keys.extend(["text", "content", "prompt"])
    text = ""
    for key in keys:
        if key in row and row[key]:
            text = str(row[key]).strip()
            break
    if not text:
        raise KeyError("Plain text row has no text field")
    text_ids = encode_text(text)
    if len(text_ids) < 8:
        raise DataQualityError("Plain text row is too short for continuation training")
    context_len = min(384, max(4, int(round(len(text_ids) * 0.20))))
    context_len = min(context_len, len(text_ids) - 4)
    user_ids = [token_ids["text_in"], *text_ids[:context_len]]
    return tokenize_modal_exchange(user_ids, text_output_ids(text_ids[context_len:]))


def choose_audio_task(row: dict, transcript: str, spec: dict[str, Any] | None = None) -> str:
    raw_mix = spec.get("audio_task_mix") if spec else None
    if isinstance(raw_mix, dict):
        weights = {str(key): float(value) for key, value in raw_mix.items()}
    elif raw_mix:
        weights = parse_json_object(str(raw_mix), {"asr": 0.25, "tts": 0.35, "audio": 0.20, "hybrid": 0.20})
    else:
        weights = parse_json_object(config.audio_task_mix, {"asr": 0.25, "tts": 0.35, "audio": 0.20, "hybrid": 0.20})
    ordered = [(key, max(0.0, value)) for key, value in weights.items()]
    total = sum(value for _, value in ordered)
    if total <= 0.0:
        return "asr"
    digest = hashlib.md5(transcript.encode("utf-8", errors="ignore")).hexdigest()
    point = (int(digest[:8], 16) / 0xFFFFFFFF) * total
    acc = 0.0
    for key, value in ordered:
        acc += value
        if point <= acc:
            return key
    return ordered[-1][0]


def format_tts_prompt(transcript: str) -> str:
    text = str(transcript).strip()
    template = (config.tts_prompt_template or "Say this aloud: {text}").strip()
    if "{text}" not in template:
        template = template.rstrip() + " {text}"
    return template.replace("{text}", text)


def text_output_ids(text_ids: list[int]) -> list[int]:
    if not text_ids:
        raise ValueError("Text output requires non-empty text ids")
    return [token_ids["text_out"], *text_ids]


def audio_output_ids(audio_ids: list[list[int]]) -> list[list[int] | int]:
    if not audio_ids:
        raise ValueError("Audio output requires non-empty audio ids")
    return [token_ids["audio_out"], *audio_ids, token_ids["audio_end"]]


def hybrid_output_ids(text_ids: list[int], audio_ids: list[list[int]]) -> list[list[int] | int]:
    if not text_ids or not audio_ids:
        raise ValueError("Hybrid output requires non-empty text and audio ids")
    return [token_ids["hybrid_out"], *text_ids, token_ids["audio_out"], *audio_ids, token_ids["audio_end"]]


def tokenize_modal_exchange(
    user_ids: list[list[int] | int],
    model_ids: list[list[int] | int],
    user_inner_weight: float | None = None,
) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if not user_ids or not model_ids:
        raise ValueError("Modal exchange requires non-empty user and model token streams")

    in_ids: list[list[int]] = []
    tr_ids: list[list[int]] = []
    weights: list[float] = []
    stats = new_target_stats()

    def add(input_id, target_id, weight_override: float | None = None) -> int:
        in_frame = input_id if isinstance(input_id, list) else [int(input_id)] + [token_ids["pad"]] * 7
        tr_frame = target_id if isinstance(target_id, list) else [int(target_id)] + [token_ids["pad"]] * 7
        add_input_stats(stats, int(in_frame[0]))
        target_main = tr_frame[0]
        w = default_loss_weight_for_target(target_main) if weight_override is None else float(weight_override)
        in_ids.append(in_frame)
        tr_ids.append(tr_frame)
        weights.append(float(w))
        add_target_stats(stats, int(target_main), float(w))
        return len(in_ids) - 1

    def set_target(index: int, target_id, weight_override: float | None = None) -> None:
        old_target = tr_ids[index][0]
        old_weight = weights[index]
        remove_target_stats(stats, int(old_target), float(old_weight))
        tr_frame = target_id if isinstance(target_id, list) else [int(target_id)] + [token_ids["pad"]] * 7
        target_main = tr_frame[0]
        new_weight = default_loss_weight_for_target(target_main) if weight_override is None else float(weight_override)
        tr_ids[index] = tr_frame
        weights[index] = float(new_weight)
        add_target_stats(stats, int(target_main), float(new_weight))

    add(token_ids["session"], token_ids["listen"])
    add(token_ids["user"], token_ids["listen"])

    last_user_idx = None
    inner_weight = config.user_inner_loss_weight if user_inner_weight is None else user_inner_weight
    for i, token_id in enumerate(user_ids):
        weight = config.listen_loss_weight if i == len(user_ids) - 1 else inner_weight
        last_user_idx = add(token_id, token_ids["listen"], weight)

    if last_user_idx is not None and config.synthesize_turn_silence and config.silence_end_tokens > 0:
        set_target(last_user_idx, token_ids["listen"], config.listen_loss_weight)
        for silence_idx in range(config.silence_end_tokens):
            is_terminal_silence = silence_idx == config.silence_end_tokens - 1
            add(
                token_ids["silence"],
                token_ids["user_end"] if is_terminal_silence else token_ids["listen"],
                config.control_loss_weight if is_terminal_silence else config.silence_token_loss_weight,
            )
    elif last_user_idx is not None:
        set_target(last_user_idx, token_ids["user_end"], config.control_loss_weight)

    add(token_ids["user_end"], token_ids["model"], config.control_loss_weight)
    add(token_ids["model"], model_ids[0])

    pending_idx = add(model_ids[0], token_ids["pad"], 0.0)
    for token_id in model_ids[1:]:
        set_target(pending_idx, token_id)
        pending_idx = add(token_id, token_ids["pad"], 0.0)
    set_target(pending_idx, token_ids["model_end"])
    add(token_ids["model_end"], token_ids["session_end"], config.control_loss_weight)
    add(token_ids["session_end"], token_ids["pad"], 0.0)

    return in_ids, tr_ids, weights, stats


def tokenize_audio_asr(row: dict, spec: dict[str, Any] | None = None) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if not config.enable_audio:
        raise RuntimeError("Audio mode is disabled")
    transcript = extract_transcript(row, spec)
    if not transcript:
        raise DataQualityError("Audio ASR row has no transcript")

    text_ids = encode_text(transcript)
    audio_ids = encode_audio_to_token_ids(row, spec)
    if not audio_ids:
        raise DataQualityError("Audio row did not produce codec tokens")

    task = choose_audio_task(row, transcript, spec)
    if task == "tts":
        prompt = format_tts_prompt(transcript)
        user_ids = [*_tokenize_modal_input_prefix("audio"), *encode_text(prompt)]
        model_ids = audio_output_ids(audio_ids)
        return tokenize_modal_exchange(user_ids, model_ids)

    if task in {"audio", "audio_audio", "speech"}:
        user_ids = [*_tokenize_modal_input_prefix("audio"), *audio_ids]
        model_ids = audio_output_ids(audio_ids)
        return tokenize_modal_exchange(user_ids, model_ids, user_inner_weight=config.listen_loss_weight)

    if task in {"hybrid", "duplex"}:
        user_ids = [*_tokenize_modal_input_prefix("hybrid"), *audio_ids]
        model_ids = hybrid_output_ids(encode_text(f"I heard: {transcript}"), audio_ids)
        return tokenize_modal_exchange(user_ids, model_ids, user_inner_weight=config.listen_loss_weight)

    user_ids = [*_tokenize_modal_input_prefix("audio"), *audio_ids]
    return tokenize_modal_exchange(user_ids, text_output_ids(text_ids), user_inner_weight=config.listen_loss_weight)


def extract_mimi_codes_array(row: dict, spec: dict[str, Any] | None = None) -> np.ndarray | None:
    keys = []
    if spec and spec.get("codes_key"):
        keys.append(str(spec["codes_key"]))
    keys.extend(["codes", "audio_codes", "mimi_codes"])
    for key in keys:
        if key not in row or row[key] is None:
            continue
        codes = np.asarray(row[key], dtype=np.int32)
        if codes.ndim != 2 or codes.size == 0:
            continue
        # Accepted source layouts:
        #   [codebooks, frames], used by shangeth/libritts-r-mimi-codes
        #   [frames, codebooks], used by some tabular pretokenized datasets
        if codes.shape[0] <= 64 and codes.shape[1] > 0:
            return codes
        if codes.shape[1] <= 64 and codes.shape[0] > 0:
            return codes.T
    return None


def pretokenized_mimi_codes_to_token_frames(codes: np.ndarray) -> list[list[int]]:
    codes_np = np.asarray(codes, dtype=np.int32)
    if codes_np.ndim != 2 or codes_np.size == 0:
        return []
    return audio_codes_to_token_frames(codes_np)


def tokenize_mimi_codes_speech_text(row: dict, spec: dict[str, Any] | None = None) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if not config.enable_audio:
        raise RuntimeError("Audio mode is disabled")
    transcript = extract_transcript(row, spec)
    if not transcript:
        raise DataQualityError("Pretokenized Mimi row has no transcript")

    codes = extract_mimi_codes_array(row, spec)
    if codes is None:
        raise DataQualityError("Pretokenized Mimi row has no codes")
    audio_ids = pretokenized_mimi_codes_to_token_frames(codes)
    if not audio_ids:
        raise DataQualityError("Pretokenized Mimi row produced no codec token frames")

    text_ids = encode_text(transcript)
    mode = str((spec or {}).get("mode", "mimi_codes_speech_text"))
    if mode == "mimi_codes_tts":
        task = "tts"
    elif mode == "mimi_codes_asr":
        task = "asr"
    else:
        task = choose_audio_task(row, transcript, spec)

    if task == "tts":
        prompt = format_tts_prompt(transcript)
        return tokenize_modal_exchange(_tokenize_modal_input_prefix("audio") + encode_text(prompt), audio_output_ids(audio_ids))

    if task in {"audio", "audio_audio", "speech"}:
        user_ids = [*_tokenize_modal_input_prefix("audio"), *audio_ids]
        return tokenize_modal_exchange(user_ids, audio_output_ids(audio_ids), user_inner_weight=config.listen_loss_weight)

    if task in {"hybrid", "duplex"}:
        user_ids = [*_tokenize_modal_input_prefix("hybrid"), *audio_ids]
        model_ids = hybrid_output_ids(encode_text(f"I heard: {transcript}"), audio_ids)
        return tokenize_modal_exchange(user_ids, model_ids, user_inner_weight=config.listen_loss_weight)

    user_ids = [*_tokenize_modal_input_prefix("audio"), *audio_ids]
    return tokenize_modal_exchange(user_ids, text_output_ids(text_ids), user_inner_weight=config.listen_loss_weight)


def first_nonempty_string(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value if str(item).strip())
        text = str(value).strip()
        if text:
            return text
    return ""


def image_text_to_patch_token_ids(text: str) -> list[int]:
    normalized = " ".join(str(text).lower().split())
    words = re.findall(r"[\w']+", normalized) or ["image"]
    tokens: list[int] = []
    for idx in range(int(config.image_tokens_per_sample)):
        word = words[idx % len(words)]
        digest = hashlib.blake2b(f"{idx}:{word}:{normalized}".encode("utf-8"), digest_size=8).digest()
        code = int.from_bytes(digest, "little") % int(config.image_patch_vocab_size)
        tokens.append(image_token_id(code))
    return tokens


def _extract_image_value(row: dict, spec: dict[str, Any] | None = None) -> Any | None:
    keys = []
    if spec and spec.get("image_key"):
        keys.append(str(spec["image_key"]))
    keys.extend(["image", "images", "frame", "camera_image", "camera_frame", "pixels"])
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _image_value_to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        if "array" in value:
            return np.asarray(value["array"])
        if "bytes" in value and value["bytes"]:
            try:
                from PIL import Image

                image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
                return np.asarray(image)
            except Exception:
                return None
        if "path" in value and value["path"]:
            value = value["path"]
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            return None
        try:
            from PIL import Image

            image = Image.open(path).convert("RGB")
            return np.asarray(image)
        except Exception:
            return None
    if hasattr(value, "convert"):
        try:
            image = value.convert("RGB")
            return np.asarray(image)
        except Exception:
            return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.size == 0:
        return None
    return array


def image_array_to_patch_token_ids(array: np.ndarray) -> list[int]:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise DataQualityError("Image row has unsupported pixel array shape")
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        array_f = np.asarray(array, dtype=np.float32)
        if array_f.size and float(np.nanmax(array_f)) <= 1.0:
            array_f = array_f * 255.0
        array = np.clip(array_f, 0.0, 255.0).astype(np.uint8)

    resolution = int(config.image_input_resolution)
    patch = int(config.image_patch_size)
    patch = max(1, min(patch, resolution))
    try:
        from PIL import Image

        image = Image.fromarray(array, mode="RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
        resized = np.asarray(image, dtype=np.uint8)
    except Exception:
        y_idx = np.linspace(0, array.shape[0] - 1, resolution).astype(np.int32)
        x_idx = np.linspace(0, array.shape[1] - 1, resolution).astype(np.int32)
        resized = array[y_idx][:, x_idx]

    tokens: list[int] = []
    for y in range(0, resolution, patch):
        for x in range(0, resolution, patch):
            block = resized[y : y + patch, x : x + patch]
            if block.size == 0:
                continue
            mean_rgb = block.reshape(-1, 3).mean(axis=0)
            r = int(mean_rgb[0] // 32) & 0x7
            g = int(mean_rgb[1] // 32) & 0x7
            b = int(mean_rgb[2] // 16) & 0xF
            tokens.append(image_token_id((r << 7) | (g << 4) | b))
            if len(tokens) >= int(config.image_tokens_per_sample):
                return tokens

    if not tokens:
        raise DataQualityError("Image row produced no patch tokens")
    while len(tokens) < int(config.image_tokens_per_sample):
        tokens.append(tokens[len(tokens) % len(tokens)])
    return tokens[: int(config.image_tokens_per_sample)]


def image_patch_token_ids(row: dict, image_text: str, spec: dict[str, Any] | None = None) -> list[int]:
    array = _image_value_to_array(_extract_image_value(row, spec))
    if array is not None:
        return image_array_to_patch_token_ids(array)
    return image_text_to_patch_token_ids(image_text)


def tokenize_image_recognition(row: dict, spec: dict[str, Any] | None = None) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if config.image_recognition_only is False:
        raise RuntimeError("Only recognition-only image supervision is implemented")

    image_key = str((spec or {}).get("image_text_key", "")).strip()
    question_key = str((spec or {}).get("question_key", "")).strip()
    answer_key = str((spec or {}).get("answer_key", "")).strip()
    image_text_keys = [image_key] if image_key else []
    image_text_keys.extend(["image_text", "caption", "description", "scene", "objects", "text"])
    question_keys = [question_key] if question_key else []
    question_keys.extend(["question", "prompt", "query"])
    answer_keys = [answer_key] if answer_key else []
    answer_keys.extend(["answer", "response", "label", "caption", "multiple_choice_answer"])

    image_text = first_nonempty_string(row, image_text_keys)
    question = first_nonempty_string(row, question_keys) or "Describe the image."
    answer = first_nonempty_string(row, answer_keys)
    if not image_text:
        if _extract_image_value(row, spec) is not None:
            image_text = "image"
    if not image_text:
        raise DataQualityError("Image recognition row has no image description or metadata text")
    if not answer:
        raise DataQualityError("Image recognition row has no answer")

    image_ids = image_patch_token_ids(row, image_text, spec)
    user_ids = [
        *_tokenize_modal_input_prefix("image"),
        *image_ids,
        token_ids["text_in"],
        *encode_text(question),
    ]
    return tokenize_modal_exchange(
        user_ids,
        text_output_ids(encode_text(answer)),
        user_inner_weight=config.listen_loss_weight,
    )


def tokenize_echox_s2s_dialogue(row: dict, spec: dict[str, Any] | None = None) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if not config.enable_audio:
        raise RuntimeError("Audio mode is disabled")
    turns = row.get("conversations")
    if not isinstance(turns, list):
        raise DataQualityError("EchoX row has no conversations list")

    target_modality = str((spec or {}).get("target_modality", "hybrid")).lower()
    inputs_all: list[list[int]] = []
    targets_all: list[list[int]] = []
    weights_all: list[float] = []
    aggregate_stats = new_target_stats()
    pending_user_audio: tuple[np.ndarray, int] | None = None

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = canonical_role(str(turn.get("from", "")))
        if role == "user":
            pending_user_audio = extract_audio_array(turn, {"audio_key": "audio"})
            continue
        if role != "assistant" or pending_user_audio is None:
            continue

        assistant_text = str(turn.get("value") or turn.get("asr") or "").strip()
        assistant_audio_raw = extract_audio_array(turn, {"audio_key": "audio"})
        user_audio, assistant_audio = encode_audio_batch_to_token_ids([pending_user_audio, assistant_audio_raw])
        if not user_audio or not assistant_audio:
            pending_user_audio = None
            continue

        if target_modality == "audio":
            model_ids = audio_output_ids(assistant_audio)
        elif target_modality == "text":
            if not assistant_text:
                pending_user_audio = None
                continue
            model_ids = text_output_ids(encode_text(assistant_text))
        else:
            if not assistant_text:
                pending_user_audio = None
                continue
            model_ids = hybrid_output_ids(encode_text(assistant_text), assistant_audio)

        user_ids = [*_tokenize_modal_input_prefix("audio"), *user_audio]
        in_ids, tr_ids, row_weights, stats = tokenize_modal_exchange(
            user_ids,
            model_ids,
            user_inner_weight=config.listen_loss_weight,
        )
        inputs_all.extend(in_ids)
        targets_all.extend(tr_ids)
        weights_all.extend(row_weights)
        for key, value in stats.items():
            aggregate_stats[key] += int(value)
        pending_user_audio = None

    if not inputs_all:
        raise DataQualityError("EchoX row produced no audio dialogue pairs")
    return inputs_all, targets_all, weights_all, aggregate_stats


def tokenize_row_by_mode(
    row: dict,
    mode: str,
    spec: dict[str, Any] | None = None,
) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]:
    if mode == "duplex_chat":
        return tokenize_duplex(row)
    if mode == "instruction_chat":
        return tokenize_instruction_chat(row)
    if mode == "dolly_instruction":
        return tokenize_dolly_instruction(row)
    if mode == "plain_text":
        return tokenize_plain_text(row, spec)
    if mode == "audio_asr":
        return tokenize_audio_asr(row, spec)
    if mode in {"mimi_codes_asr", "mimi_codes_tts", "mimi_codes_speech_text"}:
        return tokenize_mimi_codes_speech_text(row, spec)
    if mode == "image_recognition":
        return tokenize_image_recognition(row, spec)
    if mode in {"echox_s2s_dialogue", "speech_dialogue"}:
        return tokenize_echox_s2s_dialogue(row, spec)
    raise ValueError(f"Unsupported dataset mode: {mode}")


import traceback

def _worker_tokenize_row(args):
    row, mode, spec, stream_id, unroll_len = args

    # Ensure globals are initialized in case of spawn/forkserver context
    global config, tokenizer, token_ids, text_vocab_size, vocab_size
    global audio_token_start, audio_token_end, image_token_start, image_token_end
    if config is None:
        config = build_config()
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
        token_ids = ensure_special_tokens(tokenizer)
        text_vocab_size = tokenizer.get_vocab_size()
        vocab_size, audio_token_start, audio_token_end, image_token_start, image_token_end = compute_vocab_sizes(text_vocab_size)
        init_global_token_ids()

    try:
        in_ids, tr_ids, row_weights, _ = tokenize_row_by_mode(row, mode, spec)
        if len(in_ids) != len(tr_ids) or len(in_ids) != len(row_weights):
            return None
        # We must call chunk_tokenized_stream inside the worker to keep it parallel
        chunks = chunk_tokenized_stream(in_ids, tr_ids, row_weights, unroll_len)
        return stream_id, chunks
    except DataQualityError:
        # Quietly skip rows with missing/malformed data
        return None
    except Exception as e:
        log_info(f"[Worker] Unexpected exception in stream_id {stream_id}: {e}\n{traceback.format_exc()}", flush=True)
        return None


def _worker_tokenize_row_batch(args):
    rows, mode, spec, stream_id_start, unroll_len = args

    # Ensure globals are initialized in case of spawn/forkserver context
    global config, tokenizer, token_ids, text_vocab_size, vocab_size
    global audio_token_start, audio_token_end, image_token_start, image_token_end
    if config is None:
        config = build_config()
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
        token_ids = ensure_special_tokens(tokenizer)
        text_vocab_size = tokenizer.get_vocab_size()
        vocab_size, audio_token_start, audio_token_end, image_token_start, image_token_end = compute_vocab_sizes(text_vocab_size)
        init_global_token_ids()

    row_count = 0
    row_results: list[tuple[int, Any]] = []
    stream_id = int(stream_id_start)

    try:
        for row in rows:
            row_count += 1
            try:
                in_ids, tr_ids, row_weights, _ = tokenize_row_by_mode(row, mode, spec)
                if len(in_ids) != len(tr_ids) or len(in_ids) != len(row_weights):
                    row_results.append((stream_id, None))
                    stream_id += 1
                    continue
                # We must call chunk_tokenized_stream inside the worker to keep it parallel.
                chunks = chunk_tokenized_stream(in_ids, tr_ids, row_weights, unroll_len)
                row_results.append((stream_id, chunks))
            except DataQualityError:
                # Quietly skip rows with missing/malformed data but keep row cardinality.
                row_results.append((stream_id, None))
            except Exception as e:
                row_results.append((stream_id, None))
                log_info(
                    f"[Worker] Unexpected exception in stream_id {stream_id}: {e}\n{traceback.format_exc()}",
                    flush=True,
                )
            stream_id += 1
        return row_count, row_results
    except Exception:
        return 0, []


def _split_kind(split_name: str) -> str:
    return split_name.split(":", 1)[0]


def _keep_same_split_row(local_idx: int, spec: dict[str, Any], split_name: str, split: str) -> bool:
    if config.validation_skip_rows is not None:
        return True
    train_split = str(spec.get("split") or config.dataset_split)
    validation_split = str(spec.get("validation_split") or config.validation_split)
    if train_split != validation_split or split != train_split:
        return True
    stride = max(1, int(config.same_split_validation_stride))
    if stride <= 1:
        return True
    offset = int(config.same_split_validation_offset) % stride
    keep_validation = _split_kind(split_name) == "val"
    return (local_idx % stride == offset) if keep_validation else (local_idx % stride != offset)


def _merge_stats(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_run_lock() -> None:
    global _RUN_LOCK_FD, _RUN_LOCK_PATH, _RUN_LOCK_OWNER_PID
    lock_dir = cache_root_path()
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "propagator_train.active.lock"
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cwd": str(Path.cwd()),
        },
        ensure_ascii=False,
    ) + "\n"

    while True:
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.write(fd, payload.encode("utf-8"))
            _RUN_LOCK_FD = fd
            _RUN_LOCK_PATH = lock_path
            _RUN_LOCK_OWNER_PID = os.getpid()
            log_info(f"[Lock] Acquired training lock: {lock_path} pid={os.getpid()}")
            return
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", -1))
            except Exception:
                existing_pid = -1
            if pid_is_alive(existing_pid):
                raise RuntimeError(
                    f"Another propagator training/preprocessing process is active: "
                    f"pid={existing_pid}, lock={lock_path}. Stop it before starting a second run."
                )
            log_info(f"[Lock] Removing stale training lock: {lock_path}")
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def release_run_lock() -> None:
    global _RUN_LOCK_FD, _RUN_LOCK_PATH, _RUN_LOCK_OWNER_PID
    if _RUN_LOCK_OWNER_PID is not None and os.getpid() != int(_RUN_LOCK_OWNER_PID):
        return
    if _RUN_LOCK_FD is not None:
        try:
            os.close(_RUN_LOCK_FD)
        except OSError:
            pass
        _RUN_LOCK_FD = None
    if _RUN_LOCK_PATH is not None:
        try:
            _RUN_LOCK_PATH.unlink()
            log_info(f"[Lock] Released training lock: {_RUN_LOCK_PATH}")
        except FileNotFoundError:
            pass
        _RUN_LOCK_PATH = None
    _RUN_LOCK_OWNER_PID = None


atexit.register(release_run_lock)


def terminate_active_pools() -> None:
    for pool in list(_ACTIVE_POOLS):
        try:
            pool.terminate()
        except Exception:
            pass


def shutdown_pool(pool: Any, *, terminate: bool, timeout: float = 10.0) -> None:
    workers = list(getattr(pool, "_pool", []) or [])
    if terminate:
        for proc in workers:
            try:
                pid = int(getattr(proc, "pid", 0) or 0)
                if pid > 0:
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        return
    else:
        try:
            pool.close()
        except Exception:
            terminate = True
            for proc in workers:
                try:
                    if proc.is_alive():
                        proc.terminate()
                except Exception:
                    pass

    deadline = time.time() + max(0.5, float(timeout))
    while workers and time.time() < deadline:
        alive = []
        for proc in workers:
            try:
                proc.join(timeout=0.1)
                if proc.is_alive():
                    alive.append(proc)
            except Exception:
                pass
        workers = alive
        if workers:
            time.sleep(0.1)

    if workers:
        log_info(f"[Pool] Forcing exit for {len(workers)} stuck tokenization workers.")
        for proc in workers:
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        for proc in workers:
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass


def install_signal_handlers() -> None:
    def _handle_signal(signum, _frame) -> None:
        log_info(f"[Signal] Received {signum}; terminating active workers before exit.", flush=True)
        terminate_active_pools()
        release_run_lock()
        os._exit(128 + int(signum))

    try:
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    except Exception:
        pass


def acquire_pid_file_lock(lock_path: Path, label: str, poll_seconds: float = 5.0, log_events: bool = True) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label": label,
                },
                ensure_ascii=False,
            ) + "\n"
            os.write(fd, payload.encode("utf-8"))
            return fd
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", -1))
            except Exception:
                existing_pid = -1
            if not pid_is_alive(existing_pid):
                if log_events:
                    log_info(f"[Lock] Removing stale {label} lock: {lock_path}")
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if log_events:
                log_info(f"[Lock] Waiting for {label}: pid={existing_pid}, lock={lock_path}", flush=True)
            time.sleep(max(0.25, float(poll_seconds)))


def release_pid_file_lock(fd: int | None, lock_path: Path | None) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    if lock_path is not None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _echox_shard_manifest_path(shard_dir: Path, shard_index: int) -> Path:
    return shard_dir / f"shard_{shard_index:05d}.manifest.json"


def _echox_part_manifest_path(shard_dir: Path, shard_index: int, part_index: int) -> Path:
    return shard_dir / f"shard_{shard_index:05d}.part_{part_index:06d}.manifest.json"


def _valid_echox_cache_prefix(prefix: str, unroll_len: int) -> bool:
    try:
        for suffix in ("input", "target", "weight", "stream_id", "chunk_pos"):
            if not Path(str(prefix) + f".{suffix}.npy").exists():
                return False
        inputs = np.load(str(prefix) + ".input.npy", mmap_mode="r")
        return len(inputs.shape) == 3 and int(inputs.shape[1]) == int(unroll_len)
    except Exception:
        return False


def _load_json_manifest(path: Path, unroll_len: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if int(manifest.get("train_unroll_len", -1)) != int(unroll_len):
            return None
        return manifest
    except Exception:
        return None


def _load_echox_part_manifests(shard_dir: Path, shard_index: int, unroll_len: int) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    expected_part = 0
    expected_source_rows = 0
    pattern = f"shard_{shard_index:05d}.part_*.manifest.json"
    for path in sorted(shard_dir.glob(pattern)):
        manifest = _load_json_manifest(path, unroll_len)
        if manifest is None:
            continue
        part_index = int(manifest.get("part_index", -1))
        start_source_rows = int(manifest.get("start_source_rows", -1))
        end_source_rows = int(manifest.get("end_source_rows", -1))
        if part_index != expected_part or start_source_rows != expected_source_rows:
            break
        if not _valid_echox_cache_prefix(str(manifest.get("prefix", "")), unroll_len):
            break
        manifests.append(manifest)
        expected_part += 1
        expected_source_rows = end_source_rows
    return manifests


def _load_echox_shard_manifest(shard_dir: Path, shard_index: int, unroll_len: int) -> dict[str, Any] | None:
    manifest = _load_json_manifest(_echox_shard_manifest_path(shard_dir, shard_index), unroll_len)
    if manifest is None:
        return None
    if "parts" in manifest:
        for part in manifest.get("parts", []):
            if not _valid_echox_cache_prefix(str(part.get("prefix", "")), unroll_len):
                return None
        return manifest
    if _valid_echox_cache_prefix(str(manifest.get("prefix", "")), unroll_len):
        return manifest
    return None


def _worker_tokenize_echox_shard(args):
    if len(args) == 9:
        shard_index, shard_url, split_name, split, spec, shard_dir_name, stream_offset, unroll_len, progress_queue = args
    else:
        shard_index, shard_url, split_name, split, spec, shard_dir_name, stream_offset, unroll_len = args
        progress_queue = None

    global config, tokenizer, token_ids, text_vocab_size, vocab_size
    global audio_token_start, audio_token_end, image_token_start, image_token_end
    if config is None:
        config = build_config()
        tokenizer = Tokenizer.from_file(config.tokenizer_path)
        token_ids = ensure_special_tokens(tokenizer)
        text_vocab_size = tokenizer.get_vocab_size()
        vocab_size, audio_token_start, audio_token_end, image_token_start, image_token_end = compute_vocab_sizes(text_vocab_size)
        init_global_token_ids()

    shard_index = int(shard_index)
    shard_dir = Path(shard_dir_name)
    shard_dir.mkdir(parents=True, exist_ok=True)

    def progress_put(event: str, **payload: Any) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put(
                {
                    "event": event,
                    "shard_index": shard_index,
                    "time": time.time(),
                    **payload,
                }
            )
        except Exception:
            pass

    cached = _load_echox_shard_manifest(shard_dir, shard_index, int(unroll_len))
    if cached is not None:
        cached["cached"] = True
        progress_put(
            "cached",
            rows=int(cached.get("source_rows", 0)),
            chunks=int(cached.get("num_chunks", 0)),
            parts=len(cached.get("parts", [])) if isinstance(cached.get("parts"), list) else 1,
        )
        return cached

    lock_path = shard_dir / f"shard_{shard_index:05d}.lock"
    lock_fd: int | None = None
    try:
        lock_fd = acquire_pid_file_lock(lock_path, f"EchoX shard {shard_index:05d}", log_events=False)
        cached = _load_echox_shard_manifest(shard_dir, shard_index, int(unroll_len))
        if cached is not None:
            cached["cached"] = True
            progress_put(
                "cached",
                rows=int(cached.get("source_rows", 0)),
                chunks=int(cached.get("num_chunks", 0)),
                parts=len(cached.get("parts", [])) if isinstance(cached.get("parts"), list) else 1,
            )
            return cached

        for stale_tmp in shard_dir.glob(f"shard_{shard_index:05d}.*.tmp*"):
            stale_tmp.unlink(missing_ok=True)

        staged_shard_ref = stage_echox_shard_ref(str(shard_url), progress_put, shard_index)
        local_spec = dict(spec)
        local_spec["data_files"] = [str(staged_shard_ref)]
        local_spec["suppress_worker_logs"] = True
        completed_parts = _load_echox_part_manifests(shard_dir, shard_index, int(unroll_len))
        committed_source_rows = int(completed_parts[-1]["end_source_rows"]) if completed_parts else 0
        committed_chunks = sum(int(part.get("num_chunks", 0)) for part in completed_parts)
        next_part_index = len(completed_parts)
        if completed_parts:
            progress_put("resume", rows=committed_source_rows, chunks=committed_chunks, parts=len(completed_parts))

        pad_id = token_ids["pad"]
        source_rows = committed_source_rows
        emitted_chunks = committed_chunks
        last_log_time = time.time()
        last_log_rows = source_rows
        last_log_chunks = emitted_chunks
        debug_max_rows = local_spec.get("debug_max_rows")
        debug_max_rows = int(debug_max_rows) if debug_max_rows is not None else None
        spec_part_rows = int(local_spec.get("part_rows") or 0) if local_spec.get("part_rows") is not None else 0
        spec_part_chunks = int(local_spec.get("part_chunks") or 0) if local_spec.get("part_chunks") is not None else 0
        env_part_rows = int(os.environ.get("AUDIO_PREPROCESSING_PART_ROWS", "0") or "0")
        env_part_chunks = int(os.environ.get("AUDIO_PREPROCESSING_PART_CHUNKS", "0") or "0")
        part_rows = max(1, spec_part_rows or env_part_rows or 2000)
        part_chunks_limit = max(1, spec_part_chunks or env_part_chunks or 32768)

        current_part_start_rows = committed_source_rows
        current_part_stats = {
            **new_target_stats(),
            "skipped_chunks": 0,
            "source_rows": 0,
            "errors": 0,
        }
        input_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        stream_ids_local: list[int] = []
        chunk_positions_local: list[int] = []

        def write_part(end_source_rows: int) -> dict[str, Any] | None:
            nonlocal next_part_index
            nonlocal current_part_start_rows, current_part_stats
            nonlocal input_chunks, target_chunks, weight_chunks, stream_ids_local, chunk_positions_local

            if end_source_rows <= current_part_start_rows:
                return None

            part_index = next_part_index
            prefix = shard_dir / f"shard_{shard_index:05d}.part_{part_index:06d}"
            tmp_prefix = shard_dir / f"shard_{shard_index:05d}.part_{part_index:06d}.{os.getpid()}.tmp"
            manifest_path = _echox_part_manifest_path(shard_dir, shard_index, part_index)

            num_chunks = len(input_chunks)
            if num_chunks:
                np.save(str(tmp_prefix) + ".input.npy", np.stack(input_chunks, axis=0))
                np.save(str(tmp_prefix) + ".target.npy", np.stack(target_chunks, axis=0))
                np.save(str(tmp_prefix) + ".weight.npy", np.stack(weight_chunks, axis=0))
                np.save(str(tmp_prefix) + ".stream_id.npy", np.asarray(stream_ids_local, dtype=np.int64))
                np.save(str(tmp_prefix) + ".chunk_pos.npy", np.asarray(chunk_positions_local, dtype=np.int32))
            else:
                np.save(str(tmp_prefix) + ".input.npy", np.empty((0, unroll_len, 8), dtype=np.int32))
                np.save(str(tmp_prefix) + ".target.npy", np.empty((0, unroll_len, 8), dtype=np.int32))
                np.save(str(tmp_prefix) + ".weight.npy", np.empty((0, unroll_len), dtype=np.float32))
                np.save(str(tmp_prefix) + ".stream_id.npy", np.empty((0,), dtype=np.int64))
                np.save(str(tmp_prefix) + ".chunk_pos.npy", np.empty((0,), dtype=np.int32))

            for suffix in ("input", "target", "weight", "stream_id", "chunk_pos"):
                Path(str(tmp_prefix) + f".{suffix}.npy").replace(str(prefix) + f".{suffix}.npy")

            current_part_stats["source_rows"] = int(end_source_rows - current_part_start_rows)
            manifest = {
                "prefix": str(prefix),
                "shard_index": shard_index,
                "part_index": int(part_index),
                "shard_url": str(shard_url),
                "num_chunks": int(num_chunks),
                "start_source_rows": int(current_part_start_rows),
                "end_source_rows": int(end_source_rows),
                "source_rows": int(end_source_rows - current_part_start_rows),
                "train_unroll_len": int(unroll_len),
                "stats": current_part_stats,
            }
            tmp_manifest_path = Path(str(manifest_path) + f".{os.getpid()}.tmp")
            tmp_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp_manifest_path.replace(manifest_path)
            progress_put("commit", rows=end_source_rows, chunks=emitted_chunks, parts=part_index + 1)

            next_part_index += 1
            current_part_start_rows = end_source_rows
            current_part_stats = {
                **new_target_stats(),
                "skipped_chunks": 0,
                "source_rows": 0,
                "errors": 0,
            }
            input_chunks = []
            target_chunks = []
            weight_chunks = []
            stream_ids_local = []
            chunk_positions_local = []
            return manifest

        kept_rows_seen = 0
        for local_idx, row in enumerate(iter_echox_tar_rows(local_spec)):
            if not _keep_same_split_row(local_idx, spec, split_name, split):
                continue
            if kept_rows_seen < committed_source_rows:
                kept_rows_seen += 1
                if kept_rows_seen % 250 == 0 or kept_rows_seen == committed_source_rows:
                    progress_put(
                        "resume_scan",
                        rows=committed_source_rows,
                        chunks=committed_chunks,
                        parts=next_part_index,
                        scan_rows=kept_rows_seen,
                    )
                continue
            if debug_max_rows is not None and (source_rows - committed_source_rows) >= debug_max_rows:
                break

            kept_rows_seen += 1
            source_rows = kept_rows_seen
            stream_id = int(stream_offset) + shard_index * 1_000_000 + int(local_idx)
            try:
                in_ids, tr_ids, row_weights, _ = tokenize_row_by_mode(
                    row,
                    str(spec.get("mode", "echox_s2s_dialogue")),
                    spec,
                )
                if len(in_ids) != len(tr_ids) or len(in_ids) != len(row_weights):
                    current_part_stats["errors"] += 1
                    chunks = []
                else:
                    chunks = chunk_tokenized_stream(in_ids, tr_ids, row_weights, unroll_len)
            except DataQualityError:
                current_part_stats["errors"] += 1
                chunks = []
            except Exception as exc:
                current_part_stats["errors"] += 1
                progress_put("error", rows=source_rows, chunks=emitted_chunks, message=str(exc)[:240])
                chunks = []

            if not chunks:
                current_part_stats["skipped_chunks"] += 1
            else:
                for chunk_pos, (chunk_in, chunk_tr, chunk_w, chunk_stats) in enumerate(chunks):
                    input_chunks.append(np.asarray(pad_to_len(chunk_in, unroll_len, pad_id), dtype=np.int32))
                    target_chunks.append(np.asarray(pad_to_len(chunk_tr, unroll_len, pad_id), dtype=np.int32))
                    weight_chunks.append(np.asarray(pad_weights(chunk_w, unroll_len), dtype=np.float32))
                    stream_ids_local.append(stream_id)
                    chunk_positions_local.append(chunk_pos)
                    _merge_stats(current_part_stats, chunk_stats)
                    emitted_chunks += 1

            if (source_rows - current_part_start_rows) >= part_rows or len(input_chunks) >= part_chunks_limit:
                write_part(source_rows)

            if source_rows % 250 == 0:
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                progress_put(
                    "progress",
                    rows=source_rows,
                    chunks=emitted_chunks,
                    parts=next_part_index,
                    rows_per_sec=(source_rows - last_log_rows) / elapsed,
                    chunks_per_sec=(emitted_chunks - last_log_chunks) / elapsed,
                )
                last_log_time = now
                last_log_rows = source_rows
                last_log_chunks = emitted_chunks

        write_part(source_rows)
        all_parts = _load_echox_part_manifests(shard_dir, shard_index, int(unroll_len))
        aggregate_stats = {
            **new_target_stats(),
            "skipped_chunks": 0,
            "source_rows": 0,
            "errors": 0,
        }
        for part in all_parts:
            _merge_stats(aggregate_stats, part.get("stats", {}))
        aggregate_stats["source_rows"] = int(source_rows)
        num_chunks = sum(int(part.get("num_chunks", 0)) for part in all_parts)

        manifest = {
            "shard_index": shard_index,
            "shard_url": str(shard_url),
            "num_chunks": int(num_chunks),
            "source_rows": int(source_rows),
            "train_unroll_len": int(unroll_len),
            "stats": aggregate_stats,
            "parts": all_parts,
            "cached": False,
        }
        manifest_path = _echox_shard_manifest_path(shard_dir, shard_index)
        tmp_manifest_path = Path(str(manifest_path) + f".{os.getpid()}.tmp")
        tmp_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_manifest_path.replace(manifest_path)
        progress_put("done", rows=source_rows, chunks=num_chunks, parts=len(all_parts))
        return manifest
    finally:
        for tmp_file in shard_dir.glob(f"shard_{shard_index:05d}.*.{os.getpid()}.tmp*"):
            tmp_file.unlink(missing_ok=True)
        release_pid_file_lock(lock_fd, lock_path)


def cache_prefix(split_name: str, max_chunks: int, split_spec: str, skip_rows: int) -> Path:
    specs = parse_dataset_mix()
    sig_str = "|".join(
        [
            "multimodal_user_interrupt_stateful_current",
            ",".join(SPECIAL_TOKENS),
            dataset_fingerprint(specs, split_name),
            split_name,
            split_spec,
            str(skip_rows),
            str(config.train_unroll_len),
            str(vocab_size),
            str(text_vocab_size),
            str(audio_token_start),
            str(audio_token_end),
            str(image_token_start),
            str(image_token_end),
            tokenizer_fingerprint,
            str(config.tokenizer_vocab_size),
            str(max_chunks),
            str(config.user_inner_loss_weight),
            str(config.listen_loss_weight),
            str(config.control_loss_weight),
            str(config.interrupt_input_loss_weight),
            str(config.content_loss_weight),
            str(config.min_supervised_targets),
            str(config.audio_backend),
            str(config.audio_codebooks),
            str(config.audio_codebook_size),
            str(config.audio_frames_per_second),
            str(config.mimi_repo),
            str(config.mimi_filename),
            str(config.max_audio_seconds),
            str(config.max_audio_tokens_per_row),
            str(config.audio_task_mix),
            str(config.tts_prompt_template),
            str(config.audio_token_loss_weight),
            str(config.audio_codebook_loss_weight),
            str(config.audio_out_loss_weight),
            str(config.audio_end_loss_weight),
            str(config.output_modality_loss_weight),
            str(config.synthesize_turn_silence),
            str(config.silence_end_tokens),
            str(config.same_split_validation_stride),
            str(config.same_split_validation_offset),
        ]
    )
    sig = hashlib.md5(sig_str.encode()).hexdigest()[:10]
    return cache_root_path() / f"propagator_{split_name}_{sig}"


def chunk_tokenized_stream(
    in_ids: list[list[int]],
    tr_ids: list[list[int]],
    row_weights: list[float],
    unroll_len: int,
) -> list[tuple[list[list[int]], list[list[int]], list[float], dict[str, int]]]:
    chunks = []
    if not in_ids:
        return chunks

    for start in range(0, len(in_ids), unroll_len):
        chunk_in = in_ids[start : start + unroll_len]
        chunk_tr = tr_ids[start : start + unroll_len]
        chunk_w = row_weights[start : start + unroll_len]
        if not chunk_in:
            continue

        stats = new_target_stats()
        supervised = 0
        for input_id, target_id, weight in zip(chunk_in, chunk_tr, chunk_w, strict=True):
            main_input = input_id[0] if isinstance(input_id, list) else input_id
            add_input_stats(stats, int(main_input))
            main_target = target_id[0] if isinstance(target_id, list) else target_id
            if weight > 0.0 and main_target != token_ids["pad"]:
                supervised += 1
            add_target_stats(stats, int(main_target), float(weight))

        if supervised < config.min_supervised_targets:
            continue

        chunks.append((chunk_in, chunk_tr, chunk_w, stats))

    return chunks


def synthetic_control_rows(count: int, *, split_name: str) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    scenarios = [
        {
            "prompt": "What is your name?",
            "response": "I'm Propagator.",
            "interrupt": "What is your purpose?",
            "revised": "I am a research model for streaming dialogue with fixed-size memory.",
            "followup": "How do you store context?",
            "followup_response": "I update a persistent associative memory matrix as the stream advances.",
        },
        {
            "prompt": "Please summarize a robot safety checklist.",
            "response": "Check the emergency stop, clear the work area, and verify guards before operation.",
            "interrupt": "Make that shorter.",
            "revised": "Check the stop, area, and guards.",
            "followup": "What is the first item?",
            "followup_response": "Check the emergency stop.",
        },
        {
            "prompt": "Explain the memory matrix slowly.",
            "response": "The model reads from a fixed-size matrix, then writes a small update after each token.",
            "interrupt": "Use simpler words.",
            "revised": "It keeps a small memory and updates it one token at a time.",
            "followup": "Does it grow with the conversation?",
            "followup_response": "No. Its shape stays fixed.",
        },
        {
            "prompt": "What should I inspect in a failing training run?",
            "response": "Inspect data examples, per-task validation metrics, generated samples, and gradient scale.",
            "interrupt": "Focus on the data first.",
            "revised": "Inspect raw rows, transformed tokens, masks, and source proportions first.",
            "followup": "What comes after data checks?",
            "followup_response": "Check per-task loss and generated outputs.",
        },
        {
            "prompt": "Describe the audio pipeline.",
            "response": "Audio is resampled, encoded into codec frames, and trained with one target per codebook.",
            "interrupt": "Only explain the codec step.",
            "revised": "The codec converts waveform segments into synchronized discrete codebook IDs.",
            "followup": "Why must frames stay aligned?",
            "followup_response": "All codebooks describe the same audio frame and must be predicted together.",
        },
        {
            "prompt": "Give me a short answer: what is overfitting?",
            "response": "Overfitting is learning the training examples without generalizing well to new ones.",
            "interrupt": "One sentence only.",
            "revised": "Overfitting is memorizing training patterns that do not generalize.",
            "followup": "How can validation reveal it?",
            "followup_response": "Training improves while held-out performance stalls or worsens.",
        },
    ]

    rows: list[dict[str, Any]] = []
    interrupt_fraction = max(0.0, min(1.0, float(config.synthetic_interrupt_fraction)))
    interrupt_count = int(round(count * interrupt_fraction))
    for idx in range(count):
        is_interrupt = idx < interrupt_count
        scenario = scenarios[idx % len(scenarios)]
        if is_interrupt:
            rows.append(
                {
                    "allow_user_interrupts": True,
                    "output": [
                        {"role": "user", "content": scenario["prompt"]},
                        {"role": "assistant", "content": scenario["response"]},
                        {"role": "user", "content": scenario["interrupt"]},
                        {"role": "assistant", "content": scenario["revised"]},
                    ],
                }
            )
        else:
            rows.append(
                {
                    "allow_user_interrupts": False,
                    "output": [
                        {"role": "user", "content": scenario["prompt"]},
                        {"role": "assistant", "content": scenario["response"]},
                        {"role": "user", "content": scenario["followup"]},
                        {"role": "assistant", "content": scenario["followup_response"]},
                    ],
                }
            )
    rng = np.random.default_rng(config.seed + (17 if split_name == "train" else 31))
    rng.shuffle(rows)
    return rows


def build_synthetic_control_chunks(
    split_name: str,
    example_count: int,
    stream_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = synthetic_control_rows(example_count, split_name=split_name)
    unroll_len = int(config.train_unroll_len)
    pad_id = token_ids["pad"]
    input_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    stream_ids_local: list[int] = []
    chunk_positions_local: list[int] = []
    aggregate_stats = {**new_target_stats(), "source_rows": 0, "skipped_chunks": 0, "errors": 0}

    for row_idx, row in enumerate(rows):
        try:
            in_ids, tr_ids, row_weights, _ = tokenize_duplex(
                row,
                allow_user_interrupts=bool(row.get("allow_user_interrupts", False)),
            )
            chunks = chunk_tokenized_stream(in_ids, tr_ids, row_weights, unroll_len)
        except Exception:
            aggregate_stats["errors"] += 1
            continue
        if not chunks:
            aggregate_stats["skipped_chunks"] += 1
            continue
        for chunk_pos, (chunk_in, chunk_tr, chunk_w, chunk_stats) in enumerate(chunks):
            input_chunks.append(np.asarray(pad_to_len(chunk_in, unroll_len, pad_id), dtype=np.int32))
            target_chunks.append(np.asarray(pad_to_len(chunk_tr, unroll_len, pad_id), dtype=np.int32))
            weight_chunks.append(np.asarray(pad_weights(chunk_w, unroll_len), dtype=np.float32))
            stream_ids_local.append(stream_offset + row_idx)
            chunk_positions_local.append(chunk_pos)
            for key, value in chunk_stats.items():
                aggregate_stats[key] += int(value)
        aggregate_stats["source_rows"] += 1

    if not input_chunks:
        empty_inputs = np.zeros((0, unroll_len, 8), dtype=np.int32)
        empty_weights = np.zeros((0, unroll_len), dtype=np.float32)
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_pos = np.zeros((0,), dtype=np.int32)
        return empty_inputs, empty_inputs.copy(), empty_weights, empty_ids, empty_pos

    log_info(
        f"[Synthetic:{split_name}] built control chunks={len(input_chunks)} "
        f"from rows={aggregate_stats['source_rows']}: {json.dumps(aggregate_stats, ensure_ascii=False)}"
    )
    return (
        np.stack(input_chunks, axis=0),
        np.stack(target_chunks, axis=0),
        np.stack(weight_chunks, axis=0),
        np.asarray(stream_ids_local, dtype=np.int64),
        np.asarray(chunk_positions_local, dtype=np.int32),
    )


def safe_dataset_iter(dataset, repeat_count: int = 1, skip_rows: int = 0, skip_log_label: str | None = None):
    """Retrying wrapper for dataset streaming to handle transient network errors."""
    import time
    for _ in range(repeat_count):
        consumed_rows = max(0, int(skip_rows))
        max_retries = 10
        retry_delay = 5.0

        attempt = 0
        while attempt < max_retries:
            try:
                active_ds = dataset
                skip_target = int(consumed_rows)
                manual_skip = skip_target > 0
                if skip_target > 0 and hasattr(active_ds, "skip"):
                    active_ds = active_ds.skip(skip_target)
                    manual_skip = False
                elif skip_target > 0 and not config.streaming and hasattr(active_ds, "select"):
                    active_ds = active_ds.select(range(skip_target, len(active_ds)))
                    manual_skip = False

                skipped = 0
                last_skip_log = time.time()
                skip_started = last_skip_log
                label = skip_log_label or "dataset"
                if manual_skip:
                    log_info(f"[Dataset:{label}] resume_skip start rows={skip_target}")

                for row in active_ds:
                    if manual_skip and skipped < skip_target:
                        skipped += 1
                        now = time.time()
                        if now - last_skip_log >= 30.0 or skipped == skip_target:
                            elapsed = max(1e-6, now - skip_started)
                            rows_per_sec = skipped / elapsed
                            remaining = max(0, skip_target - skipped)
                            eta = remaining / max(1e-6, rows_per_sec)
                            log_info(
                                f"[Dataset:{label}] resume_skip rows={skipped}/{skip_target} "
                                f"({100.0 * skipped / max(1, skip_target):.1f}%), "
                                f"rows_per_sec={rows_per_sec:.2f}, eta={format_duration(eta)}"
                            )
                            last_skip_log = now
                        continue
                    yield row
                    consumed_rows += 1

                # If we finished the loop successfully, break the retry while
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    log_info(f"[Dataset] Fatal error after {max_retries} retries: {e}")
                    raise
                log_info(f"[Dataset] Network error during streaming (attempt {attempt}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0)

        # Reset skip_rows for the next repeat cycle
        skip_rows = 0


def _tokenization_mp_context() -> tuple[multiprocessing.context.BaseContext, str]:
    requested = str(getattr(config, "tokenize_start_method", "auto") if config is not None else "auto").lower()
    if requested not in {"auto", "fork", "spawn", "forkserver"}:
        requested = "auto"

    fallback = "spawn"
    if requested == "auto":
        if os.name == "nt":
            requested = "spawn"
        else:
            requested = "fork"

    try:
        ctx = multiprocessing.get_context(requested)
    except ValueError:
        if requested != fallback:
            log_info(f"[Tokenize] Multiprocessing start method '{requested}' unavailable; falling back to '{fallback}'.")
            ctx = multiprocessing.get_context(fallback)
            requested = fallback
        else:
            raise
    return ctx, requested


def token_cache_bytes_per_chunk(unroll_len: int) -> int:
    frame_width = 8
    input_bytes = unroll_len * frame_width * np.dtype(np.int32).itemsize
    target_bytes = unroll_len * frame_width * np.dtype(np.int32).itemsize
    weight_bytes = unroll_len * np.dtype(np.float32).itemsize
    stream_bytes = np.dtype(np.int64).itemsize
    position_bytes = np.dtype(np.int32).itemsize
    return input_bytes + target_bytes + weight_bytes + stream_bytes + position_bytes


def format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{value:.1f}TiB"


def available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def free_disk_bytes(path: Path) -> int:
    probe = path.parent if path.suffix else path
    probe.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(str(probe))
    return int(stat.f_bavail * stat.f_frsize)


def choose_token_cache_storage(cache_path: Path, max_chunks: int, unroll_len: int, resume: bool) -> str:
    requested = (config.cache_storage or "auto").lower()
    if requested not in {"auto", "disk", "memory"}:
        raise ValueError(f"cache_storage must be auto, disk, or memory; got {config.cache_storage}")
    if resume:
        return "disk"
    if requested in {"disk", "memory"}:
        return requested

    estimate = int(max_chunks) * token_cache_bytes_per_chunk(unroll_len)
    disk_free = free_disk_bytes(cache_path)
    fs_type = filesystem_type(cache_path.parent)
    mem_free = available_memory_bytes()
    disk_budget = max(0, disk_free - 8 * 1024**3)
    mem_budget = int(mem_free * 0.80) if mem_free > 0 else 0

    if estimate > disk_budget and (mem_budget <= 0 or estimate < mem_budget):
        log_info(
            f"[Cache] storage=memory path={cache_path.name} estimated={format_bytes(estimate)} "
            f"fs={fs_type} disk_budget={format_bytes(disk_budget)} mem_available={format_bytes(mem_free)} "
            "resume=process-local"
        )
        return "memory"

    log_info(
        f"[Cache] storage=disk path={cache_path.name} estimated={format_bytes(estimate)} "
        f"fs={fs_type} disk_budget={format_bytes(disk_budget)} resume=progress+meta"
    )
    return "disk"


def allocate_token_cache_arrays(
    cache_path: Path,
    max_chunks: int,
    unroll_len: int,
    storage: str,
    mmap_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape_frames = (max_chunks, unroll_len, 8)
    if storage == "memory":
        return (
            np.empty(shape_frames, dtype=np.int32),
            np.empty(shape_frames, dtype=np.int32),
            np.empty((max_chunks, unroll_len), dtype=np.float32),
            np.empty((max_chunks,), dtype=np.int64),
            np.empty((max_chunks,), dtype=np.int32),
        )

    return (
        np.memmap(str(cache_path) + ".input.bin", dtype=np.int32, mode=mmap_mode, shape=shape_frames),
        np.memmap(str(cache_path) + ".target.bin", dtype=np.int32, mode=mmap_mode, shape=shape_frames),
        np.memmap(str(cache_path) + ".weight.bin", dtype=np.float32, mode=mmap_mode, shape=(max_chunks, unroll_len)),
        np.memmap(str(cache_path) + ".stream_id.bin", dtype=np.int64, mode=mmap_mode, shape=(max_chunks,)),
        np.memmap(str(cache_path) + ".chunk_pos.bin", dtype=np.int32, mode=mmap_mode, shape=(max_chunks,)),
    )


def token_cache_file_specs(cache_path: Path, num_chunks: int, unroll_len: int) -> list[tuple[str, int]]:
    frame_bytes = int(num_chunks) * int(unroll_len) * 8 * np.dtype(np.int32).itemsize
    weight_bytes = int(num_chunks) * int(unroll_len) * np.dtype(np.float32).itemsize
    return [
        (str(cache_path) + ".input.bin", frame_bytes),
        (str(cache_path) + ".target.bin", frame_bytes),
        (str(cache_path) + ".weight.bin", weight_bytes),
        (str(cache_path) + ".stream_id.bin", int(num_chunks) * np.dtype(np.int64).itemsize),
        (str(cache_path) + ".chunk_pos.bin", int(num_chunks) * np.dtype(np.int32).itemsize),
    ]


def truncate_disk_token_cache(cache_path: Path, num_chunks: int, unroll_len: int) -> None:
    for file_name, target_size in token_cache_file_specs(cache_path, num_chunks, unroll_len):
        path = Path(file_name)
        if path.exists() and path.stat().st_size != target_size:
            with path.open("r+b") as f:
                f.truncate(target_size)


def initial_uncapped_chunk_capacity(spec: dict[str, Any] | None, dataset: Any | None = None) -> int:
    if spec and spec.get("estimated_chunks"):
        estimated = int(spec["estimated_chunks"])
        pack = active_data_pack()
        if pack is not None:
            pack_count, _ = pack
            estimated = max(1, math.ceil(estimated / max(1, pack_count)))
        return max(1, estimated)
    row_count = 0
    if dataset is not None:
        try:
            row_count = len(dataset)
        except Exception:
            row_count = 0
    repeat_count = max(1, int(spec.get("repeat", 1))) if spec else 1
    mode = str(spec.get("mode", config.dataset_mode)) if spec else str(config.dataset_mode)
    if row_count > 0:
        if mode in {"audio_asr", "mimi_codes_asr", "mimi_codes_tts", "mimi_codes_speech_text", "echox_s2s_dialogue", "speech_dialogue"} or "audio" in mode:
            chunks_per_row = max(1, math.ceil(int(config.max_audio_tokens_per_row) / max(1, int(config.train_unroll_len))))
        elif mode == "plain_text":
            chunks_per_row = 32
        else:
            chunks_per_row = 8
        return max(1024, int(row_count) * repeat_count * chunks_per_row)
    return 65_536


def flush_token_cache_arrays(*arrays: np.ndarray) -> None:
    for array in arrays:
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()


def tokenize_echox_dataset_sharded(
    split_name: str,
    cache_path: Path,
    max_chunks: int,
    spec: dict[str, Any],
    stream_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unroll_len = config.train_unroll_len
    uncapped = int(max_chunks) <= 0
    split = split_for_dataset_spec(spec, _split_kind(split_name))
    shard_urls = echox_shard_urls(spec)
    original_shard_count = len(shard_urls)
    pack = active_data_pack()
    if pack is not None:
        pack_count, pack_index = pack
        log_info(
            f"Applying staged EchoX row pack for {spec['name']} {split_name}: "
            f"row_idx % {pack_count} == {pack_index}, shards={len(shard_urls)}/{original_shard_count}"
        )
    if not shard_urls:
        raise RuntimeError("EchoX sharded tokenizer found no tar.gz shards")

    if config.validation_skip_rows is None:
        stride = max(1, int(config.same_split_validation_stride))
        offset = int(config.same_split_validation_offset) % stride
        train_split = str(spec.get("split") or config.dataset_split)
        validation_split = str(spec.get("validation_split") or config.validation_split)
        if stride > 1 and train_split == validation_split and split == train_split:
            keep_validation = _split_kind(split_name) == "val"
            label = "validation" if keep_validation else "training"
            log_info(
                f"Applying same-split {label} partition for {spec['name']} inside EchoX shard workers: "
                f"local_idx % {stride} {'==' if keep_validation else '!='} {offset}"
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_path.parent / f"{cache_path.name}.echox_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    estimated_chunks = int(spec.get("estimated_chunks", 0)) if spec.get("estimated_chunks") else 0
    if estimated_chunks > 0 and pack is not None:
        pack_count, _ = pack
        estimated_chunks = max(1, math.ceil(estimated_chunks / max(1, pack_count)))
    allocation_chunks = max(1, estimated_chunks) if uncapped and estimated_chunks > 0 else (
        initial_uncapped_chunk_capacity(spec) if uncapped else max_chunks
    )
    cache_storage = choose_token_cache_storage(cache_path, allocation_chunks, unroll_len, resume=False)
    input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions = allocate_token_cache_arrays(
        cache_path,
        allocation_chunks,
        unroll_len,
        cache_storage,
        "w+",
    )

    def ensure_merge_capacity(required_chunks: int) -> None:
        nonlocal allocation_chunks, input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions
        if required_chunks <= allocation_chunks:
            return
        new_capacity = max(required_chunks, max(allocation_chunks * 2, allocation_chunks + 1024))
        log_info(f"[Cache] Expanding merge cache for {split_name}: {allocation_chunks} -> {new_capacity} chunks")
        if cache_storage == "memory":
            new_input = np.empty((new_capacity, unroll_len, 8), dtype=np.int32)
            new_target = np.empty((new_capacity, unroll_len, 8), dtype=np.int32)
            new_weight = np.empty((new_capacity, unroll_len), dtype=np.float32)
            new_stream = np.empty((new_capacity,), dtype=np.int64)
            new_pos = np.empty((new_capacity,), dtype=np.int32)
            if actual_count:
                new_input[:actual_count] = input_tokens[:actual_count]
                new_target[:actual_count] = target_tokens[:actual_count]
                new_weight[:actual_count] = loss_weights[:actual_count]
                new_stream[:actual_count] = stream_ids[:actual_count]
                new_pos[:actual_count] = chunk_positions[:actual_count]
            input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions = (
                new_input,
                new_target,
                new_weight,
                new_stream,
                new_pos,
            )
        else:
            flush_token_cache_arrays(input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions)
            del input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions
            truncate_disk_token_cache(cache_path, new_capacity, unroll_len)
            input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions = allocate_token_cache_arrays(
                cache_path,
                new_capacity,
                unroll_len,
                "disk",
                "r+",
            )
        allocation_chunks = new_capacity

    cached_manifests: list[dict[str, Any]] = []
    partial_shard_states: dict[int, dict[str, Any]] = {}
    tasks = []
    for shard_index, shard_url in enumerate(shard_urls):
        cached = _load_echox_shard_manifest(shard_dir, shard_index, unroll_len)
        if cached is not None:
            cached["cached"] = True
            cached_manifests.append(cached)
        else:
            completed_parts = _load_echox_part_manifests(shard_dir, shard_index, unroll_len)
            if completed_parts:
                partial_shard_states[shard_index] = {
                    "status": "partial_cached",
                    "rows": int(completed_parts[-1].get("end_source_rows", 0)),
                    "chunks": sum(int(part.get("num_chunks", 0)) for part in completed_parts),
                    "parts": len(completed_parts),
                    "scan_rows": 0,
                    "raw_bytes": 0,
                    "updated": time.time(),
                }
            tasks.append((shard_index, shard_url, split_name, split, spec, str(shard_dir), stream_offset, unroll_len))

    requested_workers = config.audio_preprocessing_workers or min(os.cpu_count() or 1, 32)
    num_workers = max(1, min(int(requested_workers), len(tasks) if tasks else 1, 64))
    log_info(
        f"[EchoX] shard workers={num_workers}, requested_workers={requested_workers}, shards={len(shard_urls)}, "
        "workers independently stream tar shards and write tmpfs shard caches before merge."
    )

    manifests = list(cached_manifests)
    start_time = time.time()
    completed = len(cached_manifests)
    shard_states: dict[int, dict[str, Any]] = {}
    shard_states.update(partial_shard_states)
    for item in cached_manifests:
        shard_idx = int(item.get("shard_index", -1))
        shard_states[shard_idx] = {
            "status": "cached",
            "rows": int(item.get("source_rows", 0)),
            "chunks": int(item.get("num_chunks", 0)),
            "parts": len(item.get("parts", [])) if isinstance(item.get("parts"), list) else 1,
            "updated": time.time(),
        }

    last_main_log_time = start_time
    last_main_log_chunks = sum(int(state.get("chunks", 0)) for state in shard_states.values())
    baseline_main_log_chunks = last_main_log_chunks
    baseline_main_log_rows = sum(int(state.get("rows", 0)) for state in shard_states.values())

    def log_shard_progress() -> None:
        nonlocal last_main_log_time, last_main_log_chunks
        now = time.time()
        elapsed = max(1e-6, time.time() - start_time)
        total_chunks = sum(int(state.get("chunks", 0)) for state in shard_states.values())
        total_rows = sum(int(state.get("rows", 0)) for state in shard_states.values())
        total_parts = sum(int(state.get("parts", 0)) for state in shard_states.values())
        total_raw_bytes = sum(int(state.get("raw_bytes", 0)) for state in shard_states.values())
        total_scan_rows = sum(int(state.get("scan_rows", 0)) for state in shard_states.values())
        done_shards = sum(1 for state in shard_states.values() if state.get("status") in {"cached", "done"})
        active_shards = sum(1 for state in shard_states.values() if state.get("status") not in {"cached", "done"})
        recent_elapsed = max(1e-6, now - last_main_log_time)
        recent_chunks_per_sec = (total_chunks - last_main_log_chunks) / recent_elapsed
        new_chunks = max(0, total_chunks - baseline_main_log_chunks)
        new_rows = max(0, total_rows - baseline_main_log_rows)
        chunks_per_sec = new_chunks / elapsed
        rows_per_sec = new_rows / elapsed
        progress_total = estimated_chunks if uncapped else (min(estimated_chunks, max_chunks) if estimated_chunks > 0 else max_chunks)
        if progress_total > 0:
            remaining = max(0, progress_total - total_chunks)
            effective_rate = recent_chunks_per_sec
            eta_text = format_duration(remaining / effective_rate) if effective_rate > 0 else "unknown"
            if uncapped:
                progress = f"chunks={total_chunks}/{progress_total} ({100.0 * total_chunks / max(1, progress_total):.1f}%)"
            else:
                progress = (
                    f"chunks={total_chunks}/{progress_total} "
                    f"({100.0 * total_chunks / max(1, progress_total):.1f}%), "
                    f"target_chunks={max_chunks}"
                )
        else:
            eta_text = "unknown"
            progress = f"chunks={total_chunks}"
        log_info(
            f"[EchoX main:{split_name}] shards={done_shards}/{len(shard_urls)}, active={active_shards}, "
            f"parts={total_parts}, rows={total_rows}, resume_scan={total_scan_rows}/{baseline_main_log_rows}, "
            f"raw={format_bytes(total_raw_bytes)}, {progress}, rows_per_sec={rows_per_sec:.2f}, "
            f"chunks_per_sec={chunks_per_sec:.2f}, recent_chunks_per_sec={recent_chunks_per_sec:.2f}, "
            f"eta={eta_text}"
        )
        last_main_log_time = now
        last_main_log_chunks = total_chunks

    if cached_manifests or partial_shard_states:
        log_shard_progress()

    if tasks:
        import queue as queue_module

        ctx, mp_method = _tokenization_mp_context()
        log_info(f"[EchoX] multiprocessing method={mp_method}, maxtasks_per_child={max(0, int(config.tokenize_maxtasks_per_child or 0))}")
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        for task in tasks:
            shard_states.setdefault(
                int(task[0]),
                {
                    "status": "running",
                    "rows": 0,
                    "chunks": 0,
                    "parts": 0,
                    "scan_rows": 0,
                    "raw_bytes": 0,
                    "updated": time.time(),
                },
            )

        def drain_progress_queue() -> int:
            drained = 0
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue_module.Empty:
                    break
                except Exception:
                    break
                drained += 1
                shard_idx = int(event.get("shard_index", -1))
                if shard_idx < 0:
                    continue
                state = shard_states.setdefault(
                    shard_idx,
                    {
                        "status": "running",
                        "rows": 0,
                        "chunks": 0,
                        "parts": 0,
                        "scan_rows": 0,
                        "raw_bytes": 0,
                        "updated": time.time(),
                    },
                )
                event_name = str(event.get("event", "progress"))
                if event_name in {"cached", "done"}:
                    state["status"] = event_name
                elif event_name in {"raw_download", "raw_cached", "raw_bypass"}:
                    state["status"] = event_name
                elif event_name == "resume":
                    state["status"] = "resumed"
                elif event_name == "resume_scan":
                    state["status"] = "resume_scanning"
                elif event_name == "commit":
                    state["status"] = "running"
                elif event_name == "error":
                    state["status"] = "running_with_errors"
                else:
                    state.setdefault("status", "running")
                for key in ("rows", "chunks", "parts"):
                    if key in event:
                        state[key] = max(int(state.get(key, 0)), int(event.get(key, 0)))
                if "scan_rows" in event:
                    state["scan_rows"] = max(int(state.get("scan_rows", 0)), int(event.get("scan_rows", 0)))
                if "raw_bytes" in event:
                    state["raw_bytes"] = max(int(state.get("raw_bytes", 0)), int(event.get("raw_bytes", 0)))
                state["updated"] = float(event.get("time", time.time()))
            return drained

        async_results = []
        pool = ctx.Pool(
            processes=num_workers,
            maxtasksperchild=max(0, int(config.tokenize_maxtasks_per_child or 0)) or None,
        )
        _ACTIVE_POOLS.append(pool)
        try:
            for task in tasks:
                task_with_queue = (*task, progress_queue)
                async_results.append(pool.apply_async(_worker_tokenize_echox_shard, (task_with_queue,)))

            pending = list(async_results)
            last_periodic_log = time.time()
            while pending:
                drain_progress_queue()
                next_pending = []
                for result in pending:
                    if not result.ready():
                        next_pending.append(result)
                        continue
                    manifest = result.get()
                    manifests.append(manifest)
                    completed += 1
                    shard_idx = int(manifest.get("shard_index", -1))
                    if shard_idx >= 0:
                        shard_states[shard_idx] = {
                            "status": "done" if not manifest.get("cached") else "cached",
                            "rows": int(manifest.get("source_rows", 0)),
                            "chunks": int(manifest.get("num_chunks", 0)),
                            "parts": len(manifest.get("parts", [])) if isinstance(manifest.get("parts"), list) else 1,
                            "updated": time.time(),
                        }
                    log_shard_progress()
                pending = next_pending
                now = time.time()
                if now - last_periodic_log >= 30.0:
                    log_shard_progress()
                    last_periodic_log = now
                if pending:
                    time.sleep(1.0)
            drain_progress_queue()
            shutdown_pool(pool, terminate=False)
        except BaseException:
            shutdown_pool(pool, terminate=True)
            raise
        finally:
            if pool in _ACTIVE_POOLS:
                _ACTIVE_POOLS.remove(pool)
            manager.shutdown()

    manifests.sort(key=lambda item: int(item.get("shard_index", 0)))
    actual_count = 0
    source_rows = 0
    aggregate_stats = {
        **new_target_stats(),
        "skipped_chunks": 0,
        "source_rows": 0,
        "errors": 0,
    }
    log_info(f"[EchoX:{split_name}] Merging {len(manifests)} shard caches into {cache_path.name}")
    for manifest in manifests:
        entries = manifest.get("parts")
        if not entries:
            entries = [manifest]
        manifest_chunks = int(manifest.get("num_chunks", 0))
        if manifest_chunks <= 0:
            _merge_stats(aggregate_stats, manifest.get("stats", {}))
            source_rows += int(manifest.get("source_rows", 0))
            continue

        truncated = False
        for entry in entries:
            entry_chunks = int(entry.get("num_chunks", 0))
            if entry_chunks <= 0:
                continue
            if uncapped:
                take = entry_chunks
                ensure_merge_capacity(actual_count + take)
            else:
                take = min(entry_chunks, max_chunks - actual_count)
                if take <= 0:
                    truncated = True
                    break
            prefix = str(entry["prefix"])
            input_tokens[actual_count : actual_count + take] = np.load(prefix + ".input.npy", mmap_mode="r")[:take]
            target_tokens[actual_count : actual_count + take] = np.load(prefix + ".target.npy", mmap_mode="r")[:take]
            loss_weights[actual_count : actual_count + take] = np.load(prefix + ".weight.npy", mmap_mode="r")[:take]
            stream_ids[actual_count : actual_count + take] = np.load(prefix + ".stream_id.npy", mmap_mode="r")[:take]
            chunk_positions[actual_count : actual_count + take] = np.load(prefix + ".chunk_pos.npy", mmap_mode="r")[:take]
            actual_count += take
            if not uncapped and take < entry_chunks:
                truncated = True
                break
        source_rows += int(manifest.get("source_rows", 0))
        _merge_stats(aggregate_stats, manifest.get("stats", {}))
        if truncated:
            log_info(f"[EchoX:{split_name}] Truncated shard {manifest.get('shard_index')} at max_chunks={max_chunks}")
            break

    flush_token_cache_arrays(input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions)
    aggregate_stats["source_rows"] = source_rows
    meta = {
        "num_rows": actual_count,
        "num_chunks": actual_count,
        "source_rows": source_rows,
        "train_unroll_len": unroll_len,
        "stats": aggregate_stats,
        "dataset_name": spec.get("name"),
        "dataset_config": spec.get("config"),
        "dataset_mode": spec.get("mode"),
        "repeat": 1,
        "protocol": "multimodal_user_interrupt_stateful_echox_sharded",
        "special_tokens": SPECIAL_TOKENS,
        "text_vocab_size": text_vocab_size,
        "audio_token_start": audio_token_start,
        "audio_token_end": audio_token_end,
        "image_token_start": image_token_start,
        "image_token_end": image_token_end,
        "shards": len(manifests),
    }
    if cache_storage == "disk":
        del input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions
        truncate_disk_token_cache(cache_path, actual_count, unroll_len)
        with open(str(cache_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log_info(f"[EchoX:{split_name}] Removing merged shard cache: {shard_dir}")
        shutil.rmtree(shard_dir, ignore_errors=True)
        cached = open_tokenized_cache(cache_path, actual_count, unroll_len)
        log_info(f"Tokenized {actual_count} chunks from {source_rows} EchoX source rows for {split_name}")
        log_info(json.dumps(aggregate_stats, ensure_ascii=False, indent=2))
        return cached

    log_info(f"[EchoX:{split_name}] Removing merged shard cache: {shard_dir}")
    shutil.rmtree(shard_dir, ignore_errors=True)
    log_info(f"Tokenized {actual_count} chunks from {source_rows} EchoX source rows for {split_name}")
    log_info(json.dumps(aggregate_stats, ensure_ascii=False, indent=2))
    return (
        input_tokens[:actual_count],
        target_tokens[:actual_count],
        loss_weights[:actual_count],
        stream_ids[:actual_count],
        chunk_positions[:actual_count],
    )


def tokenize_dataset_rows(
    dataset,
    split_name: str,
    cache_path: Path,
    max_chunks: int,
    mode: str | None = None,
    spec: dict[str, Any] | None = None,
    stream_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unroll_len = config.train_unroll_len
    row_mode = mode or config.dataset_mode
    repeat_count = max(1, int(spec.get("repeat", 1))) if spec else 1
    uncapped = int(max_chunks) <= 0
    log_info(f"Tokenizing {split_name}: mode={row_mode}, train_unroll_len={unroll_len}")
    if row_mode == "echox_s2s_dialogue" and repeat_count == 1 and spec:
        return tokenize_echox_dataset_sharded(split_name, cache_path, max_chunks, spec, stream_offset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(cache_path) + ".progress.json")
    estimated_chunks = int(spec.get("estimated_chunks", 0)) if spec and spec.get("estimated_chunks") else 0
    pack = active_data_pack()
    if estimated_chunks > 0 and pack is not None:
        pack_count, _ = pack
        estimated_chunks = max(1, math.ceil(estimated_chunks / max(1, pack_count)))

    actual_count = 0
    source_rows = 0
    aggregate_stats = {
        **new_target_stats(),
        "skipped_chunks": 0,
        "source_rows": 0,
        "errors": 0,
    }

    resume = False
    if repeat_count == 1 and config.cache_resume and progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            actual_count = int(progress.get("actual_count", 0))
            source_rows = int(progress.get("source_rows", 0))
            aggregate_stats.update(progress.get("stats", {}))
            resume = actual_count > 0 or source_rows > 0
            log_info(f"Resuming {split_name} tokenization: chunks={actual_count}, source_rows={source_rows}")
        except Exception as exc:
            log_info(f"Ignoring invalid cache progress {progress_path}: {exc}")

    mmap_mode = "r+" if resume else "w+"
    effective_max_chunks = initial_uncapped_chunk_capacity(spec, dataset) if uncapped else max_chunks
    if resume and not uncapped and actual_count > max_chunks:
        effective_max_chunks = actual_count
        log_info(
            f"[Tokenize:{split_name}] Resume cache has {actual_count} chunks above target_chunks={max_chunks}; "
            "keeping the existing cache size."
        )
    if resume and uncapped:
        effective_max_chunks = max(effective_max_chunks, actual_count + max(1024, actual_count))
    cache_storage = choose_token_cache_storage(cache_path, effective_max_chunks, unroll_len, resume)
    if resume and not uncapped and cache_storage == "disk" and actual_count >= max_chunks:
        log_info(
            f"[Tokenize:{split_name}] Resume cache already reached target_chunks={max_chunks}; "
            f"finalizing existing chunks={actual_count} without truncating."
        )
        aggregate_stats["source_rows"] = source_rows
        meta = {
            "num_rows": actual_count,
            "num_chunks": actual_count,
            "source_rows": source_rows,
            "train_unroll_len": unroll_len,
            "stats": aggregate_stats,
            "dataset_name": spec.get("name") if spec else config.dataset_name,
            "dataset_config": spec.get("config") if spec else None,
            "dataset_mode": row_mode,
            "repeat": repeat_count,
            "protocol": "multimodal_user_interrupt_stateful",
            "special_tokens": SPECIAL_TOKENS,
            "text_vocab_size": text_vocab_size,
            "audio_token_start": audio_token_start,
            "audio_token_end": audio_token_end,
            "image_token_start": image_token_start,
            "image_token_end": image_token_end,
        }
        with open(str(cache_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if progress_path.exists():
            progress_path.unlink()
        cached = open_tokenized_cache(cache_path, actual_count, unroll_len)
        log_info(f"Tokenized {actual_count} chunks from {source_rows} source rows for {split_name}")
        log_info(json.dumps(aggregate_stats, ensure_ascii=False, indent=2))
        return cached

    input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions = allocate_token_cache_arrays(
        cache_path,
        effective_max_chunks,
        unroll_len,
        cache_storage,
        mmap_mode,
    )

    pad_id = token_ids["pad"]

    def ensure_token_capacity(required_chunks: int) -> None:
        nonlocal effective_max_chunks, input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions
        if required_chunks <= effective_max_chunks:
            return
        new_capacity = max(required_chunks, max(effective_max_chunks * 2, effective_max_chunks + 1024))
        log_info(f"[Cache] Expanding token cache for {split_name}: {effective_max_chunks} -> {new_capacity} chunks")
        if cache_storage == "memory":
            new_input = np.empty((new_capacity, unroll_len, 8), dtype=np.int32)
            new_target = np.empty((new_capacity, unroll_len, 8), dtype=np.int32)
            new_weight = np.empty((new_capacity, unroll_len), dtype=np.float32)
            new_stream = np.empty((new_capacity,), dtype=np.int64)
            new_pos = np.empty((new_capacity,), dtype=np.int32)
            if actual_count:
                new_input[:actual_count] = input_tokens[:actual_count]
                new_target[:actual_count] = target_tokens[:actual_count]
                new_weight[:actual_count] = loss_weights[:actual_count]
                new_stream[:actual_count] = stream_ids[:actual_count]
                new_pos[:actual_count] = chunk_positions[:actual_count]
            input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions = (
                new_input,
                new_target,
                new_weight,
                new_stream,
                new_pos,
            )
        else:
            flush_token_cache_arrays(input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions)
            del input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions
            truncate_disk_token_cache(cache_path, new_capacity, unroll_len)
            input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions = allocate_token_cache_arrays(
                cache_path,
                new_capacity,
                unroll_len,
                "disk",
                "r+",
            )
        effective_max_chunks = new_capacity

    def flush_progress() -> None:
        aggregate_stats["source_rows"] = source_rows
        flush_token_cache_arrays(input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions)
        if cache_storage != "disk":
            return
        progress = {
            "actual_count": actual_count,
            "source_rows": source_rows,
            "train_unroll_len": unroll_len,
            "max_chunks": effective_max_chunks,
            "dataset_name": spec.get("name") if spec else config.dataset_name,
            "dataset_mode": row_mode,
            "repeat": repeat_count,
            "stats": aggregate_stats,
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Use multiprocessing for parallel tokenization, preventing Tokenizer deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cpu_count = os.cpu_count() or 1
    is_audio = "audio" in row_mode or row_mode in {"audio_asr", "mimi_codes_asr", "mimi_codes_tts", "mimi_codes_speech_text", "echox_s2s_dialogue", "speech_dialogue"}
    if is_audio:
        num_workers = config.audio_preprocessing_workers or min(cpu_count, 32)
        row_batch_size = max(1, int(config.audio_preprocessing_batch_rows)) if int(config.audio_preprocessing_batch_rows) > 0 else max(
            1,
            int(config.audio_preprocessing_chunk_size),
        )
        chunk_size = max(1, int(config.audio_preprocessing_chunk_size))
        if row_mode == "echox_s2s_dialogue" and num_workers > 64:
            log_info(
                f"[Tokenize:{split_name}] Limiting EchoX audio workers from {num_workers} to 64; "
                "higher values were slower and unstable due to audio IPC pressure."
            )
            num_workers = 64
    else:
        num_workers = config.text_preprocessing_workers or min(cpu_count, 48)
        row_batch_size = max(1, int(config.text_preprocessing_batch_rows)) if int(config.text_preprocessing_batch_rows) > 0 else max(
            1,
            int(config.text_preprocessing_chunk_size),
        )
        chunk_size = max(1, min(128, int(config.text_preprocessing_chunk_size)))
        if num_workers > 64:
            log_info(
                f"[Tokenize:{split_name}] Limiting text workers from {num_workers} to 64; "
                "excessive workers often lead to deadlocks and high IPC overhead."
            )
            num_workers = 64

    log_info(
        f"Starting parallel tokenization for {split_name} with {num_workers} workers "
        f"(batch_size={row_batch_size}, chunk_size={chunk_size}, cpu_count={cpu_count}, is_audio={is_audio})"
    )

    ctx, mp_method = _tokenization_mp_context()
    env_imap = int(config.tokenize_imap_chunk_size or 0)
    if env_imap > 0:
        imap_chunk_size = max(1, env_imap)
    else:
        imap_chunk_size = max(1, min(16, row_batch_size))
    maxtasks_per_child = max(0, int(config.tokenize_maxtasks_per_child or 0))
    approx_rows_per_child = maxtasks_per_child * imap_chunk_size * row_batch_size if maxtasks_per_child > 0 else 0
    recycle_text = (
        f", approx_rows_per_child={approx_rows_per_child}"
        if approx_rows_per_child > 0
        else ", worker_recycle=disabled"
    )
    log_info(
        f"[Tokenize] multiprocessing method={mp_method}, imap_chunk_size={imap_chunk_size} "
        f"maxtasks_per_child={maxtasks_per_child}{recycle_text}"
    )

    pool = ctx.Pool(
        processes=num_workers,
        maxtasksperchild=maxtasks_per_child or None,
    )
    _ACTIVE_POOLS.append(pool)
    stop_early = False
    pool_completed = False
    display_progress_total = int(estimated_chunks) if int(estimated_chunks) > 0 else max(1, int(effective_max_chunks))
    try:
        # Prepare an iterator for the pool
        base_source_rows = source_rows

        def _row_gen_impl():
            batch: list[Any] = []
            next_stream_id = stream_offset + base_source_rows
            batch_stream_id = next_stream_id
            # Use safe_dataset_iter to handle HF streaming network errors
            for row in safe_dataset_iter(
                dataset,
                repeat_count=repeat_count,
                skip_rows=base_source_rows,
                skip_log_label=split_name,
            ):
                if not batch:
                    batch_stream_id = next_stream_id
                batch.append(row)
                next_stream_id += 1
                if len(batch) >= row_batch_size:
                    yield (batch, row_mode, spec, batch_stream_id, unroll_len)
                    batch = []
            if batch:
                yield (batch, row_mode, spec, batch_stream_id, unroll_len)

        row_gen = _row_gen_impl()

        # Use line-based periodic logs for parallel tokenization; interactive bars
        # get misleading when resumed worker progress races with main-process state.
        pbar = progress_bar(
            desc=f"Tokenizing {split_name}",
            total=None if uncapped else effective_max_chunks,
            initial=actual_count,
            disable=True,
        )
        last_log_time = time.time()
        last_log_rows = source_rows
        last_log_chunks = actual_count
        for result in pool.imap_unordered(_worker_tokenize_row_batch, row_gen, chunksize=imap_chunk_size):
            if not uncapped and actual_count >= effective_max_chunks:
                # We need to signal the pool to stop if possible, but imap is lazy
                break

            if not isinstance(result, tuple) or len(result) != 2:
                continue
            batch_rows, row_results = result
            source_rows += int(batch_rows)

            # Periodic print for log visibility without exploding long-running logs.
            now = time.time()
            if now - last_log_time >= 30.0:
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                rows_per_sec = (source_rows - last_log_rows) / elapsed
                chunks_per_sec = (actual_count - last_log_chunks) / elapsed
                if uncapped and estimated_chunks <= 0:
                    progress_total = max(1, int(effective_max_chunks))
                    progress_total_kind = "capacity"
                elif uncapped:
                    progress_total = max(int(estimated_chunks), int(actual_count))
                    progress_total_kind = "estimated"
                elif estimated_chunks > 0:
                    progress_total = min(int(estimated_chunks), int(effective_max_chunks))
                    progress_total_kind = "target"
                else:
                    progress_total = max(1, int(effective_max_chunks))
                    progress_total_kind = "target"
                if progress_total > 0:
                    progress_pct = 100.0 * actual_count / max(1, int(progress_total))
                    remaining_chunks = max(0, int(progress_total) - int(actual_count))
                    eta = remaining_chunks / max(1e-6, chunks_per_sec)
                    if uncapped:
                        progress_text = (
                            f"chunks={actual_count}/{int(progress_total)} "
                            f"({progress_pct:.1f}% {progress_total_kind})"
                        )
                    else:
                        progress_text = (
                            f"chunks={actual_count}/{int(progress_total)} ({progress_pct:.1f}%), "
                            f"target_chunks={effective_max_chunks}"
                        )
                else:
                    eta = 0.0
                    progress_text = f"chunks={actual_count} capacity={int(effective_max_chunks)}"
                eta_text = (
                    format_duration(eta) if progress_total > 0 and chunks_per_sec > 0 else "running"
                )
                log_info(
                    f"[Tokenize:{split_name}] rows={source_rows}, {progress_text}, "
                    f"rows_per_sec={rows_per_sec:.2f}, chunks_per_sec={chunks_per_sec:.2f}, "
                    f"eta={eta_text}"
                )
                last_log_time = now
                last_log_rows = source_rows
                last_log_chunks = actual_count

            if not row_results:
                aggregate_stats["errors"] += 1
                continue

            added_chunks = 0
            for stream_id, chunks in row_results:
                if chunks is None:
                    aggregate_stats["errors"] += 1
                    continue
                if not chunks:
                    aggregate_stats["skipped_chunks"] += 1
                    continue

                for chunk_pos, (chunk_in, chunk_tr, chunk_w, chunk_stats) in enumerate(chunks):
                    if not uncapped and actual_count >= effective_max_chunks:
                        break
                    if uncapped:
                        ensure_token_capacity(actual_count + 1)

                    input_tokens[actual_count] = np.asarray(pad_to_len(chunk_in, unroll_len, pad_id), dtype=np.int32)
                    target_tokens[actual_count] = np.asarray(pad_to_len(chunk_tr, unroll_len, pad_id), dtype=np.int32)
                    loss_weights[actual_count] = np.asarray(pad_weights(chunk_w, unroll_len), dtype=np.float32)
                    stream_ids[actual_count] = stream_id
                    chunk_positions[actual_count] = chunk_pos

                    _merge_stats(aggregate_stats, chunk_stats)

                    actual_count += 1
                    added_chunks += 1

            if added_chunks > 0:
                pbar.update(added_chunks)
            pbar.set_postfix({"chunks": actual_count})

            if source_rows % max(1, int(config.cache_flush_every)) == 0:
                flush_progress()
            if not uncapped and actual_count >= effective_max_chunks:
                log_info(f"[Tokenize:{split_name}] Reached target_chunks={effective_max_chunks}; terminating worker pool.")
                stop_early = True
                break

        pbar.close()
        pool_completed = True
    finally:
        if pool in _ACTIVE_POOLS:
            _ACTIVE_POOLS.remove(pool)
        shutdown_pool(pool, terminate=not (pool_completed and not stop_early))

    flush_token_cache_arrays(input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions)

    aggregate_stats["source_rows"] = source_rows

    meta = {
        "num_rows": actual_count,
        "num_chunks": actual_count,
        "source_rows": source_rows,
        "train_unroll_len": unroll_len,
        "stats": aggregate_stats,
        "dataset_name": spec.get("name") if spec else config.dataset_name,
        "dataset_config": spec.get("config") if spec else None,
        "dataset_mode": row_mode,
        "repeat": repeat_count,
        "protocol": "multimodal_user_interrupt_stateful",
        "special_tokens": SPECIAL_TOKENS,
        "text_vocab_size": text_vocab_size,
        "audio_token_start": audio_token_start,
        "audio_token_end": audio_token_end,
        "image_token_start": image_token_start,
        "image_token_end": image_token_end,
    }
    if cache_storage == "disk":
        del input_tokens, target_tokens, loss_weights, stream_ids, chunk_positions
        truncate_disk_token_cache(cache_path, actual_count, unroll_len)
        with open(str(cache_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if progress_path.exists():
            progress_path.unlink()
        cached = open_tokenized_cache(cache_path, actual_count, unroll_len)
        log_info(f"Tokenized {actual_count} chunks from {source_rows} source rows for {split_name}")
        log_info(json.dumps(aggregate_stats, ensure_ascii=False, indent=2))
        return cached
    else:
        log_info(f"[Cache] RAM cache for {split_name} is process-local and will not be reused after exit")

    log_info(f"Tokenized {actual_count} chunks from {source_rows} source rows for {split_name}")
    log_info(json.dumps(aggregate_stats, ensure_ascii=False, indent=2))

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
        log_info(f"Loading cached {split_name} chunks: chunks={num_rows}, path={cp}")

        return open_tokenized_cache(cp, num_rows, unroll_len)

    specs = parse_dataset_mix()
    if len(specs) > 1 or (config.dataset_mix or "").strip():
        return tokenize_mixed_dataset_rows(specs, split_name, cp, max_chunks)

    spec = specs[0]
    ds = load_dataset_from_spec(spec, ds_split)
    ds = apply_same_split_partition(ds, spec, split_name, ds_split)
    ds = apply_data_pack_partition(ds, spec, split_name)

    if skip_rows > 0:
        if not config.streaming and hasattr(ds, "select"):
            end = min(skip_rows + max_chunks, len(ds))
            ds = ds.select(range(skip_rows, end))
        else:
            log_info(f"Skipping {skip_rows} source rows before tokenizing {split_name}")
            ds = ds.skip(skip_rows)

    return tokenize_dataset_rows(ds, split_name, cp, max_chunks, str(spec.get("mode", config.dataset_mode)), spec)


def open_tokenized_cache(
    cache_path: Path,
    num_rows: int,
    unroll_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.memmap(str(cache_path) + ".input.bin", dtype=np.int32, mode="r", shape=(num_rows, unroll_len, 8))
    targets = np.memmap(str(cache_path) + ".target.bin", dtype=np.int32, mode="r", shape=(num_rows, unroll_len, 8))
    weights = np.memmap(str(cache_path) + ".weight.bin", dtype=np.float32, mode="r", shape=(num_rows, unroll_len))
    stream_ids = np.memmap(str(cache_path) + ".stream_id.bin", dtype=np.int64, mode="r", shape=(num_rows,))
    chunk_positions = np.memmap(str(cache_path) + ".chunk_pos.bin", dtype=np.int32, mode="r", shape=(num_rows,))
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = (
        inputs,
        targets,
        weights,
        stream_ids,
        chunk_positions,
    )

    mode = (config.cache_read_mode or "auto").lower()
    if mode not in {"auto", "mmap", "memory"}:
        raise ValueError(f"cache_read_mode must be auto, mmap, or memory; got {config.cache_read_mode}")
    total_bytes = int(num_rows) * token_cache_bytes_per_chunk(unroll_len)
    fs_type = filesystem_type(cache_path.parent)
    mem_free = available_memory_bytes()
    mem_budget = int(mem_free * max(0.0, min(0.95, float(config.cache_read_memory_fraction))))
    should_preload = mode == "memory" or (mode == "auto" and fs_type != "tmpfs" and total_bytes <= mem_budget)

    if should_preload:
        log_info(
            f"[Cache] read=memory path={cache_path.name} rows={num_rows} size={format_bytes(total_bytes)} "
            f"mem_available={format_bytes(mem_free)}"
        )
        return tuple(np.asarray(array).copy() for array in arrays)  # type: ignore[return-value]

    backing = "tmpfs/RAM" if fs_type == "tmpfs" else fs_type
    log_info(f"[Cache] read=mmap path={cache_path.name} rows={num_rows} size={format_bytes(total_bytes)} fs={backing}")
    return arrays


def find_legacy_mixed_source_cache(
    split_name: str,
    source_idx: int,
) -> Path | None:
    pattern = f"propagator_{split_name}_*.source_{source_idx}"
    candidates: list[tuple[float, Path]] = []
    for path in cache_root_path().glob(pattern):
        meta = Path(str(path) + ".meta.json")
        shard_dir = path.parent / f"{path.name}.echox_shards"
        if meta.exists() or shard_dir.exists():
            try:
                updated = max(
                    path.stat().st_mtime if path.exists() else 0.0,
                    meta.stat().st_mtime if meta.exists() else 0.0,
                    shard_dir.stat().st_mtime if shard_dir.exists() else 0.0,
                )
            except OSError:
                updated = 0.0
            candidates.append((updated, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def infer_cache_chunk_capacity() -> int:
    free_bytes = available_memory_bytes() if (config.cache_storage or "auto") == "memory" else free_disk_bytes(cache_root_path())
    reserve_bytes = 32 * 1024**3 if (config.cache_storage or "auto") == "memory" else 8 * 1024**3
    return max(1, int((free_bytes - reserve_bytes) // token_cache_bytes_per_chunk(config.train_unroll_len)))


def component_chunk_targets(specs: list[dict[str, Any]], max_chunks: int) -> list[int]:
    if max_chunks <= 0:
        targets = []
        for spec in specs:
            spec_max_chunks = spec.get("max_chunks")
            if spec_max_chunks is None:
                targets.append(-1)
                continue
            targets.append(max(1, int(spec_max_chunks)))
        return targets

    raw = [max_chunks * float(spec["weight"]) for spec in specs]
    targets = [int(math.floor(value)) for value in raw]
    remainder = max_chunks - sum(targets)
    order = np.argsort([-(value - math.floor(value)) for value in raw])
    for idx in order[:remainder]:
        targets[int(idx)] += 1
    for idx, spec in enumerate(specs):
        spec_max_chunks = spec.get("max_chunks")
        if spec_max_chunks is not None:
            targets[idx] = min(int(targets[idx]), max(1, int(spec_max_chunks)))
    return targets


def tokenize_mixed_dataset_rows(
    specs: list[dict[str, Any]],
    split_name: str,
    cache_path: Path,
    max_chunks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_targets = component_chunk_targets(specs, max_chunks)
    components = []
    component_meta = []

    for source_idx, (spec, target_chunks) in enumerate(zip(specs, source_targets, strict=True)):
        if target_chunks == 0:
            continue
        split = split_for_dataset_spec(spec, split_name)
        source_cp = source_cache_prefix(spec, split_name, source_idx)
        legacy_source_cp = Path(str(cache_path) + f".source_{source_idx}")
        legacy_source_meta = Path(str(legacy_source_cp) + ".meta.json")
        legacy_shard_dir = cache_path.parent / f"{legacy_source_cp.name}.echox_shards"
        if not Path(str(source_cp) + ".meta.json").exists() and (
            legacy_source_meta.exists() or legacy_shard_dir.exists()
        ):
            source_cp = legacy_source_cp
            log_info(
                f"[Cache] Resuming legacy mixed source cache: source={source_idx}:{spec['name']} "
                f"path={source_cp.name}"
            )
        source_meta = Path(str(source_cp) + ".meta.json")

        if source_meta.exists():
            with open(source_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            num_rows = int(meta.get("num_chunks", meta["num_rows"]))
            data = open_tokenized_cache(source_cp, num_rows, config.train_unroll_len)
            if target_chunks > 0 and num_rows > target_chunks:
                log_info(
                    f"[Cache] Reusing source cache above target without truncation: source={source_idx}:{spec['name']} "
                    f"cached_chunks={num_rows}, target_chunks={target_chunks}, path={source_cp.name}"
                )
        else:
            try:
                ds = load_dataset_from_spec(spec, split)
            except Exception as exc:
                log_info(f"Skipping dataset source {source_idx} {spec['name']} split={split}: {exc}")
                continue

            ds = apply_same_split_partition(ds, spec, split_name, split)
            ds = apply_data_pack_partition(ds, spec, split_name)

            if split_name == "val" and split == spec.get("split") and target_chunks > 0 and config.validation_skip_rows is not None:
                if config.validation_skip_rows is not None:
                    skip = int(config.validation_skip_rows)
                else:
                    skip = max(1, int(config.max_train_chunks * float(spec["weight"])))
                if skip > 0 and hasattr(ds, "skip"):
                    log_info(f"Skipping {skip} source rows for validation source {spec['name']}")
                    ds = ds.skip(skip)

            data = tokenize_dataset_rows(
                ds,
                f"{split_name}:{source_idx}:{spec['name']}",
                source_cp,
                target_chunks,
                str(spec.get("mode", config.dataset_mode)),
                spec,
                stream_offset=(source_idx + 1) * 1_000_000_000,
            )

        rows = len(data[0]) if target_chunks < 0 else min(len(data[0]), int(target_chunks))
        if rows == 0:
            log_info(f"Dataset source produced zero chunks: {spec['name']} mode={spec.get('mode')}")
            continue
        if target_chunks > 0 and rows < len(data[0]):
            log_info(
                f"[Cache] Truncating source to configured mix target: source={source_idx}:{spec['name']} "
                f"cached_chunks={len(data[0])}, selected_chunks={rows}"
            )
        components.append((spec, data, rows))
        component_meta.append({"name": spec["name"], "mode": spec.get("mode"), "chunks": rows, "target_chunks": target_chunks})

    total_rows = sum(rows for _, _, rows in components)
    if total_rows == 0:
        raise RuntimeError(f"No chunks were tokenized for mixed {split_name}")

    unroll_len = config.train_unroll_len
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    final_storage = choose_token_cache_storage(cache_path, total_rows, unroll_len, resume=False)
    inputs, targets, weights, stream_ids, chunk_positions = allocate_token_cache_arrays(
        cache_path,
        total_rows,
        unroll_len,
        final_storage,
        "w+",
    )

    cursor = 0
    for _, data, rows in components:
        src_inputs, src_targets, src_weights, src_stream_ids, src_chunk_positions = data
        inputs[cursor : cursor + rows] = src_inputs[:rows]
        targets[cursor : cursor + rows] = src_targets[:rows]
        weights[cursor : cursor + rows] = src_weights[:rows]
        stream_ids[cursor : cursor + rows] = src_stream_ids[:rows]
        chunk_positions[cursor : cursor + rows] = src_chunk_positions[:rows]
        cursor += rows

    flush_token_cache_arrays(inputs, targets, weights, stream_ids, chunk_positions)

    meta = {
        "num_rows": total_rows,
        "num_chunks": total_rows,
        "train_unroll_len": unroll_len,
        "protocol": "multimodal_user_interrupt_stateful",
        "components": component_meta,
        "special_tokens": SPECIAL_TOKENS,
        "text_vocab_size": text_vocab_size,
        "audio_token_start": audio_token_start,
        "audio_token_end": audio_token_end,
        "image_token_start": image_token_start,
        "image_token_end": image_token_end,
    }
    if final_storage == "disk":
        with open(str(cache_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    else:
        log_info(f"[Cache] RAM cache for mixed {split_name} is process-local and will not be reused after exit")

    log_info(f"Built mixed {split_name} cache: chunks={total_rows}")
    log_info(json.dumps(component_meta, ensure_ascii=False, indent=2))
    if final_storage == "disk":
        return open_tokenized_cache(cache_path, total_rows, unroll_len)
    return (
        inputs[:total_rows],
        targets[:total_rows],
        weights[:total_rows],
        stream_ids[:total_rows],
        chunk_positions[:total_rows],
    )


def load_tokenizer_and_datasets() -> tuple[np.ndarray, ...]:
    global tokenizer, vocab_size, text_vocab_size, token_ids, tokenizer_fingerprint
    global audio_token_start, audio_token_end, image_token_start, image_token_end

    tokenizer = load_or_train_tokenizer()
    token_ids = ensure_special_tokens(tokenizer)
    text_vocab_size = tokenizer.get_vocab_size()
    vocab_size, audio_token_start, audio_token_end, image_token_start, image_token_end = compute_vocab_sizes(text_vocab_size)
    tokenizer_fingerprint = tokenizer_file_fingerprint(Path(config.tokenizer_path))
    log_info(f"Tokenizer fingerprint: {tokenizer_fingerprint}")
    save_tokenizer_snapshot()

    train_split = config.dataset_split
    val_split = config.validation_split

    if config.validation_skip_rows is not None:
        val_skip = config.validation_skip_rows
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


def build_audio_candidate_token_ids() -> np.ndarray:
    ids = {token_ids["audio_end"], token_ids["model_end"], token_ids["session_end"]}
    if config.enable_audio:
        ids.update(range(audio_token_start, audio_token_end))
    return np.asarray(sorted(i for i in ids if 0 <= i < vocab_size), dtype=np.int32)


def build_audio_codebook_candidate_token_ids(codebook_idx: int, allow_stop: bool = True) -> np.ndarray:
    ids = {token_ids["audio_end"], token_ids["model_end"], token_ids["session_end"]} if allow_stop else set()
    if config.enable_audio:
        idx = int(codebook_idx) % max(1, int(config.audio_codebooks))
        start = audio_token_start + idx * int(config.audio_codebook_size)
        end = min(audio_token_end, start + int(config.audio_codebook_size))
        ids.update(range(start, end))
    return np.asarray(sorted(i for i in ids if 0 <= i < vocab_size), dtype=np.int32)


def rms_norm(x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    ms = jnp.mean(x_f32**2, axis=-1, keepdims=True)
    return (x_f32 * jax.lax.rsqrt(ms + 1e-6)).astype(x.dtype)


def apply_grouped_rope(keys: jax.Array, positions: jax.Array, base: float, position_scale: float) -> jax.Array:
    group_key_size = int(keys.shape[-1])
    rotary_dim = (group_key_size // 2) * 2
    if rotary_dim < 2:
        return keys

    x_rot = keys[..., :rotary_dim].astype(jnp.float32)
    x_tail = keys[..., rotary_dim:]
    half = rotary_dim // 2
    x_even = x_rot[..., :half]
    x_odd = x_rot[..., half:]

    inv_freq = jnp.power(
        jnp.asarray(base, dtype=jnp.float32),
        -jnp.arange(0, half, dtype=jnp.float32) / max(1, half),
    )
    scaled_pos = positions.astype(jnp.float32) / max(1.0, float(position_scale))
    angles = scaled_pos[:, None, None] * inv_freq[None, None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    rotated = jnp.concatenate([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], axis=-1)
    if x_tail.shape[-1] == 0:
        return rotated.astype(keys.dtype)
    return jnp.concatenate([rotated, x_tail], axis=-1).astype(keys.dtype)


def compute_vocab_sizes(text_size: int) -> tuple[int, int, int, int, int]:
    audio_start = int(text_size)
    audio_end = audio_start
    if config.enable_audio:
        audio_end = audio_start + int(config.audio_codebooks) * int(config.audio_codebook_size)
    image_start = audio_end
    image_end = image_start + int(config.image_patch_vocab_size)
    return image_end, audio_start, audio_end, image_start, image_end


def model_dtype() -> Any:
    if config.precision == "float16":
        return jnp.float16
    if config.precision == "bfloat16":
        return jnp.bfloat16
    return jnp.float32


def is_audio_token_id(token_id: int) -> bool:
    return bool(config.enable_audio and audio_token_start <= int(token_id) < audio_token_end)


def is_image_token_id(token_id: int) -> bool:
    if "image_token_start" not in globals() or "image_token_end" not in globals():
        return False
    return bool(image_token_start <= int(token_id) < image_token_end)


def audio_token_id(codebook: int, code: int) -> int:
    return int(audio_token_start + int(codebook) * config.audio_codebook_size + int(code))


def image_token_id(code: int) -> int:
    return int(image_token_start + (int(code) % int(config.image_patch_vocab_size)))


def audio_code_from_token_id(token_id: int) -> tuple[int, int] | None:
    if not is_audio_token_id(token_id):
        return None
    rel = int(token_id) - int(audio_token_start)
    return rel // config.audio_codebook_size, rel % config.audio_codebook_size


class PropagatorBlock(nnx.Module):
    def __init__(self, cfg: PropagatorConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        std = 1.0 / jnp.sqrt(cfg.hidden_size)
        self.groups = max(1, int(cfg.associative_groups))
        if int(cfg.memory_key_size) % self.groups != 0:
            raise ValueError("memory_key_size must be divisible by associative_groups")

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

        mlp_hidden_size = cfg.mlp_multiplier * cfg.hidden_size
        self.moe_num_experts = max(1, int(cfg.moe_num_experts))
        self.moe_top_k = max(1, min(int(cfg.moe_top_k), self.moe_num_experts))
        if self.moe_num_experts > 1:
            self.router = nnx.Linear(cfg.hidden_size, self.moe_num_experts, rngs=rngs)
            self.expert_fc1 = [nnx.Linear(cfg.hidden_size, mlp_hidden_size, rngs=rngs) for _ in range(self.moe_num_experts)]
            self.expert_fc2 = [nnx.Linear(mlp_hidden_size, cfg.hidden_size, rngs=rngs) for _ in range(self.moe_num_experts)]
            if cfg.use_swiglu:
                self.expert_gate = [nnx.Linear(cfg.hidden_size, mlp_hidden_size, rngs=rngs) for _ in range(self.moe_num_experts)]
            else:
                self.expert_gate = []
        else:
            self.fc1 = nnx.Linear(cfg.hidden_size, mlp_hidden_size, rngs=rngs)
            self.fc2 = nnx.Linear(mlp_hidden_size, cfg.hidden_size, rngs=rngs)
            if cfg.use_swiglu:
                self.fc_gate = nnx.Linear(cfg.hidden_size, mlp_hidden_size, rngs=rngs)

        self.gamma1 = nnx.Param(jnp.ones((cfg.hidden_size,)) * 0.1)
        self.gamma2 = nnx.Param(jnp.ones((cfg.hidden_size,)) * 0.1)

    def grouped_key(self, key: jax.Array, positions: jax.Array, scale: jax.Array) -> jax.Array:
        batch_size = key.shape[0]
        group_key_size = int(self.cfg.memory_key_size) // self.groups
        key = key.reshape((batch_size, self.groups, group_key_size))
        key = apply_grouped_rope(key, positions, self.cfg.rope_base, self.cfg.rope_position_scale)
        return (rms_norm(key) * scale).astype(jnp.float32)

    def mlp(self, x: jax.Array) -> jax.Array:
        dtype = model_dtype()
        if self.moe_num_experts <= 1:
            up = self.fc1(x)
            if self.cfg.use_swiglu:
                hidden = jax.nn.silu(up) * self.fc_gate(x)
            else:
                hidden = jax.nn.silu(up)
            return self.fc2(hidden.astype(dtype))

        expert_outputs = []
        for idx in range(self.moe_num_experts):
            up = self.expert_fc1[idx](x)
            if self.cfg.use_swiglu:
                hidden = jax.nn.silu(up) * self.expert_gate[idx](x)
            else:
                hidden = jax.nn.silu(up)
            expert_outputs.append(self.expert_fc2[idx](hidden.astype(dtype)))
        stacked = jnp.stack(expert_outputs, axis=1).astype(dtype)

        router_logits = self.router(x).astype(jnp.float32)
        if self.moe_top_k < self.moe_num_experts:
            top_values, top_indices = jax.lax.top_k(router_logits, self.moe_top_k)
            top_weights = jax.nn.softmax(top_values, axis=-1)
            dispatch = jnp.zeros_like(router_logits)
            dispatch = dispatch.at[jnp.arange(router_logits.shape[0])[:, None], top_indices].set(top_weights)
        else:
            dispatch = jax.nn.softmax(router_logits, axis=-1)
        return jnp.einsum("be,beh->bh", dispatch.astype(dtype), stacked)

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        valid: jax.Array,
        positions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        dtype = model_dtype()
        x = x.astype(dtype)
        memory_f32 = memory.astype(jnp.float32)

        group_key_size = int(self.cfg.memory_key_size) // self.groups
        scale = jax.lax.rsqrt(jnp.asarray(group_key_size, dtype=jnp.float32))
        memory_grouped = memory_f32.reshape((memory_f32.shape[0], self.groups, group_key_size, self.cfg.memory_value_size))

        h = self.norm1(x).astype(dtype)
        read_key = self.grouped_key(self.read_key_proj(h), positions, scale)
        if read_key.ndim == 3:
            read_key = read_key

        read_value = jnp.mean(jnp.einsum("bgkv,bgk->bgv", memory_grouped, read_key), axis=1).astype(dtype)
        x = x + (self.gamma1[...] * self.read_proj(read_value)).astype(dtype)

        mlp_in = self.norm2(x).astype(dtype)
        x = x + (self.gamma2[...] * self.mlp(mlp_in)).astype(dtype)

        w = self.norm3(x).astype(dtype)

        write_key = self.grouped_key(self.write_key_proj(w), positions, scale)
        if write_key.ndim == 3:
            write_key = write_key

        write_value = jnp.tanh(self.write_value_proj(w)).astype(jnp.float32)
        if write_value.ndim == 3:
            write_value = jnp.squeeze(write_value, 1)

        value_hat = jnp.einsum("bgkv,bgk->bgv", memory_grouped, write_key)
        err = jnp.clip(write_value[:, None, :] - value_hat, -1.0, 1.0)

        eta = jax.nn.sigmoid(self.write_gate(w)).astype(jnp.float32) * self.cfg.write_rate
        if eta.ndim == 3:
            eta = jnp.squeeze(eta, 1)

        forget = jax.nn.sigmoid(self.forget_gate(w)).astype(jnp.float32) * self.cfg.forget_rate
        if forget.ndim == 3:
            forget = jnp.squeeze(forget, 1)

        update = jnp.einsum("bgk,bgv->bgkv", write_key, err).reshape(memory_f32.shape)
        new_memory = (1.0 - forget[:, :, None]) * memory_f32 + eta[:, :, None] * update
        new_memory = jnp.clip(new_memory, -10.0, 10.0)

        valid_f = valid.astype(jnp.float32)[:, None, None]
        final_memory = valid_f * new_memory + (1.0 - valid_f) * memory_f32

        return x.astype(jnp.float32), final_memory


class PropagatorModel(nnx.Module):
    def __init__(self, cfg: PropagatorConfig, vocab_size_: int, rngs: nnx.Rngs):
        self.cfg = cfg
        self.token_emb = nnx.Embed(vocab_size_, cfg.hidden_size, rngs=rngs)
        self.audio_aux_heads = nnx.Linear(cfg.hidden_size, 7 * cfg.audio_codebook_size, rngs=rngs)
        self.blocks = [PropagatorBlock(cfg, rngs) for _ in range(cfg.num_layers)]
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
        reset = reset_mask.astype(jnp.float32)[:, None, None]
        return tuple((1.0 - reset) * m for m in memories)

    def step_hidden(
        self,
        token_ids_: jax.Array,
        memories: tuple[jax.Array, ...],
        valid: jax.Array,
        positions: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        if positions is None:
            positions = jnp.zeros((token_ids_.shape[0],), dtype=jnp.int32)
        positions = jnp.minimum(positions.astype(jnp.int32), int(self.cfg.rope_max_position))
        embeddings = self.token_emb(token_ids_)
        if embeddings.ndim == 3:
            mask = (token_ids_ != token_ids_pad)[..., None]
            active = jnp.sum(mask.astype(jnp.float32), axis=1)
            x = jnp.sum(embeddings * mask, axis=1) * jax.lax.rsqrt(jnp.maximum(active, 1.0))
        else:
            x = embeddings
        next_memories = []
        for block, memory in zip(self.blocks, memories, strict=True):
            x, next_memory = block(x, memory, valid, positions)
            next_memories.append(next_memory)
        x = self.norm(x)
        return x.astype(jnp.float32), tuple(next_memories)

    def project_full(self, hidden: jax.Array) -> jax.Array:
        return (hidden @ self.token_emb.embedding[...].T).astype(jnp.float32)

    def project_candidates(self, hidden: jax.Array, candidate_ids: jax.Array) -> jax.Array:
        candidate_embeddings = self.token_emb.embedding[candidate_ids]
        return (hidden @ candidate_embeddings.T).astype(jnp.float32)

    def project_audio_aux_head(
        self,
        depth_state: jax.Array,
        previous_token_ids: jax.Array,
        aux_index: jax.Array | int,
    ) -> tuple[jax.Array, jax.Array]:
        previous_embeddings = self.token_emb(previous_token_ids).astype(jnp.float32)
        next_depth_state = rms_norm(depth_state.astype(jnp.float32) + previous_embeddings)
        codebook_size = int(self.cfg.audio_codebook_size)
        kernel = self.audio_aux_heads.kernel[...].reshape((self.cfg.hidden_size, 7, codebook_size))
        bias = self.audio_aux_heads.bias[...].reshape((7, codebook_size))
        head_kernel = jax.lax.dynamic_index_in_dim(kernel, aux_index, axis=1, keepdims=False)
        head_bias = jax.lax.dynamic_index_in_dim(bias, aux_index, axis=0, keepdims=False)
        logits = next_depth_state @ head_kernel + head_bias
        return logits.astype(jnp.float32), next_depth_state

    def project_audio_aux_teacher(self, hidden: jax.Array, target_frame: jax.Array) -> jax.Array:
        depth_state = hidden
        previous_token_ids = target_frame[:, 0]
        logits_by_codebook = []
        for aux_index in range(7):
            logits, depth_state = self.project_audio_aux_head(depth_state, previous_token_ids, aux_index)
            logits_by_codebook.append(logits)
            previous_token_ids = target_frame[:, aux_index + 1]
        return jnp.stack(logits_by_codebook, axis=1)

    def step(
        self,
        token_ids_: jax.Array,
        memories: tuple[jax.Array, ...],
        valid: jax.Array,
        positions: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        hidden, next_memories = self.step_hidden(token_ids_, memories, valid, positions)
        return self.project_full(hidden), next_memories

    def step_candidates(
        self,
        token_ids_: jax.Array,
        memories: tuple[jax.Array, ...],
        valid: jax.Array,
        candidate_ids: jax.Array,
        positions: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        hidden, next_memories = self.step_hidden(token_ids_, memories, valid, positions)
        return self.project_candidates(hidden, candidate_ids), next_memories

    def forward_with_memories(
        self,
        inputs: jax.Array,
        targets: jax.Array,
        loss_weights: jax.Array,
        init_memories: tuple[jax.Array, ...],
        reset_mask: jax.Array,
        chunk_positions: jax.Array | None = None,
        task_ids: jax.Array | None = None,
        compute_metrics: bool = True,
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        input_mask = inputs[..., 0] != token_ids_pad if inputs.ndim == 3 else inputs != token_ids_pad
        memories = self.reset_memories(init_memories, reset_mask)
        if chunk_positions is None:
            chunk_positions = jnp.zeros((inputs.shape[0],), dtype=jnp.int32)
        step_positions = (
            chunk_positions.astype(jnp.int32)[:, None] * int(self.cfg.train_unroll_len)
            + jnp.arange(inputs.shape[1], dtype=jnp.int32)[None, :]
        )

        smooth = jnp.asarray(self.cfg.label_smoothing, dtype=jnp.float32)

        def scan_step_plain(carry, step_inputs):
            step_in, step_target, step_weight, step_valid, step_position = step_inputs

            hidden, next_carry = self.step_hidden(step_in, carry, step_valid, step_position)
            step_logits = self.project_full(hidden)

            main_target = step_target[:, 0] if step_target.ndim == 2 else step_target

            text_logits = step_logits[:, :text_vocab_size]
            text_log_probs = jax.nn.log_softmax(text_logits, axis=-1)
            text_targets = jnp.clip(main_target, 0, text_vocab_size - 1)
            text_nll = -jnp.take_along_axis(text_log_probs, text_targets[..., None], axis=-1).squeeze(-1)
            q0_start = int(audio_token_start)
            q0_end = q0_start + int(self.cfg.audio_codebook_size)
            is_audio_main_target = jnp.logical_and(main_target >= q0_start, main_target < q0_end)
            q0_logits = step_logits[:, q0_start:q0_end]
            q0_log_probs = jax.nn.log_softmax(q0_logits, axis=-1)
            q0_targets_rel = jnp.clip(main_target - q0_start, 0, int(self.cfg.audio_codebook_size) - 1)
            q0_nll = -jnp.take_along_axis(q0_log_probs, q0_targets_rel[..., None], axis=-1).squeeze(-1)
            nll = jnp.where(is_audio_main_target, q0_nll, text_nll)
            if float(self.cfg.label_smoothing) > 0.0:
                text_smooth_loss = -jnp.mean(text_log_probs, axis=-1)
                q0_smooth_loss = -jnp.mean(q0_log_probs, axis=-1)
                smooth_loss = jnp.where(is_audio_main_target, q0_smooth_loss, text_smooth_loss)
                mixed_nll = (1.0 - smooth) * nll + smooth * smooth_loss
            else:
                mixed_nll = nll

            weighted_nll = mixed_nll * step_weight

            total_aux_nll = jnp.zeros_like(weighted_nll)
            audio_codebook_correct = jnp.zeros_like(step_weight, dtype=jnp.bool_)
            aux_token_correct = jnp.zeros_like(step_weight, dtype=jnp.float32)
            aux_token_total = jnp.zeros_like(step_weight, dtype=jnp.float32)
            aux_frame_mask = jnp.zeros_like(step_weight, dtype=jnp.bool_)

            if step_target.ndim == 2:
                aux_logits = self.project_audio_aux_teacher(hidden, step_target)

                aux_targets = step_target[:, 1:8] # (batch, 7)
                is_audio_aux = jnp.logical_and(aux_targets >= audio_token_start, aux_targets < audio_token_end)

                base_offsets = audio_token_start + jnp.arange(1, 8, dtype=jnp.int32)[None, :] * int(self.cfg.audio_codebook_size)
                aux_targets_rel = aux_targets - base_offsets
                aux_targets_rel = jnp.clip(aux_targets_rel, 0, int(self.cfg.audio_codebook_size) - 1)

                aux_log_probs = jax.nn.log_softmax(aux_logits, axis=-1)
                aux_nll = -jnp.take_along_axis(aux_log_probs, aux_targets_rel[..., None], axis=-1).squeeze(-1)

                aux_nll_masked = aux_nll * is_audio_aux.astype(jnp.float32)
                aux_counts = jnp.sum(is_audio_aux.astype(jnp.float32), axis=1)
                total_aux_nll = jnp.sum(aux_nll_masked, axis=1) / jnp.maximum(1.0, aux_counts)

                if compute_metrics:
                    aux_preds = jnp.argmax(aux_logits, axis=-1).astype(jnp.int32)
                    aux_correct = jnp.logical_and(is_audio_aux, aux_preds == aux_targets_rel)
                    aux_token_correct = jnp.sum(aux_correct.astype(jnp.float32), axis=1)
                    aux_token_total = aux_counts
                    aux_frame_mask = aux_counts > 0.0
                    audio_codebook_correct = jnp.logical_and(aux_frame_mask, aux_token_correct == aux_counts)

            combined_loss = weighted_nll + total_aux_nll * float(self.cfg.audio_codebook_loss_weight) * step_weight
            combined_weight = step_weight

            if compute_metrics:
                text_pred = jnp.argmax(text_logits, axis=-1).astype(jnp.int32)
                q0_pred = jnp.argmax(q0_logits, axis=-1).astype(jnp.int32) + q0_start
                pred = jnp.where(is_audio_main_target, q0_pred, text_pred)
                supervised = step_weight > 0.0
                correct = jnp.logical_and(supervised, pred == main_target)

                decision_target = jnp.logical_or(main_target == token_ids_listen, main_target == token_ids_user_end)
                decision_target = jnp.logical_or(decision_target, main_target == token_ids_user_interrupt)
                decision_mask = jnp.logical_and(supervised, decision_target)

                listen_mask = jnp.logical_and(supervised, main_target == token_ids_listen)
                user_end_mask = jnp.logical_and(supervised, main_target == token_ids_user_end)
                interrupt_mask = jnp.logical_and(supervised, main_target == token_ids_user_interrupt)
                model_end_mask = jnp.logical_and(supervised, main_target == token_ids_model_end)

                audio_mask = jnp.logical_and(supervised, main_target >= audio_token_start)
                audio_mask = jnp.logical_and(audio_mask, main_target < audio_token_end)

                special_mask = jnp.logical_or(main_target == token_ids_listen, main_target == token_ids_user_end)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_user_interrupt)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_model_end)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_session_end)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_audio_end)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_text_out)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_audio_out)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_hybrid_out)
                special_mask = jnp.logical_or(special_mask, main_target == token_ids_pad)

                text_mask = jnp.logical_and(supervised, jnp.logical_not(audio_mask))
                text_mask = jnp.logical_and(text_mask, jnp.logical_not(special_mask))
                audio_codebook_correct = jnp.logical_and(audio_mask, audio_codebook_correct)
                audio_aux_frame_mask = jnp.logical_and(audio_mask, aux_frame_mask)
                audio_all_codebook_correct = jnp.logical_and(correct, audio_codebook_correct)
                audio_aux_token_correct = jnp.where(audio_mask, aux_token_correct, 0.0)
                audio_aux_token_total = jnp.where(audio_mask, aux_token_total, 0.0)
                aux_loss_mask = jnp.logical_and(supervised, audio_aux_frame_mask)
                weighted_aux_nll = total_aux_nll * step_weight

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
                    jnp.sum(jnp.logical_and(correct, text_mask).astype(jnp.float32)),
                    jnp.sum(text_mask.astype(jnp.float32)),
                    jnp.sum(jnp.logical_and(correct, audio_mask).astype(jnp.float32)),
                    jnp.sum(audio_mask.astype(jnp.float32)),
                    jnp.sum(audio_codebook_correct.astype(jnp.float32)),
                    jnp.sum(audio_aux_frame_mask.astype(jnp.float32)),
                    jnp.sum(audio_aux_token_correct.astype(jnp.float32)),
                    jnp.sum(audio_aux_token_total.astype(jnp.float32)),
                    jnp.sum(audio_all_codebook_correct.astype(jnp.float32)),
                    jnp.sum(audio_aux_frame_mask.astype(jnp.float32)),
                    jnp.sum(weighted_nll.astype(jnp.float32)),
                    jnp.sum(step_weight.astype(jnp.float32)),
                    jnp.sum(weighted_aux_nll.astype(jnp.float32)),
                    jnp.sum(jnp.where(aux_loss_mask, step_weight, 0.0).astype(jnp.float32)),
                )

                # Task separation (0: Text, 1: ASR, 2: TTS, 3: Duplex, 4: Image)
                task_correct = jnp.zeros(5, dtype=jnp.float32)
                task_total = jnp.zeros(5, dtype=jnp.float32)
                task_loss_sum = jnp.zeros(5, dtype=jnp.float32)
                task_weight_sum = jnp.zeros(5, dtype=jnp.float32)
                if task_ids is not None:
                    token_correct = jnp.logical_and(correct, jnp.logical_or(text_mask, audio_mask)).astype(jnp.float32)
                    token_total = jnp.logical_or(text_mask, audio_mask).astype(jnp.float32)
                    task_correct = jax.vmap(lambda i: jnp.sum(jnp.where(task_ids == i, token_correct, 0.0)))(jnp.arange(5))
                    task_total = jax.vmap(lambda i: jnp.sum(jnp.where(task_ids == i, token_total, 0.0)))(jnp.arange(5))
                    task_loss_sum = jax.vmap(lambda i: jnp.sum(jnp.where(task_ids == i, combined_loss, 0.0)))(jnp.arange(5))
                    task_weight_sum = jax.vmap(lambda i: jnp.sum(jnp.where(task_ids == i, combined_weight, 0.0)))(jnp.arange(5))

                metrics = metrics + tuple(task_correct) + tuple(task_total) + tuple(task_loss_sum) + tuple(task_weight_sum)
            else:
                metrics = tuple(jnp.zeros((), dtype=jnp.float32) for _ in range(VALIDATION_METRIC_SIZE))

            return next_carry, (combined_loss, combined_weight, metrics)

        scan_step = jax.checkpoint(scan_step_plain) if self.cfg.remat_scan_step else scan_step_plain

        final_memories, (step_losses, step_weights, metrics_t) = jax.lax.scan(
            scan_step,
            memories,
            (
                jnp.swapaxes(inputs, 0, 1),
                jnp.swapaxes(targets, 0, 1),
                loss_weights.T,
                input_mask.T,
                step_positions.T,
            ),
        )

        ce_loss = jnp.sum(step_losses) / jnp.maximum(1.0, jnp.sum(step_weights))
        reg_loss = self.cfg.memory_l2 * jnp.mean(jnp.asarray([jnp.mean(m**2) for m in final_memories]))

        metrics = tuple(jnp.sum(x, axis=0) for x in metrics_t)

        return ce_loss + reg_loss, ce_loss, tuple(jax.lax.stop_gradient(m) for m in final_memories), metrics

    def __call__(self, inputs: jax.Array, targets: jax.Array, loss_weights: jax.Array) -> tuple[jax.Array, jax.Array]:
        batch_size = inputs.shape[0]
        init_memories = self.initial_memories(batch_size)
        reset_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
        total_loss, ce_loss, _, _ = self.forward_with_memories(
            inputs,
            targets,
            loss_weights,
            init_memories,
            reset_mask,
            compute_metrics=False,
        )
        return total_loss, ce_loss


@functools.partial(nnx.jit, donate_argnums=(2, 3, 4))
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
    optimizer.update(grads)
    return ce_loss


def _train_step_stateful_impl(
    model: PropagatorModel,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    memories: tuple[jax.Array, ...],
    reset_mask: jax.Array,
    chunk_positions: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    def compute_loss(m):
        total_loss, ce_loss, final_memories, _ = m.forward_with_memories(
            inputs,
            targets,
            weights,
            memories,
            reset_mask,
            chunk_positions=chunk_positions,
            compute_metrics=False,
        )
        return total_loss, (ce_loss, final_memories)

    grads, (ce_loss, final_memories) = nnx.grad(compute_loss, has_aux=True)(model)
    optimizer.update(grads)
    return ce_loss, final_memories


def build_train_step_stateful() -> Any:
    if batch_sharding is None or vector_sharding is None or memory_sharding is None:
        if data_mesh is None:
            return nnx.jit(_train_step_stateful_impl, donate_argnums=(2, 3, 4, 5, 6))
        with jax.sharding.set_mesh(data_mesh):
            return nnx.jit(_train_step_stateful_impl, donate_argnums=(2, 3, 4, 5, 6))
    memory_shardings = tuple(memory_sharding for _ in range(config.num_layers))
    if data_mesh is None:
        return nnx.jit(
            _train_step_stateful_impl,
            in_shardings=(
                None,
                None,
                batch_sharding,
                batch_sharding,
                batch_sharding,
                memory_shardings,
                vector_sharding,
                vector_sharding,
            ),
            out_shardings=(None, memory_shardings),
            donate_argnums=(2, 3, 4, 5, 6),
        )
    with jax.sharding.set_mesh(data_mesh):
        return nnx.jit(
            _train_step_stateful_impl,
            in_shardings=(
                None,
                None,
                batch_sharding,
                batch_sharding,
                batch_sharding,
                memory_shardings,
                vector_sharding,
                vector_sharding,
            ),
            out_shardings=(None, memory_shardings),
            donate_argnums=(2, 3, 4, 5, 6),
        )


def call_train_step_stateful(
    train_step_stateful_fn: Any,
    model: PropagatorModel,
    optimizer: nnx.Optimizer,
    batch_inputs: jax.Array,
    batch_targets: jax.Array,
    batch_weights: jax.Array,
    carry_memories: tuple[jax.Array, ...],
    reset_mask: jax.Array,
    chunk_positions: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    if data_mesh is None:
        return train_step_stateful_fn(
            model,
            optimizer,
            batch_inputs,
            batch_targets,
            batch_weights,
            carry_memories,
            reset_mask,
            chunk_positions,
        )
    with jax.sharding.set_mesh(data_mesh):
        return train_step_stateful_fn(
            model,
            optimizer,
            batch_inputs,
            batch_targets,
            batch_weights,
            carry_memories,
            reset_mask,
            chunk_positions,
        )


def _validation_step_stateful_impl(
    model: PropagatorModel,
    inputs: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    memories: tuple[jax.Array, ...],
    reset_mask: jax.Array,
    chunk_positions: jax.Array,
    task_ids: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    _, ce_loss, final_memories, metrics = model.forward_with_memories(
        inputs,
        targets,
        weights,
        memories,
        reset_mask,
        chunk_positions=chunk_positions,
        task_ids=task_ids,
    )
    return ce_loss, final_memories, metrics


def build_validation_step_stateful() -> Any:
    if batch_sharding is None or vector_sharding is None or memory_sharding is None:
        if data_mesh is None:
            return nnx.jit(_validation_step_stateful_impl)
        with jax.sharding.set_mesh(data_mesh):
            return nnx.jit(_validation_step_stateful_impl)
    memory_shardings = tuple(memory_sharding for _ in range(config.num_layers))
    metrics_shardings = tuple(None for _ in range(VALIDATION_METRIC_SIZE))
    if data_mesh is None:
        return nnx.jit(
            _validation_step_stateful_impl,
            in_shardings=(
                None,
                batch_sharding,
                batch_sharding,
                batch_sharding,
                memory_shardings,
                vector_sharding,
                vector_sharding,
                vector_sharding,
            ),
            out_shardings=(None, memory_shardings, metrics_shardings),
        )
    with jax.sharding.set_mesh(data_mesh):
        return nnx.jit(
            _validation_step_stateful_impl,
            in_shardings=(
                None,
                batch_sharding,
                batch_sharding,
                batch_sharding,
                memory_shardings,
                vector_sharding,
                vector_sharding,
                vector_sharding,
            ),
            out_shardings=(None, memory_shardings, metrics_shardings),
        )


def run_validation_step_stateful(
    validation_step_stateful_fn: Any,
    model: PropagatorModel,
    batch_inputs: jax.Array,
    batch_targets: jax.Array,
    batch_weights: jax.Array,
    memories: tuple[jax.Array, ...],
    reset_mask: jax.Array,
    chunk_positions: jax.Array,
    task_ids: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    if data_mesh is None:
        return validation_step_stateful_fn(
            model,
            batch_inputs,
            batch_targets,
            batch_weights,
            memories,
            reset_mask,
            chunk_positions,
            task_ids,
        )
    with jax.sharding.set_mesh(data_mesh):
        return validation_step_stateful_fn(
            model,
            batch_inputs,
            batch_targets,
            batch_weights,
            memories,
            reset_mask,
            chunk_positions,
            task_ids,
        )


@nnx.jit
def runtime_step_full(
    model: PropagatorModel,
    input_id: jax.Array,
    memories: tuple[jax.Array, ...],
    valid: jax.Array,
    positions: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    return model.step(input_id, memories, valid, positions)


@nnx.jit
def runtime_step_candidates(
    model: PropagatorModel,
    input_id: jax.Array,
    memories: tuple[jax.Array, ...],
    valid: jax.Array,
    candidate_ids: jax.Array,
    positions: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    return model.step_candidates(input_id, memories, valid, candidate_ids, positions)


@nnx.jit
def runtime_audio_frame_step(
    model: PropagatorModel,
    input_frame: jax.Array,
    memories: tuple[jax.Array, ...],
    valid: jax.Array,
    q0_candidate_ids: jax.Array,
    positions: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    hidden, next_memories = model.step_hidden(input_frame, memories, valid, positions)
    return model.project_candidates(hidden, q0_candidate_ids), hidden, next_memories


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
        step_logits, next_carry = model.step_candidates(
            step_inputs[0],
            carry,
            step_inputs[1],
            candidate_ids,
            step_inputs[2],
        )
        return next_carry, step_logits

    positions = jnp.broadcast_to(jnp.arange(input_ids.shape[1], dtype=jnp.int32)[:, None], input_ids.T.shape)
    final_memories, logits_t = jax.lax.scan(scan_step, memories, (input_ids.T, token_mask.T, positions))
    return logits_t[-1], final_memories


@nnx.jit
def prefill_stream_full(model: PropagatorModel, input_ids: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    batch_size, _ = input_ids.shape
    token_mask = input_ids != token_ids_pad
    memories = model.initial_memories(batch_size)

    def scan_step(carry, step_inputs):
        step_logits, next_carry = model.step(step_inputs[0], carry, step_inputs[1], step_inputs[2])
        return next_carry, step_logits

    positions = jnp.broadcast_to(jnp.arange(input_ids.shape[1], dtype=jnp.int32)[:, None], input_ids.T.shape)
    final_memories, logits_t = jax.lax.scan(scan_step, memories, (input_ids.T, token_mask.T, positions))
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
def sample_audio_candidate_token_jit(
    logits: jax.Array,
    key: jax.Array,
    candidate_ids: jax.Array,
    temperature: jax.Array,
) -> jax.Array:
    scaled = logits / jnp.maximum(temperature, 1e-6)
    if config.top_k > 0:
        special_mask = candidate_ids < audio_token_start
        codebook_logits = jnp.where(special_mask[None, :], jnp.finfo(jnp.float32).min, scaled)
        values, indices = jax.lax.top_k(codebook_logits, min(config.top_k, scaled.shape[-1]))
        codebook_ids = candidate_ids[indices]

        special_values = jnp.where(special_mask[None, :], scaled, jnp.finfo(jnp.float32).min)
        special_ids = jnp.broadcast_to(candidate_ids[None, :], special_values.shape)
        combined_values = jnp.concatenate([values, special_values], axis=-1)
        combined_ids = jnp.concatenate([codebook_ids, special_ids], axis=-1)
        sampled = jax.random.categorical(key, combined_values, axis=-1)
        return jnp.take_along_axis(combined_ids, sampled[:, None], axis=-1).squeeze(-1).astype(jnp.int32)

    sampled = jax.random.categorical(key, scaled, axis=-1)
    return candidate_ids[sampled].astype(jnp.int32)


@nnx.jit
def sample_audio_aux_codes_jit(
    model: PropagatorModel,
    hidden: jax.Array,
    q0_token_ids: jax.Array,
    key: jax.Array,
    temperature: jax.Array,
) -> jax.Array:
    def scan_step(carry, aux_index):
        depth_state, previous_token_ids, rng_key = carry
        logits, next_depth_state = model.project_audio_aux_head(depth_state, previous_token_ids, aux_index)
        scaled = logits / jnp.maximum(temperature, 1e-6)
        rng_key, subkey = jax.random.split(rng_key)
        if config.top_k > 0:
            values, indices = jax.lax.top_k(scaled, min(config.top_k, scaled.shape[-1]))
            sampled_local = jax.random.categorical(subkey, values, axis=-1)
            codes = jnp.take_along_axis(indices, sampled_local[:, None], axis=-1).squeeze(-1).astype(jnp.int32)
        else:
            codes = jax.random.categorical(subkey, scaled, axis=-1).astype(jnp.int32)
        next_token_ids = audio_token_start + (aux_index + 1) * int(config.audio_codebook_size) + codes
        return (next_depth_state, next_token_ids, rng_key), codes

    (_, _, _), codes_t = jax.lax.scan(
        scan_step,
        (hidden, q0_token_ids, key),
        jnp.arange(7, dtype=jnp.int32),
    )
    return jnp.swapaxes(codes_t, 0, 1)


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
    blocked = [
        token_ids_pad,
        token_ids_unk,
        token_ids_session,
        token_ids_user,
        token_ids_model,
        token_ids_listen,
        token_ids_user_end,
        token_ids_user_interrupt,
        token_ids.get("text", -1),
        token_ids.get("audio", -1),
        token_ids_audio_in,
        token_ids.get("image", -1),
        token_ids_image_in,
        token_ids.get("hybrid", -1),
        token_ids_audio_end,
        token_ids_silence,
        token_ids_text_in,
    ]
    blocked.extend(range(audio_token_start, audio_token_end))
    blocked.extend(range(image_token_start, image_token_end))
    return blocked


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
        token_ids.get("text", -1): "[TEXT]",
        token_ids_audio_in: "[AUDIO_IN]",
        token_ids.get("audio", -1): "[AUDIO]",
        token_ids_audio_out: "[AUDIO_OUT]",
        token_ids_audio_end: "[AUDIO_END]",
        token_ids_silence: "[SILENCE]",
        token_ids_text_in: "[TEXT_IN]",
        token_ids_text_out: "[TEXT_OUT]",
        token_ids.get("image", -1): "[IMAGE]",
        token_ids.get("hybrid", -1): "[HYBRID]",
        token_ids_hybrid_out: "[HYBRID_OUT]",
        token_ids_image_in: "[IMAGE_IN]",
    }
    if token_id in names:
        return names[token_id]
    parsed_audio = audio_code_from_token_id(int(token_id))
    if parsed_audio is not None:
        codebook_idx, code = parsed_audio
        return f"<audio:{codebook_idx}:{code}>"
    if is_image_token_id(int(token_id)):
        return f"<image:{int(token_id) - int(image_token_start)}>"
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


def parse_eval_text_cases() -> list[dict[str, Any]]:
    raw = (config.eval_text_cases or "").strip()
    cases: list[dict[str, Any]] = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for idx, item in enumerate(parsed):
                    if isinstance(item, dict):
                        chunks = item.get("chunks", item.get("prompt", []))
                        if isinstance(chunks, str):
                            chunks = [chunks]
                        chunks = [str(x) for x in chunks if str(x)]
                        if chunks:
                            cases.append({"name": str(item.get("name", f"case_{idx:02d}")), "chunks": chunks})
                    elif isinstance(item, str) and item:
                        cases.append({"name": f"case_{idx:02d}", "chunks": [item]})
        except json.JSONDecodeError:
            pass
    if not cases:
        cases.append({"name": "default", "chunks": parse_sample_chunks()})
    return cases


def parse_eval_image_cases() -> list[dict[str, str]]:
    raw = (config.eval_image_cases or "").strip()
    cases: list[dict[str, str]] = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for idx, item in enumerate(parsed):
                    if not isinstance(item, dict):
                        continue
                    image_text = str(item.get("image_text") or item.get("caption") or item.get("description") or "").strip()
                    question = str(item.get("question") or item.get("prompt") or "Describe the image.").strip()
                    if image_text and question:
                        cases.append(
                            {
                                "name": str(item.get("name") or f"image_case_{idx:02d}"),
                                "image_text": image_text,
                                "question": question,
                            }
                        )
        except json.JSONDecodeError:
            pass
    if not cases:
        cases.append({"name": "red_mug", "image_text": "A red mug is on a desk.", "question": "What object is visible?"})
    return cases


def safe_filename(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:48] or "sample"


def step_runtime(
    model: PropagatorModel,
    token_id: int,
    memories: tuple[jax.Array, ...],
    use_candidate_head: bool,
    position: int,
) -> tuple[jax.Array, tuple[jax.Array, ...], np.ndarray | None]:
    input_id = jnp.asarray([int(token_id)], dtype=jnp.int32)
    valid = jnp.ones_like(input_id, dtype=jnp.bool_)
    positions = jnp.asarray([int(position)], dtype=jnp.int32)
    if use_candidate_head:
        candidate_ids = jnp.asarray(candidate_token_ids_host, dtype=jnp.int32)
        logits, memories = runtime_step_candidates(model, input_id, memories, valid, candidate_ids, positions)
        return logits, memories, np.asarray(candidate_token_ids_host, dtype=np.int32)
    logits, memories = runtime_step_full(model, input_id, memories, valid, positions)
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
    chunks: list[str] | None = None,
    case_name: str = "default",
) -> str:
    chunks = chunks if chunks is not None else parse_sample_chunks()
    key = jax.random.PRNGKey(seed)
    memories = model.initial_memories(1)
    position = 0
    lines: list[str] = []

    lines.append("# runtime loop sample")
    lines.append(f"case: {case_name}")
    lines.append("")
    lines.append("## user stream")

    logits, memories, candidate_ids_np = step_runtime(model, token_ids_session, memories, use_candidate_head, position)
    position += 1
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    lines.append(f"[SESSION] -> {token_label(user_mode_effective_decision(raw))}")

    logits, memories, candidate_ids_np = step_runtime(model, token_ids_user, memories, use_candidate_head, position)
    position += 1
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    lines.append(f"[USER] -> {token_label(user_mode_effective_decision(raw))}")

    effective_decision = token_ids_listen
    for chunk in chunks:
        tokenized = encode_text(chunk)
        if not tokenized:
            lines.append(f"{json.dumps(chunk, ensure_ascii=False)} -> [LISTEN]")
            continue

        raw = token_ids_listen
        for token_id in tokenized:
            logits, memories, candidate_ids_np = step_runtime(model, token_id, memories, use_candidate_head, position)
            position += 1
            raw = argmax_token_from_logits(logits, candidate_ids_np)

        effective_decision = user_mode_effective_decision(raw)
        lines.append(f"{json.dumps(chunk, ensure_ascii=False)} -> {token_label(effective_decision)}")

        if effective_decision == token_ids_user_end:
            break

    if effective_decision != token_ids_user_end:
        lines.append("")
        lines.append("## model stream")
        lines.append("not started because runtime policy did not receive [USER_END].")
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append("## model stream")

    logits, memories, candidate_ids_np = step_runtime(model, token_ids_user_end, memories, use_candidate_head, position)
    position += 1
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    lines.append(f"[USER_END] -> {token_label(raw)}")

    if raw != token_ids_model:
        lines.append(f"stopped: expected [MODEL], got {token_label(raw)}")
        return "\n".join(lines) + "\n"

    current_input = token_ids_model
    logits, memories, candidate_ids_np = step_runtime(model, current_input, memories, use_candidate_head, position)
    position += 1

    for _ in range(config.sample_gen_len):
        key, subkey = jax.random.split(key)
        next_token = sample_model_token_from_logits(logits, subkey, candidate_ids_np, use_candidate_head)
        lines.append(f"{token_label(current_input)} -> {token_label(next_token)}")

        if next_token in {token_ids_model_end, token_ids_session_end, token_ids_listen}:
            break

        current_input = next_token
        logits, memories, candidate_ids_np = step_runtime(model, current_input, memories, use_candidate_head, position)
        position += 1

    return "\n".join(lines) + "\n"


def generate_text_eval_samples(
    model: PropagatorModel,
    seed: int,
    out_dir: Path,
    use_candidate_head: bool,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metas: list[dict[str, Any]] = []
    combined: list[str] = []
    for idx, case in enumerate(parse_eval_text_cases()):
        name = str(case["name"])
        chunks = [str(x) for x in case["chunks"]]
        sample = generate_sample(
            model,
            seed + idx,
            use_candidate_head=use_candidate_head,
            chunks=chunks,
            case_name=name,
        )
        file_name = f"sample_{idx:02d}_{safe_filename(name)}.txt"
        (out_dir / file_name).write_text(sample, encoding="utf-8")
        combined.append(sample)
        metas.append({"name": name, "chunks": chunks, "path": str(out_dir / file_name)})
    (out_dir / "sample.txt").write_text(("\n\n" + "=" * 80 + "\n\n").join(combined), encoding="utf-8")
    (out_dir / "text_generations.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metas


def feed_runtime_tokens(
    model: PropagatorModel,
    memories: tuple[jax.Array, ...],
    token_sequence: list[int],
    use_candidate_head: bool,
    start_position: int = 0,
) -> tuple[jax.Array, tuple[jax.Array, ...], np.ndarray | None, int]:
    logits = None
    candidate_ids_np = None
    position = int(start_position)
    for token_id in token_sequence:
        logits, memories, candidate_ids_np = step_runtime(model, int(token_id), memories, use_candidate_head, position)
        position += 1
    if logits is None:
        raise ValueError("No runtime tokens were provided")
    return logits, memories, candidate_ids_np, position


def generate_image_eval_sample(
    model: PropagatorModel,
    seed: int,
    out_dir: Path,
    case: dict[str, str],
    sample_idx: int,
    use_candidate_head: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    memories = model.initial_memories(1)
    position = 0
    image_text = str(case["image_text"])
    question = str(case["question"])
    name = str(case["name"])
    lines = [
        "# image recognition runtime sample",
        f"case: {name}",
        f"image_text: {image_text}",
        f"question: {question}",
        "",
        "## user stream",
    ]

    image_ids = image_text_to_patch_token_ids(image_text)
    prefix_ids = [
        token_ids_session,
        token_ids_user,
        *_tokenize_modal_input_prefix("image"),
        *image_ids,
        token_ids_text_in,
        *encode_text(question),
    ]
    logits, memories, candidate_ids_np, position = feed_runtime_tokens(
        model,
        memories,
        prefix_ids,
        use_candidate_head,
        position,
    )
    raw = argmax_token_from_logits(logits, candidate_ids_np)
    decision = user_mode_effective_decision(raw)
    lines.append(f"[IMAGE]/[IMAGE_IN] + {len(image_ids)} image tokens + question -> {token_label(decision)}")

    generated: list[int] = []
    if decision == token_ids_user_end:
        logits, memories, candidate_ids_np = step_runtime(model, token_ids_user_end, memories, use_candidate_head, position)
        position += 1
        raw = argmax_token_from_logits(logits, candidate_ids_np)
        lines.append(f"[USER_END] -> {token_label(raw)}")
        if raw == token_ids_model:
            current_input = token_ids_model
            logits, memories, candidate_ids_np = step_runtime(model, current_input, memories, use_candidate_head, position)
            position += 1
            lines.append("")
            lines.append("## model stream")
            for _ in range(config.sample_gen_len):
                key, subkey = jax.random.split(key)
                next_token = sample_model_token_from_logits(logits, subkey, candidate_ids_np, use_candidate_head)
                generated.append(next_token)
                lines.append(f"{token_label(current_input)} -> {token_label(next_token)}")
                if next_token in {token_ids_model_end, token_ids_session_end, token_ids_listen}:
                    break
                current_input = next_token
                logits, memories, candidate_ids_np = step_runtime(model, current_input, memories, use_candidate_head, position)
                position += 1
        else:
            lines.append(f"stopped: expected [MODEL], got {token_label(raw)}")
    else:
        lines.append("not started because runtime policy did not receive [USER_END].")

    text = decode_text_token_ids_for_eval(generated)
    sample_path = out_dir / f"image_sample_{sample_idx:02d}_{safe_filename(name)}.txt"
    sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "name": name,
        "image_text": image_text,
        "question": question,
        "generated_text": text,
        "num_image_tokens": len(image_ids),
        "num_generated_tokens": len(generated),
        "path": str(sample_path),
    }


def generate_image_eval_samples(
    model: PropagatorModel,
    seed: int,
    out_dir: Path,
    use_candidate_head: bool,
) -> list[dict[str, Any]]:
    metas = [
        generate_image_eval_sample(model, seed + idx, out_dir, case, idx, use_candidate_head)
        for idx, case in enumerate(parse_eval_image_cases())
    ]
    (out_dir / "image_generations.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metas


def parse_audio_eval_prompts() -> list[str]:
    raw = (config.audio_eval_prompts or "").strip()
    prompts: list[str] = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                prompts.append(parsed)
            elif isinstance(parsed, list):
                prompts.extend(str(item) for item in parsed if str(item))
        except json.JSONDecodeError:
            prompts.extend(part.strip() for part in raw.split("|") if part.strip())
    if not prompts:
        prompts.append(config.audio_eval_prompt)
    target = max(1, int(config.eval_audio_samples))
    while len(prompts) < target:
        prompts.append(prompts[-1])
    return prompts[:target]


def generate_audio_frames_after_prefix(
    model: PropagatorModel,
    seed: int,
    prefix_ids: list[int],
    max_tokens: int,
) -> tuple[list[int], list[str], int | None, str]:
    if not prefix_ids:
        raise ValueError("Audio generation requires a non-empty prefix")

    key = jax.random.PRNGKey(seed)
    memories = model.initial_memories(1)
    position = 0
    if len(prefix_ids) > 1:
        _, memories, _, position = feed_runtime_tokens(
            model,
            memories,
            prefix_ids[:-1],
            config.eval_use_candidate_head,
            position,
        )

    current_input = jnp.asarray([prefix_ids[-1]], dtype=jnp.int32)
    valid = jnp.ones((1,), dtype=jnp.bool_)
    generated: list[int] = []
    trace: list[str] = []
    stop_token: int | None = None
    stop_reason = "max_tokens"
    min_tokens = max(0, int(config.audio_min_generation_tokens))
    max_frames = max(1, int(max_tokens) // max(1, int(config.audio_codebooks)))

    for _ in range(max_frames):
        allow_stop = len(generated) >= min_tokens
        q0_candidate_ids_np = build_audio_codebook_candidate_token_ids(0, allow_stop=allow_stop)
        q0_candidate_ids = jnp.asarray(q0_candidate_ids_np, dtype=jnp.int32)
        positions = jnp.asarray([position], dtype=jnp.int32)
        q0_logits, hidden, memories = runtime_audio_frame_step(
            model,
            current_input,
            memories,
            valid,
            q0_candidate_ids,
            positions,
        )
        position += 1

        key, q0_key, aux_key = jax.random.split(key, 3)
        q0 = sample_audio_candidate_token_jit(
            q0_logits,
            q0_key,
            q0_candidate_ids,
            jnp.asarray(config.temperature, dtype=jnp.float32),
        )
        q0_token = int(jax.device_get(q0[0]))
        if q0_token in {token_ids_audio_end, token_ids_model_end, token_ids_session_end}:
            stop_token = q0_token
            stop_reason = token_label(q0_token)
            break

        parsed_q0 = audio_code_from_token_id(q0_token)
        if parsed_q0 is None or parsed_q0[0] != 0:
            stop_reason = "invalid_q0"
            break

        aux_codes = np.asarray(
            jax.device_get(
                sample_audio_aux_codes_jit(
                    model,
                    hidden,
                    q0,
                    aux_key,
                    jnp.asarray(config.temperature, dtype=jnp.float32),
                )
            )[0],
            dtype=np.int32,
        )
        frame = [q0_token]
        frame.extend(audio_token_id(codebook_idx, int(code)) for codebook_idx, code in enumerate(aux_codes, start=1))
        generated.extend(frame)
        trace.extend(token_label(token_id) for token_id in frame)
        current_input = jnp.asarray([frame], dtype=jnp.int32)

    return generated, trace, stop_token, stop_reason


def generate_audio_eval(
    model: PropagatorModel,
    seed: int,
    out_dir: Path,
    prompt: str | None = None,
    sample_idx: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt or config.audio_eval_prompt
    prompt_ids = encode_text(prompt_text)
    silence_prefix = [token_ids_silence] * max(0, config.silence_end_tokens) if config.synthesize_turn_silence else []
    prefix_ids = [
        token_ids_session,
        token_ids_user,
        *prompt_ids,
        *silence_prefix,
        token_ids_user_end,
        token_ids_model,
        token_ids_audio_out,
    ]

    generated, trace, stop_token, stop_reason = generate_audio_frames_after_prefix(
        model,
        seed,
        prefix_ids,
        int(config.eval_audio_tokens),
    )

    raw_audio, sample_rate, decode_error = decode_audio_token_ids_to_waveform(generated)
    raw_signal_stats = audio_signal_stats(raw_audio, sample_rate)
    audio, eval_gain = normalize_audio_for_eval(raw_audio)
    wav_path = out_dir / f"audio_generation_{sample_idx:02d}.wav"
    write_wav(wav_path, audio, sample_rate)
    signal_stats = audio_signal_stats(audio, sample_rate)
    is_low_rms = raw_signal_stats["rms"] < float(config.audio_low_rms_threshold)

    meta = {
        "prompt": prompt_text,
        "sample_idx": sample_idx,
        "sample_rate": sample_rate,
        "num_samples": int(audio.shape[-1]),
        "raw_rms": raw_signal_stats["rms"],
        "raw_peak": raw_signal_stats["peak"],
        "eval_gain": eval_gain,
        "is_low_rms": is_low_rms,
        **signal_stats,
        "num_generated_tokens": len(generated),
        "num_audio_tokens": sum(1 for token_id in generated if is_audio_token_id(token_id)),
        "stop_reason": stop_reason,
        "stop_token": token_label(stop_token) if stop_token is not None else None,
        "eval_audio_seconds": float(config.eval_audio_seconds),
        "eval_audio_tokens": int(config.eval_audio_tokens),
        "audio_min_generation_seconds": float(config.audio_min_generation_seconds),
        "audio_min_generation_tokens": int(config.audio_min_generation_tokens),
        "decode_error": decode_error,
        "wav_path": str(wav_path),
        "trace": trace[:256],
    }
    if is_low_rms:
        log_info(
            f"[Audio Eval] low raw RMS sample={sample_idx}, raw_rms={raw_signal_stats['rms']:.6f}, "
            f"raw_peak={raw_signal_stats['peak']:.6f}, gain={eval_gain:.2f}, wav={wav_path}"
        )
    (out_dir / f"audio_generation_{sample_idx:02d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def generate_audio_evals(model: PropagatorModel, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    metas = [
        generate_audio_eval(model, seed + idx, out_dir, prompt=prompt, sample_idx=idx)
        for idx, prompt in enumerate(parse_audio_eval_prompts())
    ]
    (out_dir / "audio_generations.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metas


def decode_text_token_ids_for_eval(ids: list[int]) -> str:
    text_ids = [
        int(token_id)
        for token_id in ids
        if 0 <= int(token_id) < text_vocab_size and int(token_id) not in control_token_ids()
    ]
    text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
    if getattr(config, "asr_eval_case_fold", False):
        text = text.lower()
    return text


def collect_audio_target_tokens(target_frames: np.ndarray) -> list[int]:
    audio_ids: list[int] = []
    for frame in np.asarray(target_frames, dtype=np.int32):
        for token_id in frame.tolist():
            if is_audio_token_id(int(token_id)):
                audio_ids.append(int(token_id))
    return audio_ids


def collect_text_targets_after_marker(target_main: list[int], marker: int) -> list[int]:
    try:
        start = target_main.index(marker) + 1
    except ValueError:
        return []
    stop_ids = {
        token_ids_model_end,
        token_ids_session_end,
        token_ids_audio_out,
        token_ids_audio_end,
        token_ids_hybrid_out,
        token_ids_text_out,
    }
    out: list[int] = []
    for token_id in target_main[start:]:
        token_i = int(token_id)
        if token_i in stop_ids or is_audio_token_id(token_i):
            break
        if token_i not in control_token_ids():
            out.append(token_i)
    return out


def iter_validation_audio_streams(kind: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(val_stream_ids) == 0:
        return []
    stream_ids_np = np.asarray(val_stream_ids)
    boundaries = np.flatnonzero(np.diff(stream_ids_np) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(stream_ids_np)]])
    selected: list[dict[str, Any]] = []

    for start, end in zip(starts, ends, strict=True):
        input_frames = np.asarray(val_input_tokens[int(start) : int(end)]).reshape(-1, 8)
        target_frames = np.asarray(val_target_tokens[int(start) : int(end)]).reshape(-1, 8)
        input_main = [int(x) for x in input_frames[:, 0].tolist() if int(x) != token_ids_pad]
        target_main = [int(x) for x in target_frames[:, 0].tolist() if int(x) != token_ids_pad]
        has_audio_in = token_ids_audio_in in input_main
        has_text_out = token_ids_text_out in target_main
        has_audio_out = token_ids_audio_out in target_main or any(is_audio_token_id(token_id) for token_id in target_main)
        if not has_audio_in:
            continue
        if kind == "asr" and not has_text_out:
            continue
        if kind == "audio" and not has_audio_out:
            continue
        if token_ids_model not in input_main:
            continue
        selected.append(
            {
                "stream_id": int(stream_ids_np[int(start)]),
                "input_main": input_main,
                "target_main": target_main,
                "target_frames": target_frames,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def generate_text_after_prefix(
    model: PropagatorModel,
    seed: int,
    prefix_ids: list[int],
    max_tokens: int,
) -> tuple[list[int], list[str]]:
    key = jax.random.PRNGKey(seed)
    memories = model.initial_memories(1)
    position = 0
    if len(prefix_ids) > 1:
        _, memories, _, position = feed_runtime_tokens(
            model,
            memories,
            prefix_ids[:-1],
            config.eval_use_candidate_head,
            position,
        )
    logits, memories, candidate_ids_np = step_runtime(
        model,
        prefix_ids[-1],
        memories,
        config.eval_use_candidate_head,
        position,
    )
    position += 1
    generated: list[int] = []
    trace: list[str] = []
    for _ in range(max(1, int(max_tokens))):
        key, subkey = jax.random.split(key)
        token_id = sample_model_token_from_logits(logits, subkey, candidate_ids_np, config.eval_use_candidate_head)
        generated.append(token_id)
        trace.append(token_label(token_id))
        if token_id in {token_ids_model_end, token_ids_session_end, token_ids_audio_out, token_ids_audio_end}:
            break
        logits, memories, candidate_ids_np = step_runtime(
            model,
            token_id,
            memories,
            config.eval_use_candidate_head,
            position,
        )
        position += 1
    return generated, trace


def generate_audio_after_prefix(
    model: PropagatorModel,
    seed: int,
    prefix_ids: list[int],
    max_tokens: int,
) -> tuple[list[int], list[str]]:
    generated, trace, _, _ = generate_audio_frames_after_prefix(model, seed, prefix_ids, max_tokens)
    return generated, trace


def generate_audio_input_evals(model: PropagatorModel, seed: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_limit = max(0, int(config.eval_audio_input_samples))
    results: dict[str, Any] = {"asr": [], "audio_to_audio": []}

    for idx, stream in enumerate(iter_validation_audio_streams("asr", sample_limit)):
        input_main = stream["input_main"]
        model_pos = input_main.index(token_ids_model)
        prefix = input_main[: model_pos + 1]
        generated, trace = generate_text_after_prefix(
            model,
            seed + 10_000 + idx,
            prefix,
            int(config.eval_audio_input_text_tokens),
        )
        expected_ids = collect_text_targets_after_marker(stream["target_main"], token_ids_text_out)
        meta = {
            "sample_idx": idx,
            "stream_id": stream["stream_id"],
            "prefix_tokens": len(prefix),
            "expected_text": decode_text_token_ids_for_eval(expected_ids),
            "generated_text": decode_text_token_ids_for_eval(generated),
            "generated_tokens": [token_label(token_id) for token_id in generated[:128]],
            "trace": trace[:128],
        }
        results["asr"].append(meta)

    for idx, stream in enumerate(iter_validation_audio_streams("audio", sample_limit)):
        input_main = stream["input_main"]
        model_pos = input_main.index(token_ids_model)
        prefix_to_model = input_main[: model_pos + 1]
        modality_generated, modality_trace = generate_text_after_prefix(model, seed + 20_000 + idx, prefix_to_model, 4)
        if token_ids_audio_out in input_main[model_pos:]:
            audio_out_pos = model_pos + input_main[model_pos:].index(token_ids_audio_out)
            prefix = input_main[: audio_out_pos + 1]
        else:
            prefix = [*prefix_to_model, token_ids_audio_out]
        generated_audio_tokens, trace = generate_audio_after_prefix(
            model,
            seed + 30_000 + idx,
            prefix,
            int(config.eval_audio_input_audio_tokens),
        )
        raw_audio, sample_rate, decode_error = decode_audio_token_ids_to_waveform(generated_audio_tokens)
        audio, eval_gain = normalize_audio_for_eval(raw_audio)
        wav_path = out_dir / f"audio_input_audio_to_audio_{idx:02d}.wav"
        write_wav(wav_path, audio, sample_rate)
        expected_audio_tokens = collect_audio_target_tokens(stream["target_frames"])
        meta = {
            "sample_idx": idx,
            "stream_id": stream["stream_id"],
            "prefix_tokens": len(prefix),
            "model_modality_prediction": [token_label(token_id) for token_id in modality_generated],
            "model_modality_trace": modality_trace,
            "expected_audio_tokens": len(expected_audio_tokens),
            "num_generated_tokens": len(generated_audio_tokens),
            "num_audio_tokens": sum(1 for token_id in generated_audio_tokens if is_audio_token_id(token_id)),
            "decode_error": decode_error,
            "eval_gain": eval_gain,
            **audio_signal_stats(audio, sample_rate),
            "wav_path": str(wav_path),
            "trace": trace[:128],
        }
        results["audio_to_audio"].append(meta)

    (out_dir / "audio_input_evals.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


@dataclass
class StatefulChunkSampler:
    stream_ids: np.ndarray
    batch_size: int
    seed: int
    source_weights: list[float] | None = None

    def __post_init__(self):
        stream_ids_np = np.asarray(self.stream_ids)
        if len(stream_ids_np) == 0:
            raise ValueError("No chunks available for sampler")

        boundaries = np.flatnonzero(np.diff(stream_ids_np) != 0) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(stream_ids_np)]])
        self.stream_ranges = [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]

        self.rng = np.random.default_rng(self.seed)
        self.source_orders: dict[int, np.ndarray] = {}
        self.source_order_pos: dict[int, int] = {}

        source_to_streams: dict[int, list[int]] = {}
        max_source = len(self.source_weights or [])
        for stream_idx, (start, _) in enumerate(self.stream_ranges):
            stream_id = int(stream_ids_np[start])
            source_idx = int(stream_id // 1_000_000_000) - 1
            if source_idx < 0 or (max_source > 0 and source_idx >= max_source):
                source_idx = 0
            source_to_streams.setdefault(source_idx, []).append(stream_idx)

        for source_idx, indices in source_to_streams.items():
            order = np.asarray(indices, dtype=np.int64)
            self.rng.shuffle(order)
            self.source_orders[source_idx] = order
            self.source_order_pos[source_idx] = 0

        available_sources = sorted(self.source_orders)
        if not available_sources:
            raise ValueError("No source streams available for sampler")

        if self.source_weights:
            weights = np.asarray(
                [max(0.0, float(self.source_weights[idx])) if idx < len(self.source_weights) else 0.0 for idx in available_sources],
                dtype=np.float64,
            )
            if float(weights.sum()) <= 0.0:
                weights = np.ones_like(weights)
        else:
            weights = np.asarray([len(self.source_orders[idx]) for idx in available_sources], dtype=np.float64)
        weights /= weights.sum()

        raw_counts = weights * int(self.batch_size)
        lane_counts = np.floor(raw_counts).astype(np.int64)
        remainder = int(self.batch_size) - int(lane_counts.sum())
        if remainder > 0:
            fractional_order = np.argsort(-(raw_counts - lane_counts))
            lane_counts[fractional_order[:remainder]] += 1
        self.lane_sources = np.concatenate(
            [
                np.full(int(count), source_idx, dtype=np.int64)
                for source_idx, count in zip(available_sources, lane_counts, strict=True)
            ]
        )
        self.rng.shuffle(self.lane_sources)
        self.source_lane_counts = {
            int(source_idx): int(count)
            for source_idx, count in zip(available_sources, lane_counts, strict=True)
            if int(count) > 0
        }

        self.lane_pos = np.zeros((self.batch_size,), dtype=np.int64)
        self.lane_end = np.zeros((self.batch_size,), dtype=np.int64)
        self.lane_needs_reset = np.ones((self.batch_size,), dtype=np.bool_)

        for lane in range(self.batch_size):
            self._assign_stream(lane)

    def _next_stream_range(self, source_idx: int) -> tuple[int, int]:
        order = self.source_orders[source_idx]
        order_pos = self.source_order_pos[source_idx]
        if order_pos >= len(order):
            self.rng.shuffle(order)
            order_pos = 0
        stream_idx = int(order[order_pos])
        self.source_order_pos[source_idx] = order_pos + 1
        return self.stream_ranges[stream_idx]

    def _assign_stream(self, lane: int) -> None:
        start, end = self._next_stream_range(int(self.lane_sources[lane]))
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
    batch_inputs = maybe_put_batch(np.asarray(inputs_arr[indices], dtype=np.int32), np.int32)
    batch_targets = maybe_put_batch(np.asarray(targets_arr[indices], dtype=np.int32), np.int32)
    batch_weights = maybe_put_batch(np.asarray(weights_arr[indices], dtype=np.float32), np.float32)
    return batch_inputs, batch_targets, batch_weights


def get_chunk_positions_by_indices(positions_arr: np.ndarray, indices: np.ndarray) -> jax.Array:
    return maybe_put_vector(np.asarray(positions_arr[indices], dtype=np.int32), np.int32)


def sample_control_indices(total: int, batch_size: int, seed: int) -> np.ndarray:
    if total <= 0 or batch_size <= 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, total, size=(batch_size,), dtype=np.int64)


def inject_train_control_examples(
    batch_inputs_np: np.ndarray,
    batch_targets_np: np.ndarray,
    batch_weights_np: np.ndarray,
    reset_mask_np: np.ndarray,
    chunk_positions_np: np.ndarray,
    step: int,
) -> None:
    if train_control_input_tokens is None or len(train_control_input_tokens) == 0:
        return
    rate = max(0.0, min(1.0, float(config.synthetic_control_train_rate)))
    if rate <= 0.0:
        return
    lane_count = int(round(int(config.batch_size) * rate))
    lane_count = max(1, min(int(config.batch_size), lane_count))
    indices = sample_control_indices(len(train_control_input_tokens), lane_count, config.seed + 40_000 + int(step))
    rng = np.random.default_rng(config.seed + 45_000 + int(step))
    lanes = rng.choice(int(config.batch_size), size=lane_count, replace=False).astype(np.int64)
    batch_inputs_np[lanes] = np.asarray(train_control_input_tokens[indices], dtype=np.int32)
    batch_targets_np[lanes] = np.asarray(train_control_target_tokens[indices], dtype=np.int32)
    batch_weights_np[lanes] = np.asarray(train_control_loss_weights[indices], dtype=np.float32)
    if train_control_chunk_positions is not None:
        chunk_positions_np[lanes] = np.asarray(train_control_chunk_positions[indices], dtype=np.int32)
    else:
        chunk_positions_np[lanes] = 0
    reset_mask_np[lanes] = True


def get_validation_control_batch(idx: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if val_control_input_tokens is None or len(val_control_input_tokens) == 0:
        raise ValueError("No validation control chunks are available")
    indices = sample_control_indices(len(val_control_input_tokens), int(config.batch_size), config.seed + 50_000 + int(idx))
    batch_inputs = maybe_put_batch(np.asarray(val_control_input_tokens[indices], dtype=np.int32), np.int32)
    batch_targets = maybe_put_batch(np.asarray(val_control_target_tokens[indices], dtype=np.int32), np.int32)
    batch_weights = maybe_put_batch(np.asarray(val_control_loss_weights[indices], dtype=np.float32), np.float32)
    reset_mask = maybe_put_vector(np.ones((int(config.batch_size),), dtype=np.bool_), np.bool_)
    if val_control_chunk_positions is None:
        chunk_positions_np = np.zeros((int(config.batch_size),), dtype=np.int32)
    else:
        chunk_positions_np = np.asarray(val_control_chunk_positions[indices], dtype=np.int32)
    chunk_positions = maybe_put_vector(chunk_positions_np, np.int32)
    if val_control_chunk_task_ids is None:
        task_ids_np = np.zeros((int(config.batch_size),), dtype=np.int32)
    else:
        task_ids_np = np.asarray(val_control_chunk_task_ids[indices], dtype=np.int32)
    task_ids = maybe_put_vector(task_ids_np, np.int32)
    return batch_inputs, batch_targets, batch_weights, reset_mask, chunk_positions, task_ids


def get_random_batch(step_index: int, shuffled_indices: np.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    start_idx = (step_index * config.batch_size) % len(train_input_tokens)
    indices = shuffled_indices[start_idx : start_idx + config.batch_size]
    if len(indices) < config.batch_size:
        wrap = config.batch_size - len(indices)
        indices = np.concatenate([indices, shuffled_indices[:wrap]])

    return get_batch_by_indices(train_input_tokens, train_target_tokens, train_loss_weights, indices)


def get_validation_random_batch(idx: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    indices = validation_indices_for_batch(idx)
    return get_batch_by_indices(val_input_tokens, val_target_tokens, val_loss_weights, indices)


def validation_indices_for_batch(idx: int) -> np.ndarray:
    start = (idx * config.batch_size) % len(val_input_tokens)
    return (np.arange(config.batch_size) + start) % len(val_input_tokens)


def build_chunk_task_ids(inputs_arr: np.ndarray, targets_arr: np.ndarray, stream_ids_arr: np.ndarray) -> np.ndarray:
    stream_ids_np = np.asarray(stream_ids_arr)
    task_ids = np.zeros((len(stream_ids_np),), dtype=np.int32)
    if len(stream_ids_np) == 0:
        return task_ids

    boundaries = np.flatnonzero(np.diff(stream_ids_np) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(stream_ids_np)]])
    for start, end in zip(starts, ends, strict=True):
        main_inputs = np.asarray(inputs_arr[int(start) : int(end)])[..., 0]
        main_targets = np.asarray(targets_arr[int(start) : int(end)])[..., 0]
        has_audio_in = bool(np.any(main_inputs == token_ids_audio_in))
        has_image_in = bool(np.any(main_inputs == token_ids_image_in))
        has_audio_out = bool(
            np.any(main_targets == token_ids_audio_out)
            or np.any(np.logical_and(main_targets >= audio_token_start, main_targets < audio_token_end))
        )
        task_id = 0
        if has_image_in:
            task_id = 4
        elif has_audio_in and not has_audio_out:
            task_id = 1
        elif not has_audio_in and has_audio_out:
            task_id = 2
        elif has_audio_in and has_audio_out:
            task_id = 3
        task_ids[int(start) : int(end)] = task_id
    return task_ids


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
        text_correct,
        text_total,
        audio_correct,
        audio_total,
        audio_codebook_correct,
        audio_codebook_total,
        audio_aux_token_correct,
        audio_aux_token_total,
        audio_all_codebook_correct,
        audio_all_codebook_total,
        main_nll_sum,
        main_weight_sum,
        aux_nll_sum,
        aux_weight_sum,
        text_task_correct,
        asr_task_correct,
        tts_task_correct,
        duplex_task_correct,
        image_task_correct,
        text_task_total,
        asr_task_total,
        tts_task_total,
        duplex_task_total,
        image_task_total,
        text_task_nll,
        asr_task_nll,
        tts_task_nll,
        duplex_task_nll,
        image_task_nll,
        text_task_weight,
        asr_task_weight,
        tts_task_weight,
        duplex_task_weight,
        image_task_weight,
    ) = [float(x) for x in metric_sums]

    def ratio(num: float, den: float) -> float:
        return num / den if den > 0 else float("nan")

    decision_parts = [
        ratio(listen_correct, listen_total),
        ratio(user_end_correct, user_end_total),
        ratio(interrupt_correct, interrupt_total),
    ]
    decision_macro_parts = [value for value in decision_parts if math.isfinite(value)]
    decision_macro_acc = float(np.mean(decision_macro_parts)) if decision_macro_parts else float("nan")

    return {
        "decision_acc": ratio(decision_correct, decision_total),
        "decision_macro_acc": decision_macro_acc,
        "listen_acc": ratio(listen_correct, listen_total),
        "user_end_acc": ratio(user_end_correct, user_end_total),
        "interrupt_acc": ratio(interrupt_correct, interrupt_total),
        "model_end_acc": ratio(model_end_correct, model_end_total),
        "text_token_acc": ratio(text_correct, text_total),
        "audio_token_acc": ratio(audio_correct, audio_total),
        "audio_main_acc": ratio(audio_correct, audio_total),
        "audio_codebook_acc": ratio(audio_codebook_correct, audio_codebook_total),
        "audio_aux_frame_exact_acc": ratio(audio_codebook_correct, audio_codebook_total),
        "audio_aux_token_acc": ratio(audio_aux_token_correct, audio_aux_token_total),
        "audio_all_codebook_frame_exact_acc": ratio(audio_all_codebook_correct, audio_all_codebook_total),
        "main_ce": ratio(main_nll_sum, main_weight_sum),
        "aux_audio_ce": ratio(aux_nll_sum, aux_weight_sum),
        "text_task_acc": ratio(text_task_correct, text_task_total),
        "asr_task_acc": ratio(asr_task_correct, asr_task_total),
        "tts_task_acc": ratio(tts_task_correct, tts_task_total),
        "duplex_task_acc": ratio(duplex_task_correct, duplex_task_total),
        "image_task_acc": ratio(image_task_correct, image_task_total),
        "text_task_ce": ratio(text_task_nll, text_task_weight),
        "asr_task_ce": ratio(asr_task_nll, asr_task_weight),
        "tts_task_ce": ratio(tts_task_nll, tts_task_weight),
        "duplex_task_ce": ratio(duplex_task_nll, duplex_task_weight),
        "image_task_ce": ratio(image_task_nll, image_task_weight),
        "decision_total": decision_total,
        "listen_total": listen_total,
        "user_end_total": user_end_total,
        "interrupt_total": interrupt_total,
        "model_end_total": model_end_total,
        "text_token_total": text_total,
        "audio_token_total": audio_total,
        "audio_codebook_total": audio_codebook_total,
        "audio_aux_token_total": audio_aux_token_total,
        "image_task_total": image_task_total,
    }


def finite_metric(metrics: dict[str, float], key: str, default: float | None = None) -> float | None:
    value = metrics.get(key, default)
    if value is None:
        return default
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value_f):
        return default
    return value_f


def mean_score(values: list[float | None], default: float = 0.0) -> float:
    finite_values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite_values)) if finite_values else float(default)


def validation_composite_score(metrics: dict[str, float]) -> dict[str, float]:
    """Propagator-specific validation score for early stopping.

    Higher is better. This combines protocol, text semantics, speech/codebook
    behavior, duplex behavior, and coverage so early stopping cannot be driven
    only by plain weighted CE while important multimodal buckets are absent.
    """

    protocol_score = mean_score(
        [
            finite_metric(metrics, "decision_macro_acc"),
            finite_metric(metrics, "listen_acc"),
            finite_metric(metrics, "user_end_acc"),
            finite_metric(metrics, "model_end_acc"),
        ]
    )
    text_score = mean_score(
        [
            finite_metric(metrics, "text_task_acc"),
            finite_metric(metrics, "text_token_acc"),
        ]
    )
    speech_score = mean_score(
        [
            finite_metric(metrics, "asr_task_acc"),
            finite_metric(metrics, "tts_task_acc"),
            finite_metric(metrics, "audio_main_acc"),
        ]
    )
    image_score = mean_score([finite_metric(metrics, "image_task_acc")])
    audio_aux_score = mean_score([finite_metric(metrics, "audio_aux_token_acc")])
    duplex_score = finite_metric(metrics, "duplex_task_acc", 0.0) or 0.0

    coverage_checks = [
        finite_metric(metrics, "decision_total", 0.0) or 0.0,
        finite_metric(metrics, "listen_total", 0.0) or 0.0,
        finite_metric(metrics, "user_end_total", 0.0) or 0.0,
        finite_metric(metrics, "model_end_total", 0.0) or 0.0,
        finite_metric(metrics, "text_token_total", 0.0) or 0.0,
        finite_metric(metrics, "audio_token_total", 0.0) or 0.0,
        finite_metric(metrics, "audio_aux_token_total", 0.0) or 0.0,
        finite_metric(metrics, "image_task_total", 0.0) or 0.0,
    ]
    coverage_score = float(np.mean([1.0 if value > 0.0 else 0.0 for value in coverage_checks]))

    composite = (
        0.22 * protocol_score
        + 0.22 * text_score
        + 0.18 * speech_score
        + 0.10 * image_score
        + 0.10 * audio_aux_score
        + 0.10 * duplex_score
        + 0.08 * coverage_score
    )
    return {
        "validation_composite_score": float(composite),
        "validation_protocol_score": float(protocol_score),
        "validation_text_score": float(text_score),
        "validation_speech_score": float(speech_score),
        "validation_image_score": float(image_score),
        "validation_audio_aux_score": float(audio_aux_score),
        "validation_duplex_score": float(duplex_score),
        "validation_coverage_score": float(coverage_score),
    }


def run_validation(
    model: PropagatorModel,
    step: int,
    validation_step_stateful_fn: Any,
) -> tuple[float, dict[str, float]]:
    losses = []
    metric_sums = np.zeros((VALIDATION_METRIC_SIZE,), dtype=np.float64)

    if config.stateful_validation:
        sampler = StatefulChunkSampler(
            val_stream_ids,
            config.batch_size,
            config.seed + 10_000,
            source_weights=[float(spec["weight"]) for spec in parse_dataset_mix()],
        )
        memories = initial_memories_for_training(model, config.batch_size)

        for _ in range(config.validation_batches):
            indices, reset_mask_np = sampler.next_indices()
            batch_inputs, batch_targets, batch_weights = get_batch_by_indices(
                val_input_tokens,
                val_target_tokens,
                val_loss_weights,
                indices,
            )
            reset_mask = maybe_put_vector(reset_mask_np, np.bool_)
            chunk_positions = get_chunk_positions_by_indices(val_chunk_positions, indices)

            # Task ids: 0 Text->Text, 1 Audio->Text, 2 Text->Audio, 3 Audio->Audio/Hybrid, 4 Image->Text.
            task_ids_np = np.asarray(val_chunk_task_ids[indices], dtype=np.int32)
            task_ids = maybe_put_vector(task_ids_np, np.int32)

            ce_loss, memories, metrics = run_validation_step_stateful(
                validation_step_stateful_fn,
                model,
                batch_inputs,
                batch_targets,
                batch_weights,
                memories,
                reset_mask,
                chunk_positions,
                task_ids,
            )
            losses.append(float(ce_loss))
            metric_sums += np.asarray([float(jax.device_get(x)) for x in metrics], dtype=np.float64)

    else:
        for i in range(config.validation_batches):
            indices = validation_indices_for_batch(i)
            batch_inputs, batch_targets, batch_weights = get_batch_by_indices(
                val_input_tokens,
                val_target_tokens,
                val_loss_weights,
                indices,
            )
            memories = initial_memories_for_training(model, config.batch_size)
            reset_mask = maybe_put_vector(np.ones((config.batch_size,), dtype=np.bool_), np.bool_)
            chunk_positions = get_chunk_positions_by_indices(val_chunk_positions, indices)

            task_ids_np = np.asarray(val_chunk_task_ids[indices], dtype=np.int32)
            task_ids = maybe_put_vector(task_ids_np, np.int32)

            ce_loss, _, metrics = run_validation_step_stateful(
                validation_step_stateful_fn,
                model,
                batch_inputs,
                batch_targets,
                batch_weights,
                memories,
                reset_mask,
                chunk_positions,
                task_ids,
            )
            losses.append(float(ce_loss))
            metric_sums += np.asarray([float(jax.device_get(x)) for x in metrics], dtype=np.float64)

    if (
        val_control_input_tokens is not None
        and len(val_control_input_tokens) > 0
        and int(config.validation_control_batches) > 0
    ):
        for i in range(int(config.validation_control_batches)):
            batch_inputs, batch_targets, batch_weights, reset_mask, chunk_positions, task_ids = get_validation_control_batch(step + i)
            memories = initial_memories_for_training(model, config.batch_size)
            ce_loss, _, metrics = run_validation_step_stateful(
                validation_step_stateful_fn,
                model,
                batch_inputs,
                batch_targets,
                batch_weights,
                memories,
                reset_mask,
                chunk_positions,
                task_ids,
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


def format_duration(seconds: float) -> str:
    seconds_i = max(0, int(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def save_metric_plot(steps: list[int], values: list[float], path: Path, title: str, step: int) -> None:
    if not values:
        return
    if len(steps) != len(values):
        start = max(1, step - len(values) + 1)
        steps = list(range(start, start + len(values)))
    plt.figure(figsize=(10, 4))
    v_arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(v_arr)
    if not np.any(finite):
        v_arr = np.zeros_like(v_arr)
    else:
        fill_value = float(np.nanmean(v_arr[finite]))
        v_arr = np.where(finite, v_arr, fill_value)
    plt.plot(steps, v_arr, alpha=0.3, label="raw")
    if len(v_arr) > 1:
        window = max(5, len(v_arr) // 20)
        plt.plot(steps, rolling_mean(v_arr, window), linewidth=2, label=f"rolling mean ({window})")
    plt.title(f"{title} - Step {step}")
    plt.xlabel("step")
    plt.ylabel(title)
    plt.legend(loc="best")
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
    decay_steps = max(1, total_steps)

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


def save_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    model: PropagatorModel,
    optimizer: nnx.Optimizer,
    output_dir: Path,
) -> None:
    log_info(f"\n[Checkpoint] Saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_incomplete_checkpoint_dirs(output_dir)
    _, model_state = nnx.split(model)
    _, optimizer_state = nnx.split(optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    checkpointer.save(os.path.abspath(output_dir / "checkpoint"), state, force=True)
    checkpointer.wait_until_finished()
    log_info("[Checkpoint] Done\n")


def restore_checkpoint_state(checkpointer: ocp.StandardCheckpointer, checkpoint_path: Path, target_state: Any) -> Any:
    try:
        return checkpointer.restore(os.path.abspath(checkpoint_path), args=ocp.args.StandardRestore(target_state))
    except TypeError:
        return checkpointer.restore(os.path.abspath(checkpoint_path), target=target_state)


def hostify_tree(tree: Any) -> Any:
    def hostify_leaf(leaf):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            return np.asarray(jax.device_get(leaf))
        return leaf

    return jax.tree_util.tree_map(hostify_leaf, tree)


def restore_training_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    checkpoint_path: Path,
    model: PropagatorModel,
    optimizer: nnx.Optimizer,
) -> str:
    _, model_state = nnx.split(model)
    _, optimizer_state = nnx.split(optimizer)
    target = {"model": model_state, "optimizer": optimizer_state}

    try:
        restored = restore_checkpoint_state(checkpointer, checkpoint_path, target)
        if isinstance(restored, dict) and "model" in restored:
            nnx.update(model, hostify_tree(restored["model"]))
            if "optimizer" in restored:
                nnx.update(optimizer, hostify_tree(restored["optimizer"]))
                return "model+optimizer"
            return "model"
    except Exception as exc:
        log_info(f"[Checkpoint] Combined restore failed, trying model-only restore: {exc}")

    restored_model = restore_checkpoint_state(checkpointer, checkpoint_path, model_state)
    nnx.update(model, hostify_tree(restored_model))
    return "model"


_backup_threads: list[threading.Thread] = []


def configure_runtime_environment() -> None:
    global batch_sharding, vector_sharding, memory_sharding, data_mesh
    cpu_count = max(1, os.cpu_count() or 1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_count))
    log_info(f"CPU cores available for preprocessing: {cpu_count}")
    devices = jax.devices()
    log_info(f"JAX devices: {devices}")

    if os.environ.get("JAX_COORDINATOR_ADDRESS"):
        try:
            jax.distributed.initialize()
            log_info("Initialized JAX distributed runtime")
        except ValueError:
            log_info("JAX distributed runtime was already initialized")
        except Exception as exc:
            log_info(f"JAX distributed initialization skipped: {exc}")

    if config.enable_data_sharding and len(devices) > 1 and config.batch_size % len(devices) == 0:
        data_mesh = Mesh(np.asarray(devices), (config.data_axis_name,))
        batch_sharding = NamedSharding(data_mesh, P(config.data_axis_name, None))
        vector_sharding = NamedSharding(data_mesh, P(config.data_axis_name))
        memory_sharding = NamedSharding(data_mesh, P(config.data_axis_name, None, None))
        log_info(f"Enabled data-axis sharding across {len(devices)} devices")
    else:
        data_mesh = None
        batch_sharding = None
        vector_sharding = None
        memory_sharding = None
        if config.enable_data_sharding and len(devices) > 1:
            log_info(
                "Data sharding disabled because batch_size is not divisible by device count: "
                f"batch_size={config.batch_size}, devices={len(devices)}"
            )


def maybe_put_batch(array: np.ndarray, dtype: Any) -> jax.Array:
    value = np.asarray(array, dtype=dtype)
    if batch_sharding is not None:
        return jax.device_put(value, batch_sharding)
    return jnp.asarray(value)


def maybe_put_vector(array: np.ndarray, dtype: Any) -> jax.Array:
    value = np.asarray(array, dtype=dtype)
    if vector_sharding is not None:
        return jax.device_put(value, vector_sharding)
    return jnp.asarray(value)


def maybe_shard_memories(memories: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    if memory_sharding is None:
        return memories
    return tuple(jax.device_put(memory, memory_sharding) for memory in memories)


def initial_memories_for_training(model: PropagatorModel, batch_size: int) -> tuple[jax.Array, ...]:
    if memory_sharding is None:
        return model.initial_memories(batch_size)
    shape = (batch_size, model.cfg.memory_key_size, model.cfg.memory_value_size)
    return tuple(
        jax.device_put(np.zeros(shape, dtype=np.float32), memory_sharding)
        for _ in range(model.cfg.num_layers)
    )


def estimate_param_bytes(model: PropagatorModel) -> tuple[int, int]:
    _, state = nnx.split(model)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(state) if hasattr(leaf, "shape") and hasattr(leaf, "dtype")]
    fp_bytes = sum(int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize for leaf in leaves)
    params = sum(int(np.prod(leaf.shape)) for leaf in leaves)
    return params, fp_bytes


def infer_device_hbm_bytes() -> int:
    env_gb = os.environ.get("TPU_HBM_GB") or os.environ.get("JAX_DEVICE_HBM_GB")
    if env_gb:
        try:
            return int(float(env_gb) * 1024**3)
        except ValueError:
            pass
    if config.auto_batch_hbm_gb > 0:
        return int(float(config.auto_batch_hbm_gb) * 1024**3)

    devices = jax.devices()
    kind = str(getattr(devices[0], "device_kind", "")).lower() if devices else ""
    if "v5e" in kind or "v5litepod" in kind:
        return 16 * 1024**3
    if "tpu" in str(devices[0]).lower() or "v4" in kind or "v5" in kind:
        return 32 * 1024**3
    return 24 * 1024**3


def choose_auto_batch_size(model: PropagatorModel) -> int:
    env_per_device = os.environ.get("PER_DEVICE_BATCH")
    device_count = max(1, len(jax.devices()))
    if env_per_device:
        per_device = max(1, int(env_per_device))
        batch_size = per_device * device_count
        log_info(f"[Batch] Using PER_DEVICE_BATCH={per_device}; global batch_size={batch_size}")
        return batch_size

    _, param_bytes = estimate_param_bytes(model)
    hbm_bytes = infer_device_hbm_bytes()
    util = min(0.95, max(0.25, float(config.auto_batch_memory_util)))
    target_bytes = int(hbm_bytes * util)

    optimizer_and_grad_bytes = param_bytes * 3
    compile_reserve_bytes = 4 * 1024**3
    fixed_training_bytes = param_bytes + optimizer_and_grad_bytes + compile_reserve_bytes

    recurrent_bytes_per_sample = (
        int(config.num_layers)
        * int(config.memory_key_size)
        * int(config.memory_value_size)
        * np.dtype(np.float32).itemsize
    )
    activation_margin = max(1.5, 1.0 + int(config.train_unroll_len) / 64.0)
    per_sample_bytes = int(recurrent_bytes_per_sample * activation_margin)
    available_for_batch = target_bytes - fixed_training_bytes

    if available_for_batch <= 0:
        raw_per_device = max(1, int(config.auto_batch_multiple_per_device))
    else:
        raw_per_device = max(1, available_for_batch // max(1, per_sample_bytes))

    max_per_device = max(1, int(config.auto_batch_max_per_device))
    multiple = max(1, int(config.auto_batch_multiple_per_device))
    per_device = min(max_per_device, int(raw_per_device))
    per_device = max(multiple, (per_device // multiple) * multiple)
    batch_size = int(per_device * device_count)
    log_info(
        f"[Batch] Auto batch_size={batch_size} ({per_device}/device, devices={device_count}, "
        f"hbm={hbm_bytes / 1024**3:.1f}GiB, target={target_bytes / 1024**3:.1f}GiB, "
        f"fixed_est={fixed_training_bytes / 1024**3:.1f}GiB, per_sample_est={per_sample_bytes / 1024**2:.1f}MiB)"
    )
    return batch_size


def write_edge_memory_report(model: PropagatorModel, output_root: Path) -> None:
    if not config.write_edge_report:
        return
    params, fp_bytes = estimate_param_bytes(model)
    q_bytes = math.ceil(params * max(1, config.quantization_bits) / 8)
    budget_bytes = int(config.edge_vram_mb * 1024 * 1024 * config.edge_vram_util_target)
    device_count = max(1, len(jax.devices()))
    per_device_batch = math.ceil(int(config.batch_size) / device_count)
    recurrent_state_bytes = (
        int(config.num_layers)
        * int(config.memory_key_size)
        * int(config.memory_value_size)
        * np.dtype(np.float32).itemsize
    )
    recurrent_state_bytes_total_batch = recurrent_state_bytes * int(config.batch_size)
    scan_memory_matrix_bytes_per_layer_device = (
        int(config.train_unroll_len)
        * per_device_batch
        * int(config.memory_key_size)
        * int(config.memory_value_size)
        * np.dtype(np.float32).itemsize
    )
    scan_memory_matrix_bytes_all_layers_device = scan_memory_matrix_bytes_per_layer_device * int(config.num_layers)
    report = {
        "params": params,
        "training_param_bytes": fp_bytes,
        "rough_adamw_optimizer_state_bytes": fp_bytes * 2,
        "architecture": {
            "associative_groups": config.associative_groups,
            "use_swiglu": config.use_swiglu,
            "moe_num_experts": config.moe_num_experts,
            "moe_top_k": config.moe_top_k,
            "rope_base": config.rope_base,
            "rope_position_scale": config.rope_position_scale,
            "rope_max_position": config.rope_max_position,
            "image_recognition_only": config.image_recognition_only,
            "image_input_resolution": config.image_input_resolution,
            "image_max_input_resolution": config.image_max_input_resolution,
            "image_patch_size": config.image_patch_size,
        },
        "serving_recurrent_state_bytes_batch_1": recurrent_state_bytes,
        "training_recurrent_state_bytes_total_batch": recurrent_state_bytes_total_batch,
        "training_recurrent_state_bytes_per_device_if_sharded": math.ceil(recurrent_state_bytes_total_batch / device_count),
        "jax_device_count": device_count,
        "per_device_batch_if_evenly_sharded": per_device_batch,
        "remat_scan_step": bool(config.remat_scan_step),
        "scan_memory_matrix_bytes_per_layer_per_device": scan_memory_matrix_bytes_per_layer_device,
        "scan_memory_matrix_bytes_all_layers_per_device": scan_memory_matrix_bytes_all_layers_device,
        "quantization_bits": config.quantization_bits,
        "estimated_quantized_param_bytes": q_bytes,
        "edge_vram_mb": config.edge_vram_mb,
        "edge_vram_util_target": config.edge_vram_util_target,
        "target_budget_bytes": budget_bytes,
        "estimated_param_budget_fraction": q_bytes / max(1, budget_bytes),
        "fits_param_budget": q_bytes <= budget_bytes,
        "note": "Quantized edge memory is not a training-memory estimate. Without scan rematerialization, XLA may retain time-expanded memory matrices shaped like [unroll, per_device_batch, memory_key_size, memory_value_size] many times across layers and gradients. Image resolution is a preprocessing/runtime setting and is not injected into user prompts.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "edge_memory_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_info(f"Edge memory report: {json.dumps(report, ensure_ascii=False)}")


def copytree_replace(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    rsync = shutil.which("rsync")
    if rsync:
        subprocess.run([rsync, "-a", "--delete", f"{src}/", f"{tmp}/"], check=True)
    else:
        shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def prune_gcs_backups(root: Path) -> None:
    backups = []
    for child in root.glob("step_*"):
        try:
            step = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        backups.append((step, child))
    backups.sort(reverse=True)
    for _, path in backups[config.gcs_backup_keep :]:
        shutil.rmtree(path, ignore_errors=True)


def parse_gcs_backup_base() -> tuple[str, str]:
    raw = (config.gcs_backup_dir or "").strip()
    if not raw:
        raw = os.environ.get("GCS_BACKUP_DIR", "").strip()
    if not raw:
        raise RuntimeError("GCS backup target is not configured; set GCS_BACKUP_DIR or --gcs-backup-dir")

    if raw.startswith("/gcs/"):
        return "", str(Path(raw))

    if not raw.startswith("gs://"):
        return "", str(Path(raw))

    without_scheme = raw[len("gs://") :].strip("/")
    bucket, _, prefix = without_scheme.partition("/")
    target = f"gs://{bucket}"
    if prefix:
        target = f"{target}/{prefix.strip('/')}"
    return bucket, target


def parse_gcs_backup_target(step: int) -> tuple[str, str]:
    bucket, base_target = parse_gcs_backup_base()
    if base_target.startswith("gs://"):
        return bucket, f"{base_target.rstrip('/')}/sync_step_{step}"
    return bucket, str(Path(base_target) / f"sync_step_{step}")


def gcs_backup_enabled() -> bool:
    return bool((config.gcs_backup_dir or os.environ.get("GCS_BACKUP_DIR", "")).strip())


def create_gcs_bucket_if_needed(bucket: str) -> None:
    if not bucket:
        return

    location = os.environ.get("GCS_BUCKET_LOCATION", "us-central1")
    gsutil = shutil.which("gsutil")
    gcloud = shutil.which("gcloud")
    if gsutil:
        mb_cmd = [gsutil, "mb", "-l", location, f"gs://{bucket}"]
    elif gcloud:
        mb_cmd = [gcloud, "storage", "buckets", "create", f"gs://{bucket}", f"--location={location}"]
    else:
        raise RuntimeError("gsutil or gcloud is required for gs:// backup targets")

    result = subprocess.run(mb_cmd, text=True, capture_output=True)
    if result.returncode == 0:
        log_info(f"[GCS] Created bucket gs://{bucket}")
        return

    stderr = (result.stderr or "") + (result.stdout or "")
    already_exists = "already exists" in stderr.lower() or "you already own it" in stderr.lower()
    if not already_exists and gsutil and gcloud:
        fallback_cmd = [gcloud, "storage", "buckets", "create", f"gs://{bucket}", f"--location={location}"]
        result = subprocess.run(fallback_cmd, text=True, capture_output=True)
        stderr = (result.stderr or "") + (result.stdout or "")
        already_exists = "already exists" in stderr.lower() or "you already own it" in stderr.lower()
        if result.returncode == 0:
            log_info(f"[GCS] Created bucket gs://{bucket}")
            return
    if not already_exists:
        log_info(f"[GCS] Bucket create skipped/failed for gs://{bucket}: {stderr.strip()}")


def gcs_rsync_command(source: Path, target: str, *, delete: bool = True) -> list[str]:
    exclude = r"(^|/)(\.venv|__pycache__|\.git)(/|$)|(^|/)outputs/cache(/|$)"
    
    gcloud = shutil.which("gcloud")
    if gcloud:
        cmd = [gcloud, "storage", "rsync", str(source), target, "--recursive", "--exclude", exclude]
        if delete:
            cmd.append("--delete-unmatched-destination-objects")
        return cmd

    raise RuntimeError("gcloud is required for gs:// backup targets")


def backup_child_target(target: str, child: str) -> str:
    child_clean = child.strip("/")
    if target.startswith("gs://"):
        return f"{target.rstrip('/')}/{child_clean}"
    return str(Path(target) / child_clean)


def sync_backup_dir(source: Path, target: str, bucket: str) -> None:
    if target.startswith("gs://"):
        create_gcs_bucket_if_needed(bucket)
        cmd = gcs_rsync_command(source, target)
        subprocess.run(cmd, check=True)
    else:
        copytree_replace(source, Path(target))


def sync_backup_file(source: Path, target: str, bucket: str) -> None:
    if target.startswith("gs://"):
        create_gcs_bucket_if_needed(bucket)
        gcloud = shutil.which("gcloud")
        if gcloud:
            subprocess.run([gcloud, "storage", "cp", str(source), target], check=True)
            return
        gsutil = shutil.which("gsutil")
        if gsutil:
            subprocess.run([gsutil, "cp", str(source), target], check=True)
            return
        raise RuntimeError("gsutil or gcloud is required for gs:// backup targets")

    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)


def sync_checkpoint_to_gcs(step: int, step_dir: Path) -> None:
    bucket, base_target = parse_gcs_backup_base()
    checkpoint_dir = step_dir.resolve() / "checkpoint"
    if not checkpoint_dir.is_dir():
        log_info(f"[GCS] Checkpoint source missing, skipped: {checkpoint_dir}")
        return

    versioned_target = backup_child_target(parse_gcs_backup_target(step)[1], "checkpoint")
    latest_target = backup_child_target(base_target, "latest_checkpoint")
    log_info(f"[GCS] Syncing checkpoint for step {step} to {versioned_target} ...")
    sync_backup_dir(checkpoint_dir, versioned_target, bucket)
    log_info(f"[GCS] Syncing latest checkpoint pointer to {latest_target} ...")
    sync_backup_dir(checkpoint_dir, latest_target, bucket)

    with tempfile.TemporaryDirectory(prefix="propagator-latest-checkpoint-") as tmp:
        marker = Path(tmp) / "latest_checkpoint_step.txt"
        marker.write_text(f"{step}\n", encoding="utf-8")
        sync_backup_file(marker, backup_child_target(base_target, "latest_checkpoint_step.txt"), bucket)

    log_info(f"[GCS] Checkpoint backup complete for step {step}")


def start_gcs_backup(step: int, source_dir: Path) -> None:
    def worker() -> None:
        try:
            bucket, target = parse_gcs_backup_target(step)
            project_dir = Path(__file__).resolve().parent
            source_dir_resolved = source_dir.resolve()
            project_target = backup_child_target(target, "project")
            output_target = backup_child_target(target, "output")
            log_info(f"[GCS] Syncing project folder to {project_target} ...")

            sync_backup_dir(project_dir, project_target, bucket)

            if source_dir_resolved.exists():
                log_info(f"[GCS] Syncing training output to {output_target} ...")
                sync_backup_dir(source_dir_resolved, output_target, bucket)
            else:
                log_info(f"[GCS] Training output source missing, skipped: {source_dir_resolved}")

            log_info(f"[GCS] Backup complete for step {step} to {target}")
        except Exception as exc:
            log_info(f"[GCS] Backup failed for step {step}: {exc}")

    if config.gcs_async_backup:
        thread = threading.Thread(target=worker, name=f"gcs-backup-{step}", daemon=False)
        thread.start()
        _backup_threads.append(thread)
    else:
        worker()


def wait_for_backups() -> None:
    for thread in list(_backup_threads):
        thread.join()


atexit.register(wait_for_backups)


def prune_local_eval_dirs(output_root: Path, current_step: int) -> None:
    keep = max(0, config.local_eval_keep)
    if keep <= 0:
        return
    eval_dirs = []
    for child in output_root.glob("step_*"):
        if not child.is_dir():
            continue
        try:
            step = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        is_checkpoint_step = step % max(1, config.checkpoint_every) == 0 or step == current_step
        if not is_checkpoint_step:
            eval_dirs.append((step, child))
    eval_dirs.sort(reverse=True)
    for _, path in eval_dirs[keep:]:
        shutil.rmtree(path, ignore_errors=True)


def remove_incomplete_checkpoint_dirs(output_dir: Path) -> None:
    for tmp_path in output_dir.glob("checkpoint.orbax-checkpoint-tmp*"):
        if tmp_path.is_dir():
            log_info(f"[Checkpoint] Removing incomplete checkpoint temp dir: {tmp_path}")
            shutil.rmtree(tmp_path, ignore_errors=True)


def prune_local_checkpoint_dirs(output_root: Path) -> None:
    keep = max(1, config.local_checkpoint_keep)
    checkpoint_dirs = []
    for child in output_root.glob("step_*"):
        if not child.is_dir():
            continue
        try:
            step = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        if (child / "checkpoint").is_dir():
            checkpoint_dirs.append((step, child))

    checkpoint_dirs.sort(reverse=True)
    for _, path in checkpoint_dirs[keep:]:
        checkpoint_path = path / "checkpoint"
        log_info(f"[Checkpoint] Pruning old local checkpoint: {checkpoint_path}")
        shutil.rmtree(checkpoint_path, ignore_errors=True)


def append_metrics_jsonl(output_root: Path, record: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_global_token_ids() -> None:
    global token_ids_pad, token_ids_unk, token_ids_session, token_ids_user, token_ids_model
    global token_ids_listen, token_ids_user_end, token_ids_model_end, token_ids_session_end, token_ids_user_interrupt
    global token_ids_audio_in, token_ids_audio_out, token_ids_audio_end, token_ids_silence, token_ids_text_in, token_ids_text_out
    global token_ids_hybrid_out, token_ids_image_in

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
    token_ids_audio_in = token_ids["audio_in"]
    token_ids_audio_out = token_ids["audio_out"]
    token_ids_audio_end = token_ids["audio_end"]
    token_ids_silence = token_ids["silence"]
    token_ids_text_in = token_ids["text_in"]
    token_ids_text_out = token_ids["text_out"]
    token_ids_hybrid_out = token_ids["hybrid_out"]
    token_ids_image_in = token_ids["image_in"]


def main() -> None:
    global config, train_input_tokens, train_target_tokens, train_loss_weights, train_stream_ids, train_chunk_positions
    global val_input_tokens, val_target_tokens, val_loss_weights, val_stream_ids, val_chunk_positions, val_chunk_task_ids
    global train_control_input_tokens, train_control_target_tokens, train_control_loss_weights, train_control_stream_ids
    global train_control_chunk_positions, train_control_chunk_task_ids
    global val_control_input_tokens, val_control_target_tokens, val_control_loss_weights, val_control_stream_ids
    global val_control_chunk_positions, val_control_chunk_task_ids
    global candidate_token_ids_host, audio_candidate_token_ids_host

    log_info(f"[{datetime.now().isoformat()}] Main started. Initializing config and tokenization...")
    config = build_config()
    install_signal_handlers()
    acquire_run_lock()

    # Load tokenizer and datasets BEFORE JAX initialization to avoid fork deadlocks
    loaded = load_tokenizer_and_datasets()

    # Initialize JAX and sharding
    log_info(f"Tokenization done. Initializing JAX...")
    configure_runtime_environment()

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
    val_chunk_task_ids = build_chunk_task_ids(val_input_tokens, val_target_tokens, val_stream_ids)
    (
        train_control_input_tokens,
        train_control_target_tokens,
        train_control_loss_weights,
        train_control_stream_ids,
        train_control_chunk_positions,
    ) = build_synthetic_control_chunks(
        "train",
        int(config.synthetic_control_train_examples),
        9_000_000_000,
    )
    train_control_chunk_task_ids = build_chunk_task_ids(
        train_control_input_tokens,
        train_control_target_tokens,
        train_control_stream_ids,
    )
    (
        val_control_input_tokens,
        val_control_target_tokens,
        val_control_loss_weights,
        val_control_stream_ids,
        val_control_chunk_positions,
    ) = build_synthetic_control_chunks(
        "val",
        int(config.synthetic_control_val_examples),
        9_100_000_000,
    )
    val_control_chunk_task_ids = build_chunk_task_ids(
        val_control_input_tokens,
        val_control_target_tokens,
        val_control_stream_ids,
    )

    candidate_token_ids_host = build_candidate_token_ids(vocab_size)
    audio_candidate_token_ids_host = build_audio_candidate_token_ids()

    log_info(f"Tokenizer path: {config.tokenizer_path}")
    log_info(f"Tokenizer text vocab size: {text_vocab_size}")
    log_info(f"Model vocab size: {vocab_size}")
    log_info(f"Audio token range: [{audio_token_start}, {audio_token_end})")
    log_info(f"Image token range: [{image_token_start}, {image_token_end})")
    log_info(f"Tokenizer fingerprint: {tokenizer_fingerprint}")
    log_info(f"Token ids: {json.dumps(token_ids, ensure_ascii=False, indent=2)}")
    log_info(f"Candidate inference head size: {len(candidate_token_ids_host)} / {vocab_size}")
    log_info(f"Audio eval candidate head size: {len(audio_candidate_token_ids_host)} / {vocab_size}")
    log_info(f"Stateful train: {config.stateful_train}, stateful validation: {config.stateful_validation}")

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for step_dir in output_root.glob("step_*"):
        if step_dir.is_dir():
            remove_incomplete_checkpoint_dirs(step_dir)
    prune_local_checkpoint_dirs(output_root)

    model = PropagatorModel(config, vocab_size, nnx.Rngs(config.seed))
    if int(config.batch_size) <= 0:
        config = config.model_copy(update={"batch_size": choose_auto_batch_size(model)})
    elif len(jax.devices()) > 1 and config.enable_data_sharding and int(config.batch_size) % len(jax.devices()) != 0:
        adjusted = int(math.ceil(int(config.batch_size) / len(jax.devices())) * len(jax.devices()))
        log_info(f"[Batch] Adjusting batch_size from {config.batch_size} to {adjusted} for {len(jax.devices())} devices")
        config = config.model_copy(update={"batch_size": adjusted})

    steps_per_epoch = max(1, len(train_input_tokens) // config.batch_size)
    epoch_total_steps = config.epochs * steps_per_epoch
    total_steps = epoch_total_steps
    if int(config.max_steps) > 0:
        total_steps = min(epoch_total_steps, int(config.max_steps))
    log_info(
        f"[Schedule] steps_per_epoch={steps_per_epoch}, epoch_total_steps={epoch_total_steps}, "
        f"max_steps={int(config.max_steps)}, total_steps={total_steps}"
    )
    (output_root / "run_config.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    optimizer = nnx.Optimizer(model, build_optimizer(total_steps), wrt=nnx.Param)
    checkpointer = ocp.StandardCheckpointer()

    start_step = 0
    train_loss_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    val_decision_accs: list[float] = []
    val_decision_macro_accs: list[float] = []
    val_user_end_accs: list[float] = []
    val_interrupt_accs: list[float] = []
    val_text_accs: list[float] = []
    val_audio_accs: list[float] = []
    val_audio_codebook_accs: list[float] = []
    val_audio_aux_token_accs: list[float] = []
    val_audio_all_codebook_frame_accs: list[float] = []
    val_main_ces: list[float] = []
    val_aux_audio_ces: list[float] = []
    val_text_task_accs: list[float] = []
    val_asr_task_accs: list[float] = []
    val_tts_task_accs: list[float] = []
    val_duplex_task_accs: list[float] = []
    val_image_task_accs: list[float] = []
    val_image_task_ces: list[float] = []
    val_composite_scores: list[float] = []
    best_early_stop_score = float("-inf")
    evals_without_improvement = 0

    if config.resume_checkpoint:
        # Resume from latest checkpoint if exists.
        ckpt_dirs = sorted(list(output_root.glob("step_*")), key=lambda x: int(x.name.split("_")[1]), reverse=True)
        for ckpt_dir in ckpt_dirs:
            ckpt_path = ckpt_dir / "checkpoint"
            if ckpt_path.exists():
                try:
                    step = int(ckpt_dir.name.split("_")[1])
                    log_info(f"[Checkpoint] Resuming from step {step} at {ckpt_path}")
                    restored_kind = restore_training_checkpoint(checkpointer, ckpt_path, model, optimizer)
                    log_info(f"[Checkpoint] Restored {restored_kind}")
                    start_step = step

                    # Try to recover training metrics history.
                    metrics_path = output_root / "metrics.jsonl"
                    if metrics_path.exists():
                        train_by_step: dict[int, float] = {}
                        val_by_step: dict[int, tuple[float, dict[str, float]]] = {}
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    record = json.loads(line)
                                    s = int(record.get("step", 0))
                                    if s <= start_step:
                                        train_by_step[s] = float(record.get("train_loss", 0.0))
                                        if "val_loss" in record:
                                            val_by_step[s] = (float(record["val_loss"]), record.get("metrics", {}))
                                except Exception:
                                    continue
                        for s in sorted(train_by_step):
                            train_loss_steps.append(s)
                            train_losses.append(train_by_step[s])
                        for s in sorted(val_by_step):
                            v_loss, m = val_by_step[s]
                            val_steps.append(s)
                            val_losses.append(v_loss)
                            val_decision_accs.append(m.get("decision_acc", 0.0))
                            val_decision_macro_accs.append(m.get("decision_macro_acc", float("nan")))
                            val_user_end_accs.append(m.get("user_end_acc", 0.0))
                            val_interrupt_accs.append(m.get("interrupt_acc", float("nan")))
                            val_text_accs.append(m.get("text_token_acc", 0.0))
                            val_audio_accs.append(m.get("audio_token_acc", 0.0))
                            val_audio_codebook_accs.append(m.get("audio_codebook_acc", 0.0))
                            val_audio_aux_token_accs.append(m.get("audio_aux_token_acc", float("nan")))
                            val_audio_all_codebook_frame_accs.append(m.get("audio_all_codebook_frame_exact_acc", float("nan")))
                            val_main_ces.append(m.get("main_ce", float("nan")))
                            val_aux_audio_ces.append(m.get("aux_audio_ce", float("nan")))
                            val_text_task_accs.append(m.get("text_task_acc", 0.0))
                            val_asr_task_accs.append(m.get("asr_task_acc", 0.0))
                            val_tts_task_accs.append(m.get("tts_task_acc", 0.0))
                            val_duplex_task_accs.append(m.get("duplex_task_acc", 0.0))
                            val_image_task_accs.append(m.get("image_task_acc", 0.0))
                            val_image_task_ces.append(m.get("image_task_ce", float("nan")))
                            composite_record = validation_composite_score(m)
                            val_composite_scores.append(
                                float(m.get("validation_composite_score", composite_record["validation_composite_score"]))
                            )
                        if val_composite_scores:
                            best_early_stop_score = max(val_composite_scores)
                    break
                except Exception as e:
                    log_info(f"[Checkpoint] Failed to restore from {ckpt_path}: {e}")
    else:
        log_info("[Checkpoint] Resume disabled; starting from fresh model and optimizer state")

    write_edge_memory_report(model, output_root)

    shuffled = shuffle_data_for_epoch(start_step // steps_per_epoch)

    if config.stateful_train:
        train_sampler = StatefulChunkSampler(
            train_stream_ids,
            config.batch_size,
            config.seed + start_step,
            source_weights=[float(spec["weight"]) for spec in parse_dataset_mix()],
        )
        source_specs = parse_dataset_mix()
        source_lane_summary = {
            f"{idx}:{source_specs[idx]['name']}": count
            for idx, count in train_sampler.source_lane_counts.items()
            if idx < len(source_specs)
        }
        log_info(f"[Sampler] Stateful source lanes: {json.dumps(source_lane_summary, ensure_ascii=False)}")
        carry_memories = initial_memories_for_training(model, config.batch_size)
    else:
        train_sampler = None
        carry_memories = None

    train_step_stateful_fn = build_train_step_stateful()
    validation_step_stateful_fn = build_validation_step_stateful()

    pbar = progress_bar(range(start_step, total_steps), desc="Training", initial=start_step, total=total_steps)
    train_wall_start = time.time()
    last_train_log_time = train_wall_start
    last_train_log_step = start_step
    sharding_retry_count = 0

    for step in pbar:
        should_early_stop = False
        if not config.stateful_train and step > 0 and step % steps_per_epoch == 0:
            shuffled = shuffle_data_for_epoch(step // steps_per_epoch)
            log_info(f"\n[Shuffle] Epoch {step // steps_per_epoch} started")

        if config.stateful_train:
            assert train_sampler is not None
            assert carry_memories is not None

            indices, reset_mask_np = train_sampler.next_indices()
            batch_inputs_np = np.asarray(train_input_tokens[indices], dtype=np.int32).copy()
            batch_targets_np = np.asarray(train_target_tokens[indices], dtype=np.int32).copy()
            batch_weights_np = np.asarray(train_loss_weights[indices], dtype=np.float32).copy()
            chunk_positions_np = np.asarray(train_chunk_positions[indices], dtype=np.int32).copy()
            inject_train_control_examples(
                batch_inputs_np,
                batch_targets_np,
                batch_weights_np,
                reset_mask_np,
                chunk_positions_np,
                step,
            )
            batch_inputs = maybe_put_batch(batch_inputs_np, np.int32)
            batch_targets = maybe_put_batch(batch_targets_np, np.int32)
            batch_weights = maybe_put_batch(batch_weights_np, np.float32)
            reset_mask = maybe_put_vector(reset_mask_np, np.bool_)
            chunk_positions = maybe_put_vector(chunk_positions_np, np.int32)
            for attempt in (0, 1):
                try:
                    ce_loss_val, carry_memories = call_train_step_stateful(
                        train_step_stateful_fn,
                        model,
                        optimizer,
                        batch_inputs,
                        batch_targets,
                        batch_weights,
                        carry_memories,
                        reset_mask,
                        chunk_positions,
                    )
                    break
                except ValueError as exc:
                    msg = str(exc)
                    is_jit_sharding_error = (
                        "Sharding passed to pjit does not match the sharding on the respective arg." in msg
                        or "Received incompatible devices for jitted computation." in msg
                    )
                    if (not is_jit_sharding_error) or attempt == 1:
                        raise
                    sharding_retry_count += 1
                    log_info(
                        f"[Train] Retrying stateful step at global step {step} after jit sharding mismatch "
                        f"#{sharding_retry_count}: {msg.splitlines()[0]}"
                    )
                    train_step_stateful_fn = build_train_step_stateful()
        else:
            batch_inputs, batch_targets, batch_weights = get_random_batch(step, shuffled)
            ce_loss_val = train_step_stateless(model, optimizer, batch_inputs, batch_targets, batch_weights)

        act_step = step + 1
        should_record_train_loss = (
            (config.train_log_every > 0 and act_step % config.train_log_every == 0)
            or (config.eval_every > 0 and act_step % config.eval_every == 0)
            or act_step == total_steps
        )
        if should_record_train_loss:
            latest_train_loss = float(ce_loss_val)
            train_loss_steps.append(act_step)
            train_losses.append(latest_train_loss)
            pbar.set_postfix({"loss": f"{latest_train_loss:.4f}"})

        if config.train_log_every > 0 and act_step % config.train_log_every == 0:
            now = time.time()
            elapsed = max(1e-6, now - train_wall_start)
            completed = act_step - start_step
            interval_steps = max(1, act_step - last_train_log_step)
            interval_elapsed = max(1e-6, now - last_train_log_time)
            avg_sps = completed / elapsed
            interval_sps = interval_steps / interval_elapsed
            remaining_steps = max(0, total_steps - act_step)
            eta_seconds = remaining_steps / max(1e-6, interval_sps)
            latest_train_loss = train_losses[-1] if train_losses else float("nan")
            log_info(
                f"[Train] step={act_step}/{total_steps}, loss={latest_train_loss:.4f}, "
                f"steps_per_sec={avg_sps:.4f}, interval_steps_per_sec={interval_sps:.4f}, "
                f"elapsed={format_duration(elapsed)}, eta={format_duration(eta_seconds)}"
            )
            last_train_log_time = now
            last_train_log_step = act_step

        if act_step % config.eval_every == 0:
            for attempt in (0, 1):
                try:
                    v_loss, v_metrics = run_validation(model, act_step, validation_step_stateful_fn)
                    break
                except ValueError as exc:
                    msg = str(exc)
                    is_jit_sharding_error = (
                        "Sharding passed to pjit does not match the sharding on the respective arg." in msg
                        or "Received incompatible devices for jitted computation." in msg
                    )
                    if (not is_jit_sharding_error) or attempt == 1:
                        raise
                    validation_step_stateful_fn = build_validation_step_stateful()
                    log_info(
                        f"[Validation] Retrying stateful validation at global step {act_step} after jit sharding mismatch "
                        f"#{attempt + 1}: {msg.splitlines()[0]}"
                    )

            v_metrics.update(validation_composite_score(v_metrics))
            v_score = float(v_metrics["validation_composite_score"])
            if v_score > best_early_stop_score + float(config.early_stopping_min_delta):
                best_early_stop_score = v_score
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            should_early_stop = (
                int(config.early_stopping_patience) > 0
                and evals_without_improvement >= int(config.early_stopping_patience)
            )

            val_steps.append(act_step)
            val_losses.append(v_loss)
            val_decision_accs.append(v_metrics["decision_acc"])
            val_decision_macro_accs.append(v_metrics.get("decision_macro_acc", float("nan")))
            val_user_end_accs.append(v_metrics["user_end_acc"])
            val_interrupt_accs.append(v_metrics.get("interrupt_acc", float("nan")))
            val_text_accs.append(v_metrics["text_token_acc"])
            val_audio_accs.append(v_metrics["audio_token_acc"])
            val_audio_codebook_accs.append(v_metrics["audio_codebook_acc"])
            val_audio_aux_token_accs.append(v_metrics.get("audio_aux_token_acc", float("nan")))
            val_audio_all_codebook_frame_accs.append(v_metrics.get("audio_all_codebook_frame_exact_acc", float("nan")))
            val_main_ces.append(v_metrics.get("main_ce", float("nan")))
            val_aux_audio_ces.append(v_metrics.get("aux_audio_ce", float("nan")))
            val_text_task_accs.append(v_metrics.get("text_task_acc", float("nan")))
            val_asr_task_accs.append(v_metrics.get("asr_task_acc", float("nan")))
            val_tts_task_accs.append(v_metrics.get("tts_task_acc", float("nan")))
            val_duplex_task_accs.append(v_metrics.get("duplex_task_acc", float("nan")))
            val_image_task_accs.append(v_metrics.get("image_task_acc", float("nan")))
            val_image_task_ces.append(v_metrics.get("image_task_ce", float("nan")))
            val_composite_scores.append(v_score)

            out_dir = output_root / f"step_{act_step}"
            out_dir.mkdir(parents=True, exist_ok=True)

            save_metric_plot(train_loss_steps, train_losses, out_dir / "train_loss.png", "Train weighted CE", act_step)
            save_metric_plot(val_steps, val_losses, out_dir / "val_loss.png", "Validation weighted CE", act_step)
            save_metric_plot(val_steps, val_decision_accs, out_dir / "val_decision_acc.png", "Validation decision accuracy", act_step)
            save_metric_plot(
                val_steps,
                val_decision_macro_accs,
                out_dir / "val_decision_macro_acc.png",
                "Validation decision macro accuracy",
                act_step,
            )
            save_metric_plot(val_steps, val_user_end_accs, out_dir / "val_user_end_acc.png", "Validation user_end accuracy", act_step)
            save_metric_plot(val_steps, val_interrupt_accs, out_dir / "val_interrupt_acc.png", "Validation interrupt accuracy", act_step)
            save_metric_plot(val_steps, val_text_accs, out_dir / "val_text_token_acc.png", "Validation text token accuracy", act_step)
            save_metric_plot(val_steps, val_audio_accs, out_dir / "val_audio_token_acc.png", "Validation audio token accuracy", act_step)
            save_metric_plot(
                val_steps,
                val_audio_codebook_accs,
                out_dir / "val_audio_codebook_acc.png",
                "Validation aux frame exact accuracy",
                act_step,
            )
            save_metric_plot(
                val_steps,
                val_audio_aux_token_accs,
                out_dir / "val_audio_aux_token_acc.png",
                "Validation aux codebook token accuracy",
                act_step,
            )
            save_metric_plot(
                val_steps,
                val_audio_all_codebook_frame_accs,
                out_dir / "val_audio_all_codebook_frame_acc.png",
                "Validation all-codebook frame exact accuracy",
                act_step,
            )
            save_metric_plot(val_steps, val_main_ces, out_dir / "val_main_ce.png", "Validation main CE", act_step)
            save_metric_plot(val_steps, val_aux_audio_ces, out_dir / "val_aux_audio_ce.png", "Validation aux audio CE", act_step)
            save_metric_plot(val_steps, val_text_task_accs, out_dir / "val_text_task_acc.png", "Validation Text Task Accuracy", act_step)
            save_metric_plot(val_steps, val_asr_task_accs, out_dir / "val_asr_task_acc.png", "Validation ASR Task Accuracy", act_step)
            save_metric_plot(val_steps, val_tts_task_accs, out_dir / "val_tts_task_acc.png", "Validation TTS Task Accuracy", act_step)
            save_metric_plot(val_steps, val_duplex_task_accs, out_dir / "val_duplex_task_acc.png", "Validation Duplex Task Accuracy", act_step)
            save_metric_plot(val_steps, val_image_task_accs, out_dir / "val_image_task_acc.png", "Validation Image Task Accuracy", act_step)
            save_metric_plot(val_steps, val_image_task_ces, out_dir / "val_image_task_ce.png", "Validation Image Task CE", act_step)
            save_metric_plot(
                val_steps,
                val_composite_scores,
                out_dir / "val_composite_score.png",
                "Validation Propagator Composite Score",
                act_step,
            )

            text_meta = generate_text_eval_samples(
                model,
                config.seed + 30_000,
                out_dir,
                use_candidate_head=config.eval_use_candidate_head,
            )
            image_meta = generate_image_eval_samples(
                model,
                config.seed + 40_000,
                out_dir,
                use_candidate_head=config.eval_use_candidate_head,
            )
            audio_meta = None
            audio_input_meta = None
            if config.enable_audio and config.eval_audio_every > 0 and act_step % config.eval_audio_every == 0:
                audio_meta = generate_audio_evals(model, config.seed + 99_000, out_dir)
                audio_input_meta = generate_audio_input_evals(model, config.seed + 199_000, out_dir)

            (out_dir / "validation_metrics.json").write_text(
                json.dumps(v_metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            append_metrics_jsonl(
                output_root,
                {
                    "step": act_step,
                    "train_loss": train_losses[-1] if train_losses else float("nan"),
                    "val_loss": v_loss,
                    "metrics": v_metrics,
                    "text_eval": text_meta,
                    "image_eval": image_meta,
                    "audio_eval": audio_meta,
                    "audio_input_eval": audio_input_meta,
                    "time": time.time(),
                },
            )
            prune_local_eval_dirs(output_root, act_step)

            log_info(f"\n[Eval] CE={v_loss:.4f}, composite={v_score:.4f}")
            log_info(
                f"[Early Stop] best_composite={best_early_stop_score:.4f}, "
                f"without_improvement={evals_without_improvement}/{config.early_stopping_patience}"
            )
            log_info(json.dumps(v_metrics, ensure_ascii=False, indent=2))
            if audio_meta is not None:
                audio_summary = [
                    {k: item[k] for k in ("sample_idx", "num_generated_tokens", "num_audio_tokens", "decode_error", "wav_path")}
                    for item in audio_meta
                ]
                log_info(f"[Audio Eval] {json.dumps(audio_summary, ensure_ascii=False)}")
            if audio_input_meta is not None:
                input_summary = {
                    "asr_samples": len(audio_input_meta.get("asr", [])),
                    "audio_to_audio_samples": len(audio_input_meta.get("audio_to_audio", [])),
                }
                log_info(f"[Audio Input Eval] {json.dumps(input_summary, ensure_ascii=False)}")
            log_info(f"[Sample] wrote {len(text_meta)} text samples to {out_dir}")
            log_info(f"[Image Eval] wrote {len(image_meta)} image samples to {out_dir}")

        if act_step % config.checkpoint_every == 0 or act_step == total_steps or should_early_stop:
            ckpt_dir = output_root / f"step_{act_step}"
            save_checkpoint(checkpointer, model, optimizer, ckpt_dir)
            checkpoint_backup_ok = True
            if gcs_backup_enabled() and config.gcs_sync_every > 0 and (
                act_step % config.gcs_sync_every == 0 or should_early_stop
            ):
                try:
                    sync_checkpoint_to_gcs(act_step, ckpt_dir)
                except Exception as exc:
                    checkpoint_backup_ok = False
                    log_info(f"[GCS] Checkpoint backup failed for step {act_step}: {exc}")
            if checkpoint_backup_ok:
                prune_local_checkpoint_dirs(output_root)
            else:
                log_info("[Checkpoint] Local checkpoint pruning skipped because GCS checkpoint backup failed")

        if gcs_backup_enabled() and config.gcs_sync_every > 0 and (
            act_step % config.gcs_sync_every == 0 or should_early_stop
        ):
            start_gcs_backup(act_step, output_root / f"step_{act_step}")

        if should_early_stop:
            log_info(
                f"[Early Stop] stopping at step {act_step}: validation composite did not improve by "
                f"{config.early_stopping_min_delta} for {config.early_stopping_patience} evaluations"
            )
            break

    wait_for_backups()


if __name__ == "__main__":
    main()

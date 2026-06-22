import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path("/mnt/disks/propagator-cache/cache")
CACHE_DIR = Path(os.environ.get("PROPAGATOR_CACHE_DIR", DEFAULT_CACHE_DIR))
TMP_DIR = Path(os.environ.get("HF_UPLOAD_TMP_DIR", "/mnt/disks/propagator-cache/hf_upload_tmp"))
SHARD_BYTES = int(os.environ.get("HF_UPLOAD_SHARD_BYTES", str(16 * 1024**3)))
COPY_BUFFER_BYTES = int(os.environ.get("HF_UPLOAD_COPY_BUFFER_BYTES", str(64 * 1024**2)))
PUBLIC_DATASET = os.environ.get("HF_DATASET_PUBLIC", "1").lower() not in {"0", "false", "no", "off"}
DELETE_UNSHARDED_REMOTE = os.environ.get("HF_DELETE_UNSHARDED_REMOTE", "1").lower() not in {"0", "false", "no", "off"}
SKIP_EXISTING = os.environ.get("HF_UPLOAD_SKIP_EXISTING", "1").lower() not in {"0", "false", "no", "off"}
DEFAULT_DATASET_REPO_NAME = "propagator-multimodal-pretraining-data"

DATA_SUFFIXES = (
    ".input.bin",
    ".target.bin",
    ".weight.bin",
    ".stream_id.bin",
    ".chunk_pos.bin",
    ".meta.json",
)


def load_token() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in environment or .env", flush=True)
        sys.exit(1)
    return token


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def read_meta(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"meta_read_error": str(exc)}


def completed_cache_groups(cache_dir: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for meta_path in sorted(cache_dir.glob("*.meta.json")):
        prefix = meta_path.name[: -len(".meta.json")]
        files = [cache_dir / f"{prefix}{suffix}" for suffix in DATA_SUFFIXES]
        existing = [path for path in files if path.exists()]
        if meta_path not in existing:
            continue
        groups.append(
            {
                "prefix": prefix,
                "files": existing,
                "missing_files": [path.name for path in files if not path.exists()],
                "total_bytes": sum(file_size(path) for path in existing),
                "meta": read_meta(meta_path),
            }
        )
    return groups


def file_group_name(path: Path) -> str:
    name = path.name
    for suffix in DATA_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def is_binary_cache(path: Path) -> bool:
    return path.name.endswith(".bin")


def repo_path_for_whole_file(path: Path) -> str:
    group = file_group_name(path)
    return f"shards/{group}/{path.name}"


def repo_path_for_part(path: Path, part_index: int, part_count: int) -> str:
    group = file_group_name(path)
    return f"shards/{group}/{path.name}.part-{part_index:05d}-of-{part_count:05d}"


def build_manifest(groups: list[dict[str, Any]]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_shards = 0
    for group in groups:
        for path in group["files"]:
            size = file_size(path)
            part_count = math.ceil(size / SHARD_BYTES) if is_binary_cache(path) and size > SHARD_BYTES else 1
            total_shards += part_count
            files.append(
                {
                    "path": path.name,
                    "bytes": size,
                    "group": group["prefix"],
                    "sharded": part_count > 1,
                    "part_count": part_count,
                    "repo_paths": [
                        repo_path_for_part(path, idx, part_count) if part_count > 1 else repo_path_for_whole_file(path)
                        for idx in range(part_count)
                    ],
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(CACHE_DIR),
        "layout": "sharded-v1",
        "shard_bytes": SHARD_BYTES,
        "group_count": len(groups),
        "source_file_count": len(files),
        "repo_file_count": total_shards + 2,
        "total_bytes": sum(item["bytes"] for item in files),
        "groups": [
            {
                "prefix": group["prefix"],
                "total_bytes": group["total_bytes"],
                "missing_files": group["missing_files"],
                "dataset_name": group["meta"].get("dataset_name"),
                "dataset_config": group["meta"].get("dataset_config"),
                "dataset_mode": group["meta"].get("dataset_mode"),
                "source_rows": group["meta"].get("source_rows"),
                "token_stats": group["meta"].get("token_stats"),
            }
            for group in groups
        ],
        "files": files,
    }


SOURCE_DESCRIPTIONS = {
    "HuggingFaceFW/fineweb-edu": ("Language", "educational web text"),
    "wikimedia/wikipedia": ("Language", "encyclopedic long-form text"),
    "HuggingFaceM4/VQAv2": ("Vision-language", "image question answering and recognition"),
    "xinrongzhang2022/Duplex-UltraChat": ("Dialogue", "multi-turn conversational text"),
    "databricks/databricks-dolly-15k": ("Instruction", "instruction-following examples"),
    "shangeth/libritts-r-mimi-codes": ("Speech-language", "LibriTTS-R speech/text Mimi code examples"),
    "shangeth/librispeech-mimi-codes": ("Speech-language", "LibriSpeech speech/text Mimi code examples"),
    "shangeth/vctk-mimi-codes": ("Speech-language", "VCTK speech/text Mimi code examples"),
    "shangeth/jenny-mimi-codes": ("Speech-language", "Jenny speech/text Mimi code examples"),
    "shangeth/ljspeech-mimi-codes": ("Speech-language", "LJSpeech speech/text Mimi code examples"),
    "json": ("Curated", "local prepared text and vision-language examples"),
}


def hf_dataset_link(name: str) -> str:
    if name == "json":
        return "Local prepared rows"
    return f"[{name}](https://huggingface.co/datasets/{name})"


def source_table(manifest: dict[str, Any]) -> str:
    rows: dict[str, dict[str, Any]] = {}
    for group in manifest["groups"]:
        name = group.get("dataset_name")
        if not name:
            continue
        item = rows.setdefault(name, {"rows": 0, "modes": set(), "configs": set()})
        item["rows"] += int(group.get("source_rows") or 0)
        if group.get("dataset_mode"):
            item["modes"].add(str(group["dataset_mode"]))
        if group.get("dataset_config"):
            item["configs"].add(str(group["dataset_config"]))

    order = [
        "HuggingFaceFW/fineweb-edu",
        "wikimedia/wikipedia",
        "HuggingFaceM4/VQAv2",
        "xinrongzhang2022/Duplex-UltraChat",
        "databricks/databricks-dolly-15k",
        "shangeth/libritts-r-mimi-codes",
        "shangeth/librispeech-mimi-codes",
        "shangeth/vctk-mimi-codes",
        "shangeth/jenny-mimi-codes",
        "shangeth/ljspeech-mimi-codes",
        "json",
    ]
    names = [name for name in order if name in rows] + sorted(name for name in rows if name not in order)
    lines = [
        "| Source dataset | Modality | Contribution | Prepared rows | Preprocessing mode |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for name in names:
        item = rows[name]
        modality, contribution = SOURCE_DESCRIPTIONS.get(name, ("Mixed", "processed training examples"))
        config_suffix = ""
        if item["configs"]:
            config_suffix = f" ({', '.join(sorted(item['configs']))})"
        modes = ", ".join(f"`{mode}`" for mode in sorted(item["modes"])) or "recorded in manifest"
        lines.append(
            f"| {hf_dataset_link(name)}{config_suffix} | {modality} | {contribution} | "
            f"{item['rows']:,} | {modes} |"
        )
    return "\n".join(lines)


def dataset_readme(manifest: dict[str, Any]) -> str:
    tib = manifest["total_bytes"] / 1024**4
    shard_gib = manifest["shard_bytes"] / 1024**3
    visibility = "public" if PUBLIC_DATASET else "private"
    sources = source_table(manifest)
    return f"""---
license: other
pretty_name: Propagator Multimodal Pretraining Data
language:
- en
tags:
- multimodal
- pretraining
- tokenized
- sharded
- text
- image
- speech
task_categories:
- text-generation
- question-answering
- image-to-text
- automatic-speech-recognition
configs:
- config_name: sharded
  data_files:
  - "shards/**/*.bin*"
  - "shards/**/*.json"
---

# Propagator Multimodal Pretraining Data

This {visibility} dataset contains tokenized multimodal pretraining data prepared for the Propagator model family. It combines language, image-grounded, and speech/audio-token examples into a single training format.

This is not a raw text or image browsing dataset. The examples have already been converted into compact binary token frames for model training, with a manifest that records the source groups and file layout.

## What's Included

- **Language:** web text, encyclopedic text, instruction-following, and dialogue data.
- **Vision-language:** image recognition and image question-answering style examples represented as image patch tokens plus text tokens.
- **Speech-language:** speech/text examples represented with Mimi-style audio code tokens for ASR, TTS, and duplex audio-text training.

## Source Datasets

{sources}

The table lists source families represented in the current prepared package. Exact split names, file groups, byte sizes, and reconstruction order are recorded in `propagator_cache_manifest.json`.

## Intended Use

This repository is intended for training or reproducing Propagator-style multimodal models that consume the packed Propagator frame format. It is useful if you want a ready-to-stream pretraining corpus rather than rebuilding tokenization and modality packing from the original upstream datasets.

It is not intended as a general-purpose dataset viewer, example gallery, or raw media archive.

## Format

Large binary cache files are split under `shards/<cache_group>/` as:

`<original-file>.part-00000-of-NNNNN`

The split layout keeps each object at about `{shard_gib:.0f} GiB`, which is friendlier for resumable upload/download and parallel reads. The manifest records the exact reconstruction order and original byte sizes.

Each prepared data group contains the same file family:

- `*.input.bin`: int32 token frames with shape `[num_chunks, unroll_length, 8]`.
- `*.target.bin`: int32 next-token target frames with shape `[num_chunks, unroll_length, 8]`.
- `*.weight.bin`: float32 loss weights with shape `[num_chunks, unroll_length]`.
- `*.stream_id.bin`: int64 stream identifiers for sequence boundaries.
- `*.chunk_pos.bin`: int32 chunk positions within each stream.
- `*.meta.json`: source rows, chunk counts, unroll length, frame width, and preprocessing metadata.

The first lane carries the main text/control stream. Additional lanes carry modality-specific codebooks, including image patch tokens and audio code tokens where applicable.

## Loading

For each file, read `propagator_cache_manifest.json` and concatenate the listed `repo_paths` in order, or stream those parts directly if your loader supports sharded reads. Validate the final byte count against the manifest before memory-mapping.

## Current Package

- Total prepared data: `{tib:.2f} TiB`
- Prepared data groups: `{manifest["group_count"]}`
- Original cache files: `{manifest["source_file_count"]}`
- Repository objects after splitting: `{manifest["repo_file_count"]}`
- Manifest: `propagator_cache_manifest.json`

## License and Source Terms

This dataset is a processed training artifact assembled from multiple upstream datasets. Check the upstream dataset licenses and terms listed in the manifest before redistribution or commercial use.
"""


def upload_text(api: HfApi, repo_id: str, repo_type: str, path_in_repo: str, text: str, message: str) -> None:
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=message,
    )


def ensure_public_dataset(api: HfApi, repo_id: str, token: str) -> None:
    create_repo(repo_id=repo_id, repo_type="dataset", private=not PUBLIC_DATASET, token=token, exist_ok=True)
    if PUBLIC_DATASET:
        api.update_repo_settings(repo_id, repo_type="dataset", private=False, token=token)
        print(f"Dataset repo is public: {repo_id}", flush=True)


def delete_unsharded_remote_files(api: HfApi, repo_id: str, existing_files: set[str]) -> set[str]:
    if not DELETE_UNSHARDED_REMOTE:
        return existing_files
    protected = {"README.md", "propagator_cache_manifest.json", ".gitattributes"}
    for path in sorted(existing_files):
        if path in protected or path.startswith("shards/"):
            continue
        print(f"DELETE remote unsharded file {path}", flush=True)
        api.delete_file(
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Delete unsharded {path}",
        )
        existing_files.discard(path)
    return existing_files


def upload_path(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    path: Path,
    path_in_repo: str,
    existing_files: set[str],
    message: str,
) -> bool:
    if SKIP_EXISTING and path_in_repo in existing_files:
        print(f"SKIP existing {path_in_repo}", flush=True)
        return False
    start = time.time()
    size_gib = file_size(path) / 1024**3
    print(f"UPLOAD {path_in_repo} ({size_gib:.2f} GiB)", flush=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=message,
    )
    elapsed = max(time.time() - start, 1e-6)
    print(f"DONE {path_in_repo} in {elapsed:.1f}s ({size_gib / elapsed * 3600:.2f} GiB/h)", flush=True)
    existing_files.add(path_in_repo)
    return True


def copy_range_to_temp(src: Path, offset: int, length: int, dst: Path) -> str:
    sha = hashlib.sha256()
    remaining = length
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        f_in.seek(offset)
        while remaining > 0:
            chunk = f_in.read(min(COPY_BUFFER_BYTES, remaining))
            if not chunk:
                break
            f_out.write(chunk)
            sha.update(chunk)
            remaining -= len(chunk)
    if remaining != 0:
        raise IOError(f"Failed to read requested range from {src}: remaining={remaining}")
    return sha.hexdigest()


def upload_sharded_file(api: HfApi, repo_id: str, path: Path, existing_files: set[str]) -> list[dict[str, Any]]:
    size = file_size(path)
    if not is_binary_cache(path) or size <= SHARD_BYTES:
        repo_path = repo_path_for_whole_file(path)
        upload_path(api, repo_id, "dataset", path, repo_path, existing_files, f"Upload {repo_path}")
        return [{"repo_path": repo_path, "offset": 0, "bytes": size, "sha256": None}]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    part_count = math.ceil(size / SHARD_BYTES)
    parts: list[dict[str, Any]] = []
    for part_index in range(part_count):
        offset = part_index * SHARD_BYTES
        length = min(SHARD_BYTES, size - offset)
        repo_path = repo_path_for_part(path, part_index, part_count)
        part_record = {"repo_path": repo_path, "offset": offset, "bytes": length, "sha256": None}
        if SKIP_EXISTING and repo_path in existing_files:
            print(f"SKIP existing {repo_path}", flush=True)
            parts.append(part_record)
            continue
        tmp_path = TMP_DIR / Path(repo_path).name
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"SHARD {path.name} part {part_index + 1}/{part_count} offset={offset} bytes={length}", flush=True)
        sha256 = copy_range_to_temp(path, offset, length, tmp_path)
        part_record["sha256"] = sha256
        try:
            upload_path(api, repo_id, "dataset", tmp_path, repo_path, existing_files, f"Upload shard {repo_path}")
            parts.append(part_record)
        finally:
            tmp_path.unlink(missing_ok=True)
    return parts


def write_local_upload_plan(manifest: dict[str, Any]) -> Path:
    out_path = PROJECT_ROOT / "outputs" / "propagator-multimodal" / "hf_sharded_upload_plan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    token = load_token()
    if not CACHE_DIR.exists():
        print(f"ERROR: cache dir not found: {CACHE_DIR}", flush=True)
        sys.exit(1)

    api = HfApi(token=token)
    username = api.whoami().get("name")
    if not username:
        print("ERROR: could not determine Hugging Face username", flush=True)
        sys.exit(1)

    tokenizer_repo = f"{username}/propagator-tokenizer"
    dataset_repo = os.environ.get(
        "HF_DATASET_REPO",
        f"{username}/{os.environ.get('HF_DATASET_REPO_NAME', DEFAULT_DATASET_REPO_NAME)}",
    )
    print(f"Authenticated as {username}", flush=True)
    print(f"Tokenizer repo: {tokenizer_repo}", flush=True)
    print(f"Dataset repo: {dataset_repo}", flush=True)
    print(f"Dataset visibility target: {'public' if PUBLIC_DATASET else 'private'}", flush=True)
    print(f"Shard size: {SHARD_BYTES / 1024**3:.2f} GiB", flush=True)

    create_repo(repo_id=tokenizer_repo, repo_type="model", private=True, token=token, exist_ok=True)
    ensure_public_dataset(api, dataset_repo, token)

    tokenizer_path = PROJECT_ROOT / "assets" / "tokenizer-byte-bpe-16000.json"
    tokenizer_existing = set(api.list_repo_files(tokenizer_repo, repo_type="model"))
    if tokenizer_path.exists():
        upload_path(api, tokenizer_repo, "model", tokenizer_path, "tokenizer.json", tokenizer_existing, "Upload tokenizer")
    upload_text(
        api,
        tokenizer_repo,
        "model",
        "README.md",
        "# Propagator Tokenizer\n\nByte-level BPE tokenizer with Propagator protocol and multimodal special tokens.\n",
        "Update tokenizer README",
    )

    groups = completed_cache_groups(CACHE_DIR)
    manifest = build_manifest(groups)
    plan_path = write_local_upload_plan(manifest)
    print(
        "Completed cache groups: "
        f"{manifest['group_count']} groups, {manifest['source_file_count']} source files, "
        f"{manifest['repo_file_count']} repo files, {manifest['total_bytes'] / 1024**4:.2f} TiB",
        flush=True,
    )
    print(f"Local upload plan: {plan_path}", flush=True)

    existing_dataset_files = set(api.list_repo_files(dataset_repo, repo_type="dataset"))
    existing_dataset_files = delete_unsharded_remote_files(api, dataset_repo, existing_dataset_files)
    upload_text(api, dataset_repo, "dataset", "README.md", dataset_readme(manifest), "Update sharded dataset README")
    upload_text(
        api,
        dataset_repo,
        "dataset",
        "propagator_cache_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "Update sharded cache manifest",
    )
    existing_dataset_files.update({"README.md", "propagator_cache_manifest.json"})

    uploaded_parts: list[dict[str, Any]] = []
    for group in groups:
        for path in group["files"]:
            uploaded_parts.extend(upload_sharded_file(api, dataset_repo, path, existing_dataset_files))

    upload_result = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": dataset_repo,
        "public": PUBLIC_DATASET,
        "uploaded_or_existing_parts": uploaded_parts,
    }
    result_path = PROJECT_ROOT / "outputs" / "propagator-multimodal" / "hf_sharded_upload_result.json"
    result_path.write_text(json.dumps(upload_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    upload_text(
        api,
        dataset_repo,
        "dataset",
        "propagator_cache_upload_result.json",
        json.dumps(upload_result, ensure_ascii=False, indent=2) + "\n",
        "Update sharded upload result",
    )
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("HF sharded upload finished", flush=True)


if __name__ == "__main__":
    main()

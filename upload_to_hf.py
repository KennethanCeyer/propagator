import json
import os
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
STAGING_DIR = Path(os.environ.get("HF_UPLOAD_STAGING_DIR", "/mnt/disks/propagator-cache/hf_upload_staging"))
SKIP_EXISTING = os.environ.get("HF_UPLOAD_SKIP_EXISTING", "1").lower() not in {"0", "false", "no", "off"}
UPLOAD_WORKERS = int(os.environ.get("HF_UPLOAD_WORKERS", "2"))

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
        missing = [path.name for path in files if not path.exists()]
        total_bytes = sum(file_size(path) for path in existing)
        meta = read_meta(meta_path)
        groups.append(
            {
                "prefix": prefix,
                "files": existing,
                "missing_files": missing,
                "total_bytes": total_bytes,
                "meta": meta,
            }
        )
    return groups


def build_manifest(groups: list[dict[str, Any]]) -> dict[str, Any]:
    files = []
    for group in groups:
        for path in group["files"]:
            files.append(
                {
                    "path": path.name,
                    "bytes": file_size(path),
                    "group": group["prefix"],
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(CACHE_DIR),
        "group_count": len(groups),
        "file_count": len(files),
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


def dataset_readme(manifest: dict[str, Any]) -> str:
    gib = manifest["total_bytes"] / 1024**3
    return f"""---
configs:
- config_name: default
  data_files:
  - "*.bin"
  - "*.json"
---

# Propagator Preprocessed Multimodal Cache

This private dataset contains completed preprocessed cache groups for Propagator training.

## Contents

- Cache groups: `{manifest["group_count"]}`
- Files: `{manifest["file_count"]}`
- Total size: `{gib:.2f} GiB`
- Manifest: `propagator_cache_manifest.json`

Each completed cache group is keyed by a fingerprinted prefix and includes tokenized inputs, targets, weights, stream ids, chunk positions, and metadata when present:

- `*.input.bin`
- `*.target.bin`
- `*.weight.bin`
- `*.stream_id.bin`
- `*.chunk_pos.bin`
- `*.meta.json`

The aggregate groups `propagator_train_*` and `propagator_val_*` are the full train/validation caches used by the current training run. Source groups are retained so individual dataset contributions can be audited or rebuilt independently.
"""


def upload_text(api: HfApi, repo_id: str, repo_type: str, path_in_repo: str, text: str, message: str) -> None:
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=message,
    )


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def prepare_dataset_staging(groups: list[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {"README.md", "propagator_cache_manifest.json"}
    for group in groups:
        for path in group["files"]:
            wanted.add(path.name)

    for path in STAGING_DIR.iterdir():
        if path.is_file() and path.name not in wanted:
            path.unlink()

    (STAGING_DIR / "README.md").write_text(dataset_readme(manifest), encoding="utf-8")
    (STAGING_DIR / "propagator_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for group in groups:
        for path in group["files"]:
            hardlink_or_copy(path, STAGING_DIR / path.name)
    return STAGING_DIR


def upload_path(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    path: Path,
    path_in_repo: str,
    existing_files: set[str],
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
        commit_message=f"Upload {path_in_repo}",
    )
    elapsed = max(time.time() - start, 1e-6)
    print(f"DONE {path_in_repo} in {elapsed:.1f}s ({size_gib / elapsed * 3600:.2f} GiB/h)", flush=True)
    return True


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
    dataset_repo = f"{username}/propagator-preprocessed"
    print(f"Authenticated as {username}", flush=True)
    print(f"Tokenizer repo: {tokenizer_repo}", flush=True)
    print(f"Dataset repo: {dataset_repo}", flush=True)

    create_repo(repo_id=tokenizer_repo, repo_type="model", private=True, token=token, exist_ok=True)
    create_repo(repo_id=dataset_repo, repo_type="dataset", private=True, token=token, exist_ok=True)

    tokenizer_path = PROJECT_ROOT / "assets" / "tokenizer-byte-bpe-16000.json"
    tokenizer_existing = set(api.list_repo_files(tokenizer_repo, repo_type="model"))
    if tokenizer_path.exists():
        upload_path(api, tokenizer_repo, "model", tokenizer_path, "tokenizer.json", tokenizer_existing)
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
    print(
        "Completed cache groups: "
        f"{manifest['group_count']} groups, {manifest['file_count']} files, "
        f"{manifest['total_bytes'] / 1024**4:.2f} TiB",
        flush=True,
    )

    staging_dir = prepare_dataset_staging(groups, manifest)
    print(f"Prepared staging folder: {staging_dir}", flush=True)
    print(f"Starting large-folder upload with workers={UPLOAD_WORKERS}", flush=True)
    api.upload_large_folder(
        repo_id=dataset_repo,
        repo_type="dataset",
        folder_path=staging_dir,
        private=True,
        num_workers=UPLOAD_WORKERS,
        print_report=True,
        print_report_every=60,
    )
    print("HF large-folder upload finished", flush=True)


if __name__ == "__main__":
    main()

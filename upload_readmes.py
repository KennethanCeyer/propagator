import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_REPO_NAME = "propagator-multimodal-pretraining-shards"

load_dotenv(PROJECT_ROOT / ".env")
token = os.environ.get('HF_TOKEN')

if not token:
    print("ERROR: HF_TOKEN not found in .env!")
    sys.exit(1)

api = HfApi(token=token)
username = api.whoami().get('name')

# README for propagator-tokenizer
tokenizer_readme = """# Propagator Tokenizer

This repository contains the byte-level BPE tokenizer used to process text sequences for training the **Propagator** model.

## Properties

*   **Vocab Size**: 16,000
*   **Type**: Byte-Level Byte-Pair Encoding (Byte BPE)
*   **File**: `tokenizer.json`
*   **Special Tokens**: Configured with custom protocol markers for session boundaries, speaker turn-taking, and multimodal alignment.
"""

# README for propagator-multimodal-pretraining-shards
dataset_readme = """---
license: other
pretty_name: Propagator Multimodal Pretraining Shards
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

# Propagator Multimodal Pretraining Shards

This public dataset contains sharded, tokenized multimodal pretraining caches for the **Propagator** model family. It is a model-training artifact rather than a raw, human-readable dataset.

The cache mixes text, image-grounded, and speech/audio-token workloads:

* text generation, instruction following, dialogue, and long-context plain text;
* image recognition and image-plus-prompt answer generation through image patch tokens;
* speech/text examples encoded as Mimi-style audio code tokens for ASR, TTS, and duplex audio-text behavior.

## Dataset Structure

Large files are split under `shards/<cache_group>/` as `<original-file>.part-00000-of-NNNNN`. Reconstruct files by concatenating the `repo_paths` listed in `propagator_cache_manifest.json`.

*   `*.input.bin`: Array of input token frames (shape: `[num_chunks, unroll_length, 8]`).
*   `*.target.bin`: Array of target token frames (shape: `[num_chunks, unroll_length, 8]`).
*   `*.weight.bin`: Array of loss weights (shape: `[num_chunks, unroll_length]`).
*   `*.stream_id.bin`: Sequence identifiers mapping chunks to source database streams.
*   `*.chunk_pos.bin`: Order position of the chunk in the sequence.
*   `*.meta.json`: Metadata detailing the count of chunks, unroll length, source row counts, vocabulary properties, and token stats.

## Included Source Families

The manifest records exact source names, row counts, preprocessing modes, and cache groups. Current source families include FineWeb-Edu, Wikipedia, VQAv2/image recognition data, instruction/dialogue datasets, Mimi-code speech corpora, and local Propagator seed/identity JSON groups. Source licenses and usage terms follow the upstream datasets listed in the manifest.
"""

# Upload Tokenizer README
try:
    api.upload_file(
        path_or_fileobj=tokenizer_readme.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=f"{username}/propagator-tokenizer",
        repo_type="model"
    )
    print("Uploaded README.md to propagator-tokenizer")
except Exception as e:
    print(f"Error uploading tokenizer README: {e}")

# Upload Preprocessed Dataset README
try:
    api.upload_file(
        path_or_fileobj=dataset_readme.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=f"{username}/{DATASET_REPO_NAME}",
        repo_type="dataset"
    )
    print(f"Uploaded README.md to {DATASET_REPO_NAME}")
except Exception as e:
    print(f"Error uploading dataset README: {e}")

print("Done uploading READMEs!")

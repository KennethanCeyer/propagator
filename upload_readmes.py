import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv('/home/user/propagator/.env')
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

# README for propagator-preprocessed
dataset_readme = """---
configs:
- config_name: default
  data_files:
  - "*.bin"
  - "*.json"
---

# Propagator Preprocessed Datasets

This repository contains preprocessed (tokenized) datasets ready for training the **Propagator** model, a stateful language model architecture built around persistent matrix memory.

## Dataset Structure

Each preprocessed dataset consists of binary files packed and padded to the model's training unroll length (`64` tokens per frame):

*   `*.input.bin`: Array of input token frames (shape: `[num_chunks, unroll_length, 8]`).
*   `*.target.bin`: Array of target token frames (shape: `[num_chunks, unroll_length, 8]`).
*   `*.weight.bin`: Array of loss weights (shape: `[num_chunks, unroll_length]`).
*   `*.stream_id.bin`: Sequence identifiers mapping chunks to source database streams.
*   `*.chunk_pos.bin`: Order position of the chunk in the sequence.
*   `*.meta.json`: Metadata detailing the count of chunks, unroll length, source row counts, vocabulary properties, and token stats.

## Included Sources

1.  **xinrongzhang2022/Duplex-UltraChat**: Stateful multi-turn conversation logs preprocessed in `duplex_chat` mode.
2.  **databricks/databricks-dolly-15k**: Instruction following dataset preprocessed in `dolly_instruction` mode.
3.  **Local JSON Caches**:
    *   Identity files (Propagator name and architecture descriptions).
    *   Clean instruction balanced seeds (formatting, recall, extraction logic).
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
        repo_id=f"{username}/propagator-preprocessed",
        repo_type="dataset"
    )
    print("Uploaded README.md to propagator-preprocessed")
except Exception as e:
    print(f"Error uploading dataset README: {e}")

print("Done uploading READMEs!")

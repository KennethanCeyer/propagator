import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_REPO_NAME = "propagator-multimodal-pretraining-data"

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

# README for propagator-multimodal-pretraining-data
dataset_readme = """---
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

This public dataset contains tokenized multimodal pretraining data prepared for the **Propagator** model family. It combines language, image-grounded, and speech/audio-token examples into a single training format.

This is not a raw text or image browsing dataset. The examples have already been converted into compact binary token frames for model training, with a manifest that records the source groups and file layout.

## What's Included

*   **Language:** web text, encyclopedic text, instruction-following, and dialogue data.
*   **Vision-language:** image recognition and image question-answering style examples represented as image patch tokens plus text tokens.
*   **Speech-language:** speech/text examples represented with Mimi-style audio code tokens for ASR, TTS, and duplex audio-text training.

## Source Datasets

| Source dataset | Modality | Contribution | Prepared rows | Preprocessing mode |
| --- | --- | --- | ---: | --- |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | Language | educational web text | 9,672,101 | `plain_text` |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en) | Language | encyclopedic long-form text | 4,601,098 | `plain_text` |
| [HuggingFaceM4/VQAv2](https://huggingface.co/datasets/HuggingFaceM4/VQAv2) | Vision-language | image question answering and recognition | 658,111 | `image_recognition` |
| [xinrongzhang2022/Duplex-UltraChat](https://huggingface.co/datasets/xinrongzhang2022/Duplex-UltraChat) | Dialogue | multi-turn conversational text | 5,973,182 | `duplex_chat` |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Instruction | instruction-following examples | 15,011 | `dolly_instruction` |
| [shangeth/libritts-r-mimi-codes](https://huggingface.co/datasets/shangeth/libritts-r-mimi-codes) | Speech-language | LibriTTS-R speech/text Mimi code examples | 365,042 | `mimi_codes_speech_text` |
| [shangeth/librispeech-mimi-codes](https://huggingface.co/datasets/shangeth/librispeech-mimi-codes) | Speech-language | LibriSpeech speech/text Mimi code examples | 286,808 | `mimi_codes_speech_text` |
| [shangeth/vctk-mimi-codes](https://huggingface.co/datasets/shangeth/vctk-mimi-codes) | Speech-language | VCTK speech/text Mimi code examples | 44,283 | `mimi_codes_speech_text` |
| [shangeth/jenny-mimi-codes](https://huggingface.co/datasets/shangeth/jenny-mimi-codes) | Speech-language | Jenny speech/text Mimi code examples | 20,978 | `mimi_codes_speech_text` |
| [shangeth/ljspeech-mimi-codes](https://huggingface.co/datasets/shangeth/ljspeech-mimi-codes) | Speech-language | LJSpeech speech/text Mimi code examples | 13,100 | `mimi_codes_speech_text` |
| Local prepared rows | Curated | local prepared text and vision-language examples | 491,356 | `duplex_chat`, `image_recognition` |

## Intended Use

This repository is intended for training or reproducing Propagator-style multimodal models that consume the packed Propagator frame format. It is useful if you want a ready-to-stream pretraining corpus rather than rebuilding tokenization and modality packing from the original upstream datasets.

It is not intended as a general-purpose dataset viewer, example gallery, or raw media archive.

## Format

Large files are split under `shards/<cache_group>/` as `<original-file>.part-00000-of-NNNNN`. Reconstruct files by concatenating the `repo_paths` listed in `propagator_cache_manifest.json`.

*   `*.input.bin`: Array of input token frames (shape: `[num_chunks, unroll_length, 8]`).
*   `*.target.bin`: Array of target token frames (shape: `[num_chunks, unroll_length, 8]`).
*   `*.weight.bin`: Array of loss weights (shape: `[num_chunks, unroll_length]`).
*   `*.stream_id.bin`: Sequence identifiers mapping chunks to source database streams.
*   `*.chunk_pos.bin`: Order position of the chunk in the sequence.
*   `*.meta.json`: Metadata detailing the count of chunks, unroll length, source row counts, vocabulary properties, and token stats.

## License and Source Terms

This dataset is a processed training artifact assembled from multiple upstream datasets. Check the upstream dataset licenses and terms listed in the manifest before redistribution or commercial use.
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

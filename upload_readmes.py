import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_REPO_NAME = "propagator-multimodal-pretraining-data"
TOKENIZER_REPO_NAME = "propagator-tokenizer"
GITHUB_REPO_URL = "https://github.com/KennethanCeyer/propagator"

SPECIAL_TOKEN_TABLE = """| Token id | Token | Purpose |
| ---: | --- | --- |
| 0 | `[PAD]` | Padding token |
| 1 | `[UNK]` | Unknown token |
| 2 | `[SESSION]` | Conversation/session start marker |
| 3 | `[USER]` | User turn marker |
| 4 | `[MODEL]` | Model turn marker |
| 5 | `[LISTEN]` | Listening/continuation marker |
| 6 | `[USER_END]` | User turn boundary |
| 7 | `[MODEL_END]` | Model turn boundary |
| 8 | `[SESSION_END]` | Conversation/session boundary |
| 9 | `[USER_INTERRUPT]` | User interruption marker |
| 10 | `[AUDIO_IN]` | Audio input span marker |
| 11 | `[AUDIO_OUT]` | Audio output span marker |
| 12 | `[AUDIO_END]` | Audio span boundary |
| 13 | `[SILENCE]` | Silence marker |
| 14 | `[TEXT_IN]` | Text input span marker |
| 15 | `[TEXT_OUT]` | Text output span marker |
| 16000 | `[HYBRID_OUT]` | Mixed-modality output marker |
| 16001 | `[IMAGE_IN]` | Image input span marker |
| 16002 | `[TEXT]` | Text modality marker |
| 16003 | `[AUDIO]` | Audio modality marker |
| 16004 | `[IMAGE]` | Image modality marker |
| 16005 | `[HYBRID]` | Mixed-modality marker |"""

load_dotenv(PROJECT_ROOT / ".env")
token = os.environ.get('HF_TOKEN')

if not token:
    print("ERROR: HF_TOKEN not found in .env!")
    sys.exit(1)

api = HfApi(token=token)
username = api.whoami().get('name')

# README for propagator-tokenizer
tokenizer_readme = f"""---
license: other
library_name: tokenizers
tags:
- tokenizer
- byte-level-bpe
- multimodal
- propagator
---

# Propagator Tokenizer

This repository contains the tokenizer used with the [Propagator Multimodal Pretraining Data](https://huggingface.co/datasets/{username}/{DATASET_REPO_NAME}).

The tokenizer is a byte-level BPE text tokenizer with a small set of special tokens for conversation boundaries and modality markers. It is intended for projects that need the same text and marker vocabulary used by the Propagator multimodal dataset.

## Source Code

*   GitHub: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})
*   Related dataset: [Propagator Multimodal Pretraining Data](https://huggingface.co/datasets/{username}/{DATASET_REPO_NAME})

## Files

*   `tokenizer.json`: Hugging Face `tokenizers` JSON file.

## Token Space

*   Base text BPE vocabulary: 16,000 tokens.
*   This repository only contains the text tokenizer. Multimodal numeric ids used by the paired dataset are documented in the dataset card.

## Special Tokens

{SPECIAL_TOKEN_TABLE}

## Quick Start

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repo_id = "{username}/{TOKENIZER_REPO_NAME}"
tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

encoded = tokenizer.encode("[SESSION] [USER] [TEXT_IN] Describe the image. [USER_END]")
print(encoded.ids)
```

## Related Dataset

*   Dataset: [Propagator Multimodal Pretraining Data](https://huggingface.co/datasets/{username}/{DATASET_REPO_NAME})

Use the dataset card for source coverage, binary frame layout, and reconstruction notes.
"""

# README for propagator-multimodal-pretraining-data
dataset_readme = f"""---
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

## Source Code

*   GitHub: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})
*   Tokenizer: [Propagator Tokenizer](https://huggingface.co/{username}/{TOKENIZER_REPO_NAME})

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

## Tokenizer

Text and marker tokens in this package use the [Propagator Tokenizer](https://huggingface.co/{username}/{TOKENIZER_REPO_NAME}). The tokenizer repository contains the Hugging Face `tokenizers` JSON file; non-text modality ids are described in this dataset card.

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

## Loading

```python
import json
from huggingface_hub import hf_hub_download

manifest_path = hf_hub_download(
    repo_id="{username}/{DATASET_REPO_NAME}",
    filename="propagator_cache_manifest.json",
    repo_type="dataset",
)
manifest = json.load(open(manifest_path, encoding="utf-8"))

first_binary = next(item for item in manifest["files"] if item["path"].endswith(".input.bin"))
print(first_binary["repo_paths"][:3])
```

## License and Source Terms

This dataset is a processed training artifact assembled from multiple upstream datasets. Check the upstream dataset licenses and terms listed in the manifest before redistribution or commercial use.
"""

# Upload Tokenizer README
try:
    api.upload_file(
        path_or_fileobj=tokenizer_readme.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=f"{username}/{TOKENIZER_REPO_NAME}",
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

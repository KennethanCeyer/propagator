import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv('/home/user/propagator/.env')
token = os.environ.get('HF_TOKEN')

if not token:
    print("ERROR: HF_TOKEN not found in .env!")
    sys.exit(1)

api = HfApi(token=token)
username = api.whoami().get('name')
print(f"Authenticated as user: {username}")

# 1. Create and Upload Tokenizer (Model Repo)
tokenizer_repo = f"{username}/propagator-tokenizer"
print(f"Creating/Checking private model repository: {tokenizer_repo}...")
try:
    create_repo(repo_id=tokenizer_repo, repo_type="model", private=True, token=token)
    print("Repository created.")
except Exception as e:
    print(f"Repository already exists or error: {e}")

tokenizer_file = Path('/home/user/propagator/assets/tokenizer-byte-bpe-16000.json')
if tokenizer_file.exists():
    print(f"Uploading tokenizer file: {tokenizer_file.name}...")
    api.upload_file(
        path_or_fileobj=str(tokenizer_file),
        path_in_repo="tokenizer.json",
        repo_id=tokenizer_repo,
        repo_type="model"
    )
    print("Tokenizer uploaded successfully.")
else:
    print(f"ERROR: Tokenizer file not found at {tokenizer_file}!")


# 2. Create and Upload Preprocessed Cache files (Dataset Repo)
dataset_repo = f"{username}/propagator-preprocessed"
print(f"Creating/Checking private dataset repository: {dataset_repo}...")
try:
    create_repo(repo_id=dataset_repo, repo_type="dataset", private=True, token=token)
    print("Dataset repository created.")
except Exception as e:
    print(f"Dataset repository already exists or error: {e}")

cache_dir = Path('/mnt/disks/propagator-cache/cache')
meta_files = list(cache_dir.glob('*.meta.json'))

print(f"Found {len(meta_files)} completed dataset caches.")

for meta_file in meta_files:
    prefix = meta_file.name[:-10] # strip .meta.json
    print(f"Processing cache group: {prefix}")
    # Find all files belonging to this cache group
    files = list(cache_dir.glob(f"{prefix}*"))
    for f in files:
        print(f"Uploading {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)...")
        start_time = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=dataset_repo,
                repo_type="dataset"
            )
            elapsed = time.time() - start_time
            print(f"Successfully uploaded {f.name} in {elapsed:.2f} seconds.")
        except Exception as e:
            print(f"ERROR uploading {f.name}: {e}")

print("HF Upload Script finished successfully!")

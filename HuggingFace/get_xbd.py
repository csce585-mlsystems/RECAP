"""
Partial downloader for the PUBLIC dataset aryananand/xBD -> ./data/xBD

- Cross-platform (Windows/macOS/Linux)
- Choose how many files per folder to download (MAX_PER_FOLDER)
- Batches allow_patterns to avoid huge single-requests
- Moderate parallelism to avoid rate limits; retries with backoff
- Auto-installs huggingface_hub; tries optional speedups (hf_transfer, hf_xet)

Change the constants in the "CONFIG" section to your liking.
"""

import os
import sys
import time
import math
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# ===================== CONFIG =====================
HF_REPO_ID       = "aryananand/xBD"
REVISION         = "v1"        # change to v2 later
DEST             = Path(__file__).resolve().parents[1] / "data" / "xBD"

SPLITS           = ["train", "test"]
FOLDERS          = ["images", "labels", "target"]
EXTS             = [".png"]    # add others if needed (e.g., ".json", ".tif")

MAX_PER_FOLDER   = 2000        # set None to get ALL; lower if you want smaller pulls
BATCH_SIZE       = 500         # how many files per snapshot_download call
MAX_WORKERS      = 8           # concurrency per call (balance speed vs 429)
MAX_ATTEMPTS     = 5           # retries per batch
BASE_SLEEP       = 3           # seconds (exponential backoff)
# ==================================================

MIN_PY = (3, 8)
if sys.version_info < MIN_PY:
    sys.exit(f"Please run with Python {MIN_PY[0]}.{MIN_PY[1]}+")

def ensure(pkg_spec):
    """pip install/upgrade a package if missing."""
    try:
        __import__(pkg_spec.split("[")[0].split("==")[0].split(">=")[0])
        return
    except Exception:
        pass
    print(f"[setup] Installing {pkg_spec} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg_spec])

# 1) Core lib
ensure("huggingface_hub>=0.25.0")

# 2) Optional accelerators (best-effort)
try:
    ensure("hf_transfer")  # Rust parallel downloader
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    print("[setup] Could not install hf_transfer; continuing without it.")
try:
    ensure("hf_xet")       # Xet-optimized backend
except Exception:
    print("[setup] Could not install hf_xet; continuing without it.")

from huggingface_hub import HfApi, snapshot_download

DEST.mkdir(parents=True, exist_ok=True)

def list_repo_filepaths(api: HfApi) -> List[str]:
    print("[list] Listing files from repo (this is quick metadata, not downloading data)...")
    files = api.list_repo_files(repo_id=HF_REPO_ID, revision=REVISION, repo_type="dataset")
    return files

def group_and_cap(files: List[str]) -> Dict[Tuple[str, str], List[str]]:
    """
    Group files by (split, folder) and cap to MAX_PER_FOLDER if set.
    Return dict: (split, folder) -> [paths...]
    """
    wanted: Dict[Tuple[str, str], List[str]] = {}
    norm_exts = tuple(EXTS)
    for split in SPLITS:
        for folder in FOLDERS:
            prefix = f"{split}/{folder}/"
            candidates = [f for f in files if f.startswith(prefix) and f.lower().endswith(norm_exts)]
            candidates.sort()  # stable subset
            if MAX_PER_FOLDER is not None:
                candidates = candidates[:MAX_PER_FOLDER]
            wanted[(split, folder)] = candidates
            print(f"[plan] {split}/{folder}: {len(candidates)} files to fetch")
    return wanted

def is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    code = getattr(getattr(err, "response", None), "status_code", None)
    transient = any(x in msg for x in ["429", "500", "502", "503", "504", "too many requests",
                                       "gateway time-out", "service unavailable"])
    return transient or (code in {429, 500, 502, 503, 504})

def download_batch(paths: List[str], attempt: int) -> None:
    """Download a batch (list of exact repo-relative paths) with retries."""
    if not paths:
        return
    # snapshot_download will honor allow_patterns relative to repo root
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(DEST),
            local_dir_use_symlinks=False,
            allow_patterns=paths,       # list of exact paths to fetch
            max_workers=MAX_WORKERS,
        )
    except Exception as e:
        if is_transient(e) and attempt < MAX_ATTEMPTS:
            sleep_s = min(60, BASE_SLEEP * (2 ** (attempt - 1)))
            print(f"[warn] Transient error on batch (attempt {attempt}/{MAX_ATTEMPTS}): {e}")
            print(f"       Backing off {sleep_s}s and retrying...")
            time.sleep(sleep_s)
            return download_batch(paths, attempt + 1)
        raise

def main():
    api = HfApi()
    files = list_repo_filepaths(api)
    plan = group_and_cap(files)

    total_to_get = sum(len(v) for v in plan.values())
    if total_to_get == 0:
        print("[done] Nothing to download (check your filters or revision).")
        return

    print(f"[start] Will fetch ~{total_to_get} files into {DEST}")
    done = 0
    for (split, folder), paths in plan.items():
        if not paths:
            continue
        print(f"[group] {split}/{folder} — {len(paths)} files")
        # Batch into chunks of BATCH_SIZE
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i+BATCH_SIZE]
            print(f"  [batch] {i//BATCH_SIZE + 1}/{math.ceil(len(paths)/BATCH_SIZE)} "
                  f"({len(batch)} files, workers={MAX_WORKERS})")
            download_batch(batch, attempt=1)
            done += len(batch)
            print(f"  [prog] {done}/{total_to_get} files fetched")

    print(f"[ok] Finished. Data is at: {DEST}")

    # Ensure expected dirs exist (handy if you capped and a folder is empty locally)
    for p in [
        "train/images", "train/labels", "train/target",
        "test/images",  "test/labels",  "test/target",
    ]:
        (DEST / p).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()

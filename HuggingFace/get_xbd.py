"""
Public dataset downloader for aryananand/xBD@v1 -> ./data/xBD
- Cross-platform (Windows/macOS/Linux)
- Auto-installs huggingface_hub
- Tries to enable speed-ups (hf_xet, hf_transfer) if available
- Limits concurrency to avoid 429 rate limits
- Retries with exponential backoff on transient errors
"""

import os
import sys
import time
import subprocess
from pathlib import Path

HF_REPO_ID = "aryananand/xBD"   # public dataset
REVISION   = "v1"               # bump to v2 later if needed

MIN_PY = (3, 8)
if sys.version_info < MIN_PY:
    sys.exit(f"Please run with Python {MIN_PY[0]}.{MIN_PY[1]}+")

def ensure(pkg_spec):
    """pip install a package if missing."""
    try:
        __import__(pkg_spec.split("[")[0].split("==")[0].split(">=")[0])
        return
    except Exception:
        pass
    print(f"[setup] Installing {pkg_spec} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])

# 1) Core library
ensure("huggingface_hub>=0.25.0")

# 2) Optional speed-ups:
#    - hf_xet: enables Xet-backed optimized storage on HF (fewer/ faster requests)
#    - hf_transfer: Rust-based parallel downloader (faster, fewer Python-level requests)
# If these fail to install, we just continue with regular HTTP.
try:
    ensure("hf_xet")
except Exception:
    print("[setup] Could not install hf_xet; continuing without it.")
try:
    # either install hf_transfer directly or via extra
    # ensure("huggingface_hub[hf_transfer]")  # alternatively
    ensure("hf_transfer")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    print("[setup] Could not install hf_transfer; continuing without it.")

from huggingface_hub import snapshot_download, HfHubHTTPError

DEST = Path(__file__).resolve().parents[1] / "data" / "xBD"
DEST.mkdir(parents=True, exist_ok=True)

def download_with_retries(max_attempts=6, base_sleep=3, max_workers=4):
    """
    Try snapshot_download with throttled workers and retry on 429/5xx.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[download] {HF_REPO_ID}@{REVISION} → {DEST} (attempt {attempt}/{max_attempts}, workers={max_workers})")
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=REVISION,
                local_dir=str(DEST),
                local_dir_use_symlinks=False,  # Windows-friendly
                max_workers=max_workers,       # throttle parallelism to reduce 429s
                # cache_dir=...                 # optionally move HF cache
            )
            print(f"[ok] xBD ready at: {DEST}")
            return
        except HfHubHTTPError as e:
            msg = str(e).lower()
            last_err = e
            # Backoff only on rate-limit (429) or transient server errors (5xx)
            if "429" in msg or "503" in msg or "504" in msg or "500" in msg or "502" in msg:
                sleep_s = min(60, base_sleep * (2 ** (attempt - 1)))
                print(f"[warn] Transient HTTP error ({e}). Backing off {sleep_s}s and retrying...")
                time.sleep(sleep_s)
                # Also reduce workers once if we still get 429
                if "429" in msg and max_workers > 1:
                    max_workers = max(1, max_workers // 2)
                continue
            # Non-retryable error
            raise
        except Exception as e:
            last_err = e
            sleep_s = min(60, base_sleep * (2 ** (attempt - 1)))
            print(f"[warn] Unexpected error: {e}. Backing off {sleep_s}s and retrying...")
            time.sleep(sleep_s)
            continue
    # If we reach here, all attempts failed
    raise SystemExit(f"[error] Download failed after retries: {last_err}")

if __name__ == "__main__":
    download_with_retries()
    # Light structure sanity check (won't fail)
    expected = [
        "train/images", "train/labels", "train/target",
        "test/images",  "test/labels",  "test/target",
    ]
    missing = [p for p in expected if not (DEST / p).exists()]
    if missing:
        for p in missing:
            (DEST / p).mkdir(parents=True, exist_ok=True)
        print(f"[note] Created empty dirs (not found in snapshot): {', '.join(missing)}")
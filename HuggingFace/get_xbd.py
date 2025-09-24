
"""
Public dataset downloader for aryananand/xBD@v1 -> ./data/xBD
- Cross-platform (Windows/macOS/Linux)
- Auto-installs huggingface_hub
- Tries to enable speed-ups (hf_xet, hf_transfer) if available
- Limits concurrency to reduce 429s; retries with backoff on transient errors
"""

import os
import sys
import time
import subprocess
from pathlib import Path

HF_REPO_ID = "aryananand/xBD"   # public dataset
REVISION   = "v1"               # bump to v2 later if needed
MIN_PY     = (3, 8)

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
    ensure("hf_xet")  # enables Xet-backed optimized fetch on HF
except Exception:
    print("[setup] Could not install hf_xet; continuing without it.")
try:
    ensure("hf_transfer")  # faster parallel download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    print("[setup] Could not install hf_transfer; continuing without it.")

from huggingface_hub import snapshot_download  # import after installs

DEST = Path(__file__).resolve().parents[1] / "data" / "xBD"
DEST.mkdir(parents=True, exist_ok=True)

def is_transient(err: Exception) -> bool:
    """Detect rate-limit/server errors without relying on specific exception types."""
    msg = str(err).lower()
    # Try to read an HTTP status code if present
    code = None
    resp = getattr(err, "response", None)
    if getattr(resp, "status_code", None):
        code = int(resp.status_code)
    # Consider 429 + common 5xx transient
    return ("429" in msg) or (code in {429, 500, 502, 503, 504}) or any(x in msg for x in ["429", "500", "502", "503", "504", "too many requests", "gateway time-out", "service unavailable"])

def download_with_retries(max_attempts=7, base_sleep=3, max_workers=4):
    """Throttled snapshot_download with exponential backoff."""
    last_err = None
    workers = max_workers
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[download] {HF_REPO_ID}@{REVISION} -> {DEST} (attempt {attempt}/{max_attempts}, workers={workers})")
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=REVISION,
                local_dir=str(DEST),
                local_dir_use_symlinks=False,  # Windows-friendly
                max_workers=workers,           # throttle to avoid 429
            )
            print(f"[ok] xBD ready at: {DEST}")
            return
        except Exception as e:
            last_err = e
            if is_transient(e) and attempt < max_attempts:
                # exponential backoff; tighten workers on 429-like errors
                sleep_s = min(90, base_sleep * (2 ** (attempt - 1)))
                print(f"[warn] Transient error: {e}\n        Backing off {sleep_s}s and retrying...")
                if "429" in str(e) and workers > 1:
                    workers = max(1, workers // 2)
                time.sleep(sleep_s)
                continue
            # Non-transient or out of attempts
            raise SystemExit(f"[error] Download failed: {e}")

if __name__ == "__main__":
    download_with_retries()
    # Optional: ensure expected dirs exist so downstream code never breaks
    for p in [
        "train/images", "train/labels", "train/target",
        "test/images",  "test/labels",  "test/target",
    ]:
        (DEST / p).mkdir(parents=True, exist_ok=True)

#!/usr/bin/env python3
"""
Download the PUBLIC dataset aryananand/xBD@v1 into ./data/xBD (same path on macOS/Linux/Windows).
- Auto-installs huggingface_hub if missing
- No token required (public repo)
- Idempotent: re-running resumes / skips already-downloaded files
"""

import sys
from pathlib import Path

HF_REPO_ID = "aryananand/xBD"   # <-- public dataset repo
REVISION   = "v1"               # <-- change to v2 later if you publish a new tag

MIN_PY = (3, 8)
if sys.version_info < MIN_PY:
    sys.exit(f"Please run with Python {MIN_PY[0]}.{MIN_PY[1]}+")

# Ensure dependency (auto-install if missing)
try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    import subprocess
    print("[setup] Installing huggingface_hub ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.25.0"])
    from huggingface_hub import snapshot_download

DEST = Path(__file__).resolve().parents[1] / "data" / "xBD"
DEST.mkdir(parents=True, exist_ok=True)

print(f"[download] Fetching {HF_REPO_ID}@{REVISION} → {DEST}")
try:
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(DEST),           # materialize files under ./data/xBD
        local_dir_use_symlinks=False,  # real files (Windows-friendly)
        # token=None  # public: no token
    )
except Exception as e:
    sys.exit(f"[error] Download failed: {e}")

# Optional: light structure sanity check (won't fail download)
expected = [
    "train/images", "train/labels", "train/target",
    "test/images",  "test/labels",  "test/target",
]
missing = [p for p in expected if not (DEST / p).exists()]
if missing:
    # Create empty dirs so downstream code never breaks if a split is absent
    for p in missing:
        (DEST / p).mkdir(parents=True, exist_ok=True)
    print(f"[warn] Created missing empty dirs: {', '.join(missing)}")

print(f"[ok] xBD is ready at: {DEST}")

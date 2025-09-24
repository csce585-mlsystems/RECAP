
import os
import sys
from pathlib import Path

# ==== EDIT ONLY IF YOU CHANGE REPO/TAG ====
HF_REPO_ID = "aryananand/xBD"
REVISION   = "v1"
# ==========================================

MIN_PY = (3, 8)
if sys.version_info < MIN_PY:
    sys.exit(f"Please run with Python {MIN_PY[0]}.{MIN_PY[1]} or newer.")

# Ensure dependency is present (auto-install if missing)
try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    import subprocess
    print("[setup] Installing huggingface_hub ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.25.0"])
    from huggingface_hub import snapshot_download

# Require token for PRIVATE datasets
TOKEN = os.getenv("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") or os.getenv("HF_TOKEN")
if not TOKEN:
    sys.exit(
        "No token found.\n"
        "Set HUGGINGFACE_HUB_TOKEN to your Hugging Face Read token (starts with hf_).\n"
        "macOS/Linux: export HUGGINGFACE_HUB_TOKEN=hf_XXXX\n"
        "Windows PS : $env:HUGGINGFACE_HUB_TOKEN=\"hf_XXXX\""
    )

# Destination folder: ./data/xBD (same on every OS)
DEST = Path(__file__).resolve().parents[1] / "data" / "xBD"
DEST.mkdir(parents=True, exist_ok=True)

print(f"[download] Fetching {HF_REPO_ID}@{REVISION} → {DEST}")
try:
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(DEST),
        local_dir_use_symlinks=False,  # write real files (Windows-friendly)
        token=TOKEN,
        # You can optionally set cache_dir=... if you want to move the HF cache
    )
except Exception as e:
    # A friendly, single-line error that’s easy to report
    sys.exit(f"[error] Download failed: {e}")

print(f"[ok] xBD is ready at: {DEST}")

# from huggingface_hub import HfApi
# import os

# api = HfApi(token=os.getenv("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"))
# api.upload_folder(
#     folder_path="/Users/aryananand/Documents/RECAP/data/xBD Dataset",
#     repo_id="aryananand/xBD",
#     repo_type="dataset",
# )

"""
Ultra-simple uploader for xBD -> Hugging Face Hub.
1) Edit the CONSTANTS below.
2) Run:  python3 upload_xbd_simple.py
"""

# ========== EDIT THESE CONSTANTS ==========
HF_USER        = "aryananand"            # e.g., "aryananand"
HF_TOKEN       = ""     # Write-scope token; or leave "" to use env var
LOCAL_XBD_DIR  = "/Users/aryananand/Documents/RECAP/data/xBD Dataset"      # e.g., r"/Users/you/Documents/RECAP/data/xBD"
PRIVATE_REPO   = True                        # True for private, False for public
TAG            = "v1"                          # dataset version tag
# ==========================================

import os, sys, time, subprocess
from pathlib import Path

REPO_ID = f"{HF_USER}/xBD"

def ensure_deps():
    try:
        import huggingface_hub  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.25.0"])

def upload_subfolder(api, token, local_root: Path, subpath: str):
    from huggingface_hub import upload_folder
    src = (local_root / subpath).resolve()
    if not src.exists():
        print(f"[-] Skipping (not found): {src}")
        return
    print(f"[+] Uploading {src}  →  hf://{REPO_ID}/{subpath}")
    # simple retry loop for transient errors like 504
    for attempt in range(1, 4):
        try:
            upload_folder(
                folder_path=str(src),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=subpath,                 # commit into same path on the hub
                commit_message=f"Add/update {subpath}",
                token=token,
            )
            print(f"[✓] Done: {subpath}")
            return
        except Exception as e:
            print(f"[!] Attempt {attempt} failed on {subpath}: {e}")
            if attempt == 3:
                raise
            time.sleep(5 * attempt)  # backoff then retry

def main():
    if sys.version_info < (3, 8):
        sys.exit("Please run with Python 3.8+")

    ensure_deps()
    from huggingface_hub import HfApi

    token = HF_TOKEN or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        sys.exit("No HF token provided. Set HF_TOKEN constant or HUGGINGFACE_HUB_TOKEN env var.")

    local_root = Path(LOCAL_XBD_DIR).resolve()
    if not local_root.exists():
        sys.exit(f"Local folder not found: {local_root}")

    api = HfApi(token=token)

    # 1) Create repo (idempotent)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=PRIVATE_REPO, exist_ok=True)
    print(f"[1/3] Repo ready: https://huggingface.co/datasets/{REPO_ID} (private={PRIVATE_REPO})")

    # 2) Upload in small commits to avoid 504s
    print("[2/3] Uploading in batches...")
    subfolders = [
        "train/images",
        "train/labels",
        "train/target",
        "test/images",
        "test/labels",
        "test/target",
    ]
    for sub in subfolders:
        upload_subfolder(api, token, local_root, sub)

    # 3) Tag final state
    try:
        api.delete_tag(repo_id=REPO_ID, repo_type="dataset", tag=TAG)
    except Exception:
        pass
    api.create_tag(repo_id=REPO_ID, repo_type="dataset", tag=TAG)
    print(f"[3/3] Done. Tagged {REPO_ID}@{TAG}")

if __name__ == "__main__":
    main()
    
  
# precompute_chips_parallel.py
"""
Parallel precompute script for RECAP chips.

Saves combined pre+post chips as .npy files under data/chips/{building_id}.npy

Usage:
    python precompute_chips_parallel.py

Notes:
 - This is CPU-bound (image decode + crop + resize). GPU is NOT used.
 - Resume-friendly: existing .npy files are skipped.
 - On Windows, run this from the command line (not inside some IDE run panels).
"""

import os
import sys
import csv
import math
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Config
INDEX_FILE = "info/index.csv"   # input index
OUT_DIR = "data/chips"         # where .npy chips are written
CHIP_SIZE = 224
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)  # default: one less than CPU count
SAVE_COMPRESSED = False        # if True uses np.savez_compressed -> smaller files, slower loads
SKIP_EXISTING = True          # skip files if they already exist
MISSING_LOG = os.path.join(OUT_DIR, "missing_ids.txt")
BATCH_WRITE_FLUSH = 100       # flush missing log every N writes

# Worker function must be top-level picklable
def process_row(row_tuple, chip_size=CHIP_SIZE, out_dir=OUT_DIR, save_compressed=SAVE_COMPRESSED):
    """
    row_tuple: (building_id, pre_path, post_path, polygon_wkt)
    Returns: (building_id, status, message)
      status = "ok" | "missing" | "error"
    """
    import cv2
    import numpy as np
    from shapely import wkt
    from shapely.geometry import Polygon

    building_id, pre_path, post_path, polygon_wkt = row_tuple
    out_file = os.path.join(out_dir, f"{building_id}.npy")
    out_file_npz = os.path.join(out_dir, f"{building_id}.npz")

    # If a file already exists, skip (caller should have checked but double-check for race)
    if os.path.exists(out_file) or os.path.exists(out_file_npz):
        return (building_id, "ok", "exists")

    # Basic validation of paths
    if not isinstance(pre_path, str) or not isinstance(post_path, str):
        return (building_id, "missing", "invalid_path")

    pre_img = cv2.imread(pre_path, cv2.IMREAD_COLOR)
    post_img = cv2.imread(post_path, cv2.IMREAD_COLOR)
    if pre_img is None or post_img is None:
        return (building_id, "missing", "image_missing")

    try:
        h, w, _ = pre_img.shape
    except Exception as e:
        return (building_id, "error", f"bad_image_shape: {e}")

    # centroid from WKT polygon; fallback to image center
    try:
        poly = wkt.loads(polygon_wkt)
        cen = poly.centroid
        cx, cy = int(cen.x), int(cen.y)
    except Exception:
        cx, cy = w // 2, h // 2

    half = chip_size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)

    pre_chip = pre_img[y1:y2, x1:x2]
    post_chip = post_img[y1:y2, x1:x2]

    try:
        pre_chip_resized = cv2.resize(pre_chip, (chip_size, chip_size))
        post_chip_resized = cv2.resize(post_chip, (chip_size, chip_size))
    except Exception as e:
        return (building_id, "error", f"resize_failed: {e}")

    combined = np.concatenate([pre_chip_resized, post_chip_resized], axis=2)  # HxWx6 uint8

    try:
        if save_compressed:
            # save as .npz
            np.savez_compressed(out_file_npz, chip=combined)
        else:
            np.save(out_file, combined)
    except Exception as e:
        return (building_id, "error", f"save_failed: {e}")

    return (building_id, "ok", "saved")

def load_index_rows(index_file=INDEX_FILE):
    """
    Returns list of tuples (building_id, pre_path, post_path, polygon_wkt)
    """
    df = pd.read_csv(index_file)
    # sanity: drop rows missing at least pre/post
    required = ["building_id", "pre_path", "post_path", "polygon_wkt"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Index missing required column: {c}")
    rows = []
    for _, r in df.iterrows():
        rows.append((str(r["building_id"]), r["pre_path"], r["post_path"], r["polygon_wkt"]))
    return rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_index_rows(INDEX_FILE)
    total = len(rows)
    print(f"Index rows: {total}. Writing chips to: {OUT_DIR}")
    print(f"Workers: {NUM_WORKERS}, Chip size: {CHIP_SIZE}, Compressed save: {SAVE_COMPRESSED}")

    # Optionally pre-filter to only rows missing output, to shrink worklist (saves some IPC)
    worklist = []
    for r in rows:
        b_id = r[0]
        out_file = os.path.join(OUT_DIR, f"{b_id}.npy")
        out_file_npz = os.path.join(OUT_DIR, f"{b_id}.npz")
        if SKIP_EXISTING and (os.path.exists(out_file) or os.path.exists(out_file_npz)):
            continue
        worklist.append(r)

    print(f"Worklist size (skipping existing): {len(worklist)}")

    missing_log = []
    saved_count = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as exe:
        # submit all tasks
        futures = {exe.submit(process_row, r): r[0] for r in worklist}

        # iterate with tqdm
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chips"):
            b_id = futures[fut]
            try:
                building_id, status, msg = fut.result()
            except Exception as e:
                # Unexpected failure in worker
                missing_log.append((b_id, "error", str(e)))
                continue

            if status != "ok":
                missing_log.append((building_id, status, msg))
            else:
                saved_count += 1

            # periodically flush missing log (avoid losing progress if interrupted)
            if len(missing_log) >= BATCH_WRITE_FLUSH:
                with open(MISSING_LOG, "a", newline="") as mf:
                    for item in missing_log:
                        mf.write(",".join(map(str, item)) + "\n")
                missing_log = []

    # flush any remaining missing
    if missing_log:
        with open(MISSING_LOG, "a", newline="") as mf:
            for item in missing_log:
                mf.write(",".join(map(str, item)) + "\n")

    print(f"Done. Saved chips: {saved_count}. Missing/failed entries logged to: {MISSING_LOG}")

if __name__ == "__main__":
    # On Windows, ensure safe multiprocessing
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()

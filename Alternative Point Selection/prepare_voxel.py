"""
prepare_voxel.py

Preprocesses raw .ply scans without registration using voxel grid
downsampling only — no FPS involved.

For every subject in the manifest:
  1. Load each .ply scan
  2. Normalize scale into unit sphere
  3. Voxel grid downsample (uniform spatial coverage)
  4. Truncate or pad to exactly N_POINTS without FPS
  5. Average all scans for subject into one (N, 3) array
  6. Save as OUT_DIR/<FBID>.npy

Resume-safe: skips subjects whose .npy already exists.

Dependencies: pip install --user numpy pandas trimesh
"""

import os
os.environ["OMP_NUM_THREADS"]      = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import pandas as pd
import trimesh
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST = "/home/changj9/projects/facebase/manifest.csv"
PLY_DIR  = Path("/home/changj9/projects/facebase/FB-TJ0_files/scans")
OUT_DIR  = Path("/home/changj9/projects/facebase/no_reg_voxel")

# ── Tunable parameters ────────────────────────────────────────────────────────
VOXEL_SIZE = 0.05   # in unit sphere space — smaller = more points kept
N_POINTS   = 4096   # fixed output size

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_verts(ply_path: Path) -> np.ndarray:
    mesh = trimesh.load(str(ply_path), process=False, skip_materials=True)
    return np.asarray(mesh.vertices, dtype=np.float32)


def normalize_scale(verts: np.ndarray) -> np.ndarray:
    centroid = verts.mean(axis=0)
    v = verts - centroid
    scale = np.linalg.norm(v, axis=1).mean()
    if scale > 0:
        v /= scale
    return v


def voxel_downsample(verts: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel grid downsampling — divide unit sphere space into cubic voxels,
    keep the centroid of all points within each occupied voxel.
    Fast O(N) operation giving spatially uniform coverage.
    Output size varies with density and voxel_size.
    """
    voxel_coords = np.floor(verts / voxel_size).astype(np.int32)
    voxel_dict = {}
    for i, coord in enumerate(map(tuple, voxel_coords)):
        if coord not in voxel_dict:
            voxel_dict[coord] = []
        voxel_dict[coord].append(i)
    centroids = np.array([
        verts[indices].mean(axis=0)
        for indices in voxel_dict.values()
    ], dtype=np.float32)
    return centroids


def fix_size(verts: np.ndarray, n: int) -> np.ndarray:
    """
    Truncate or pad to exactly n points without FPS.
    Truncation: random subset. Padding: repeat random existing points.
    """
    if len(verts) >= n:
        idx = np.random.choice(len(verts), n, replace=False)
        return verts[idx]
    else:
        pad_idx = np.random.choice(len(verts), n - len(verts), replace=True)
        return np.concatenate([verts, verts[pad_idx]], axis=0)


# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST)
subjects = list(df.groupby("FBID"))
total = len(subjects)

n_success = 0
n_skipped = 0
n_missing = 0
n_failed  = 0

print(f"Method     : Voxel only (no FPS)")
print(f"Voxel size : {VOXEL_SIZE}")
print(f"Output N   : {N_POINTS}")
print(f"Processing {total} subjects...\n")

for i, (fbid, group) in enumerate(subjects, 1):
    out_path = OUT_DIR / f"{fbid}.npy"

    if out_path.exists():
        print(f"[{i}/{total}] {fbid} — already done, skipping")
        n_success += 1
        continue

    print(f"[{i}/{total}] {fbid} ({len(group)} scan(s))")

    scan_list = []
    for _, row in group.iterrows():
        ply_path = PLY_DIR / row["Scan_name"]
        if not ply_path.exists():
            print(f"  MISSING : {ply_path.name}")
            n_missing += 1
            continue
        try:
            verts = load_verts(ply_path)
            verts = normalize_scale(verts)
            verts = voxel_downsample(verts, VOXEL_SIZE)
            verts = fix_size(verts, N_POINTS)
            scan_list.append(verts)
        except Exception as e:
            print(f"  FAILED  : {ply_path.name} — {e}")
            n_failed += 1

    if not scan_list:
        print(f"  SKIPPED : no valid scans for {fbid}")
        n_skipped += 1
        continue

    subject_array = np.mean(np.stack(scan_list), axis=0)
    np.save(out_path, subject_array)
    print(f"  SAVED   : shape {subject_array.shape}")
    n_success += 1

print(f"\nDone.")
print(f"  Succeeded : {n_success}")
print(f"  Skipped   : {n_skipped}")
print(f"  Missing   : {n_missing}")
print(f"  Failed    : {n_failed}")

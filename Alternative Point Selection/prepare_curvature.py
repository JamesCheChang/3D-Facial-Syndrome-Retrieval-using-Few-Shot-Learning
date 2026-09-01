"""
prepare_curvature.py

Preprocesses raw .ply scans without registration using curvature-weighted
point sampling only — no FPS involved.

For every subject in the manifest:
  1. Load each .ply scan
  2. Normalize scale into unit sphere
  3. Estimate per-point curvature via PCA on local neighborhood
  4. Sample N_POINTS with probability proportional to curvature^ALPHA
  5. Average all scans for subject into one (N, 3) array
  6. Save as OUT_DIR/<FBID>.npy

Resume-safe: skips subjects whose .npy already exists.

Dependencies: pip install --user numpy pandas trimesh scipy
"""

import os
os.environ["OMP_NUM_THREADS"]      = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import KDTree
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST = "/home/changj9/projects/facebase/manifest.csv"
PLY_DIR  = Path("/home/changj9/projects/facebase/FB-TJ0_files/scans")
OUT_DIR  = Path("/home/changj9/projects/facebase/no_reg_curvature")

# ── Tunable parameters ────────────────────────────────────────────────────────
N_POINTS    = 4096
K_NEIGHBORS = 20     # neighbors used for curvature estimation
ALPHA       = 2.0    # curvature weighting strength
                     # 1.0 = proportional to curvature
                     # 2.0 = squared (more aggressive)
                     # 0.0 = uniform random sampling

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


def estimate_curvature(verts: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    """
    Per-point curvature via PCA on local k-neighborhood.
    Returns smallest eigenvalue ratio — ~0 for flat, ~0.33 for corners/edges.
    """
    tree = KDTree(verts)
    _, idx = tree.query(verts, k=k + 1)
    idx = idx[:, 1:]  # exclude self

    curvatures = np.zeros(len(verts), dtype=np.float32)
    for i in range(len(verts)):
        neighbors = verts[idx[i]]
        centered  = neighbors - neighbors.mean(axis=0)
        cov       = (centered.T @ centered) / k
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = np.maximum(eigenvals, 0)
        total     = eigenvals.sum()
        if total > 0:
            curvatures[i] = eigenvals[0] / total
    return curvatures


def curvature_weighted_sample(verts: np.ndarray, n: int) -> np.ndarray:
    """
    Sample n points weighted by curvature^ALPHA.
    No FPS involved — pure probability-weighted sampling.
    """
    curvatures = estimate_curvature(verts)
    weights    = curvatures ** ALPHA
    weights   += 1e-6
    weights   /= weights.sum()

    if len(verts) <= n:
        pad_idx = np.random.choice(len(verts), n - len(verts),
                                    replace=True, p=weights)
        return np.concatenate([verts, verts[pad_idx]], axis=0)

    idx = np.random.choice(len(verts), n, replace=False, p=weights)
    return verts[idx]


# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST)
subjects = list(df.groupby("FBID"))
total = len(subjects)

n_success = 0
n_skipped = 0
n_missing = 0
n_failed  = 0

print(f"Method     : Curvature-weighted sampling only (no FPS)")
print(f"K neighbors: {K_NEIGHBORS}")
print(f"Alpha      : {ALPHA}")
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
            verts = curvature_weighted_sample(verts, N_POINTS)
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

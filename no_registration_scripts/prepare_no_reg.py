"""
prepare_no_reg.py

For every subject in the manifest:
  1. Load each of their .ply scans
  2. Normalize scale into unit sphere
  3. Optionally apply FPS downsampling
  4. Average all scans for the subject into one (N, 3) array
  5. Save as /home/changj9/projects/facebase/no_registration/raw_numpy/<FBID>.npy

No cropping, no ICP, no CPD — raw normalized point clouds ready for PointNet++.

Resume-safe: skips subjects whose .npy already exists.

FPS_N controls downsampling:
  - Set to None to keep all vertices (no downsampling)
  - Set to an integer e.g. 20000 to FPS downsample to that many points

Dependencies:
    pip install --user numpy pandas trimesh scipy
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
OUT_DIR  = Path("/home/changj9/projects/facebase/no_registration_numpy_files")

# ── Tunable ───────────────────────────────────────────────────────────────────
# Set to None to keep all vertices, or an integer to FPS downsample
# e.g. FPS_N = 20000 downsamples to ~20k points
FPS_N = 20000

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_verts(ply_path: Path) -> np.ndarray:
    mesh = trimesh.load(str(ply_path), process=False, skip_materials=True)
    return np.asarray(mesh.vertices, dtype=np.float32)


def normalize_scale(verts: np.ndarray) -> np.ndarray:
    """Center at origin, scale so mean vertex distance from centroid = 1."""
    centroid = verts.mean(axis=0)
    v = verts - centroid
    scale = np.linalg.norm(v, axis=1).mean()
    if scale > 0:
        v /= scale
    return v


def fps(verts: np.ndarray, n: int) -> np.ndarray:
    """Farthest Point Sampling — selects n points maximally spread across
    the surface, guaranteeing even spatial coverage regardless of density."""
    if len(verts) <= n:
        pad_idx = np.random.choice(len(verts), n - len(verts), replace = True)
        return np.concatenate([verts, verts[pad_idx]], axis=0)
    N = len(verts)
    selected  = np.zeros(n, dtype=np.int64)
    selected[0] = np.random.randint(0, N)
    min_dists = np.full(N, np.inf)
    for i in range(1, n):
        last = verts[selected[i - 1]]
        dists_to_last = np.sum((verts - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        selected[i] = np.argmax(min_dists)
    return verts[selected]


# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST)
subjects = list(df.groupby("FBID"))
total = len(subjects)

n_success = 0
n_skipped = 0
n_missing = 0
n_failed  = 0

if FPS_N is None:
    print(f"FPS disabled — keeping all vertices")
else:
    print(f"FPS enabled — downsampling to {FPS_N} points")

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

            if FPS_N is not None:
                verts = fps(verts, FPS_N)
                print(f"  FPS     : {ply_path.name} → {len(verts)} verts")

            scan_list.append(verts)
        except Exception as e:
            print(f"  FAILED  : {ply_path.name} — {e}")
            n_failed += 1

    if not scan_list:
        print(f"  SKIPPED : no valid scans for {fbid}")
        n_skipped += 1
        continue

    # Check if all scans have same vertex count for averaging
    counts = [len(v) for v in scan_list]
    if len(set(counts)) == 1:
        subject_array = np.mean(np.stack(scan_list), axis=0)
    else:
        # Different counts (only happens if FPS_N is None and scans vary)
        # FPS each to the minimum count then average
        min_count = min(counts)
        print(f"  VARYING counts {counts} — FPS all to {min_count}")
        resampled = [fps(v, min_count) for v in scan_list]
        subject_array = np.mean(np.stack(resampled), axis=0)
        subject_array = normalize_scale(subject_array)

    np.save(out_path, subject_array)
    print(f"  SAVED   : shape {subject_array.shape}")
    n_success += 1

print(f"\nDone.")
print(f"  Succeeded : {n_success}")
print(f"  Skipped   : {n_skipped}")
print(f"  Missing   : {n_missing}")
print(f"  Failed    : {n_failed}")

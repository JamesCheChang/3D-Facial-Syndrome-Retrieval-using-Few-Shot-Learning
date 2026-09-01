"""
split_data_no_reg.py

Assigns syndromes to tiers based on registered subject count, then splits into:
  - test  : 5 syndromes randomly drawn from T1+T2 pool (>=10 registered subjects)
  - val   : 5 syndromes randomly drawn from remaining T1+T2 pool
  - train : all remaining syndromes (leftover T1/T2 + all T3/T4)

Only considers syndromes where .npy files actually exist in no_registration_numpy_files.
Saves split to /home/changj9/projects/facebase/split_no_reg.json

Dependencies: pandas
"""

import json
import random
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST = "/home/changj9/projects/facebase/manifest.csv"
REG_DIR  = Path("/home/changj9/projects/facebase/no_reg_curvature")
OUT      = "/home/changj9/projects/facebase/split_no_reg_curvature.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED    = 42
N_SHOT  = 5
N_QUERY = 4  # minimum subjects needed = N_SHOT + N_QUERY = 9

random.seed(SEED)

# ── Load manifest ─────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST).drop_duplicates("FBID")[["FBID", "Syndrome"]]

# ── Filter to only subjects with .npy files ───────────────────────────────────
registered = set(p.stem for p in REG_DIR.glob("*.npy"))
df = df[df["FBID"].isin(registered)]

print(f"Registered subjects available: {len(df)}")

# ── Count subjects per syndrome ───────────────────────────────────────────────
subject_counts = df.groupby("Syndrome")["FBID"].nunique()
subject_counts = subject_counts[subject_counts >= N_SHOT + N_QUERY]

print(f"Syndromes with >={N_SHOT + N_QUERY} registered subjects: {len(subject_counts)}")

# ── Assign tiers ──────────────────────────────────────────────────────────────
t1 = subject_counts[subject_counts >= 20].index.tolist()
t2 = subject_counts[(subject_counts >= 10) & (subject_counts < 20)].index.tolist()
t3 = subject_counts[(subject_counts >= 5)  & (subject_counts < 10)].index.tolist()
t4 = subject_counts[(subject_counts >= 2)  & (subject_counts < 5)].index.tolist()

print(f"\nT1 (>=20 subjects) : {len(t1)} syndromes")
print(f"T2 (10-19 subjects): {len(t2)} syndromes")
print(f"T3 (5-9 subjects)  : {len(t3)} syndromes")
print(f"T4 (2-4 subjects)  : {len(t4)} syndromes")

# ── Sample test and val from T1+T2 pool ──────────────────────────────────────
t1_t2_pool = t1 + t2

if len(t1_t2_pool) < 10:
    raise ValueError(f"Not enough T1+T2 syndromes for test+val. "
                     f"Have {len(t1_t2_pool)}, need 10.")

test_syndromes    = random.sample(t1_t2_pool, 5)
remaining_t1_t2   = [s for s in t1_t2_pool if s not in set(test_syndromes)]
val_syndromes     = random.sample(remaining_t1_t2, 5)

# ── Remaining syndromes go to train ──────────────────────────────────────────
held_out = set(test_syndromes) | set(val_syndromes)
train_syndromes = [s for s in subject_counts.index.tolist()
                   if s not in held_out]

# ── Save ─────────────────────────────────────────────────────────────────────
split = {
    "train": sorted(train_syndromes),
    "val"  : sorted(val_syndromes),
    "test" : sorted(test_syndromes)
}

with open(OUT, "w") as f:
    json.dump(split, f, indent=2)

print(f"\nTrain syndromes : {len(train_syndromes)}")
print(f"Val syndromes   : {len(val_syndromes)}")
print(f"Test syndromes  : {len(test_syndromes)}")
print(f"Saved to        : {OUT}")

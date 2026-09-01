"""
split_data.py

Assigns syndromes to tiers based on registered subject count, then splits into:
  - test  : 5 syndromes randomly drawn from T2 pool (9-19 registered subjects)
  - val   : 5 syndromes randomly drawn from T1 pool (>=20 registered subjects)
  - train : all remaining syndromes

*** Minimum subject threshold ***
Any syndrome with fewer than N_SHOT + N_QUERY (= 9) registered subjects is
excluded entirely — it cannot appear in test, val, or train.  This guarantees
every episode can always draw 5 support + 4 query subjects without overlap.

Only considers syndromes where registered .npy files actually exist.
Saves split to /home/changj9/projects/facebase/split.json

Dependencies: pandas
"""

import json
import random
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST = "/home/changj9/projects/facebase/manifest.csv"
REG_DIR  = Path("/home/changj9/projects/facebase/registered_r6_FPS")
OUT      = "/home/changj9/projects/facebase/FPS_5n5k_5n20/split_r_6.json"
SEED     = 42
N_SHOT   = 5
N_QUERY  = 4   # every syndrome must have >= N_SHOT + N_QUERY = 9 subjects

random.seed(SEED)

# ── Load manifest ─────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST).drop_duplicates("FBID")[["FBID", "Syndrome"]]

# ── Filter to only subjects with registered .npy files ───────────────────────
registered = set(p.stem for p in REG_DIR.glob("*.npy"))
df = df[df["FBID"].isin(registered)]

print(f"Registered subjects available : {len(df)}")

# ── Count registered subjects per syndrome ────────────────────────────────────
# Hard minimum: syndrome must have enough subjects for at least one full episode
MIN_SUBJECTS = N_SHOT + N_QUERY   # = 9
subject_counts = df.groupby("Syndrome")["FBID"].nunique()

excluded = subject_counts[subject_counts < MIN_SUBJECTS]
subject_counts = subject_counts[subject_counts >= MIN_SUBJECTS]

print(f"Syndromes excluded (<{MIN_SUBJECTS} subjects) : {len(excluded)}")
print(f"Syndromes eligible (>={MIN_SUBJECTS} subjects): {len(subject_counts)}")

# ── Assign tiers based on registered subject count ───────────────────────────
# T4 (2-4) and T3 (5-8) are now impossible given the >=9 floor.
# T2 becomes the "lower" tier from which test syndromes are drawn, so that
# the harder (fewer-subject) classes end up in the test set — matching the
# original intent of using smaller classes for evaluation.
t1 = subject_counts[subject_counts >= 20].index.tolist()
t2 = subject_counts[(subject_counts >= MIN_SUBJECTS) & (subject_counts < 20)].index.tolist()

print(f"\nT1 (>=20 subjects)          : {len(t1)} syndromes")
print(f"T2 ({MIN_SUBJECTS}-19 subjects) : {len(t2)} syndromes")

# ── Sample test syndromes from T2 pool (harder — fewer subjects) ──────────────
if len(t2) < 5:
    raise ValueError(f"Not enough T2 syndromes for test set. Have {len(t2)}, need 5.")
test_syndromes = random.sample(t2, 5)

# ── Sample val syndromes from T1 pool (easier — more subjects) ───────────────
if len(t1) < 5:
    raise ValueError(f"Not enough T1 syndromes for val set. Have {len(t1)}, need 5.")
val_syndromes = random.sample(t1, 5)

# ── Remaining syndromes go to train ──────────────────────────────────────────
test_set = set(test_syndromes)
val_set  = set(val_syndromes)
train_syndromes = [s for s in subject_counts.index.tolist()
                   if s not in test_set and s not in val_set]

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

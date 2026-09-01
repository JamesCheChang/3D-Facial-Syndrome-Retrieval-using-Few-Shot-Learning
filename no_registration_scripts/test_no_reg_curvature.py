"""
test_no_reg.py

Evaluates trained PointNet++ on test syndromes using raw normalized numpy files.

  - Loads best_model_no_reg.pth from pointNetTrainOutput/
  - Runs 1,000 test episodes (5-way 5-shot, 4 queries per class)
  - Reports mean accuracy and 95% confidence interval

Dependencies: torch, numpy, pandas
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append("/home/changj9/projects/facebase/scripts")
from model_no_reg import PointNetPlusPlus, prototypical_loss

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST  = "/home/changj9/projects/facebase/manifest.csv"
SPLIT     = "/home/changj9/projects/facebase/split_no_reg_curvature.json"
REG_DIR   = Path("/home/changj9/projects/facebase/no_reg_curvature")
MODEL_PTH = Path("/home/changj9/projects/facebase/curvature_train/best_model_no_reg.pth")

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_WAY           = 5
N_SHOT          = 5
N_QUERY         = 4
N_TEST_EPISODES = 1000
EMBED_DIM       = 256
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load manifest and split ───────────────────────────────────────────────────
df = pd.read_csv(MANIFEST).drop_duplicates("FBID")[["FBID", "Syndrome"]]

with open(SPLIT) as f:
    split = json.load(f)

test_syndromes = split["test"]

# ── Build test map ────────────────────────────────────────────────────────────
test_map = {}
for s in test_syndromes:
    fbids = df[df["Syndrome"] == s]["FBID"].tolist()
    fbids = [f for f in fbids if (REG_DIR / f"{f}.npy").exists()]
    if len(fbids) >= N_SHOT + N_QUERY:
        test_map[s] = fbids

print(f"Test syndromes with enough subjects: {len(test_map)}")

if len(test_map) < N_WAY:
    raise ValueError(f"Not enough test syndromes with {N_SHOT + N_QUERY} subjects. "
                     f"Need {N_WAY}, have {len(test_map)}.")

# ── Episode sampler ───────────────────────────────────────────────────────────
def sample_episode(syndrome_map):
    syndromes = random.sample(list(syndrome_map.keys()), N_WAY)
    support_list, query_list = [], []
    for s in syndromes:
        fbids = random.sample(syndrome_map[s], N_SHOT + N_QUERY)
        for fbid in fbids[:N_SHOT]:
            support_list.append(np.load(REG_DIR / f"{fbid}.npy"))
        for fbid in fbids[N_SHOT:]:
            query_list.append(np.load(REG_DIR / f"{fbid}.npy"))
    support = torch.tensor(np.stack(support_list), dtype=torch.float32).to(device)
    query   = torch.tensor(np.stack(query_list),   dtype=torch.float32).to(device)
    return support, query

# ── Load model ────────────────────────────────────────────────────────────────
model = PointNetPlusPlus(embed_dim=EMBED_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PTH}\n")

# ── Test loop ─────────────────────────────────────────────────────────────────
print(f"Running {N_TEST_EPISODES} test episodes...\n")

accs = []
with torch.no_grad():
    for episode in range(1, N_TEST_EPISODES + 1):
        support, query = sample_episode(test_map)
        all_points  = torch.cat([support, query], dim=0)
        embeddings  = model(all_points)
        support_emb = embeddings[:N_WAY * N_SHOT]
        query_emb   = embeddings[N_WAY * N_SHOT:]
        _, acc = prototypical_loss(support_emb, query_emb, N_WAY, N_SHOT, N_QUERY)
        accs.append(acc)

        if episode % 100 == 0:
            print(f"Episode {episode:4d}/{N_TEST_EPISODES} | "
                  f"Running Acc: {np.mean(accs):.4f}")

# ── Results ───────────────────────────────────────────────────────────────────
mean_acc = np.mean(accs)
std_acc  = np.std(accs)
ci95     = 1.96 * std_acc / np.sqrt(N_TEST_EPISODES)

print(f"\n── Test Results ──────────────────────────────")
print(f"Mean Accuracy : {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"95% CI        : ± {ci95:.4f}")
print(f"Std Dev       : {std_acc:.4f}")
print(f"Random chance : {1/N_WAY:.4f} ({100/N_WAY:.1f}%)")

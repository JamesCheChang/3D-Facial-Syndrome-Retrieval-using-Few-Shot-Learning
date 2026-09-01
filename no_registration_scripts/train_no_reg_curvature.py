"""
train_no_reg.py

Episodic training of PointNet++ with prototypical networks.
Uses raw normalized numpy files from no_registration_numpy_files/.

  - 5,000 training episodes (5-way 5-shot, 4 queries per class)
  - Validates every 500 episodes (200 val episodes each check)
  - Learning rate halved every 500 episodes
  - Saves best model to pointNetTrainOutput/best_model_no_reg.pth
  - Logs to pointNetTrainOutput/train_log_no_reg.csv

Dependencies: torch, numpy, pandas
"""

import json
import random
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

sys.path.append("/home/changj9/projects/facebase/scripts")
from model_no_reg import PointNetPlusPlus, prototypical_loss

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST = "/home/changj9/projects/facebase/manifest_no_reg.csv"
SPLIT    = "/home/changj9/projects/facebase/split_no_reg_curvature.json"
REG_DIR  = Path("/home/changj9/projects/facebase/no_reg_curvature")
OUT_DIR  = Path("/home/changj9/projects/facebase/curvature_train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_WAY            = 5
N_SHOT           = 5
N_QUERY          = 4
N_TRAIN_EPISODES = 5000
N_VAL_EPISODES   = 200
VAL_EVERY        = 500
EMBED_DIM        = 256
LR               = 1e-3
LR_STEP          = 500
LR_GAMMA         = 0.5
SEED             = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load manifest and split ───────────────────────────────────────────────────
df = pd.read_csv(MANIFEST).drop_duplicates("FBID")[["FBID", "Syndrome"]]

with open(SPLIT) as f:
    split = json.load(f)

train_syndromes = split["train"]
val_syndromes   = split["val"]

# ── Build syndrome → FBID maps ────────────────────────────────────────────────
def build_syndrome_map(syndromes):
    mapping = {}
    for s in syndromes:
        fbids = df[df["Syndrome"] == s]["FBID"].tolist()
        fbids = [f for f in fbids if (REG_DIR / f"{f}.npy").exists()]
        if len(fbids) >= N_SHOT + N_QUERY:
            mapping[s] = fbids
    return mapping

train_map = build_syndrome_map(train_syndromes)
val_map   = build_syndrome_map(val_syndromes)

print(f"Train syndromes with enough subjects: {len(train_map)}")
print(f"Val syndromes with enough subjects  : {len(val_map)}")

if len(train_map) < N_WAY:
    raise ValueError(f"Not enough train syndromes. Need {N_WAY}, have {len(train_map)}.")

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

# ── Model, optimizer, scheduler ──────────────────────────────────────────────
model     = PointNetPlusPlus(embed_dim=EMBED_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = OUT_DIR / "train_log_no_reg.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "train_loss", "train_acc", "val_acc", "lr"])

best_val_acc = 0.0

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting training for {N_TRAIN_EPISODES} episodes...\n")

for episode in range(1, N_TRAIN_EPISODES + 1):

    model.train()
    support, query = sample_episode(train_map)

    all_points  = torch.cat([support, query], dim=0)
    embeddings  = model(all_points)
    support_emb = embeddings[:N_WAY * N_SHOT]
    query_emb   = embeddings[N_WAY * N_SHOT:]

    loss, acc = prototypical_loss(support_emb, query_emb, N_WAY, N_SHOT, N_QUERY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"Episode {episode:5d}/{N_TRAIN_EPISODES} | "
          f"Loss: {loss.item():.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}", end="")

    val_acc = None
    if episode % VAL_EVERY == 0 and len(val_map) >= N_WAY:
        model.eval()
        val_accs = []
        with torch.no_grad():
            for _ in range(N_VAL_EPISODES):
                support, query = sample_episode(val_map)
                all_points  = torch.cat([support, query], dim=0)
                embeddings  = model(all_points)
                support_emb = embeddings[:N_WAY * N_SHOT]
                query_emb   = embeddings[N_WAY * N_SHOT:]
                _, acc_v = prototypical_loss(support_emb, query_emb, N_WAY, N_SHOT, N_QUERY)
                val_accs.append(acc_v)
        val_acc = np.mean(val_accs)
        print(f" | Val Acc: {val_acc:.4f}", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "best_model_no_reg.pth")
            print(" ✓ saved", end="")

    print()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, loss.item(), acc,
                         val_acc if val_acc is not None else "", current_lr])

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
print(f"Model saved to: {OUT_DIR / 'best_model_no_reg.pth'}")

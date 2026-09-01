"""
train_mixed.py

Trains a single PointNet++ encoder by mixing episodes from 9 configurations:
  5-way  × K ∈ {1, 3, 5}
  10-way × K ∈ {1, 3, 5}
  20-way × K ∈ {1, 3, 5}

Improvements over v1:
  - 15000 episodes (was 5000)
  - Slower LR decay: step=2000, gamma=0.7 (was step=500, gamma=0.5)
  - Curriculum learning: first 5000 episodes favour 5-way,
    next 5000 mix 5-way and 10-way, final 5000 use all configs
  - Larger embedding dim: 512 (was 256)
  - Validation uses mixed configs (5-way and 10-way) for better signal
  - Resume-safe: continues from last episode if job restarts

Saves:
  <OUT_DIR>/best_model_mixed.pth
  <OUT_DIR>/train_log_mixed.csv
  <OUT_DIR>/split_mixed.json

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

sys.path.append("/home/changj9/projects/facebase/all_config_one_model/scripts")
from model import PointNetPlusPlus, prototypical_loss

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST = "/home/changj9/projects/facebase/manifest.csv"
REG_DIR  = Path("/home/changj9/projects/facebase/no_registration_numpy_files")
OUT_DIR  = Path("/home/changj9/projects/facebase/all_config_one_model/train_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Training configurations ───────────────────────────────────────────────────
TRAIN_CONFIGS = [
    (5,   1), (5,   3), (5,   5),
    (10,  1), (10,  3), (10,  5),
    
]
N_QUERY = 4

# ── Curriculum phases ─────────────────────────────────────────────────────────
# Phase 1 (ep 1-7500)    : only 5-way — easy, builds basic embedding space
# Phase 2 (ep 7501-15000): 5-way + 10-way — full difficulty
PHASE1_END = 7500
PHASE2_END = 15000

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_TRAIN_EPISODES = 15000
N_VAL_EPISODES   = 200
VAL_EVERY        = 500
EMBED_DIM        = 512   # increased from 256
LR               = 1e-3
LR_STEP          = 2000  # slower decay (was 500)
LR_GAMMA         = 0.7   # gentler decay (was 0.5)
N_TEST_HELD_OUT  = 10
N_VAL_HELD_OUT   = 10
SEED             = 30

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── Load manifest ─────────────────────────────────────────────────────────────
df = pd.read_csv(MANIFEST).drop_duplicates("FBID")[["FBID", "Syndrome"]]
registered = set(p.stem for p in REG_DIR.glob("*.npy"))
df = df[df["FBID"].isin(registered)]
subject_counts = df.groupby("Syndrome")["FBID"].nunique()

# ── Load or generate split ────────────────────────────────────────────────────
split_out = OUT_DIR / "split_mixed.json"

if split_out.exists():
    print("Loading existing split...")
    with open(split_out) as f:
        split = json.load(f)
    test_syndromes = split["test"]
    val_syndromes  = split["val"]
else:
    max_needed = 20 + N_QUERY  # guarantee enough subjects for K=20 eval
    eligible   = subject_counts[subject_counts >= max_needed].index.tolist()
    print(f"Syndromes with >={max_needed} subjects (eligible for test/val): {len(eligible)}")

    if len(eligible) < N_TEST_HELD_OUT + N_VAL_HELD_OUT:
        raise ValueError(f"Not enough eligible syndromes. Have {len(eligible)}, "
                         f"need {N_TEST_HELD_OUT + N_VAL_HELD_OUT}.")

    test_syndromes = random.sample(eligible, N_TEST_HELD_OUT)
    remaining      = [s for s in eligible if s not in set(test_syndromes)]
    val_syndromes  = random.sample(remaining, N_VAL_HELD_OUT)

held_out = set(test_syndromes) | set(val_syndromes)
print(f"Test syndromes: {test_syndromes}")
print(f"Val syndromes : {val_syndromes}\n")

# ── Build syndrome maps ───────────────────────────────────────────────────────
def build_map(syndromes, min_subj):
    mapping = {}
    for s in syndromes:
        fbids = df[df["Syndrome"] == s]["FBID"].tolist()
        fbids = [f for f in fbids if (REG_DIR / f"{f}.npy").exists()]
        if len(fbids) >= min_subj:
            mapping[s] = fbids
    return mapping

config_maps = {}
for n_way, n_shot in TRAIN_CONFIGS:
    min_subj = n_shot + N_QUERY
    if min_subj not in config_maps:
        usable = subject_counts[subject_counts >= min_subj].index.tolist()
        train_syndromes = [s for s in usable if s not in held_out]
        config_maps[min_subj] = build_map(train_syndromes, min_subj)

# Val map — use 5-shot threshold so all val syndromes qualify
val_map_5way  = build_map(val_syndromes, 5 + N_QUERY)
val_map_10way = build_map(val_syndromes, 5 + N_QUERY)  # same threshold, sample 10-way if enough

# Resolve actual n_way
def get_actual_n_way(n_way, n_shot):
    min_subj  = n_shot + N_QUERY
    train_map = config_maps[min_subj]
    return len(train_map) if n_way == "all" else n_way

# Check feasibility
feasible_configs = []
for n_way, n_shot in TRAIN_CONFIGS:
    min_subj     = n_shot + N_QUERY
    train_map    = config_maps[min_subj]
    actual_n_way = get_actual_n_way(n_way, n_shot)
    if len(train_map) >= actual_n_way:
        feasible_configs.append((n_way, n_shot, actual_n_way))
        print(f"  {n_way}w{n_shot}s: {len(train_map)} train syndromes, {actual_n_way}-way")
    else:
        print(f"  {n_way}w{n_shot}s: SKIPPED (only {len(train_map)} syndromes)")

# Separate by difficulty for curriculum
easy_configs   = [(w, s, a) for w, s, a in feasible_configs if w == 5]
medium_configs = [(w, s, a) for w, s, a in feasible_configs if w in [5, 10]]

print(f"\nCurriculum phases:")
print(f"  Phase 1 (ep    1-{PHASE1_END:5d}): {len(easy_configs)} configs (5-way only)")
print(f"  Phase 2 (ep {PHASE1_END+1:5d}-{N_TRAIN_EPISODES:5d}): {len(medium_configs)} configs (5+10-way)")

if not feasible_configs:
    raise ValueError("No feasible training configs found.")

# ── Save split ────────────────────────────────────────────────────────────────
if not split_out.exists():
    all_train = set()
    for n_way, n_shot, actual_n_way in feasible_configs:
        min_subj = n_shot + N_QUERY
        all_train.update(config_maps[min_subj].keys())
    split_data = {
        "train"  : sorted(all_train),
        "val"    : sorted(val_syndromes),
        "test"   : sorted(test_syndromes),
        "configs": [f"{nw}w{ns}s" for nw, ns, _ in feasible_configs]
    }
    with open(split_out, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"\nSplit saved to {split_out}")

# ── Episode sampler ───────────────────────────────────────────────────────────
def sample_episode(n_way, n_shot, n_query, actual_n_way):
    min_subj  = n_shot + n_query
    train_map = config_maps[min_subj]
    syndromes = random.sample(list(train_map.keys()), actual_n_way)
    support_list, query_list = [], []
    for s in syndromes:
        fbids = random.sample(train_map[s], n_shot + n_query)
        for fbid in fbids[:n_shot]:
            support_list.append(np.load(REG_DIR / f"{fbid}.npy"))
        for fbid in fbids[n_shot:]:
            query_list.append(np.load(REG_DIR / f"{fbid}.npy"))
    support = torch.tensor(np.stack(support_list), dtype=torch.float32).to(device)
    query   = torch.tensor(np.stack(query_list),   dtype=torch.float32).to(device)
    return support, query, actual_n_way, n_shot, n_query

# ── Resume logic ──────────────────────────────────────────────────────────────
model_out     = OUT_DIR / "best_model_mixed.pth"
log_out       = OUT_DIR / "train_log_mixed.csv"
start_episode = 1
best_val_acc  = 0.0
log_mode      = "w"

if model_out.exists() and log_out.exists():
    try:
        existing_log = pd.read_csv(log_out)
        last_episode = int(existing_log["episode"].max())
        if last_episode >= N_TRAIN_EPISODES:
            print(f"Already complete ({last_episode} episodes). Exiting.")
            sys.exit(0)
        start_episode = last_episode + 1
        log_mode      = "a"
        val_rows = existing_log[existing_log["val_acc"].notna()]
        if len(val_rows) > 0:
            best_val_acc = float(val_rows["val_acc"].max())
        print(f"Resuming from episode {start_episode} "
              f"(best val acc so far: {best_val_acc:.4f})")
    except Exception as e:
        print(f"Could not read log ({e}) — starting fresh")

# ── Model ─────────────────────────────────────────────────────────────────────
model     = PointNetPlusPlus(embed_dim=EMBED_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

if start_episode > 1 and model_out.exists():
    model.load_state_dict(torch.load(model_out, map_location=device))
    print(f"Loaded existing model weights")
    for _ in range(start_episode - 1):
        scheduler.step()

# ── Log header ────────────────────────────────────────────────────────────────
if log_mode == "w":
    with open(log_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "phase", "config",
                         "train_loss", "train_acc", "val_acc", "lr"])

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting mixed training for {N_TRAIN_EPISODES} episodes...\n")

for episode in range(start_episode, N_TRAIN_EPISODES + 1):
    model.train()

    # Curriculum: pick config pool based on episode
    if episode <= PHASE1_END:
        phase = 1
        pool  = easy_configs
    else:
        phase = 2
        pool  = medium_configs

    n_way, n_shot, actual_n_way = random.choice(pool)
    config_tag = f"{n_way}w{n_shot}s"

    support, query, actual_n_way, n_shot, n_query = sample_episode(
        n_way, n_shot, N_QUERY, actual_n_way)

    all_points  = torch.cat([support, query], dim=0)
    embeddings  = model(all_points)
    support_emb = embeddings[:actual_n_way * n_shot]
    query_emb   = embeddings[actual_n_way * n_shot:]

    loss, acc = prototypical_loss(support_emb, query_emb,
                                  actual_n_way, n_shot, N_QUERY)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"Ep {episode:6d}/{N_TRAIN_EPISODES} [P{phase}:{config_tag}] | "
          f"Loss: {loss.item():.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}", end="")

    # ── Validation — mix 5-way and 10-way ─────────────────────────────────────
    val_acc = None
    if episode % VAL_EVERY == 0:
        model.eval()
        val_accs = []
        with torch.no_grad():
            for vi in range(N_VAL_EPISODES):
                # Alternate between 5-way and 10-way validation
                if vi % 2 == 0 and len(val_map_5way) >= 5:
                    n_way_v, n_shot_v = 5, 5
                    vmap = val_map_5way
                elif len(val_map_10way) >= 5:
                    n_way_v, n_shot_v = min(5, len(val_map_10way)), 5
                    vmap = val_map_10way
                else:
                    continue

                syn = random.sample(list(vmap.keys()), n_way_v)
                s_list, q_list = [], []
                for s in syn:
                    fbids = random.sample(vmap[s], n_shot_v + N_QUERY)
                    for fbid in fbids[:n_shot_v]:
                        s_list.append(np.load(REG_DIR / f"{fbid}.npy"))
                    for fbid in fbids[n_shot_v:]:
                        q_list.append(np.load(REG_DIR / f"{fbid}.npy"))
                sup = torch.tensor(np.stack(s_list), dtype=torch.float32).to(device)
                qry = torch.tensor(np.stack(q_list), dtype=torch.float32).to(device)
                emb = model(torch.cat([sup, qry], dim=0))
                _, av = prototypical_loss(emb[:n_way_v * n_shot_v],
                                          emb[n_way_v * n_shot_v:],
                                          n_way_v, n_shot_v, N_QUERY)
                val_accs.append(av)

        if val_accs:
            val_acc = np.mean(val_accs)
            print(f" | Val Acc: {val_acc:.4f}", end="")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_out)
                print(" ✓", end="")

    print()

    with open(log_out, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, phase, config_tag, loss.item(), acc,
                         val_acc if val_acc is not None else "", current_lr])

if best_val_acc == 0.0:
    print("No val checkpoint — saving final model")
    torch.save(model.state_dict(), model_out)

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
print(f"Model → {model_out}")

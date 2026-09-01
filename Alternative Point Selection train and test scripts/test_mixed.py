"""
test_mixed.py

Evaluates the single mixed-trained PointNet++ model across all 15 way×K
configurations and prints R@1 and R@5 tables matching the reference slide.

Reads:
  all_config_train_output/best_model_mixed.pth
  all_config_train_output/split_mixed.json

Writes:
  all_config_test_output/test_results_mixed_<tag>.csv  — per-episode results
  all_config_test_output/test_summary_mixed.csv         — full summary

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
from model import PointNetPlusPlus, prototypical_loss

# ── Paths ─────────────────────────────────────────────────────────────────────
MANIFEST   = "/home/changj9/projects/facebase/manifest.csv"
REG_DIR    = Path("/home/changj9/projects/facebase/no_registration_numpy_files")
TRAIN_DIR  = Path("/home/changj9/projects/facebase/all_config_one_model/train_output")
TEST_OUT   = Path("/home/changj9/projects/facebase/all_config_one_model/test_output")
TEST_OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = TRAIN_DIR / "best_model_mixed.pth"
SPLIT_PATH = TRAIN_DIR / "split_mixed.json"

# ── Evaluation configurations ─────────────────────────────────────────────────
EVAL_WAYS  = [5, 10]
EVAL_SHOTS = [1, 3, 5]
N_QUERY    = 4
N_EPISODES = 1000
EMBED_DIM  = 512
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── Load manifest and split ───────────────────────────────────────────────────
df = pd.read_csv(MANIFEST).drop_duplicates("FBID")[["FBID", "Syndrome"]]

with open(SPLIT_PATH) as f:
    split = json.load(f)

test_syndromes = split["test"]
print(f"Test syndromes: {test_syndromes}\n")

# ── Load model ────────────────────────────────────────────────────────────────
model = PointNetPlusPlus(embed_dim=EMBED_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_test_map(n_shot, n_query):
    test_map = {}
    for s in test_syndromes:
        fbids = df[df["Syndrome"] == s]["FBID"].tolist()
        fbids = [f for f in fbids if (REG_DIR / f"{f}.npy").exists()]
        if len(fbids) >= n_shot + n_query:
            test_map[s] = fbids
    return test_map


def sample_episode(syndrome_map, n_way, n_shot, n_query):
    syndromes = random.sample(list(syndrome_map.keys()), n_way)
    support_list, query_list = [], []
    for s in syndromes:
        fbids = random.sample(syndrome_map[s], n_shot + n_query)
        for fbid in fbids[:n_shot]:
            support_list.append(np.load(REG_DIR / f"{fbid}.npy"))
        for fbid in fbids[n_shot:]:
            query_list.append(np.load(REG_DIR / f"{fbid}.npy"))
    support = torch.tensor(np.stack(support_list), dtype=torch.float32).to(device)
    query   = torch.tensor(np.stack(query_list),   dtype=torch.float32).to(device)
    return support, query


def compute_r_at_k(support_emb, query_emb, n_way, n_shot, n_query, k):
    embed_dim  = support_emb.shape[-1]
    support    = support_emb.view(n_way, n_shot, embed_dim)
    prototypes = support.mean(dim=1)
    queries    = query_emb.unsqueeze(1)
    protos     = prototypes.unsqueeze(0)
    dists      = torch.sum((queries - protos) ** 2, dim=-1)
    device_    = support_emb.device
    labels     = torch.arange(n_way, device=device_).repeat_interleave(n_query)
    topk_k     = min(k, n_way)
    _, topk_idx = torch.topk(dists, topk_k, dim=-1, largest=False)
    labels_exp  = labels.unsqueeze(1).expand_as(topk_idx)
    correct     = (topk_idx == labels_exp).any(dim=-1).float()
    return correct.mean().item()


# ── Evaluate all way×K combinations ──────────────────────────────────────────
summary_rows = []

for n_shot in EVAL_SHOTS:
    test_map = build_test_map(n_shot, N_QUERY)
    print(f"\nK={n_shot}: {len(test_map)} test syndromes with >={n_shot+N_QUERY} subjects")

    for n_way in EVAL_WAYS:
        actual_n_way = len(test_map) if n_way == "all" else n_way
        tag = f"{n_way}w{n_shot}s"

        if len(test_map) < actual_n_way:
            print(f"  [{tag}] SKIPPED — only {len(test_map)} test syndromes")
            summary_rows.append({
                "config": tag, "n_way": actual_n_way, "n_shot": n_shot,
                "R@1": None, "R@1_ci95": None,
                "R@5": None, "R@5_ci95": None,
                "chance": round(1/actual_n_way, 4) if actual_n_way > 0 else None
            })
            continue

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        r1_list, r5_list = [], []
        with torch.no_grad():
            for _ in range(N_EPISODES):
                support, query = sample_episode(test_map, actual_n_way, n_shot, N_QUERY)
                all_pts  = torch.cat([support, query], dim=0)
                emb      = model(all_pts)
                s_emb    = emb[:actual_n_way * n_shot]
                q_emb    = emb[actual_n_way * n_shot:]
                r1_list.append(compute_r_at_k(s_emb, q_emb, actual_n_way, n_shot, N_QUERY, k=1))
                r5_list.append(compute_r_at_k(s_emb, q_emb, actual_n_way, n_shot, N_QUERY, k=5))

        mean_r1 = np.mean(r1_list)
        mean_r5 = np.mean(r5_list)
        ci95_r1 = 1.96 * np.std(r1_list) / np.sqrt(N_EPISODES)
        ci95_r5 = 1.96 * np.std(r5_list) / np.sqrt(N_EPISODES)
        chance  = 1 / actual_n_way

        print(f"  [{tag}] R@1: {mean_r1:.4f} ± {ci95_r1:.4f} | "
              f"R@5: {mean_r5:.4f} ± {ci95_r5:.4f} (chance: {chance:.4f})")

        pd.DataFrame({"episode": range(1, N_EPISODES+1),
                      "r_at_1": r1_list, "r_at_5": r5_list})\
          .to_csv(TEST_OUT / f"test_results_mixed_{tag}.csv", index=False)

        summary_rows.append({
            "config": tag, "n_way": actual_n_way, "n_shot": n_shot,
            "R@1": round(mean_r1, 4), "R@1_ci95": round(ci95_r1, 4),
            "R@5": round(mean_r5, 4), "R@5_ci95": round(ci95_r5, 4),
            "chance": round(chance, 4)
        })

# ── Save summary ──────────────────────────────────────────────────────────────
summary = pd.DataFrame(summary_rows)
summary.to_csv(TEST_OUT / "test_summary_mixed.csv", index=False)

# ── Print tables ──────────────────────────────────────────────────────────────
def print_table(metric):
    print(f"\n── {metric} Table ──────────────────────────────────────────")
    print(f"{'way':<12} {'K=1':>8} {'K=3':>8} {'K=5':>8} {'K=10':>8} {'K=20':>8}")
    print("-" * 52)
    for n_way in [5, 10, "all"]:
        row = []
        for n_shot in [1, 3, 5, 10, 20]:
            tag   = f"{n_way}w{n_shot}s"
            match = summary[summary["config"] == tag]
            if len(match) > 0 and match[metric].values[0] is not None:
                row.append(f"{match[metric].values[0]:.3f}")
            else:
                row.append("N/A")
        label = f"{n_way}-way" if n_way != "all" else "all-way"
        print(f"{label:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

print_table("R@1")
print_table("R@5")
print(f"\nFull results saved to: {TEST_OUT}")

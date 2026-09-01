"""
model.py

Defines the PointNet++ encoder and prototypical network.
Imported by both train.py and test.py.

PointNet++ encoder:
  Input : (B, N, 3) — batch of point clouds
  Output: (B, 256)  — embedding vectors

Dependencies: torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────
def farthest_point_sample(xyz, n_points):
    """Sample n_points from xyz using farthest point sampling.
    xyz: (B, N, 3)
    returns: (B, n_points) indices"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return centroids


def index_points(points, idx):
    """Index points by idx.
    points: (B, N, C)
    idx   : (B, S) or (B, S, K)
    returns indexed points"""
    B = points.shape[0]
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, dtype=torch.long, device=device)\
        .view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def ball_query(radius, n_sample, xyz, new_xyz):
    """Find n_sample neighbors within radius for each point in new_xyz.
    xyz    : (B, N, 3)
    new_xyz: (B, S, 3)
    returns: (B, S, n_sample) indices"""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device

    group_idx = torch.arange(N, dtype=torch.long, device=device)\
        .view(1, 1, N).repeat(B, S, 1)
    sqrdists = torch.sum(
        (new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :n_sample]

    # Replace out-of-radius indices with first valid index
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, n_sample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(n_points, radius, n_sample, xyz, points):
    """FPS + ball query grouping.
    xyz   : (B, N, 3)
    points: (B, N, C) or None
    returns new_xyz (B, n_points, 3), new_points (B, n_points, n_sample, 3+C)"""
    B, N, _ = xyz.shape
    S = n_points

    fps_idx  = farthest_point_sample(xyz, S)        # (B, S)
    new_xyz  = index_points(xyz, fps_idx)            # (B, S, 3)
    idx      = ball_query(radius, n_sample, xyz, new_xyz)  # (B, S, K)
    grouped_xyz = index_points(xyz, idx)             # (B, S, K, 3)
    grouped_xyz -= new_xyz.unsqueeze(2)              # relative coords

    if points is not None:
        grouped_points = index_points(points, idx)   # (B, S, K, C)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


# ── Set Abstraction Layer ─────────────────────────────────────────────────────
class SetAbstraction(nn.Module):
    def __init__(self, n_points, radius, n_sample, in_channel, mlp_channels):
        super().__init__()
        self.n_points  = n_points
        self.radius    = radius
        self.n_sample  = n_sample

        layers = []
        last_c = in_channel
        for out_c in mlp_channels:
            layers += [
                nn.Conv2d(last_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """xyz: (B,N,3), points: (B,N,C) or None
        returns new_xyz (B,S,3), new_points (B,S,C')"""
        new_xyz, new_points = sample_and_group(
            self.n_points, self.radius, self.n_sample, xyz, points)
        # new_points: (B, S, K, C) → (B, C, K, S) for Conv2d
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0]  # (B, C', S)
        new_points = new_points.permute(0, 2, 1)       # (B, S, C')
        return new_xyz, new_points


# ── PointNet++ Encoder ────────────────────────────────────────────────────────
class PointNetPlusPlus(nn.Module):
    """
    Two set abstraction layers + global pooling → 256-dim embedding.
    Input : (B, N, 3)
    Output: (B, 256)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.sa1 = SetAbstraction(
            n_points=512, radius=0.2, n_sample=32,
            in_channel=3, mlp_channels=[64, 64, 128])
        self.sa2 = SetAbstraction(
            n_points=128, radius=0.4, n_sample=64,
            in_channel=128 + 3, mlp_channels=[128, 128, 256])

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim)
        )

    def forward(self, xyz):
        """xyz: (B, N, 3)"""
        xyz1, pts1 = self.sa1(xyz, None)
        _,    pts2 = self.sa2(xyz1, pts1)
        x = pts2.mean(dim=1)   # global average pooling → (B, 256)
        x = self.fc(x)
        return x               # (B, embed_dim)


# ── Prototypical Network ──────────────────────────────────────────────────────
def prototypical_loss(support_embeddings, query_embeddings,
                      n_way, n_shot, n_query):
    """
    Compute prototypical network loss and accuracy.

    support_embeddings: (n_way * n_shot,  embed_dim)
    query_embeddings  : (n_way * n_query, embed_dim)
    """
    embed_dim = support_embeddings.shape[-1]

    # Reshape and compute prototypes
    support = support_embeddings.view(n_way, n_shot, embed_dim)
    prototypes = support.mean(dim=1)                          # (n_way, embed_dim)

    # Euclidean distances from each query to each prototype
    queries = query_embeddings.unsqueeze(1)                   # (n_way*n_query, 1, D)
    protos  = prototypes.unsqueeze(0)                         # (1, n_way, D)
    dists   = torch.sum((queries - protos) ** 2, dim=-1)      # (n_way*n_query, n_way)

    # Labels: query i belongs to class i // n_query
    device = support_embeddings.device
    labels = torch.arange(n_way, device=device).repeat_interleave(n_query)

    loss = F.cross_entropy(-dists, labels)
    preds = torch.argmin(dists, dim=-1)
    acc = (preds == labels).float().mean().item()

    return loss, acc

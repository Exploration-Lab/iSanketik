"""
Video-level pose embedding extraction for ISL topic modeling.

Loads a trained self-supervised pose encoder (MoCo-style backbone) and runs
inference over a directory of merged, video-level pose tensors, producing a
single embedding matrix along with row-aligned metadata. The resulting
embeddings and metadata are intended as input to downstream topic modeling
(e.g. BERTopic clustering) over the ISL video corpus.

Expected inputs:
    - A backbone checkpoint (either raw encoder weights or a full training
      checkpoint containing an "encoder_q_state" entry).
    - A directory of per-video .pt files, each holding a padded pose sequence
      with an accompanying validity mask (see VideoLevelPoseDataset for the
      exact schema).

Outputs (written to --output_dir):
    - embeddings.npy      : [N, output_embed_dim] float32 array
    - topic_modeling.csv         : per-video metadata, row-aligned with embeddings
"""

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Model definition
#
# Mirrors the architecture used in the training script. Kept self-contained
# here (rather than imported) so this extraction script has no dependency
# on the training codebase and can be run/shared independently.
# ---------------------------------------------------------------------------

class Config:
    pose_dim = 522
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    d_feedforward = 2048
    dropout = 0.1
    output_embed_dim = 256
    proj_dim = 128
    max_position_embeddings = 4096


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalPooling(nn.Module):
    """Attention-weighted pooling over the time axis, respecting a per-frame validity mask."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x, mask_frame: Optional[torch.Tensor] = None):
        logits = self.attn(x).squeeze(-1)
        if mask_frame is not None:
            # Guard against sequences with no valid frames at all, which would
            # otherwise soften every logit to -inf and produce a NaN softmax.
            all_bad = ~mask_frame.any(dim=1, keepdim=True)
            safe = mask_frame | all_bad
            logits = logits.masked_fill(~safe, float("-inf"))
        w = F.softmax(logits, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class PoseEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.input_proj = nn.Linear(config.pose_dim, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_position_embeddings, config.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.pooling = TemporalPooling(config.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_embed_dim),
        )
        self.recon_head = nn.Linear(config.d_model, config.pose_dim)

    def encode(self, x, mask_frame=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        pad_mask = (~mask_frame) if mask_frame is not None else None
        seq = self.transformer(x, src_key_padding_mask=pad_mask)
        pooled = self.pooling(seq, mask_frame)
        emb = self.output_proj(pooled)
        return emb

    def forward(self, x, mask_frame=None):
        return self.encode(x, mask_frame)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoLevelPoseDataset(Dataset):
    """
    Reads video-level .pt files produced by the merge_video preprocessing step.

    Each file is expected to contain:
        video_id     : str
        X            : Tensor [L, 522]  -- pose keypoint coordinates
        M            : Tensor [L, 522]  -- boolean validity mask
        C            : Tensor [L, 522]  -- per-keypoint confidence (unused here)
        root         : Tensor [L, ...]
        scale        : Tensor [L, ...]
        L            : Tensor([total_L]) -- single-element tensor holding sequence length
        src_fps      : float
        num_segments : int
    """

    def __init__(self, pt_files: List[str], max_seq_len: int = 4096):
        self.pt_files = pt_files
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx: int) -> Dict:
        path = self.pt_files[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)

        X = data["X"]
        M = data["M"]
        L = int(data["L"].item())

        # Clip to the encoder's positional embedding range; longer videos are
        # rare and truncating the tail keeps memory usage predictable.
        if L > self.max_seq_len:
            X = X[:self.max_seq_len]
            M = M[:self.max_seq_len]
            L = self.max_seq_len

        return {
            "X": X,
            "M": M,
            "L": L,
            "video_id": str(data["video_id"]),
            "src_fps": float(data["src_fps"]) if "src_fps" in data else -1.0,
            "num_segments": int(data["num_segments"]) if "num_segments" in data else -1,
            "file": os.path.basename(path),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Pads a batch of variable-length pose sequences and derives a frame-level mask."""
    B = len(batch)
    pose_dim = 522
    max_len = max(s["L"] for s in batch)

    Xp = torch.zeros(B, max_len, pose_dim)
    Mfp = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(batch):
        l = s["L"]
        Xp[i, :l] = s["X"]
        # A frame counts as valid if at least one of its keypoints is valid.
        Mfp[i, :l] = s["M"].any(dim=1)

    return {
        "X": Xp,
        "M_frame": Mfp,
        "L": [s["L"] for s in batch],
        "video_id": [s["video_id"] for s in batch],
        "src_fps": [s["src_fps"] for s in batch],
        "num_segments": [s["num_segments"] for s in batch],
        "file": [s["file"] for s in batch],
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_backbone(checkpoint_path: str, device: str) -> PoseEncoder:
    """Loads the trained encoder in eval mode, accepting either a raw or full-checkpoint state dict."""
    cfg = Config()
    enc = PoseEncoder(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Full training checkpoints (checkpoint_epoch_XXX.pt) nest the encoder
    # weights under this key; best_backbone.pt stores them at the top level.
    if "encoder_q_state" in state:
        state = state["encoder_q_state"]

    enc.load_state_dict(state)
    enc.eval()
    print(f"[backbone] Loaded  {checkpoint_path}")
    print(f"[backbone] Params  {sum(p.numel() for p in enc.parameters()):,}")
    return enc


def extract(
    encoder: PoseEncoder,
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
) -> None:
    """Runs the encoder over every video in data_dir and writes embeddings, metadata, and stats."""
    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    pt_files = [f for f in pt_files if os.path.isfile(f)]
    print(f"[data] Found {len(pt_files)} .pt files in {data_dir}")
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {data_dir}")

    dataset = VideoLevelPoseDataset(pt_files, max_seq_len=Config.max_position_embeddings)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # preserve file order so metadata row i matches embeddings row i
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    all_embs: List[np.ndarray] = []
    all_video_ids: List[str] = []
    all_lengths: List[int] = []
    all_src_fps: List[float] = []
    all_num_segs: List[int] = []
    all_files: List[str] = []

    t0 = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            X = batch["X"].to(device)
            M_frame = batch["M_frame"].to(device)

            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                emb = encoder(X, M_frame)  # [B, output_embed_dim]

            all_embs.append(emb.cpu().float().numpy())
            all_video_ids.extend(batch["video_id"])
            all_lengths.extend(batch["L"])
            all_src_fps.extend(batch["src_fps"])
            all_num_segs.extend(batch["num_segments"])
            all_files.extend(batch["file"])

            if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
                done = min((bi + 1) * batch_size, len(pt_files))
                elapsed = time.time() - t0
                speed = done / max(elapsed, 1e-6)
                eta = (len(pt_files) - done) / max(speed, 1e-6)
                print(f"  [{bi + 1}/{len(loader)}]  "
                      f"{done}/{len(pt_files)} videos  "
                      f"{elapsed:.1f}s elapsed  ETA {eta:.0f}s")

    embeddings = np.concatenate(all_embs, axis=0)
    print(f"\n[extract] Shape={embeddings.shape}  "
          f"dtype={embeddings.dtype}  "
          f"time={time.time() - t0:.1f}s")

    emb_path = os.path.join(output_dir, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"[save] embeddings  -> {emb_path}")

    out_meta = pd.DataFrame({
        "video_id": all_video_ids,
        "length": all_lengths,
        "src_fps": all_src_fps,
        "num_segments": all_num_segs,
        "file": all_files,
    })
    meta_path = os.path.join(output_dir, "metadata.csv")
    out_meta.to_csv(meta_path, index=False)
    print(f"[save] metadata    -> {meta_path}")

    # Basic sanity checks before handing embeddings off to topic modeling.
    emb_std = embeddings.std(axis=0)
    norms = np.linalg.norm(embeddings, axis=1)
    nan_rows = int(np.isnan(embeddings).any(axis=1).sum())
    inf_rows = int(np.isinf(embeddings).any(axis=1).sum())
    dead_dims = int((emb_std < 1e-6).sum())

    stats = {
        "n_videos": int(embeddings.shape[0]),
        "embed_dim": int(embeddings.shape[1]),
        "nan_rows": nan_rows,
        "inf_rows": inf_rows,
        "dead_dims": dead_dims,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "emb_global_mean": float(embeddings.mean()),
        "emb_global_std": float(emb_std.mean()),
        "length_mean": float(np.mean(all_lengths)),
        "length_max": int(np.max(all_lengths)),
        "length_min": int(np.min(all_lengths)),
    }
    stats_path = os.path.join(output_dir, "embedding_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] stats       -> {stats_path}")

    print("\n" + "=" * 52)
    print("EMBEDDING HEALTH CHECK")
    print("=" * 52)
    print(f"  Videos extracted  : {stats['n_videos']}")
    print(f"  NaN rows          : {nan_rows}  {'WARNING' if nan_rows > 0 else 'OK'}")
    print(f"  Inf rows          : {inf_rows}  {'WARNING' if inf_rows > 0 else 'OK'}")
    print(f"  Dead dims (0 var) : {dead_dims}/256  "
          f"{'WARNING' if dead_dims > 10 else 'OK'}")
    print(f"  L2 norm  mean/std : {stats['norm_mean']:.3f} / {stats['norm_std']:.3f}")
    print(f"  Emb std (avg/dim) : {stats['emb_global_std']:.4f}")
    print(f"  Video length      : min={stats['length_min']}  "
          f"mean={stats['length_mean']:.0f}  max={stats['length_max']} frames")
    print("=" * 52)
    if nan_rows == 0 and inf_rows == 0 and dead_dims < 10:
        print("Embeddings look healthy. Ready for topic modeling.")
    else:
        print("Issues found — inspect embedding_stats.json before proceeding.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract video-level pose embeddings for ISL topic modeling")
    p.add_argument("--backbone", default="./best_backbone.pt",
                    help="Path to best_backbone.pt")
    p.add_argument("--data_dir", default="./video_level_pose",
                    help="Directory containing {video_id}.pt files (output of the merge_video pipeline)")
    p.add_argument("--output_dir", default="./train_encoder_output/embeddings",
                    help="Where to write embeddings.npy and metadata.csv")
    p.add_argument("--batch_size", type=int, default=32,
                    help="Reduce if OOM on long videos (default 32)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    encoder = load_backbone(args.backbone, device=device)
    extract(
        encoder=encoder,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )


if __name__ == "__main__":
    main()
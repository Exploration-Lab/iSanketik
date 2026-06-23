"""
extract_I3D_embeddings.py
─────────────────────────
Extract 1024-d video embeddings using the pretrained I3D RGB backbone.
  Architecture : Inflated 3D ConvNet (Inception-V1 base)
  Reference    : Carreira & Zisserman, CVPR 2017
  Weights repo : https://github.com/piergiaj/pytorch-i3d
                 RGB Kinetics pretrained → rgb_kinetics.pt
                 RGB ImageNet+Kinetics   → rgb_imagenet.pt

HOW TO USE
──────────
1.  Download pretrained weights from the repo above.
    Recommended: rgb_imagenet.pt  (better generalisation)
    Place at  CKPT_PATH  below (or pass  --ckpt  on the CLI).

2.  Run:
        python extract_I3D_embeddings.py

    Or with explicit paths:
        python extract_I3D_embeddings.py \\
            --ckpt   /path/to/rgb_imagenet.pt \\
            --csv    /path/to/split.csv \\
            --videos /path/to/mp4_folder \\
            --output /path/to/output_dir

OUTPUT
──────
    <output_dir>/embeddings.npz
        uids        : string array  [N]
        embeddings  : float32 array [N, 1024]

    Also saved per-video as  <output_dir>/<uid>.npy  for easy lookup.
"""

# ── 0. standard imports ───────────────────────────────────────────────────────
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── 1. default paths ──────────────────────────────────────────────────────────
DEFAULT_CKPT = "/DATA3/ashishu23/ashishu23_2/pytorch-i3d/models/rgb_imagenet.pt"
DEFAULT_VIDEOS = (
    "/DATAS3/Sign_language_data_extended/FINAL_data/All_video_sign_language"
    "/Dictonary_dataset/ISLRTC_Vocab_cropped_renamed"
)
DEFAULT_CSV = (
    "/DATAS3/Sign_language_data_extended/FINAL_data/All_video_sign_language"
    "/Workshop_Shared_task/Word_gloss_recognition/dataset/val.csv"
)
DEFAULT_OUTPUT = "/DATA3/ashishu23/ashishu23_2/embeddings_islrtc_val_I3D"

NUM_FRAMES = 64
IMG_SIZE   = 224
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet mean/std used by the official I3D weights
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ── 2. I3D architecture (self-contained) ──────────────────────────────────────
# Faithful re-implementation of piergiaj/pytorch-i3d for RGB stream.

class Unit3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0, activation_fn=F.relu,
                 use_batch_norm=True, use_bias=False, name="unit_3d"):
        super().__init__()
        self._use_batch_norm   = use_batch_norm
        self._activation_fn    = activation_fn
        self.conv3d = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=use_bias)
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x, inplace=False)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        assert len(out_channels) == 6
        self.b0 = Unit3D(in_channels,      out_channels[0], name=name + "/Branch_0/Conv3d_0a_1x1")
        self.b1a = Unit3D(in_channels,     out_channels[1], name=name + "/Branch_1/Conv3d_0a_1x1")
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_size=(3, 3, 3), padding=1,
                          name=name + "/Branch_1/Conv3d_0b_3x3")
        self.b2a = Unit3D(in_channels,     out_channels[3], name=name + "/Branch_2/Conv3d_0a_1x1")
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_size=(3, 3, 3), padding=1,
                          name=name + "/Branch_2/Conv3d_0b_3x3")
        self.b3a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.b3b = Unit3D(in_channels,     out_channels[5], name=name + "/Branch_3/Conv3d_0b_1x1")

    def forward(self, x):
        b0  = self.b0(x)
        b1  = self.b1b(self.b1a(x))
        b2  = self.b2b(self.b2a(x))
        b3  = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """
    I3D RGB model up to and including Mixed_5c (before logits).
    forward() returns the feature tensor [B, 1024, T', H', W'].
    """

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Conv3d_2b_1x1",
        "Conv3d_2c_3x3", "MaxPool3d_3a_3x3",
        "Mixed_3b", "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b", "Mixed_4c", "Mixed_4d", "Mixed_4e", "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b", "Mixed_5c",
    )

    def __init__(self):
        super().__init__()
        # Stem
        self.Conv3d_1a_7x7   = Unit3D(3,   64, kernel_size=(7,7,7), stride=(2,2,2), padding=3)
        self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.Conv3d_2b_1x1   = Unit3D(64,  64)
        self.Conv3d_2c_3x3   = Unit3D(64, 192, kernel_size=(3,3,3), padding=1)
        self.MaxPool3d_3a_3x3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        # Mixed 3x
        self.Mixed_3b = InceptionModule( 192, [64,  96, 128,  16,  32,  32], "Mixed_3b")
        self.Mixed_3c = InceptionModule( 256, [128, 128, 192,  32,  96,  64], "Mixed_3c")
        self.MaxPool3d_4a_3x3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        # Mixed 4x
        self.Mixed_4b = InceptionModule( 480, [192,  96, 208,  16,  48,  64], "Mixed_4b")
        self.Mixed_4c = InceptionModule( 512, [160, 112, 224,  24,  64,  64], "Mixed_4c")
        self.Mixed_4d = InceptionModule( 512, [128, 128, 256,  24,  64,  64], "Mixed_4d")
        self.Mixed_4e = InceptionModule( 512, [112, 144, 288,  32,  64,  64], "Mixed_4e")
        self.Mixed_4f = InceptionModule( 528, [256, 160, 320,  32, 128, 128], "Mixed_4f")
        self.MaxPool3d_5a_2x2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=0)
        # Mixed 5x  → 1024 channels
        # Mixed_5b(832) → 256+320+128+128 = 832 out
        # Mixed_5c(832) → 384+384+128+128 = 1024 out
        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128], "Mixed_5b")
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128], "Mixed_5c")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] → [B, 1024, T', H', W']"""
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        return x   # [B, 1024, T', H', W']


# ── 3. model wrapper ──────────────────────────────────────────────────────────

class I3DEmbedder(nn.Module):
    """
    Wraps InceptionI3d for embedding extraction.

    forward(videos) → (B, 1024)  via global average pooling on Mixed_5c output.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.backbone = InceptionI3d()
        self._load_weights(ckpt_path)
        self.backbone.to(device).eval()
        # Keep normalisation tensors on the right device
        self._mean = _MEAN.to(device)
        self._std  = _STD.to(device)

    def _load_weights(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            print(
                f"\n[WARNING] Checkpoint not found: {ckpt_path}\n"
                "  → Running with random weights.\n"
                "  → Download from https://github.com/piergiaj/pytorch-i3d\n"
                "    rgb_imagenet.pt  or  rgb_kinetics.pt\n"
            )
            return

        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")

        # state dict may be nested under a key
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        # strip common prefixes
        for prefix in ("module.", "backbone.", "i3d."):
            if all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]: v for k, v in state.items()}
                break

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(unexpected)
        print(f"  Loaded {n_loaded}/{len(state)} weight tensors.")
        if missing:
            print(f"  Missing  ({len(missing)}): {missing[:4]} …")
        if unexpected:
            print(f"  Ignored  ({len(unexpected)}): {unexpected[:4]} …")

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            videos : [B, C, T, H, W]  float32 in [0, 1]
        Returns:
            embeddings : [B, 1024]  float32  (on CPU)
        """
        videos = videos.to(self.device)
        # ImageNet normalisation
        videos = (videos - self._mean) / self._std
        feat   = self.backbone(videos)               # [B, 1024, T', H', W']
        pooled = F.adaptive_avg_pool3d(feat, 1)      # [B, 1024, 1, 1, 1]
        emb    = pooled.flatten(1)                   # [B, 1024]
        return emb.cpu().float()


# ── 4. video preprocessing ────────────────────────────────────────────────────

def _decode_all_frames(cap: cv2.VideoCapture) -> list:
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def preprocess_video(path: str,
                     num_frames: int = NUM_FRAMES,
                     img_size:   int = IMG_SIZE) -> torch.Tensor:
    """
    Load an MP4 and return a float32 tensor  [C, T, H, W]  in [0, 1].

    Sampling strategy (matches NLA-SLR test-time 'origin' mode):
      - If video has ≥ num_frames: uniform sample across full length.
      - If video is shorter      : repeat-pad to num_frames.
    Each frame is centre-cropped to a square then resized to img_size×img_size.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        frames = _decode_all_frames(cap)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"Empty video: {path}")

    total = len(frames)
    if total >= num_frames:
        indices = np.round(np.linspace(0, total - 1, num_frames)).astype(int)
    else:
        reps   = (num_frames // total) + 1
        frames = (frames * reps)[:num_frames]
        indices = list(range(num_frames))

    out = []
    for i in indices:
        f    = frames[i]
        h, w = f.shape[:2]
        side = min(h, w)
        y0   = (h - side) // 2
        x0   = (w - side) // 2
        crop = f[y0: y0 + side, x0: x0 + side]
        out.append(cv2.resize(crop, (img_size, img_size),
                              interpolation=cv2.INTER_LINEAR))

    arr    = np.stack(out).astype(np.float32) / 255.0   # [T, H, W, 3]
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # [C, T, H, W]
    return tensor


# ── 5. dataset processing ─────────────────────────────────────────────────────

def run_extraction(
    df:         pd.DataFrame,
    video_dir:  str,
    embedder:   I3DEmbedder,
    output_dir: str,
    batch_size: int = BATCH_SIZE,
    num_frames: int = NUM_FRAMES,
) -> None:
    """
    Iterate all rows in df, extract embeddings in mini-batches, and save:
      <output_dir>/embeddings.npz   — all-in-one (uids + embeddings)
      <output_dir>/<uid>.npy        — per-video for easy random access
    """
    os.makedirs(output_dir, exist_ok=True)

    uids        = df["uid"].tolist()
    all_embs    = []
    valid_uids  = []
    skipped     = []

    batch_tensors: list = []
    batch_uids:    list = []

    def _flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors, dim=0)   # [B, C, T, H, W]
        embs  = embedder(batch)                      # [B, 1024]
        for uid, emb in zip(batch_uids, embs):
            np.save(os.path.join(output_dir, f"{uid}.npy"), emb.numpy())
        all_embs.append(embs.numpy())
        valid_uids.extend(batch_uids)

    for uid in tqdm(uids, desc="Extracting embeddings", unit="video"):
        video_path = os.path.join(video_dir, f"{uid}.mp4")
        if not os.path.isfile(video_path):
            skipped.append(uid)
            continue

        try:
            tensor = preprocess_video(video_path, num_frames=num_frames)
        except Exception as exc:
            print(f"  [skip] {uid}: {exc}")
            skipped.append(uid)
            continue

        batch_tensors.append(tensor)
        batch_uids.append(uid)

        if len(batch_tensors) >= batch_size:
            _flush()
            batch_tensors.clear()
            batch_uids.clear()

    _flush()

    if not all_embs:
        print("\nNo embeddings extracted. Check VIDEO_DIR and CKPT_PATH.")
        return

    embeddings = np.concatenate(all_embs, axis=0)
    npz_path   = os.path.join(output_dir, "embeddings.npz")
    np.savez(npz_path,
             uids       = np.array(valid_uids),
             embeddings = embeddings)

    print(f"\n{'─'*50}")
    print(f"  Saved {embeddings.shape[0]} embeddings → {npz_path}")
    print(f"  Embedding shape : {embeddings.shape}   (each video = 1024-d vector)")
    if skipped:
        print(f"  Skipped {len(skipped)} videos (missing files / decode errors)")
    print(f"{'─'*50}")


# ── 6. CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract I3D embeddings from sign language videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",   default=DEFAULT_CKPT,   help="I3D checkpoint (.pt / .pth)")
    p.add_argument("--videos", default=DEFAULT_VIDEOS, help="Directory with UID.mp4 files")
    p.add_argument("--csv",    default=DEFAULT_CSV,    help="CSV file with 'uid' column")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    p.add_argument("--device", default=DEVICE,         help="cuda or cpu")
    p.add_argument("--batch",  type=int, default=BATCH_SIZE, help="Batch size")
    p.add_argument("--frames", type=int, default=NUM_FRAMES,
                   help="Frames sampled per video")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Device      : {args.device}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Video dir   : {args.videos}")
    print(f"CSV         : {args.csv}")
    print(f"Output dir  : {args.output}")
    print(f"Batch size  : {args.batch}   Frames: {args.frames}")
    print()

    embedder = I3DEmbedder(ckpt_path=args.ckpt, device=args.device)

    df = pd.read_csv(args.csv)
    print(f"\nLoaded CSV: {len(df)} rows\n")

    run_extraction(
        df         = df,
        video_dir  = args.videos,
        embedder   = embedder,
        output_dir = args.output,
        batch_size = args.batch,
        num_frames = args.frames,
    )


# ── 7. convenience function for notebook/interactive use ─────────────────────

def extract_single(video_path: str,
                   ckpt_path:  str = DEFAULT_CKPT,
                   device:     str = DEVICE) -> np.ndarray:
    """
    Extract a 1024-d embedding for a single video.  Useful in a notebook.

    Example:
        from extract_I3D_embeddings import extract_single
        emb = extract_single("/path/to/video.mp4")
        print(emb.shape)   # (1024,)
    """
    embedder = I3DEmbedder(ckpt_path=ckpt_path, device=device)
    tensor   = preprocess_video(video_path)
    emb      = embedder(tensor.unsqueeze(0))   # [1, 1024]
    return emb.numpy()[0]                      # (1024,)

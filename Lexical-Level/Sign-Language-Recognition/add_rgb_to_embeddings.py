#!/usr/bin/env python3
"""
Add RGB pixel-derived features from .mp4 videos to existing embeddings.

For each sample in the embeddings metadata (matched by uid), this script:
1. Loads the corresponding .mp4 from the video directory (expected: {uid}.mp4).
2. Samples frames uniformly and extracts RGB (resize to a small spatial size).
3. Builds a compact RGB feature vector (mean per frame or flattened pixels).
4. Concatenates the RGB feature to the existing embedding and saves augmented
   embeddings + metadata.

Usage:
    python add_rgb_to_embeddings.py \\
      --embeddings /DATAS3/ashishu23_2/embeddings_islrtc_test/embeddings.npy \\
      --metadata   /DATAS3/ashishu23_2/embeddings_islrtc_test/metadata.json \\
      --video_dir  "/DATAS3/Sign_language_data_extended/FINAL_data/All_video_sign_language/Dictonary_dataset/ISLRTC_Vocab_cropped_renamed" \\
      --output_dir /DATAS3/ashishu23_2/embeddings_islrtc_test_with_rgb

Requirements: opencv-python (cv2), numpy
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python: pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_embeddings_and_metadata(emb_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict]]:
    emb_path = Path(emb_path)
    meta_path = Path(meta_path)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    emb = np.load(str(emb_path))
    with open(str(meta_path), "r") as f:
        meta = json.load(f)
    return emb, meta


def get_uid(entry: Dict) -> str:
    return str(entry.get("uid", ""))


def extract_rgb_from_video(
    video_path: str,
    num_frames: int = 256,
    resize_h: int = 32,
    resize_w: int = 32,
    mode: str = "mean",
) -> np.ndarray:
    """
    Sample frames from video and extract a compact RGB feature vector.

    Args:
        video_path: Path to .mp4 file.
        num_frames: Number of frames to sample uniformly.
        resize_h, resize_w: Spatial size for each frame.
        mode: "mean" -> (num_frames, 3) mean R,G,B per frame, flattened to 3*num_frames.
              "flatten" -> (num_frames * resize_h * resize_w * 3,) flattened pixels.

    Returns:
        1D float32 array (normalized to [0,1] from 0-255).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(
            (3 * num_frames if mode == "mean" else num_frames * resize_h * resize_w * 3),
            dtype=np.float32,
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros(
            (3 * num_frames if mode == "mean" else num_frames * resize_h * resize_w * 3),
            dtype=np.float32,
        )

    # Uniform frame indices
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames_list: List[np.ndarray] = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or frame is None:
            frames_list.append(np.zeros((resize_h, resize_w, 3), dtype=np.uint8))
            continue
        # BGR -> RGB, resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(
            frame_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR
        )
        frames_list.append(frame_small)

    cap.release()
    stack = np.stack(frames_list, axis=0)  # (num_frames, H, W, 3)
    # Normalize to [0, 1]
    stack = stack.astype(np.float32) / 255.0

    if mode == "mean":
        # (num_frames, 3)
        feature = np.mean(stack, axis=(1, 2))
        return feature.flatten()
    else:
        # flatten
        return stack.flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Add RGB features from .mp4 videos to embeddings."
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to existing embeddings.npy",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to existing metadata.json",
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory containing .mp4 files named {uid}.mp4",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for augmented embeddings.npy and metadata.json",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to sample per video (default: 32)",
    )
    parser.add_argument(
        "--resize_h",
        type=int,
        default=32,
        help="Frame resize height (default: 32)",
    )
    parser.add_argument(
        "--resize_w",
        type=int,
        default=32,
        help="Frame resize width (default: 32)",
    )
    parser.add_argument(
        "--mode",
        choices=("mean", "flatten"),
        default="mean",
        help="RGB feature: 'mean' = mean R,G,B per frame (compact); 'flatten' = all pixels (default: mean)",
    )
    parser.add_argument(
        "--missing",
        choices=("zero", "skip"),
        default="zero",
        help="If video not found: 'zero' = use zero vector; 'skip' = skip sample (default: zero)",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings and metadata...")
    emb, meta = load_embeddings_and_metadata(args.embeddings, args.metadata)
    n_samples, emb_dim = emb.shape
    print(f"Embeddings shape: {emb.shape}")

    # Compute RGB feature dimension
    if args.mode == "mean":
        rgb_dim = args.num_frames * 3
    else:
        rgb_dim = args.num_frames * args.resize_h * args.resize_w * 3
    print(f"RGB feature dim: {rgb_dim} (mode={args.mode})")

    rgb_features = np.zeros((n_samples, rgb_dim), dtype=np.float32)
    valid = np.ones(n_samples, dtype=bool)
    missing_uids: List[str] = []

    for i, entry in enumerate(tqdm(meta, desc="Videos", unit="video")):
        uid = get_uid(entry)
        video_path = video_dir / f"{uid}.mp4"
        if not video_path.exists():
            missing_uids.append(uid)
            if args.missing == "skip":
                valid[i] = False
            continue
        rgb_features[i] = extract_rgb_from_video(
            str(video_path),
            num_frames=args.num_frames,
            resize_h=args.resize_h,
            resize_w=args.resize_w,
            mode=args.mode,
        )

    if missing_uids:
        print(f"Warning: {len(missing_uids)} videos not found (first 5: {missing_uids[:5]})")
    if args.missing == "skip" and not valid.all():
        n_valid = int(valid.sum())
        print(f"Skipping {n_samples - n_valid} samples with missing video.")
        emb = emb[valid]
        rgb_features = rgb_features[valid]
        meta = [m for m, v in zip(meta, valid) if v]
        n_samples = n_valid

    # Concatenate: [original_embedding | rgb_feature]
    augmented = np.concatenate([emb, rgb_features], axis=1)
    print(f"Augmented embeddings shape: {augmented.shape}")

    out_emb = output_dir / "embeddings.npy"
    out_meta = output_dir / "metadata.json"
    np.save(str(out_emb), augmented)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    # Save a small config so you know how augmentation was done
    config = {
        "source_embeddings": str(Path(args.embeddings).resolve()),
        "source_metadata": str(Path(args.metadata).resolve()),
        "video_dir": str(video_dir.resolve()),
        "original_embedding_dim": int(emb_dim),
        "rgb_feature_dim": int(rgb_dim),
        "num_frames": args.num_frames,
        "resize_h": args.resize_h,
        "resize_w": args.resize_w,
        "mode": args.mode,
        "augmented_dim": int(augmented.shape[1]),
    }
    with open(output_dir / "rgb_augment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved augmented embeddings to {out_emb}")
    print(f"Saved metadata to {out_meta}")
    print(f"Saved config to {output_dir / 'rgb_augment_config.json'}")


if __name__ == "__main__":
    main()

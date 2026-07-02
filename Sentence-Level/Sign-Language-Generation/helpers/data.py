"""
ISL Pose Dataset
Loads pose shards and CSV text translations.
Handles variable-length sequences, normalization using root/scale,
confidence masking and text tokenization.

Pose dimensionality: 190 (was 210)
   Body  : 23 joints x 2 = 46 dims
   L hand: 21 joints x 2 = 42 dims
   R hand: 21 joints x 2 = 42 dims
   Face  : 30 joints x 2 = 60 dims

Face reduction: keeping only linguistically relevant facial signals for sign language:
   Eyebrows (3 pts each x 2): inner, arch-peak, outer, captures raise/furrow
   Eyes     (6 pts each x 2): corners, lid apex, lid nadir, captures aperture
   Lips     (8 pts)          : corners, top/bottom centers, mid-points
   Nose     (4 pts)          : bridge, tip, base-left, base-right
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, OrderedDict
import random


# Constants

BODY_JOINTS = 23
LHAND_JOINTS = 21
RHAND_JOINTS = 21
FACE_JOINTS = 30

TOTAL_JOINTS = BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS + FACE_JOINTS  # 95
POSE_DIM = TOTAL_JOINTS * 2  # 190

# Slices in the 190-dim flat vector (after joint selection + xy-only)
BODY_SLICE = slice(0, BODY_JOINTS * 2)  # 0:46
LHAND_SLICE = slice(BODY_JOINTS * 2, (BODY_JOINTS + LHAND_JOINTS) * 2)  # 46:88
RHAND_SLICE = slice((BODY_JOINTS + LHAND_JOINTS) * 2,
                    (BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS) * 2)  # 88:130
FACE_SLICE = slice((BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS) * 2, POSE_DIM)  # 130:190


# Body joint selection (upper body only)
_KEEP_BODY_INDICES = np.arange(23, dtype=np.int64)  # landmarks 0-22


# Face joint selection
_COMPACT_FACE_LOCAL = np.array([
     0,  2,  4,  6,  7,  9, 10, 13, 15, 19,
    22, 29, 30, 32, 34, 42, 53, 54, 55, 56,
    59, 61, 65, 68, 75, 76, 78, 80, 87, 98
], dtype=np.int64)  # 30 values

_KEEP_JOINT_INDICES = np.concatenate([
    _KEEP_BODY_INDICES,                # 23 upper-body joints (0-22)
    np.arange(33, 54, dtype=np.int64), # left hand (33-53)
    np.arange(54, 75, dtype=np.int64), # right hand (54-74)
    75 + _COMPACT_FACE_LOCAL           # compact face in global joint space
])  # shape (95,)

assert len(_KEEP_JOINT_INDICES) == TOTAL_JOINTS, \
    f"Expected {TOTAL_JOINTS} joints, got {len(_KEEP_JOINT_INDICES)}"


# Length-aware weighted sampler

class LengthAwareShardSampler(torch.utils.data.Sampler):
    """
    Shard-local weighted sampler that upweights longer sentences.

    weight(n) = 1 + alpha * max(0, n - pivot) / pivot
    pivot is the word count below which weight = 1.0.
    """

    def __init__(
        self,
        dataset,
        shuffle_shards: bool = True,
        pivot: float = 7.0,
        alpha: float = 1.5,
        seed: int = None,
    ):
        self.dataset = dataset
        self.shuffle_shards = shuffle_shards
        self.pivot = pivot
        self.alpha = alpha
        self.seed = seed
        self._weights = self._compute_weights()

    # Weight computation
    def _word_count(self, uid: str) -> int:
        text = self.dataset.uid2text.get(uid, "")
        return max(1, len(text.split()))

    def _sample_weight(self, n_words: int) -> float:
        """Linear upweight above pivot; minimum weight is always 1.0."""
        return 1.0 + self.alpha * max(0.0, (n_words - self.pivot) / self.pivot)

    def _compute_weights(self) -> list:
        weights = []
        for shard_path, local_idx in self.dataset.index:
            uid = None  # resolved lazily in _ensure_uid_list
            weights.append(None)
        return weights

    def _ensure_uid_list(self):
        if hasattr(self, "_uid_list"):
            return
        if getattr(self.dataset, "uid_list", None) is not None:
            self._uid_list = self.dataset.uid_list
        else:
            uid_list = []
            shard_cache = {}
            for shard_path, local_idx in self.dataset.index:
                if shard_path not in shard_cache:
                    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                    shard_cache[shard_path] = [str(u).strip() for u in shard["uid"]]
                uid_list.append(shard_cache[shard_path][local_idx])
            self._uid_list = uid_list

        self._weights = [
            self._sample_weight(self._word_count(uid))
            for uid in self._uid_list
        ]
        wt = np.array(self._weights, dtype=np.float32)
        print(
            f"[LengthAwareSampler] weight stats: "
            f"min={wt.min():.2f} mean={wt.mean():.2f} "
            f"max={wt.max():.2f} (pivot={self.pivot}, alpha={self.alpha})"
        )

    # Sampling
    def __iter__(self):
        self._ensure_uid_list()

        # Group indices by shard, preserving shard-local I/O locality
        shard_to_indices = defaultdict(list)
        for i, (shard_path, _) in enumerate(self.dataset.index):
            shard_to_indices[shard_path].append(i)

        shard_ids = list(shard_to_indices.keys())
        if self.shuffle_shards:
            random.shuffle(shard_ids)

        rng = np.random.default_rng(self.seed)
        indices = []

        for sid in shard_ids:
            idxs = np.array(shard_to_indices[sid])
            weights = np.array([self._weights[i] for i in idxs], dtype=np.float64)
            probs = weights / weights.sum()

            drawn = rng.choice(idxs, size=len(idxs), replace=True, p=probs)
            indices.extend(drawn.tolist())

        return iter(indices)

    def __len__(self):
        return len(self.dataset)


# Original ShardAwareSampler

class ShardAwareSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle_shards=True):
        self.dataset = dataset
        self.shuffle_shards = shuffle_shards

    def __iter__(self):
        shard_to_indices = defaultdict(list)
        for i, (shard_path, _) in enumerate(self.dataset.index):
            shard_to_indices[shard_path].append(i)
        shard_ids = list(shard_to_indices.keys())
        if self.shuffle_shards:
            random.shuffle(shard_ids)
        indices = []
        for sid in shard_ids:
            idxs = shard_to_indices[sid].copy()
            random.shuffle(idxs)
            indices.extend(idxs)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


# Shard index builder

def build_shard_index(shard_dir: str, csv_path: str,
                      split: str = "train", index_path: str = None):
    df = pd.read_csv(csv_path)
    df["uid"] = df["uid"].astype(str).str.strip()
    uid2text = dict(zip(df["uid"], df["text"].astype(str)))
    print(f"[{split}] Loaded {len(uid2text)} text entries from CSV.")

    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
    assert len(shard_paths) > 0, f"No shards found in {shard_dir}"
    id_to_path = {
        int(os.path.basename(p).replace("shard_", "").replace(".pt", "")): p
        for p in shard_paths
    }

    if index_path and os.path.exists(index_path):
        saved = torch.load(index_path, map_location="cpu", weights_only=False)
        raw_index = saved["index"]
        index = [(id_to_path[shard_id], local_id) for shard_id, local_id in raw_index]
        uid_list = saved.get("uid_list", None)  # None for old index files
        print(f"[{split}] Loaded {len(index)} samples from index.")
        return index, uid2text, uid_list

    # No index file: scan shards from scratch
    print(f"[{split}] No index file found, scanning shards from scratch...")
    index = []
    uid_list = []
    missing_text = 0
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        uids = shard["uid"]
        for i in range(len(uids)):
            uid = str(uids[i]).strip()
            if uid in uid2text:
                index.append((shard_path, i))
                uid_list.append(uid)
            else:
                missing_text += 1
    print(f"[{split}] Built index: {len(index)} valid samples "
          f"({missing_text} skipped, no text).")
    return index, uid2text, uid_list


# Shard cache (LRU)

class ShardCache:
    def __init__(self, max_shards: int = 8):
        self.cache = OrderedDict()
        self.max_shards = max_shards

    def get(self, shard_path: str):
        if shard_path in self.cache:
            self.cache.move_to_end(shard_path)
            return self.cache[shard_path]
        if len(self.cache) >= self.max_shards:
            self.cache.popitem(last=False)
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        self.cache[shard_path] = shard
        return shard


# Main dataset

class ISLPoseDataset(Dataset):
    """
    Returns:
        pose           : (T, 190) float32, normalized 2D pose (95 joints x 2)
        mask           : (T,) float32, valid frame mask
        length         : int, actual T
        input_ids      : (L,) int64
        attention_mask : (L,) int64
        uid            : str
    """

    def __init__(
        self,
        shard_dir: str,
        csv_path: str,
        split: str = "train",
        index_path: str = None,
        tokenizer_name: str = "google/flan-t5-base",
        max_text_len: int = 64,
        max_pose_len: int = 512,
        max_shard_cache: int = 8,
        augment: bool = False,
        vel_aug_min: float = 0.85,
        vel_aug_max: float = 1.40,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.max_pose_len = max_pose_len
        self.augment = augment
        self.split = split
        self.vel_aug_min = vel_aug_min
        self.vel_aug_max = vel_aug_max

        self.index, self.uid2text, self.uid_list = build_shard_index(
            shard_dir, csv_path, split, index_path
        )
        self.shard_cache = ShardCache(max_shard_cache)

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

    def __len__(self):
        return len(self.index)

    def _normalize_pose(self, X: np.ndarray, root: np.ndarray, scale) -> np.ndarray:
        """Center pose on root and divide by scale. root/scale may be per-frame or global."""
        X = X.copy().astype(np.float32)
        if root.ndim == 1:
            root = root[np.newaxis, :]  # (1, 2)
        root_expanded = np.tile(root, (1, POSE_DIM // 2))  # (1 or T, 190)
        if np.isscalar(scale) or (isinstance(scale, np.ndarray) and scale.ndim == 0):
            scale_val = float(scale) + 1e-6
        else:
            scale_val = scale.reshape(-1, 1) + 1e-6  # (T, 1)
        return (X - root_expanded) / scale_val

    def _velocity_augment(self, pose: np.ndarray) -> np.ndarray:
        """
        Temporal speed perturbation, resamples the sequence at a random rate.
        Range is right-skewed toward factor > 1 (slower signer, longer sequence)
        so training sees more long-duration samples, matching val/test better.
        """
        T = pose.shape[0]
        factor = np.random.uniform(self.vel_aug_min, self.vel_aug_max)
        new_T = max(10, int(T * factor))
        old_t = np.linspace(0, T - 1, T)
        new_t = np.linspace(0, T - 1, new_T)
        out = np.zeros((new_T, pose.shape[1]), dtype=np.float32)
        for d in range(pose.shape[1]):
            out[:, d] = np.interp(new_t, old_t, pose[:, d])
        return out

    def _mirror_augment(self, pose: np.ndarray) -> np.ndarray:
        """Flip left/right: negate x coords, swap lhand/rhand blocks."""
        pose = pose.copy()
        pose[:, 0::2] = -pose[:, 0::2]
        lh = pose[:, LHAND_SLICE].copy()
        rh = pose[:, RHAND_SLICE].copy()
        pose[:, LHAND_SLICE] = rh
        pose[:, RHAND_SLICE] = lh
        return pose

    def __getitem__(self, idx):
        shard_path, local_idx = self.index[idx]
        shard = self.shard_cache.get(shard_path)

        def to_np(v):
            if isinstance(v, torch.Tensor):
                return v.numpy()
            return np.array(v, dtype=np.float32)

        X = to_np(shard["X"][local_idx]).astype(np.float32)
        C = to_np(shard["C"][local_idx]).astype(np.float32)
        root = to_np(shard["root"][local_idx]).astype(np.float32)
        scale = to_np(shard["scale"][local_idx]).astype(np.float32)

        # Joint selection + xy-only
        T_orig = X.shape[0]
        X = (X.reshape(T_orig, 174, 3)
              [:, _KEEP_JOINT_INDICES, :]  # (T, 95, 3)
              [:, :, :2]                   # drop z
              .reshape(T_orig, POSE_DIM))  # (T, 190)

        # root: keep only xy
        if root.ndim == 1:
            root = root[:2]
        else:
            root = root[:, :2]

        if scale.ndim == 0:
            scale = float(scale)

        T = X.shape[0]
        X = self._normalize_pose(X, root, scale)

        # Per-frame confidence mask
        if C.ndim == 2:
            frame_conf = C.mean(axis=1)
        else:
            frame_conf = np.ones(T, dtype=np.float32)
        frame_mask = (frame_conf > 0.3).astype(np.float32)

        # NaN guard
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.zeros_like(X)
            frame_mask = np.zeros(T, dtype=np.float32)

        # Augmentation (train only)
        if self.augment and self.split == "train":
            if np.random.random() < 0.5:
                X = self._velocity_augment(X)
                T = X.shape[0]
                frame_mask = np.ones(T, dtype=np.float32)
            if np.random.random() < 0.5:
                X = self._mirror_augment(X)

        # Hard cap
        if T > self.max_pose_len:
            X = X[:self.max_pose_len]
            frame_mask = frame_mask[:self.max_pose_len]
            T = self.max_pose_len

        pose = torch.from_numpy(X)
        mask = torch.from_numpy(frame_mask)

        uid = str(shard["uid"][local_idx]).strip()
        text = self.uid2text.get(uid, "")
        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pose": pose,
            "mask": mask,
            "length": T,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "uid": uid,
        }


# Collate

def collate_fn(batch):
    poses = [b["pose"] for b in batch]
    masks = [b["mask"] for b in batch]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)

    poses_padded = pad_sequence(poses, batch_first=True, padding_value=0.0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0.0)

    return {
        "pose": poses_padded,          # (B, T_max, 190)
        "mask": masks_padded,          # (B, T_max)
        "lengths": lengths,            # (B,)
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "uids": [b["uid"] for b in batch],
    }


# Worker init

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    np.random.seed(worker_info.seed % (2**31))
    random.seed(worker_info.seed % (2**31))
    dataset = worker_info.dataset
    if not hasattr(dataset, 'shard_cache') or dataset.shard_cache is None:
        dataset.shard_cache = ShardCache(max_shards=8)


# DataLoader

def get_dataloader(
    shard_dir: str,
    csv_path: str,
    split: str = "train",
    index_path: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    tokenizer_name: str = "google/flan-t5-base",
    max_text_len: int = 64,
    max_pose_len: int = 512,
    augment: bool = False,
    pin_memory: bool = True,
    length_aware: bool = True,
    sampler_pivot: float = 7.0,
    sampler_alpha: float = 1.5,
    vel_aug_min: float = 0.85,
    vel_aug_max: float = 1.40,
):
    dataset = ISLPoseDataset(
        shard_dir=shard_dir,
        csv_path=csv_path,
        split=split,
        index_path=index_path,
        tokenizer_name=tokenizer_name,
        max_text_len=max_text_len,
        max_pose_len=max_pose_len,
        augment=augment,
        vel_aug_min=vel_aug_min,
        vel_aug_max=vel_aug_max,
    )

    if split == "train":
        if length_aware:
            sampler = LengthAwareShardSampler(
                dataset,
                shuffle_shards=True,
                pivot=sampler_pivot,
                alpha=sampler_alpha,
            )
        else:
            sampler = ShardAwareSampler(dataset)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset
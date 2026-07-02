import io
import re
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import webdataset as wds


# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Default shard directory — can be overridden per loader via shard_dir parameter
WEBDATASET_DIR = "/indian-slp/dattatreya/isign1.0plus2.0"
SHUFFLE_BUFFER = 1000
BATCH_SIZE     = 1024
NUM_WORKERS    = 4


# ─── DECODE ───────────────────────────────────────────────────────────────────

def decode_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def decode_sample(sample: dict) -> dict:
    uid = sample["__key__"]
    if isinstance(uid, bytes):
        uid = uid.decode("utf-8")

    label = sample["label.txt"]
    if isinstance(label, bytes):
        label = label.decode("utf-8")

    return {
        "uid":   uid.strip(),
        "data":  decode_npy(sample["data.npy"]),
        "label": label.strip(),
    }


# ─── AUGMENTATION ─────────────────────────────────────────────────────────────

def add_gaussian_noise(keypoints: np.ndarray, mean=0, std=0.06) -> np.ndarray:
    return keypoints + np.random.normal(mean, std, keypoints.shape)


# ─── WINDOW DATASET ───────────────────────────────────────────────────────────

NUM_SUFFIX_RE = re.compile(r"(.*?)(\d+)$")

def segment_sort_key(uid: str):
    m = NUM_SUFFIX_RE.match(uid)
    if m:
        return (1, int(m.group(2)), uid)
    return (0, -1, uid)


class WebDatasetWrapper(torch.utils.data.Dataset):
    """
    Collects all streamed WebDataset samples into memory,
    then builds windowed samples.

    Three modes:
      - random_window=False, cross_video_random=False (default):
            gap-aware consecutive sliding windows within same video

      - random_window=True,  cross_video_random=False:
            randomly sample window_size segments from same video_id pool
            number of windows preserved = same as sliding formula
            seed-controlled for reproducibility

      - cross_video_random=True:
            randomly merge clips from ANY video (no video_id constraint)
            each clip appears at least ws-1 and at most ws times
            total windows = same as Method 2 total
            seed-controlled for reproducibility

    shard_dir: path to directory containing split subfolders with .tar shards.
               Defaults to WEBDATASET_DIR if not provided.

    save_merge_csv: if True, saves a CSV with merge info for all samples.
    merge_csv_path: path to save the CSV file.
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        uid_to_videoid: dict,
        max_frames: int,
        max_length_decoder: int = 64,
        step_frames: int = 1,
        add_noise: bool = False,
        window_size: int = 2,
        stride: int = 1,
        shuffle: bool = True,
        random_window: bool = False,
        cross_video_random: bool = False,
        seed: int = 42,
        shard_dir: str = None,
        save_merge_csv: bool = False,           # ← save merge info to CSV
        merge_csv_path: str = "merge_info.csv", # ← path to save CSV
    ):
        base_dir    = shard_dir if shard_dir is not None else WEBDATASET_DIR
        print("Directory using:",base_dir)
        shard_paths = sorted(glob.glob(f"{base_dir}/{split}/*.tar"))
        assert len(shard_paths) > 0, f"No shards found for split '{split}' in '{base_dir}'"

        self.tokenizer          = tokenizer
        self.max_frames         = max_frames
        self.max_length_decoder = max_length_decoder
        self.step_frames        = step_frames
        self.add_noise          = add_noise
        self.window_size        = int(window_size)
        self.stride             = int(stride)
        self.random_window      = random_window
        self.cross_video_random = cross_video_random
        self.seed               = seed
        self.save_merge_csv     = save_merge_csv
        self.merge_csv_path     = merge_csv_path

        # ── determine method name for CSV ──
        if cross_video_random:
            method_name = "cross_video_random"
        elif random_window:
            method_name = "random_same_video"
        else:
            method_name = "consecutive"

        # ── Stream all samples into memory ──
        print(f"[{split}] Loading all samples from shards into memory...")
        print(f"[{split}] Shard dir: {base_dir}")
        dataset = (
            wds.WebDataset(shard_paths, shardshuffle=shuffle)
            .map(decode_sample)
        )

        uid_to_data     = {}
        uid_to_label    = {}
        uid_to_nframes  = {}   # ← track raw frame count per clip
        vid_to_uids     = {}

        for sample in dataset:
            uid = sample["uid"]

            # skip if uid not in csv — ensures only csv-specified samples are used
            if uid not in uid_to_videoid:
                continue

            uid_to_data[uid]    = sample["data"]
            uid_to_label[uid]   = sample["label"]
            uid_to_nframes[uid] = len(sample["data"])   # raw frames before step_frames
            video_id            = uid_to_videoid[uid]
            vid_to_uids.setdefault(video_id, []).append(uid)

        print(f"[{split}] Loaded {len(uid_to_data)} segments. Building windows...")

        # ── Build windowed samples ──
        self.samples     = []
        self.sample_uids = []
        merge_rows       = []   # ← rows for CSV

        W   = self.window_size
        S   = self.stride
        rng = random.Random(seed)

        def build_sample_from_window(pose_uids):
            """Helper — builds one sample dict and one CSV row from a list of uids."""
            merged_text     = " ".join([uid_to_label[u] for u in pose_uids]).strip()
            merged_text_bos = f"<s>{merged_text}</s>"

            token_ids = self.tokenizer(
                merged_text_bos,
                max_length=self.max_length_decoder,
                padding="max_length",
                truncation=True,
            )["input_ids"]

            token_ids = [
                tid if tid != self.tokenizer.pad_token_id else -100
                for tid in token_ids
            ]

            sample_uid = "+".join(pose_uids)

            self.samples.append({
                "pose_uids":  pose_uids,
                "labels":     token_ids,
                "sample_uid": sample_uid,
            })
            self.sample_uids.append(sample_uid)

            if save_merge_csv:
                merge_rows.append({
                    "sample_uid":  sample_uid,
                    "merged_text": merged_text,
                    "split":       split,
                    "method":      method_name,
                    "video_ids":   "+".join([uid_to_videoid.get(u, u) for u in pose_uids]),
                    "num_frames":  "+".join([str(uid_to_nframes.get(u, 0)) for u in pose_uids]),
                })

        # ── METHOD 3: Cross-video random merging ──
        if cross_video_random:

            # Step 1: compute target T using gap-aware consecutive run logic
            # (matches Method 1 exactly so window counts are comparable)
            T = 0
            for vid, uids in vid_to_uids.items():
                uids_sorted = sorted(uids, key=segment_sort_key)

                runs = []
                current_run = [uids_sorted[0]]
                for prev, curr in zip(uids_sorted, uids_sorted[1:]):
                    m_prev = NUM_SUFFIX_RE.match(prev)
                    m_curr = NUM_SUFFIX_RE.match(curr)
                    if m_prev and m_curr and int(m_curr.group(2)) - int(m_prev.group(2)) == 1:
                        current_run.append(curr)
                    else:
                        runs.append(current_run)
                        current_run = [curr]
                runs.append(current_run)

                for run in runs:
                    if len(run) < W:
                        T += 1
                    else:
                        T += len(range(0, len(run) - W + 1, S))

            # Step 2: all clips in one flat pool
            all_uids = list(uid_to_data.keys())
            N        = len(all_uids)

            print(f"[{split}] Cross-video random | Total clips: {N} | Target windows: {T}")

            # Step 3: decide how many clips get W appearances vs W-1 appearances
            # x clips appear W times, y clips appear W-1 times
            # x + y = N  and  x*W + y*(W-1) = T*W
            # solving → x = N - W*(N-T),  y = W*(N-T)
            num_full    = N - W * (N - T)   # clips appearing W times
            num_partial = W * (N - T)       # clips appearing W-1 times

            assert num_full >= 0,    f"num_full={num_full} is negative — check T vs N"
            assert num_partial >= 0, f"num_partial={num_partial} is negative — check T vs N"

            # Step 4: randomly assign which clips get W vs W-1 appearances (seeded)
            shuffled_uids = all_uids.copy()
            rng.shuffle(shuffled_uids)

            full_clips    = shuffled_uids[:num_full]
            partial_clips = shuffled_uids[num_full:]

            # Step 5: build pool respecting appearance constraints
            pool = full_clips * W + partial_clips * (W - 1)
            rng.shuffle(pool)

            # Step 6: group into T windows of size W and build samples
            for i in range(0, T * W, W):
                pose_uids = pool[i:i + W]
                build_sample_from_window(pose_uids)

        # ── METHOD 1 & 2: Per-video merging ──
        else:
            for vid, uids in vid_to_uids.items():
                uids_sorted = sorted(uids, key=segment_sort_key)

                if random_window:
                    # Method 2: randomly sample W segments from same video pool
                    num_windows = max(1, len(uids_sorted) - W + 1)
                    windows = [
                        rng.sample(uids_sorted, min(W, len(uids_sorted)))
                        for _ in range(num_windows)
                    ]

                else:
                    # Method 1: gap-aware consecutive sliding windows
                    runs = []
                    current_run = [uids_sorted[0]]
                    for prev, curr in zip(uids_sorted, uids_sorted[1:]):
                        m_prev = NUM_SUFFIX_RE.match(prev)
                        m_curr = NUM_SUFFIX_RE.match(curr)
                        if m_prev and m_curr and int(m_curr.group(2)) - int(m_prev.group(2)) == 1:
                            current_run.append(curr)
                        else:
                            runs.append(current_run)
                            current_run = [curr]
                    runs.append(current_run)

                    windows = []
                    for run in runs:
                        if len(run) < W:
                            windows.append(run)
                        else:
                            windows += [run[i:i+W] for i in range(0, len(run) - W + 1, S)]

                for pose_uids in windows:
                    build_sample_from_window(pose_uids)

        self.uid_to_data = uid_to_data
        print(f"[{split}] Built {len(self.samples)} windowed samples.")

        # ── Save merge CSV ──
        if save_merge_csv and merge_rows:
            df = pd.DataFrame(merge_rows, columns=[
                "sample_uid", "merged_text", "split", "method", "video_ids", "num_frames"
            ])
            df.to_csv(merge_csv_path, index=False)
            print(f"[{split}] Merge info saved → {merge_csv_path} ({len(df)} rows)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        feats = []
        for uid in sample["pose_uids"]:
            data = self.uid_to_data.get(uid, None)
            if data is None:
                continue

            if self.step_frames and self.step_frames > 1:
                data = data[::self.step_frames]

            if self.add_noise:
                data = add_gaussian_noise(data)

            feats.append(data.astype(np.float32))

        if len(feats) == 0:
            feature_vector = np.zeros((0, 152), dtype=np.float32)
        elif len(feats) == 1:
            feature_vector = feats[0]
        else:
            feature_vector = np.concatenate(feats, axis=0)

        T        = len(feature_vector)
        feat_dim = feature_vector.shape[1] if T > 0 else 152

        attn = np.ones(T, dtype=np.float32)

        if T < self.max_frames:
            pad            = np.zeros((self.max_frames - T, feat_dim), dtype=np.float32)
            feature_vector = np.vstack([feature_vector, pad])
            attn           = np.concatenate([attn, np.zeros(self.max_frames - T, dtype=np.float32)])
        else:
            feature_vector = feature_vector[:self.max_frames]
            attn           = attn[:self.max_frames]

        return {
            "input_ids":      torch.tensor(feature_vector, dtype=torch.float),
            "attention_mask": torch.tensor(attn,           dtype=torch.float),
            "labels":         torch.tensor(sample["labels"], dtype=torch.long),
        }


# ─── DATALOADER BUILDER ───────────────────────────────────────────────────────

def build_dataloader(
    split: str,
    tokenizer,
    max_frames: int,
    max_length_decoder: int = 64,
    step_frames: int = 1,
    add_noise: bool = False,
    window_size: int = 2,
    stride: int = 1,
    shuffle: bool = True,
    random_window: bool = False,
    cross_video_random: bool = False,
    seed: int = 42,
    shard_dir: str = None,
    save_merge_csv: bool = False,
    merge_csv_path: str = "merge_info.csv",
) -> DataLoader:

    dataset = WebDatasetWrapper(
        split=split,
        tokenizer=tokenizer,
        max_frames=max_frames,
        max_length_decoder=max_length_decoder,
        step_frames=step_frames,
        add_noise=add_noise,
        window_size=window_size,
        stride=stride,
        shuffle=shuffle,
        random_window=random_window,
        cross_video_random=cross_video_random,
        seed=seed,
        shard_dir=shard_dir,
        save_merge_csv=save_merge_csv,
        merge_csv_path=merge_csv_path,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# ─── EXAMPLE USAGE ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import PreTrainedTokenizerFast

    tokenizer_target = PreTrainedTokenizerFast(tokenizer_file="tokenizer_filew/isign_tokenizer.json")
    tokenizer_target.add_special_tokens({
        "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "pad_token": "<pad>", "mask_token": "<mask>",
        "additional_special_tokens": ["<PERSON>", "<UNKNOWN>"]
    })

    # ── Method 1: gap-aware consecutive ──
    train_loader_m1 = build_dataloader(
        split="train", tokenizer=tokenizer_target,
        max_frames=160, max_length_decoder=64, step_frames=5,
        window_size=2, stride=1, shuffle=True,
        random_window=False, cross_video_random=False,
        shard_dir="/indian-slp/dattatreya/isign1.0plus2.0",
        save_merge_csv=True,
        merge_csv_path="merge_info_train_consecutive.csv",
    )

    # ── Method 2: random same-video ──
    train_loader_m2 = build_dataloader(
        split="train", tokenizer=tokenizer_target,
        max_frames=160, max_length_decoder=64, step_frames=5,
        window_size=2, stride=1, shuffle=True,
        random_window=True, cross_video_random=False, seed=42,
        shard_dir="/indian-slp/dattatreya/isign1.0plus2.0",
        save_merge_csv=True,
        merge_csv_path="merge_info_train_random_same_video.csv",
    )

    # ── Method 3: cross-video random ──
    train_loader_m3 = build_dataloader(
        split="train", tokenizer=tokenizer_target,
        max_frames=160, max_length_decoder=64, step_frames=5,
        window_size=2, stride=1, shuffle=True,
        random_window=False, cross_video_random=True, seed=42,
        shard_dir="/indian-slp/dattatreya/isign1.0plus2.0",
        save_merge_csv=True,
        merge_csv_path="merge_info_train_cross_video_random.csv",
    )

    print("Method 1 samples:", len(train_loader_m1.dataset))
    print("Method 2 samples:", len(train_loader_m2.dataset))
    print("Method 3 samples:", len(train_loader_m3.dataset))

    b = next(iter(train_loader_m3))
    print("input_ids:     ", b["input_ids"].shape)
    print("attention_mask:", b["attention_mask"].shape)
    print("labels:        ", b["labels"].shape)
import os
import io
import random
import pandas as pd
import numpy as np
import webdataset as wds
from pose_format import Pose
from tqdm import tqdm


# ─── CONFIG ───────────────────────────────────────────────────────────────────

POSE_DIRS = [ "/indian-slp/Dataset/ISL_Dataset/translation/isign1.0/iSign-poses_v1.1",
             "/indian-slp/Dataset/ISL_Dataset/translation/isign2.0/English/english_videos_pose"
]

OUTPUT_BASE_DIR = "/indian-slp/dattatreya/coreference/shards/coreference_shards"

TRAIN_CSV = "/data/dept_share/sanjeet/Khushi/coreference/datasets/train.csv"
VAL_CSV   = "/data/dept_share/sanjeet/Khushi/coreference/datasets/val.csv"
TEST_CSV  = "/data/dept_share/sanjeet/Khushi/coreference/datasets/test.csv"

SHARD_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB per shard
SEED             = 42


# ─── KEYPOINT INDICES ────────────────────────────────────────────────────────

mediapipe_exclude = [
    26, 28, 30, 32,
    25, 27, 29, 31,
    1, 3, 4, 6,
    9, 10,
    17, 18, 19, 20,
    21, 22, 23, 24,
]
POSE_INDICES = np.array([i for i in range(33) if i not in mediapipe_exclude], dtype=np.int32)

FACE_INDICES = np.array([
    61, 291,
    17, 0,
    70, 105, 107,
    300, 334, 336,
    161, 158,
    33, 163, 153, 133,
    388, 385,
    263, 390, 380, 362,
    9,
], dtype=np.int32)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

from typing import Optional

def find_pose_file(uid: str, pose_dirs: list) -> Optional[str]:
    for pose_dir in pose_dirs:
        candidate = os.path.join(pose_dir, uid + ".pose")
        if os.path.exists(candidate):
            return candidate
    return None


def numpy_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def find_nearest_unmasked_frame(data, confidence, start_idx):
    for offset in range(1, len(data)):
        fwd = start_idx + offset
        bwd = start_idx - offset
        if fwd < len(data) and not np.any(data.mask[fwd]) and np.min(confidence[fwd]) > 0.8:
            return fwd
        if bwd >= 0 and not np.any(data.mask[bwd]) and np.min(confidence[bwd]) > 0.8:
            return bwd
    return start_idx


def fix_masked_frames(data, confidence):
    out_data = []
    out_conf = []
    for i in range(len(data)):
        if np.any(data.mask[i]):
            idx = find_nearest_unmasked_frame(data, confidence, i)
            out_data.append(data[idx])
            out_conf.append(confidence[idx])
        else:
            out_data.append(data[i])
            out_conf.append(confidence[i])
    return np.stack(out_data), np.stack(out_conf)


def normalize_minmax(keypoints: np.ndarray) -> np.ndarray:
    min_vals = np.min(keypoints, axis=(0, 1))
    max_vals = np.max(keypoints, axis=(0, 1))
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    return 2 * (keypoints - min_vals) / denom - 1


def get_component(pose, component_name: str, kp_indices=None) -> np.ndarray:
    p = pose.get_components([component_name])
    confidence = np.squeeze(np.array(p.body.confidence), axis=1)

    if component_name == "POSE_LANDMARKS":
        p.normalize(p.header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        ))
    elif component_name == "FACE_LANDMARKS":
        p.normalize(p.header.normalization_info(
            p1=("FACE_LANDMARKS", "48"),
            p2=("FACE_LANDMARKS", "278"),
        ))
    elif component_name == "LEFT_HAND_LANDMARKS":
        p.normalize(p.header.normalization_info(
            p1=("LEFT_HAND_LANDMARKS", "INDEX_FINGER_MCP"),
            p2=("LEFT_HAND_LANDMARKS", "PINKY_MCP"),
        ))
    elif component_name == "RIGHT_HAND_LANDMARKS":
        p.normalize(p.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP"),
            p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
        ))

    data = p.body.data

    if kp_indices is not None:
        data       = data[:, :, kp_indices, :]
        confidence = confidence[:, kp_indices]

    data = data[:, 0, :, :]
    data, _ = fix_masked_frames(data, confidence)

    keypoints = np.array(data[:, :, :2], dtype=np.float32)
    keypoints = normalize_minmax(keypoints)
    return keypoints.reshape(keypoints.shape[0], -1)


def process_pose(pose_path: str) -> np.ndarray:
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    left_hand  = get_component(pose, "LEFT_HAND_LANDMARKS",  kp_indices=None)
    right_hand = get_component(pose, "RIGHT_HAND_LANDMARKS", kp_indices=None)
    pose_kps   = get_component(pose, "POSE_LANDMARKS",       kp_indices=POSE_INDICES)
    face_kps   = get_component(pose, "FACE_LANDMARKS",       kp_indices=FACE_INDICES)

    return np.concatenate([left_hand, right_hand, pose_kps, face_kps], axis=1)


# ─── UNIQUE UID EXTRACTION ───────────────────────────────────────────────────

def get_unique_uids(csv_path: str) -> list:
    """
    Read uid_A and uid_B columns, take the union of unique values,
    and return as a sorted list of strings.
    """
    df = pd.read_csv(csv_path)
    uids_a = set(df["uid_A"].dropna().astype(str).unique())
    uids_b = set(df["uid_B"].dropna().astype(str).unique())
    unique_uids = sorted(uids_a | uids_b)
    print(f"  uid_A unique: {len(uids_a)} | uid_B unique: {len(uids_b)} | combined unique: {len(unique_uids)}")
    return unique_uids


# ─── CONVERSION ──────────────────────────────────────────────────────────────

def convert_split(csv_path: str, split_name: str, pose_dirs: list, output_base: str):
    output_dir = os.path.join(output_base, split_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[{split_name}] Extracting unique UIDs from uid_A and uid_B ...")
    unique_uids = get_unique_uids(csv_path)

    random.seed(SEED)
    random.shuffle(unique_uids)

    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    skipped = 0
    written = 0

    with wds.ShardWriter(shard_pattern, maxsize=SHARD_SIZE_BYTES) as sink:
        for uid in tqdm(unique_uids, desc=f"Converting {split_name}"):

            pose_path = find_pose_file(uid, pose_dirs)
            if pose_path is None:
                skipped += 1
                continue

            try:
                keypoints = process_pose(pose_path)
            except Exception as e:
                print(f"\nFailed: {uid} — {e}")
                skipped += 1
                continue

            sink.write({
                "__key__":  uid,
                "data.npy": numpy_to_bytes(keypoints),
            })
            written += 1

    print(f"\n{split_name}: {written} written, {skipped} skipped")
    print(f"Shards saved to: {output_dir}\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    convert_split(TRAIN_CSV, "train", POSE_DIRS, OUTPUT_BASE_DIR)
    convert_split(VAL_CSV,   "val",   POSE_DIRS, OUTPUT_BASE_DIR)
    convert_split(TEST_CSV,  "test",  POSE_DIRS, OUTPUT_BASE_DIR)

    print("All splits converted successfully.")


if __name__ == "__main__":
    main()
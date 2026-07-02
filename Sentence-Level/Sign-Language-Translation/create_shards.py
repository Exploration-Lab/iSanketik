import os
import io
import random
import pandas as pd
import numpy as np
import webdataset as wds
from pose_format import Pose
from tqdm import tqdm


# ─── CONFIG ───────────────────────────────────────────────────────────────────

POSE_DIR        = '/indian-slp/Dataset/ISL_Dataset/translation/isign2.0/English/english_videos_pose'
OUTPUT_BASE_DIR = "/indian-slp/dattatreya/isign2.1"

TRAIN_CSV = "/indian-slp/dattatreya/translation/tokenization/isign2_data_excluding_isign1.csv"
VAL_CSV   ="/indian-slp/Dataset/ISL_Dataset/translation/isign2.0/Hindi/Hindi_csv/hindi_isign_val.csv"
TEST_CSV  = "/indian-slp/Dataset/ISL_Dataset/translation/isign2.0/Hindi/Hindi_csv/hindi_isign_test.csv"


SHARD_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB per shard
SEED             = 42


# ─── KEYPOINT INDICES (same as process_data_kmeansCISLR.py) ──────────────

mediapipe_exclude = [
    26, 28, 30, 32,     # right leg/foot
    25, 27, 29, 31,     # left leg/foot
    1, 3, 4, 6,         # eyes
    9, 10,              # mouth
    17, 18, 19, 20,     # fingers
    21, 22, 23, 24,     # hips + fingers
]
POSE_INDICES = np.array([i for i in range(33) if i not in mediapipe_exclude], dtype=np.int32)
# 11 keypoints

FACE_INDICES = np.array([
    61, 291,             # mouth corners
    17, 0,               # lips
    70, 105, 107,        # right eyebrow
    300, 334, 336,       # left eyebrow
    161, 158,            # right eye upper
    33, 163, 153, 133,   # right eye lower
    388, 385,            # left eye upper
    263, 390, 380, 362,  # left eye lower
    9,                   # nose tip
], dtype=np.int32)
# 23 keypoints


# ─── HELPERS ──────────────────────────────────────────────────────────────────

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
    """Replace masked frames with nearest unmasked frame (confidence > 0.8)."""
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
    """Min-max normalize to [-1, 1] across all frames and keypoints."""
    min_vals = np.min(keypoints, axis=(0, 1))
    max_vals = np.max(keypoints, axis=(0, 1))
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0  # avoid division by zero
    return 2 * (keypoints - min_vals) / denom - 1


def get_component(pose, component_name: str, kp_indices=None) -> np.ndarray:
    """
    Extract, normalize, fix masked frames for one pose component.
    Returns (frames, kps*2) float32 array — x,y only, flattened.
    """
    p = pose.get_components([component_name])
    confidence = np.squeeze(np.array(p.body.confidence), axis=1)  # (frames, kps)

    # Component-specific normalization
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

    data = p.body.data  # masked array (frames, 1, kps, dims)

    # Slice keypoints if specified
    if kp_indices is not None:
        data       = data[:, :, kp_indices, :]
        confidence = confidence[:, kp_indices]

    # Squeeze person dim → (frames, kps, dims)
    data = data[:, 0, :, :]

    # Fix masked frames
    data, _ = fix_masked_frames(data, confidence)

    # Keep only x, y → (frames, kps, 2)
    keypoints = np.array(data[:, :, :2], dtype=np.float32)

    # Min-max normalize to [-1, 1]
    keypoints = normalize_minmax(keypoints)

    # Flatten → (frames, kps*2)
    return keypoints.reshape(keypoints.shape[0], -1)


def process_pose(pose_path: str) -> np.ndarray:
    """
    Full pipeline: load .pose → extract 4 components → concatenate.
    Returns (frames, 152) float32 array.
    Concatenation order: left_hand + right_hand + pose + face
    (matches your existing DataLoader exactly)
    """
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    left_hand  = get_component(pose, "LEFT_HAND_LANDMARKS",  kp_indices=None)          # (T, 42)
    right_hand = get_component(pose, "RIGHT_HAND_LANDMARKS", kp_indices=None)          # (T, 42)
    pose_kps   = get_component(pose, "POSE_LANDMARKS",       kp_indices=POSE_INDICES)  # (T, 22)
    face_kps   = get_component(pose, "FACE_LANDMARKS",       kp_indices=FACE_INDICES)  # (T, 46)

    # (T, 152)
    return np.concatenate([left_hand, right_hand, pose_kps, face_kps], axis=1)


# ─── CONVERSION ───────────────────────────────────────────────────────────────

def convert_split(csv_path: str, split_name: str, pose_dir: str, output_base: str):
    output_dir = os.path.join(output_base, split_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    samples = list(zip(df["uid"].astype(str), df["text"].astype(str)))

    random.seed(SEED)
    random.shuffle(samples)

    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    skipped = 0
    written = 0

    with wds.ShardWriter(shard_pattern, maxsize=SHARD_SIZE_BYTES) as sink:
        for uid, label in tqdm(samples, desc=f"Converting {split_name}"):
            pose_path = os.path.join(pose_dir, uid + ".pose")

            if not os.path.exists(pose_path):
                skipped += 1
                continue

            try:
                keypoints = process_pose(pose_path)  # (frames, 152) — done once, forever
            except Exception as e:
                print(f"\nFailed: {uid} — {e}")
                skipped += 1
                continue

            sink.write({
                "__key__":   uid,
                "data.npy":  numpy_to_bytes(keypoints),  # (frames, 152) float32
                "label.txt": label,
            })
            written += 1

    print(f"\n{split_name}: {written} written, {skipped} skipped")
    print(f"Shards saved to: {output_dir}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    convert_split(TRAIN_CSV, "train", POSE_DIR, OUTPUT_BASE_DIR)
    #convert_split(VAL_CSV,   "val",   POSE_DIR, OUTPUT_BASE_DIR)
    #convert_split(TEST_CSV,  "test",  POSE_DIR, OUTPUT_BASE_DIR)

    print("All splits converted successfully.")


if __name__ == "__main__":
    main()


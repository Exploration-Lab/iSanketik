import os
import io
import random
import pandas as pd
import numpy as np
import webdataset as wds
from pose_format import Pose
from tqdm import tqdm
from typing import Optional

# ─── CONFIG — EDIT THESE PATHS ───────────────────────────────────────────────

POSE_DIRS = [
     "/DATAS3/Sign_language_data_extended/FINAL_data/All_video_sign_language/english_videos_pose"
]

OUTPUT_BASE_DIR =  "/indian-slp/dattatreya/emotion_may23/shards/emotionshards" 

TRAIN_CSV = "/data/dept_share/sanjeet/Khushi/emotion/dataset/train.csv"
VAL_CSV   = "/data/dept_share/sanjeet/Khushi/emotion/dataset/val.csv"
TEST_CSV  = "/data/dept_share/sanjeet/Khushi/emotion/dataset/test.csv"

SHARD_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB per shard
SEED             = 42

# Valid emotion labels — lowercase
VALID_LABELS = {
    "neutral", "happy", "sad", "frustrated",
    "angry", "fear", "shock", "surprise", "excited"
}

# ─── KEYPOINT INDICES — identical to fingerspelling ──────────────────────────

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


# ─── HELPERS — identical to fingerspelling ───────────────────────────────────

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
    denom    = max_vals - min_vals
    denom[denom == 0] = 1.0
    return 2 * (keypoints - min_vals) / denom - 1


def get_component(pose, component_name: str, kp_indices=None) -> np.ndarray:
    p          = pose.get_components([component_name])
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

    data        = data[:, 0, :, :]
    data, _     = fix_masked_frames(data, confidence)
    keypoints   = np.array(data[:, :, :2], dtype=np.float32)
    keypoints   = normalize_minmax(keypoints)
    return keypoints.reshape(keypoints.shape[0], -1)


def process_pose(pose_path: str) -> np.ndarray:
    """
    Identical pipeline to fingerspelling.
    Returns (frames, 152) float32 — compatible with translation encoder.
    """
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    left_hand  = get_component(pose, "LEFT_HAND_LANDMARKS",  kp_indices=None)
    right_hand = get_component(pose, "RIGHT_HAND_LANDMARKS", kp_indices=None)
    pose_kps   = get_component(pose, "POSE_LANDMARKS",       kp_indices=POSE_INDICES)
    face_kps   = get_component(pose, "FACE_LANDMARKS",       kp_indices=FACE_INDICES)

    return np.concatenate([left_hand, right_hand, pose_kps, face_kps], axis=1)


# ─── CONVERSION ──────────────────────────────────────────────────────────────

def convert_split(csv_path: str, split_name: str, pose_dirs: list, output_base: str):
    output_dir = os.path.join(output_base, split_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Lowercase the label, drop rows with missing or invalid labels
    df["gemini_label"] = df["gemini_label"].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[df["gemini_label"].isin(VALID_LABELS)].reset_index(drop=True)
    after  = len(df)
    if before != after:
        print(f"  [{split_name}] Dropped {before - after} rows with invalid/unknown labels")

    samples = list(zip(df["uid"].astype(str), df["gemini_label"].astype(str)))

    random.seed(SEED)
    random.shuffle(samples)

    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    skipped = 0
    written = 0

    with wds.ShardWriter(shard_pattern, maxsize=SHARD_SIZE_BYTES) as sink:
        for uid, label in tqdm(samples, desc=f"Converting {split_name}"):

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
                "__key__":   uid,
                "data.npy":  numpy_to_bytes(keypoints),
                "label.txt": label,
            })
            written += 1

    # Write size.txt so the training script can auto-detect dataset size
    size_path = os.path.join(output_dir, "size.txt")
    with open(size_path, "w") as f:
        f.write(str(written))

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


# import pandas as pd
# import os

# # ── edit these ────────────────────────────────────────────────────────────────
# POSE_DIRS = [
#     "/DATAS3/Sign_language_data_extended/FINAL_data/All_video_sign_language/english_videos_pose"
#     #  "/indian-slp/Dataset/ISL_Dataset/translation/isign1.0",
#     #  "/indian-slp/Dataset/ISL_Dataset/translation/isign2.0"
# ]

# OUTPUT_BASE_DIR =  "/indian-slp/dattatreya/emotion_may23/shards/emotionshards"  # edit this

# TRAIN_CSV = "/data/dept_share/sanjeet/Khushi/emotion/dataset/train.csv"
# VAL_CSV   = "/data/dept_share/sanjeet/Khushi/emotion/dataset/val.csv"
# TEST_CSV  = "/data/dept_share/sanjeet/Khushi/emotion/dataset/test.csv"
# # ─────────────────────────────────────────────────────────────────────────────

# VALID_LABELS = {
#     "neutral", "happy", "sad", "frustrated",
#     "angry", "fear", "shock", "surprise", "excited"
# }

# df = pd.read_csv(TRAIN_CSV)
# print("Total rows:", len(df))
# print("\nColumns:", df.columns.tolist())
# print("\nFirst 5 rows:\n", df[["uid", "gemini_label"]].head())

# # Check label distribution
# df["gemini_label"] = df["gemini_label"].astype(str).str.strip().str.lower()
# print("\nUnique labels found:", df["gemini_label"].unique().tolist())
# print("Rows with valid labels:  ", df[df["gemini_label"].isin(VALID_LABELS)].shape[0])
# print("Rows with INVALID labels:", df[~df["gemini_label"].isin(VALID_LABELS)].shape[0])

# # Check actual files across all pose dirs
# print("\n── Pose directory contents ──────────────────────────────────────")
# for pose_dir in POSE_DIRS:
#     if not os.path.exists(pose_dir):
#         print(f"  DIR NOT FOUND: {pose_dir}")
#         continue
#     actual_files = [f for f in os.listdir(pose_dir) if f.endswith(".pose")]
#     print(f"  {pose_dir}")
#     print(f"    .pose files: {len(actual_files)}")
#     if actual_files:
#         print(f"    Examples   : {actual_files[:3]}")

# # Check pose file matching across all dirs
# df_valid     = df[df["gemini_label"].isin(VALID_LABELS)]
# found        = 0
# missing      = 0
# missing_examples = []

# for uid in df_valid["uid"].astype(str):
#     match = False
#     for pose_dir in POSE_DIRS:
#         if os.path.exists(os.path.join(pose_dir, uid + ".pose")):
#             match = True
#             break
#     if match:
#         found += 1
#     else:
#         missing += 1
#         if len(missing_examples) < 5:
#             missing_examples.append(uid)

# print(f"\n── Matching results ─────────────────────────────────────────────")
# print(f"  Pose files found   : {found}")
# print(f"  Pose files missing : {missing}")
# if missing_examples:
#     print(f"  Missing uid examples: {missing_examples}")
#     # Show what the uid looks like vs actual filename to spot format mismatch
#     for pose_dir in POSE_DIRS:
#         actual = [f for f in os.listdir(pose_dir) if f.endswith(".pose")]
#         if actual:
#             print(f"\n  CSV uid  : '{missing_examples[0]}'")
#             print(f"  File name: '{actual[0].replace('.pose', '')}'")
#             break
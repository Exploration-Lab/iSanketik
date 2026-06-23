"""
temp.py — Convert NLA-SLR embeddings.npz → embeddings.npy + metadata.json
so that final.py can consume them directly.

For each split (train / val / test):
  Input  : <emb_dir>/embeddings.npz   keys: 'uids' [N], 'embeddings' [N, D]
  Output : <emb_dir>/embeddings.npy   float32 array [N, D]
           <emb_dir>/metadata.json    list of {uid, word} dicts

Run:
    python3 npz_to_npy.py
"""

import json
import os
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = (
    "/DATAS3/Sign_language_data_extended/FINAL_data/All_video_sign_language"
    "/Workshop_Shared_task/Word_gloss_recognition/dataset"
)

SPLITS = {
    "train": {
        "emb_dir": "/DATA3/ashishu23/ashishu23_2/embeddings_islrtc_train_I3D",
        "csv":     os.path.join(DATASET_DIR, "train.csv"),
    },
    "val": {
        "emb_dir": "/DATA3/ashishu23/ashishu23_2/embeddings_islrtc_val_I3D",
        "csv":     os.path.join(DATASET_DIR, "val.csv"),
    },
    "test": {
        "emb_dir": "/DATA3/ashishu23/ashishu23_2/embeddings_islrtc_test_I3D",
        "csv":     os.path.join(DATASET_DIR, "test.csv"),
    },
}
# ──────────────────────────────────────────────────────────────────────────────


def convert_split(split_name: str, emb_dir: str, csv_path: str) -> None:
    npz_path  = os.path.join(emb_dir, "embeddings.npz")
    npy_path  = os.path.join(emb_dir, "embeddings.npy")
    meta_path = os.path.join(emb_dir, "metadata.json")

    if not os.path.isfile(npz_path):
        print(f"[{split_name}] SKIP — not found: {npz_path}")
        return

    data       = np.load(npz_path, allow_pickle=True)
    uids       = data["uids"].astype(str)       # [N]
    embeddings = data["embeddings"].astype(np.float32)  # [N, D]
    print(f"[{split_name}] {len(uids)} embeddings  shape={embeddings.shape}")

    df       = pd.read_csv(csv_path)
    uid2word = dict(zip(df["uid"].astype(str), df["word"].astype(str)))

    missing  = []
    metadata = []
    for uid in uids:
        word = uid2word.get(uid)
        if word is None:
            missing.append(uid)
        metadata.append({"uid": uid, "word": word})

    if missing:
        print(f"  WARNING: {len(missing)} UIDs missing from CSV "
              f"(first 5: {missing[:5]})")

    np.save(npy_path, embeddings)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  → {npy_path}")
    print(f"  → {meta_path}")


def main():
    for split_name, cfg in SPLITS.items():
        convert_split(split_name, cfg["emb_dir"], cfg["csv"])

    print("\nDone. Run final.py with:")
    print(
        "  python3 final.py \\\n"
        "    --train_embeddings /DATAS3/ashishu23_2/embeddings_islrtc_train_NLA_SLR/embeddings.npy \\\n"
        "    --train_metadata   /DATAS3/ashishu23_2/embeddings_islrtc_train_NLA_SLR/metadata.json \\\n"
        "    --eval_embeddings  /DATAS3/ashishu23_2/embeddings_islrtc_val_NLA_SLR/embeddings.npy  \\\n"
        "    --eval_metadata    /DATAS3/ashishu23_2/embeddings_islrtc_val_NLA_SLR/metadata.json"
    )


if __name__ == "__main__":
    main()

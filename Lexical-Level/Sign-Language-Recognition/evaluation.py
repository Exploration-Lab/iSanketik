#!/usr/bin/env python3
"""Nearest-neighbor classifier using cosine similarity and Euclidean distance between embeddings.

This script evaluates the validation dataset using the combined train and test datasets as reference sets.

Usage:
    python3 evaluation2.py \
      --train_embeddings /DATA3/ashishu23/ashishu23_2/embeddings_islrtc_train_I3D/embeddings.npy \
      --train_metadata  /DATA3/ashishu23/ashishu23_2/embeddings_islrtc_train_I3D/metadata.json \
      --test_embeddings /DATA3/ashishu23/ashishu23_2/embeddings_islrtc_test_I3D/embeddings.npy \
      --test_metadata  /DATA3/ashishu23/ashishu23_2/embeddings_islrtc_test_I3D/metadata.json \
      --val_embeddings /DATA3/ashishu23/ashishu23_2/embeddings_islrtc_val_I3D/embeddings.npy  \
      --val_metadata  /DATA3/ashishu23/ashishu23_2/embeddings_islrtc_val_I3D/metadata.json \
      --output predictions.json

The script computes cosine similarity, Euclidean distance, Manhattan (L1) distance,
dot product, and correlation distance between each validation embedding and all
combined train+test reference embeddings. For each metric, it picks the nearest
neighbor, outputs predicted class and scores, and prints top-1/2/5 accuracy.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def load_embeddings_and_metadata(emb_path: str, meta_path: str):
    emb_path = Path(emb_path)
    meta_path = Path(meta_path)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    emb = np.load(str(emb_path))
    with open(str(meta_path), 'r') as f:
        meta = json.load(f)
    return emb, meta


def get_label_from_meta(entry: Dict) -> str:
    # common keys: 'word', 'label', 'class'
    for k in ('word', 'label', 'class'):
        if k in entry:
            return entry[k]
    # fallback: try 'uid'
    return str(entry.get('uid', ''))


def normalize_rows(x: np.ndarray) -> np.ndarray:
    # L2 normalize rows, avoid division by zero
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def knn_cosine(train_emb: np.ndarray, eval_emb: np.ndarray, batch_size: int = 128, topk: int = 5):
    """Compute top-k nearest neighbor indices in train_emb for each eval_emb using cosine similarity.

    Returns: indices (n_eval, topk), sims (n_eval, topk) where sims are cosine similarities.
    """
    train_norm = normalize_rows(train_emb)
    eval_norm = normalize_rows(eval_emb)

    n_eval = eval_norm.shape[0]
    nn_idx = np.empty((n_eval, topk), dtype=np.int64)
    nn_sim = np.empty((n_eval, topk), dtype=np.float32)

    for start in range(0, n_eval, batch_size):
        end = min(n_eval, start + batch_size)
        batch = eval_norm[start:end]  # (B, D)
        # similarity: (B, D) @ (D, N) -> (B, N)
        sims = batch.dot(train_norm.T)
        # get topk indices (argsort desc)
        if topk >= sims.shape[1]:
            # all columns
            idx = np.argsort(-sims, axis=1)
            top_idx = idx[:, :topk]
        else:
            top_idx = np.argpartition(-sims, topk - 1, axis=1)[:, :topk]
            # sort those topk
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_idx = top_idx[row_idx, np.argsort(-sims[row_idx, top_idx], axis=1)]

        top_sims = sims[np.arange(sims.shape[0])[:, None], top_idx]
        nn_idx[start:end, :] = top_idx
        nn_sim[start:end, :] = top_sims

    return nn_idx, nn_sim


def knn_euclidean(train_emb: np.ndarray, eval_emb: np.ndarray, batch_size: int = 128, topk: int = 5):
    """Compute top-k nearest neighbor indices in train_emb for each eval_emb using Euclidean distance.

    Returns: indices (n_eval, topk), dists (n_eval, topk) where dists are Euclidean distances (smaller = closer).
    """
    n_eval = eval_emb.shape[0]
    nn_idx = np.empty((n_eval, topk), dtype=np.int64)
    nn_dist = np.empty((n_eval, topk), dtype=np.float32)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    train_sq = np.sum(train_emb ** 2, axis=1)  # (N,)

    for start in range(0, n_eval, batch_size):
        end = min(n_eval, start + batch_size)
        batch = eval_emb[start:end]  # (B, D)
        eval_sq = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
        dists_sq = eval_sq + train_sq[None, :] - 2.0 * batch.dot(train_emb.T)
        dists_sq = np.maximum(dists_sq, 0.0)  # numerical clamp
        dists = np.sqrt(dists_sq)

        if topk >= dists.shape[1]:
            top_idx = np.argsort(dists, axis=1)[:, :topk]
        else:
            top_idx = np.argpartition(dists, topk - 1, axis=1)[:, :topk]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_idx = top_idx[row_idx, np.argsort(dists[row_idx, top_idx], axis=1)]

        top_dists = dists[np.arange(dists.shape[0])[:, None], top_idx]
        nn_idx[start:end, :] = top_idx
        nn_dist[start:end, :] = top_dists

    return nn_idx, nn_dist


def knn_manhattan(train_emb: np.ndarray, eval_emb: np.ndarray, batch_size: int = 128, topk: int = 5):
    """Compute top-k nearest neighbors using Manhattan (L1) distance (smaller = closer).

    Uses a per-sample inner loop to avoid the memory cost of a (B, N, D) broadcast.
    """
    n_eval = eval_emb.shape[0]
    nn_idx  = np.empty((n_eval, topk), dtype=np.int64)
    nn_dist = np.empty((n_eval, topk), dtype=np.float32)

    for start in range(0, n_eval, batch_size):
        end = min(n_eval, start + batch_size)
        batch = eval_emb[start:end]
        b = end - start
        dists = np.empty((b, train_emb.shape[0]), dtype=np.float32)
        for bi in range(b):
            dists[bi] = np.abs(train_emb - batch[bi]).sum(axis=1)

        if topk >= dists.shape[1]:
            top_idx = np.argsort(dists, axis=1)[:, :topk]
        else:
            top_idx = np.argpartition(dists, topk - 1, axis=1)[:, :topk]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_idx = top_idx[row_idx, np.argsort(dists[row_idx, top_idx], axis=1)]

        top_dists = dists[np.arange(b)[:, None], top_idx]
        nn_idx[start:end, :]  = top_idx
        nn_dist[start:end, :] = top_dists

    return nn_idx, nn_dist


def knn_dot_product(train_emb: np.ndarray, eval_emb: np.ndarray, batch_size: int = 128, topk: int = 5):
    """Compute top-k nearest neighbors using raw (unnormalized) dot product (larger = closer).

    Unlike cosine similarity, embedding magnitudes matter here.
    """
    n_eval = eval_emb.shape[0]
    nn_idx = np.empty((n_eval, topk), dtype=np.int64)
    nn_sim = np.empty((n_eval, topk), dtype=np.float32)

    for start in range(0, n_eval, batch_size):
        end = min(n_eval, start + batch_size)
        batch = eval_emb[start:end]
        sims = batch.dot(train_emb.T)  # (B, N) — no normalization

        if topk >= sims.shape[1]:
            idx = np.argsort(-sims, axis=1)
            top_idx = idx[:, :topk]
        else:
            top_idx = np.argpartition(-sims, topk - 1, axis=1)[:, :topk]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_idx = top_idx[row_idx, np.argsort(-sims[row_idx, top_idx], axis=1)]

        top_sims = sims[np.arange(sims.shape[0])[:, None], top_idx]
        nn_idx[start:end, :] = top_idx
        nn_sim[start:end, :] = top_sims

    return nn_idx, nn_sim


def knn_correlation(train_emb: np.ndarray, eval_emb: np.ndarray, batch_size: int = 128, topk: int = 5):
    """Compute top-k nearest neighbors using correlation distance = 1 - Pearson r (smaller = closer).

    Equivalent to cosine similarity on mean-centered vectors. Values in [0, 2].
    """
    # Mean-center each row, then L2-normalize (done once outside the loop for train)
    train_c = normalize_rows((train_emb - train_emb.mean(axis=1, keepdims=True)).astype(np.float32))
    eval_c  = (eval_emb - eval_emb.mean(axis=1, keepdims=True)).astype(np.float32)

    n_eval = eval_c.shape[0]
    nn_idx  = np.empty((n_eval, topk), dtype=np.int64)
    nn_dist = np.empty((n_eval, topk), dtype=np.float32)

    for start in range(0, n_eval, batch_size):
        end = min(n_eval, start + batch_size)
        batch_c = normalize_rows(eval_c[start:end].copy())
        sims  = batch_c.dot(train_c.T)          # Pearson r in [-1, 1]
        dists = (1.0 - sims).astype(np.float32) # correlation distance in [0, 2]

        if topk >= dists.shape[1]:
            top_idx = np.argsort(dists, axis=1)[:, :topk]
        else:
            top_idx = np.argpartition(dists, topk - 1, axis=1)[:, :topk]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_idx = top_idx[row_idx, np.argsort(dists[row_idx, top_idx], axis=1)]

        top_dists = dists[np.arange(dists.shape[0])[:, None], top_idx]
        nn_idx[start:end, :]  = top_idx
        nn_dist[start:end, :] = top_dists

    return nn_idx, nn_dist


def main():
    parser = argparse.ArgumentParser(description='Nearest-neighbor by cosine similarity, using combined train+test as reference for validation')
    parser.add_argument('--train_embeddings', required=True)
    parser.add_argument('--train_metadata', required=True)
    parser.add_argument('--test_embeddings', required=True)
    parser.add_argument('--test_metadata', required=True)
    parser.add_argument('--val_embeddings', required=True)
    parser.add_argument('--val_metadata', required=True)
    parser.add_argument('--output', default='predictions.json')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--topk', type=int, default=5, help='number of top neighbors to consider for accuracy')

    args = parser.parse_args()

    print('Loading train embeddings...')
    train_emb, train_meta = load_embeddings_and_metadata(args.train_embeddings, args.train_metadata)
    print('Loading test embeddings...')
    test_emb, test_meta = load_embeddings_and_metadata(args.test_embeddings, args.test_metadata)
    print('Loading val embeddings...')
    val_emb, val_meta = load_embeddings_and_metadata(args.val_embeddings, args.val_metadata)

    print(f'Train embeddings: {train_emb.shape}, Test embeddings: {test_emb.shape}, Val embeddings: {val_emb.shape}')

    # Combine train and test as reference
    combined_emb = np.concatenate([train_emb, test_emb], axis=0)
    combined_meta = train_meta + test_meta
    combined_labels = [get_label_from_meta(m) for m in combined_meta]

    print(f'Combined reference embeddings: {combined_emb.shape}')

    # Build val labels
    val_labels = [get_label_from_meta(m) for m in val_meta]

    topk = max(1, args.topk)

    predictions = []

    # Evaluate using combined as reference
    print(f'Computing nearest neighbors (cosine similarity) for val using combined, topk={topk}...')
    nn_idx_cos, nn_sim = knn_cosine(combined_emb, val_emb, batch_size=args.batch_size, topk=topk)

    print(f'Computing nearest neighbors (Euclidean distance) for val using combined, topk={topk}...')
    nn_idx_euc, nn_dist = knn_euclidean(combined_emb, val_emb, batch_size=args.batch_size, topk=topk)

    print(f'Computing nearest neighbors (Manhattan distance) for val using combined, topk={topk}...')
    nn_idx_man, nn_dist_man = knn_manhattan(combined_emb, val_emb, batch_size=args.batch_size, topk=topk)

    print(f'Computing nearest neighbors (dot product) for val using combined, topk={topk}...')
    nn_idx_dot, nn_sim_dot = knn_dot_product(combined_emb, val_emb, batch_size=args.batch_size, topk=topk)

    print(f'Computing nearest neighbors (correlation distance) for val using combined, topk={topk}...')
    nn_idx_corr, nn_dist_corr = knn_correlation(combined_emb, val_emb, batch_size=args.batch_size, topk=topk)

    # Build predictions and compute accuracies
    cos_top1 = cos_top2 = cos_top5 = 0
    euc_top1 = euc_top2 = euc_top5 = 0
    man_top1 = man_top2 = man_top5 = 0
    dot_top1 = dot_top2 = dot_top5 = 0
    corr_top1 = corr_top2 = corr_top5 = 0

    total = len(val_meta)

    for i, meta in enumerate(val_meta):
        true_label = get_label_from_meta(meta)

        cos_idxs = [int(x) for x in nn_idx_cos[i]]
        cos_labels = [combined_labels[j] for j in cos_idxs]
        cos_sims = [float(x) for x in nn_sim[i]]

        euc_idxs = [int(x) for x in nn_idx_euc[i]]
        euc_labels = [combined_labels[j] for j in euc_idxs]
        euc_dists = [float(x) for x in nn_dist[i]]

        man_idxs = [int(x) for x in nn_idx_man[i]]
        man_labels = [combined_labels[j] for j in man_idxs]
        man_dists = [float(x) for x in nn_dist_man[i]]

        dot_idxs = [int(x) for x in nn_idx_dot[i]]
        dot_labels = [combined_labels[j] for j in dot_idxs]
        dot_sims = [float(x) for x in nn_sim_dot[i]]

        corr_idxs = [int(x) for x in nn_idx_corr[i]]
        corr_labels = [combined_labels[j] for j in corr_idxs]
        corr_dists = [float(x) for x in nn_dist_corr[i]]

        predictions.append({
            'uid': meta.get('uid', None),
            'true': true_label,
            'cosine_pred': cos_labels[0],
            'cosine_topk_preds': cos_labels,
            'cosine_topk_sims': cos_sims,
            'cosine_topk_indices': cos_idxs,
            'euclidean_pred': euc_labels[0],
            'euclidean_topk_preds': euc_labels,
            'euclidean_topk_dists': euc_dists,
            'euclidean_topk_indices': euc_idxs,
            'manhattan_pred': man_labels[0],
            'manhattan_topk_preds': man_labels,
            'manhattan_topk_dists': man_dists,
            'manhattan_topk_indices': man_idxs,
            'dotprod_pred': dot_labels[0],
            'dotprod_topk_preds': dot_labels,
            'dotprod_topk_sims': dot_sims,
            'dotprod_topk_indices': dot_idxs,
            'corr_pred': corr_labels[0],
            'corr_topk_preds': corr_labels,
            'corr_topk_dists': corr_dists,
            'corr_topk_indices': corr_idxs,
        })

        if true_label == cos_labels[0]:
            cos_top1 += 1
        if true_label in cos_labels[:2]:
            cos_top2 += 1
        if true_label in cos_labels[:5]:
            cos_top5 += 1

        if true_label == euc_labels[0]:
            euc_top1 += 1
        if true_label in euc_labels[:2]:
            euc_top2 += 1
        if true_label in euc_labels[:5]:
            euc_top5 += 1

        if true_label == man_labels[0]:
            man_top1 += 1
        if true_label in man_labels[:2]:
            man_top2 += 1
        if true_label in man_labels[:5]:
            man_top5 += 1

        if true_label == dot_labels[0]:
            dot_top1 += 1
        if true_label in dot_labels[:2]:
            dot_top2 += 1
        if true_label in dot_labels[:5]:
            dot_top5 += 1

        if true_label == corr_labels[0]:
            corr_top1 += 1
        if true_label in corr_labels[:2]:
            corr_top2 += 1
        if true_label in corr_labels[:5]:
            corr_top5 += 1

    def pct(n): return n / total * 100 if total > 0 else 0.0

    print('\n--- Using Combined Train+Test as Reference ---')
    print('--- Cosine Similarity ---')
    print(f'Top-1 Accuracy: {pct(cos_top1):.2f}% ({cos_top1}/{total})')
    print(f'Top-2 Accuracy: {pct(cos_top2):.2f}% ({cos_top2}/{total})')
    print(f'Top-5 Accuracy: {pct(cos_top5):.2f}% ({cos_top5}/{total})')

    print('\n--- Euclidean Distance ---')
    print(f'Top-1 Accuracy: {pct(euc_top1):.2f}% ({euc_top1}/{total})')
    print(f'Top-2 Accuracy: {pct(euc_top2):.2f}% ({euc_top2}/{total})')
    print(f'Top-5 Accuracy: {pct(euc_top5):.2f}% ({euc_top5}/{total})')

    print('\n--- Manhattan (L1) Distance ---')
    print(f'Top-1 Accuracy: {pct(man_top1):.2f}% ({man_top1}/{total})')
    print(f'Top-2 Accuracy: {pct(man_top2):.2f}% ({man_top2}/{total})')
    print(f'Top-5 Accuracy: {pct(man_top5):.2f}% ({man_top5}/{total})')

    print('\n--- Dot Product (Inner Product) ---')
    print(f'Top-1 Accuracy: {pct(dot_top1):.2f}% ({dot_top1}/{total})')
    print(f'Top-2 Accuracy: {pct(dot_top2):.2f}% ({dot_top2}/{total})')
    print(f'Top-5 Accuracy: {pct(dot_top5):.2f}% ({dot_top5}/{total})')

    print('\n--- Correlation Distance (1 - Pearson r) ---')
    print(f'Top-1 Accuracy: {pct(corr_top1):.2f}% ({corr_top1}/{total})')
    print(f'Top-2 Accuracy: {pct(corr_top2):.2f}% ({corr_top2}/{total})')
    print(f'Top-5 Accuracy: {pct(corr_top5):.2f}% ({corr_top5}/{total})')

    # Save predictions
    outp = Path(args.output)
    with outp.open('w') as f:
        json.dump(predictions, f, indent=2)

    print(f'Predictions saved to: {outp}')


if __name__ == '__main__':
    main()
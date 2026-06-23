# =============================================================================
# Sign Language Identification — Transformer Classifier
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# ---------------------
# Trains a Masked Transformer to classify skeleton pose sequences into one
# of 8 sign languages:
#   ISL · ASL · CSL · DGS · DSGS · VRT · BSL · SSL
#
# Input  : WebDataset .tar shards (pre-processed by convert_to_shards.py)
# Output : trained model checkpoint, per-epoch metrics CSV, classification
#          reports, confusion matrices, attention rollout CSVs
#
# =============================================================================
# PIPELINE OVERVIEW
# =============================================================================
#
#   Shard dirs (train / val / test)
#        │
#        ▼
#   build_shard_loader           — WebDataset pipeline, decodes pre-processed
#        │                         .npy arrays, pads/truncates to max_frames
#        ▼
#   MaskedTransformerClassifier — Linear proj → Positional Encoding →
#        │                         TransformerEncoder → masked mean pool
#        │                         → classification head
#        ▼
#   Training Loop               — AdamW + linear warmup + cosine decay
#        │                         + gradient clipping + EMA weights
#        │                         + early stopping on val macro-F1
#        ▼
#   Evaluation                  — accuracy / macro-F1 / MCC / kappa
#        │                         per-language breakdown
#        │                         bootstrap 95% CI on test set
#        ▼
#   Interpretability            — attention rollout (Abnar & Zuidema 2020)
#                                  per-frame importance scores saved as CSV
#
# =============================================================================
# DIRECTORY STRUCTURE  (auto-created on first run)
# =============================================================================
#
#   checkpoints/
#     run_YYYYMMDD_HHMMSS/
#       best.pt          ← best val macro-F1 checkpoint (EMA weights)
#       last.pt          ← latest epoch checkpoint
#
#   logs/
#     run_YYYYMMDD_HHMMSS/
#       config.json                   ← full arg dump for reproducibility
#       run_YYYYMMDD_HHMMSS.log       ← full training log (DEBUG level)
#       epoch_metrics.csv             ← one row per epoch, easy to plot
#       val_report_e001.txt           ← sklearn classification report
#       val_cm_e001.csv               ← labeled confusion matrix
#       val_per_language_e001.csv     ← per-class F1 / precision / recall
#       final_test_report.txt         ← final test report with bootstrap CI
#       interpretability/
#         val_0_rollout.csv           ← per-frame attention importance
#         val_0_rollout.txt           ← ASCII bar chart (quick sanity check)
#
# =============================================================================
# USAGE
# =============================================================================
#
#   Train from scratch:
#     python train_transformer.py \
#       --mode train \
#       --train_csv data/train.csv \
#       --val_csv   data/val.csv   \
#       --test_csv  data/test.csv  \
#       --epochs 50 --batch_size 256 --lr 3e-4
#
#   Resume from checkpoint:
#     python train_transformer.py \
#       --mode train \
#       --train_csv data/train.csv \
#       --val_csv   data/val.csv   \
#       --test_csv  data/test.csv  \
#       --resume checkpoints/run_YYYYMMDD_HHMMSS/last.pt
#
#   Test only (loads checkpoint, runs eval + rollout, no training):
#     python train_transformer.py \
#       --mode test \
#       --test_csv data/test.csv   \
#       --resume checkpoints/run_YYYYMMDD_HHMMSS/best.pt
#
# =============================================================================
# KEY ARGUMENTS
# =============================================================================
#
#   --batch_size      int     default 512   training batch size
#   --epochs          int     default 30    max training epochs
#   --lr              float   default 3e-4  peak learning rate (AdamW)
#   --weight_decay    float   default 1e-2  AdamW weight decay
#   --max_frames      int     default 80    frames per sequence (pad/truncate)
#   --warmup_frac     float   default 0.05  fraction of steps for LR warmup
#   --grad_clip       float   default 1.0   gradient norm clipping threshold
#   --label_smoothing float   default 0.1   cross-entropy label smoothing
#   --ema_decay       float   default 0.9999 EMA weight decay
#   --patience        int     default 7     early stopping patience (epochs)
#   --layers          int     default 4     number of Transformer encoder layers
#   --seed            int     default 42    global random seed
#   --num_workers     int     default 4     DataLoader worker processes
#   --device          str     default cuda:0
#
# =============================================================================
# DEPENDENCIES
# =============================================================================
#
#   torch >= 2.0      torchvision    numpy    pandas    scikit-learn
#   tqdm              webdataset
#
#   Install:
#     pip install torch numpy pandas scikit-learn tqdm webdataset
#
# =============================================================================
# CHECKPOINTING NOTES
# =============================================================================
#
#   - best.pt  is saved whenever val macro-F1 improves (not accuracy).
#     Macro-F1 is a better criterion for imbalanced multi-class problems.
#   - Both best.pt and last.pt store: model weights, EMA shadow weights,
#     optimizer state, scheduler state, epoch, global_step, best_val_f1.
#   - Test evaluation runs from a deepcopy of the EMA model so the live
#     training model is never touched mid-loop (fixes original state leak).
#
# =============================================================================
# INTERPRETABILITY NOTES
# =============================================================================
#
#   Attention rollout (Abnar & Zuidema, ACL 2020) approximates which input
#   frames most influenced the final prediction by multiplying per-layer
#   attention matrices together (with residual connection added at each step).
#
#   Output:
#     - importance[t] ∈ [0, 1]  for each real (non-padded) frame t
#     - 1.0 = most attended, 0.0 = least attended
#
#   Cost: ~1-2 seconds for 5 samples at end of training. Not run during
#   the training loop so it has zero impact on training speed.
#
# =============================================================================
# AUTHORS / REFERENCES
# =============================================================================
#
#   Attention rollout:
#     Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in
#     Transformers. ACL 2020. https://arxiv.org/abs/2005.00928
#
#   EMA:
#     Polyak, B. & Juditsky, A. (1992). Acceleration of stochastic
#     approximation by averaging. SIAM Journal on Control and Optimization.
#
# =============================================================================

import argparse
import os
import random
import math
import csv
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import glob
import webdataset as wds

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# ── Bootstrap CI ──────────────────────────────────────────────────────────────
from sklearn.utils import resample

LABEL_MAP = {
    "ISL": 0, "ASL": 1, "CSL": 2, "DGS": 3,
    "DSGS": 4, "VRT": 5, "BSL": 6, "SSL": 7
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP  (file + console)
# ══════════════════════════════════════════════════════════════════════════════

def build_logger(log_dir: str, run_name: str) -> logging.Logger:
    """
    Dual-sink logger: INFO to console, DEBUG to rotating file.
    Every run gets its own timestamped log file so nothing is overwritten.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler ─ INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler ─ DEBUG and above (captures everything)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{run_name}.log"), encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  (save / restore for reproducibility)
# ══════════════════════════════════════════════════════════════════════════════

def save_config(args, run_dir: str):
    cfg = vars(args)
    cfg["run_dir"] = run_dir
    cfg["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET  — WebDataset shard-based loader
# ══════════════════════════════════════════════════════════════════════════════
#
# Each shard sample contains:
#   data.npy   — float32 array (frames, 152) pre-processed by convert_to_shards.py
#   class.txt  — sign language label string e.g. "ISL"
#
# Preprocessing (normalization, masked frame fixing) was already done at
# shard-creation time, so this loader only does:
#   1. Pad / truncate to max_frames
#   2. Build the padding mask
#   3. Map label string → integer via LABEL_MAP
#
# ── Why glob pattern for shards ──────────────────────────────────────────────
# WebDataset accepts a glob string or a list of tar paths.
# Pass the shard directory and it expands shard-*.tar automatically.
# ─────────────────────────────────────────────────────────────────────────────

def _decode_sample(sample: dict, max_frames: int) -> dict:
    """
    Decode one WebDataset sample dict into model-ready tensors.
    Called inside the wds pipeline via .map().

    sample keys after wds decoding:
        "__key__"  : str  (filename stem)
        "data.npy" : np.ndarray  (frames, feat_dim)  float32
        "class.txt": str  (label, e.g. "ISL")
    """
    feats = sample["data.npy"]                        # (T, D)  already float32
    label = sample["class.txt"].strip().upper()

    if label not in LABEL_MAP:
        return None                                   # corrupted sample — wds will skip

    y = LABEL_MAP[label]
    # T, D = feats.shape
    # mask = np.ones(T, dtype=np.float32)

    # if T < max_frames:
    #     pad   = max_frames - T
    #     feats = np.vstack([feats, np.zeros((pad, D), dtype=np.float32)])
    #     mask  = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
    # else:
    #     feats = feats[:max_frames]
    #     mask  = mask[:max_frames]
    
    T, D = feats.shape

    # Step sampling — uniformly sample max_frames indices across full length
    # This mirrors the original PoseLangClsDataset step_frames logic:
    # instead of taking first 80 frames, take 80 evenly spaced frames
    # so the model sees the entire video compressed into max_frames slots.
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames).astype(int)
        feats   = feats[indices]
        T       = max_frames

    mask = np.ones(T, dtype=np.float32)

    if T < max_frames:
        pad   = max_frames - T
        feats = np.vstack([feats, np.zeros((pad, D), dtype=np.float32)])
        mask  = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

    return {
        "x":    torch.tensor(feats, dtype=torch.float32),
        "mask": torch.tensor(mask,  dtype=torch.float32),
        "y":    torch.tensor(y,     dtype=torch.long),
    }


def build_shard_loader(shard_dir: str,
                       max_frames: int,
                       batch_size: int,
                       num_workers: int,
                       shuffle: bool = False,
                       shuffle_buffer: int = 2000) -> tuple:
    """
    Build a WebDataset pipeline + DataLoader for one split.

    Returns (loader, approx_dataset_size).
    approx_dataset_size is read from a companion size.txt file written by
    convert_to_shards.py — used for tqdm and scheduler step count.
    If the file is absent, returns None for size (tqdm will still work,
    just without a total count).

    Args:
        shard_dir      : directory containing shard-*.tar files
        max_frames     : pad/truncate target length
        batch_size     : DataLoader batch size
        num_workers    : DataLoader worker count
        shuffle        : whether to shuffle (True for train, False for val/test)
        shuffle_buffer : how many samples to buffer for shuffle (train only)
    """
    # Glob pattern WebDataset understands
    shard_pattern = os.path.join(shard_dir, "shard-*.tar")
    shard_files   = sorted(glob.glob(shard_pattern))

    if not shard_files:
        raise FileNotFoundError(
            f"No shard files found at: {shard_pattern} — Run convert_to_shards.py first."
        )

    # ── Read dataset size from companion file (written by convert_to_shards) ──
    size_file = os.path.join(shard_dir, "size.txt")
    dataset_size = None
    if os.path.exists(size_file):
        with open(size_file) as f:
            dataset_size = int(f.read().strip())

    # ── Build WebDataset pipeline ─────────────────────────────────────────────
    dataset = (
        wds.WebDataset(shard_files, shardshuffle=shuffle, nodesplitter=wds.split_by_node)
        .decode()                                         # auto-decodes .npy and .txt
        .map(lambda s: _decode_sample(s, max_frames))    # pad/truncate + label map
        .select(lambda s: s is not None)                 # drop corrupted samples
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)        # in-memory shuffle buffer

    # wds.batched default_collation_fn expects list/tuple — pass torch collate
    # explicitly so our dict samples are collated correctly into batched dicts.
    dataset = dataset.batched(
        batch_size,
        partial=True,
        collation_fn=torch.utils.data.default_collate
    )

    loader = wds.WebLoader(
        dataset,
        batch_size  = None,     # batching already done above in .batched()
        num_workers = num_workers,
        pin_memory  = True,
    )

    return loader, dataset_size, len(shard_files)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (unchanged from original — no model changes per request)
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MaskedTransformerClassifier(nn.Module):

    def __init__(self, input_dim, num_classes,
                 d_model=256, nhead=8,
                 num_layers=4, ff_dim=512,
                 dropout=0.1, max_len=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos  = PositionalEncoding(d_model, max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask, return_attn=False):
        x        = self.proj(x)
        x        = self.pos(x)
        pad_mask = (mask == 0)

        if return_attn:
            # ── Attention rollout: collect per-layer attention weights ────────
            attn_weights = []

            def _hook(module, inp, out):
                # TransformerEncoderLayer in PyTorch >= 2.0 returns (x, attn)
                # when need_weights=True; here we capture via a forward hook.
                # We store the input to the self_attn sublayer instead.
                pass

            x = self.encoder(x, src_key_padding_mask=pad_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=pad_mask)

        mask_3d = mask.unsqueeze(-1)
        x       = x * mask_3d
        feat    = x.sum(1) / mask_3d.sum(1).clamp(min=1)
        return self.head(feat)


# ══════════════════════════════════════════════════════════════════════════════
# EMA  (Exponential Moving Average of model weights)
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Maintains a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    Use the EMA model for evaluation — it is typically more stable
    than the live model, especially with small datasets.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE — warmup + cosine, resume-aware
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """
    Linear warmup for `warmup_steps`, then cosine decay to 0.
    Works correctly with resume because it's step-based, not epoch-based.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(labels, preds, metric_fn, n=1000, ci=0.95) -> tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval for any sklearn metric.
    Returns (lower, upper) bounds.
    """
    scores = []
    for _ in range(n):
        idx = resample(range(len(labels)), replace=True)
        try:
            scores.append(metric_fn(labels[idx], preds[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    alpha  = (1 - ci) / 2
    return float(np.percentile(scores, 100 * alpha)), \
           float(np.percentile(scores, 100 * (1 - alpha)))


@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader: DataLoader,
                     device: torch.device,
                     run_dir: str = None,
                     split: str = "val",
                     epoch: int = None,
                     logger: logging.Logger = None,
                     bootstrap: bool = False) -> dict:
    """
    Full evaluation pass.

    Additions over original:
      - Saves classification report and confusion matrix to disk
      - Computes per-class metrics explicitly
      - Optionally computes bootstrap 95% CIs for macro-F1 and accuracy
      - Returns structured dict suitable for CSV logging
    """
    model.eval()
    preds_list, labels_list = [], []

    for b in loader:
        x   = b["x"].to(device)
        m   = b["mask"].to(device)
        y   = b["y"].to(device)
        out = model(x, m).argmax(-1)
        preds_list.append(out.cpu().numpy())
        labels_list.append(y.cpu().numpy())

    preds  = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    wt_f1    = f1_score(labels, preds, average="weighted")
    bal_acc  = balanced_accuracy_score(labels, preds)
    kappa    = cohen_kappa_score(labels, preds)
    mcc      = matthews_corrcoef(labels, preds)
    report   = classification_report(
        labels, preds, target_names=list(LABEL_MAP.keys()), digits=4
    )
    cm = confusion_matrix(labels, preds)

    result = {
        "accuracy":          acc,
        "macro_f1":          macro_f1,
        "micro_f1":          micro_f1,
        "weighted_f1":       wt_f1,
        "balanced_accuracy": bal_acc,
        "kappa":             kappa,
        "mcc":               mcc,
        "report":            report,
        "cm":                cm,
        "preds":             preds,    # raw arrays for per_language_breakdown
        "labels":            labels,
    }

    # ── Bootstrap CIs (optional — slow for large test sets) ──────────────────
    if bootstrap:
        acc_lo,  acc_hi  = bootstrap_ci(labels, preds,
                                         accuracy_score)
        f1_lo,   f1_hi   = bootstrap_ci(labels, preds,
                                         lambda l, p: f1_score(l, p, average="macro"))
        result["acc_ci"]  = (acc_lo, acc_hi)
        result["f1_ci"]   = (f1_lo,  f1_hi)

    # ── Save to disk ──────────────────────────────────────────────────────────
    if run_dir is not None:
        epoch_tag = f"_e{epoch:03d}" if epoch is not None else ""

        # Classification report (text)
        report_path = os.path.join(run_dir, f"{split}_report{epoch_tag}.txt")
        with open(report_path, "w") as f:
            f.write(report)
            if bootstrap and "acc_ci" in result:
                f.write(f"\nAcc  95% CI : [{result['acc_ci'][0]:.4f}, {result['acc_ci'][1]:.4f}]")
                f.write(f"\nF1   95% CI : [{result['f1_ci'][0]:.4f},  {result['f1_ci'][1]:.4f}]")

        # Confusion matrix (CSV)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{k}"  for k in LABEL_MAP.keys()],
            columns=[f"Pred_{k}" for k in LABEL_MAP.keys()]
        )
        cm_df.to_csv(os.path.join(run_dir, f"{split}_cm{epoch_tag}.csv"))

    # ── Console / logger output ───────────────────────────────────────────────
    if logger is not None:
        logger.info(f"[{split.upper()}] Acc={acc:.4f}  MacroF1={macro_f1:.4f}  "
                    f"MCC={mcc:.4f}  Kappa={kappa:.4f}")
        logger.debug(f"\n{report}")
        if bootstrap and "acc_ci" in result:
            logger.info(f"[{split.upper()}] Acc CI=[{result['acc_ci'][0]:.4f}, "
                        f"{result['acc_ci'][1]:.4f}]  "
                        f"F1 CI=[{result['f1_ci'][0]:.4f}, {result['f1_ci'][1]:.4f}]")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION ROLLOUT  (frame-level interpretability)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def attention_rollout(model: nn.Module,
                      sample: dict,
                      device: torch.device,
                      run_dir: str,
                      sample_id: str = "sample"):
    """
    Approximates per-frame importance by rolling attention matrices
    across Transformer encoder layers (Abnar & Zuidema, 2020).

    Saves a per-frame importance CSV and a simple text visualisation
    so you can inspect which frames drive the prediction.

    Note: PyTorch's TransformerEncoderLayer does not expose attention
    weights by default. This implementation patches the encoder to
    collect them via forward hooks.
    """
    model.eval()
    attn_maps = []

    hooks = []
    for layer in model.encoder.layers:
        def _hook(mod, inp, out, _store=attn_maps):
            # MultiheadAttention returns (attn_output, attn_weights)
            # when need_weights=True — we re-run just the self-attn call.
            q = k = v = inp[0]
            with torch.no_grad():
                _, w = mod.forward(q, k, v, need_weights=True,average_attn_weights=True)
            _store.append(w.cpu())  # (B, T, T)

        hooks.append(layer.self_attn.register_forward_hook(_hook))

    x = sample["x"].unsqueeze(0).to(device)
    m = sample["mask"].unsqueeze(0).to(device)
    logits = model(x, m)
    pred   = logits.argmax(-1).item()

    for h in hooks:
        h.remove()

    if not attn_maps:
        return

    # ── Rollout ───────────────────────────────────────────────────────────────
    T    = attn_maps[0].shape[-1]
    roll = torch.eye(T)

    for A in attn_maps:
        A    = A[0]                          # (T, T)
        A    = A + torch.eye(T)              # add residual connection
        A    = A / A.sum(dim=-1, keepdim=True)
        roll = A @ roll

    # Mean over all rows → per-frame importance
    importance = roll.mean(0).numpy()        # (T,)

    # Only consider real (non-padded) frames
    n_real = int(sample["mask"].sum().item())
    importance = importance[:n_real]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_dir = os.path.join(run_dir, "interpretability")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        "frame":      range(n_real),
        "importance": importance
    })
    df.to_csv(os.path.join(out_dir, f"{sample_id}_rollout.csv"), index=False)

    # ── ASCII bar chart (quick sanity check in log) ───────────────────────────
    top_k   = min(20, n_real)
    top_idx = np.argsort(importance)[-top_k:][::-1]

    lines = [f"Attention rollout — predicted: {IDX_TO_LABEL[pred]}",
             f"Top-{top_k} important frames (of {n_real}):"]
    for rank, fi in enumerate(top_idx, 1):
        bar = "█" * int(importance[fi] * 30)
        lines.append(f"  #{rank:2d}  frame {fi:4d}  {importance[fi]:.3f}  {bar}")

    rollout_txt = "\n".join(lines)

    with open(os.path.join(out_dir, f"{sample_id}_rollout.txt"), "w") as f:
        f.write(rollout_txt)

    return importance


# ══════════════════════════════════════════════════════════════════════════════
# PER-LANGUAGE  breakdown  (for error analysis)
# ══════════════════════════════════════════════════════════════════════════════

def per_language_breakdown(labels: np.ndarray,
                           preds:  np.ndarray,
                           run_dir: str,
                           split:  str,
                           epoch:  int = None,
                           logger: logging.Logger = None) -> pd.DataFrame:
    """
    Per-class precision / recall / F1 / support saved as CSV.
    Highlights any class with F1 < 0.5 so weak languages are obvious.
    """
    p, r, f, s = precision_recall_fscore_support(labels, preds, average=None,
                                                  labels=list(range(NUM_CLASSES)))
    rows = []
    for i, name in IDX_TO_LABEL.items():
        rows.append({
            "language":  name,
            "precision": round(float(p[i]), 4),
            "recall":    round(float(r[i]), 4),
            "f1":        round(float(f[i]), 4),
            "support":   int(s[i]),
            "weak":      f[i] < 0.5
        })

    df = pd.DataFrame(rows)

    epoch_tag = f"_e{epoch:03d}" if epoch is not None else ""
    df.to_csv(
        os.path.join(run_dir, f"{split}_per_language{epoch_tag}.csv"),
        index=False
    )

    if logger:
        logger.info(f"\n[{split.upper()}] Per-language F1:\n" +
                    df[["language", "f1", "support", "weak"]].to_string(index=False))
        weak = df[df["weak"]]["language"].tolist()
        if weak:
            logger.warning(f"Weak languages (F1 < 0.5): {weak}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CSV METRIC LOGGER  (one row per epoch, easy to plot later)
# ══════════════════════════════════════════════════════════════════════════════

class EpochCSVLogger:
    """
    Appends one row per epoch to a CSV file.
    Creates the file with headers on first call.
    """

    def __init__(self, path: str):
        self.path    = path
        self._header = False

    def log(self, row: dict):
        write_header = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training when the monitored metric has not improved for
    `patience` epochs. Improvement threshold is configurable.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline(args):
    set_seed(args.seed)

    # ── Run directory (timestamped so nothing is overwritten) ─────────────────
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(args.log_dir, run_name)
    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = build_logger(run_dir, run_name)
    logger.info(f"Run: {run_name}")
    logger.info(f"Args: {vars(args)}")

    save_config(args, run_dir)

    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    csv_log   = EpochCSVLogger(os.path.join(run_dir, "epoch_metrics.csv"))
    es        = EarlyStopping(patience=args.patience)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Data  (WebDataset shard loaders) ─────────────────────────────────────
    train_loader, train_size, train_shards = build_shard_loader(
        args.train_shards, args.max_frames, args.batch_size,
        args.num_workers, shuffle=True, shuffle_buffer=args.shuffle_buffer
    )
    val_loader,  val_size,  val_shards  = build_shard_loader(
        args.val_shards, args.max_frames, args.batch_size,
        args.num_workers, shuffle=False
    )
    test_loader, test_size, test_shards = build_shard_loader(
        args.test_shards, args.max_frames, args.batch_size,
        args.num_workers, shuffle=False
    )

    # input_dim is fixed by convert_to_shards.py: 152
    # (left_hand 42 + right_hand 42 + pose 22 + face 46)
    input_dim = 152

    logger.info(
        f"Input dim: {input_dim}  |  "
        f"Train shards: {train_shards} (~{train_size} samples)  |  "
        f"Val shards: {val_shards} (~{val_size} samples)  |  "
        f"Test shards: {test_shards} (~{test_size} samples)"
    )

    # Steps per epoch — needed for scheduler.
    # With WebDataset we don't have len(loader) so compute from dataset size.
    if train_size is not None:
        steps_per_epoch = math.ceil(train_size / args.batch_size)
    else:
        # fallback: user can pass --steps_per_epoch explicitly
        steps_per_epoch = args.steps_per_epoch
        if steps_per_epoch is None:
            raise ValueError(
                "Cannot determine steps_per_epoch: no size.txt found in "
                f"{args.train_shards} and --steps_per_epoch not set."
            )
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # Build a small fixed rollout buffer from val shards for interpretability.
    # WebDataset is not indexable, so we collect 5 samples once at startup.
    rollout_samples = []
    for sample in val_loader:
        for i in range(sample["x"].shape[0]):
            rollout_samples.append({
                "x":    sample["x"][i],
                "mask": sample["mask"][i],
                "y":    sample["y"][i],
            })
            if len(rollout_samples) >= 5:
                break
        if len(rollout_samples) >= 5:
            break

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MaskedTransformerClassifier(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        num_layers=args.layers,
        max_len=args.max_frames
    ).to(device)

    ema = EMA(model, decay=args.ema_decay)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler    = build_scheduler(opt, total_steps, warmup_steps)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # FIX: use torch.amp (not deprecated torch.cuda.amp)
    use_amp = device.type == "cuda"
    scaler  = GradScaler(device.type) if use_amp else None

    best_val_f1 = -1.0       # FIX: checkpoint on macro F1, not accuracy
    start_epoch = 1
    global_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume is not None:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed at epoch {start_epoch}  best_val_f1={best_val_f1:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # EPOCH LOOP
    # ══════════════════════════════════════════════════════════════════════════

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        total_loss    = 0.0
        correct       = 0
        total         = 0
        steps_in_epoch = 0

        for b in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                      total=steps_per_epoch, leave=False):
            x = b["x"].to(device)
            m = b["mask"].to(device)
            y = b["y"].to(device)

            opt.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    out  = model(x, m)
                    loss = loss_fn(out, y)
                scaler.scale(loss).backward()
                # FIX: gradient clipping (unscale first, then clip)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                out  = model(x, m)
                loss = loss_fn(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            scheduler.step()      # step-based, not epoch-based
            ema.update(model)     # update EMA shadow weights

            total_loss    += loss.item()
            pred           = out.argmax(-1)
            correct       += (pred == y).sum().item()
            total         += y.size(0)
            global_step   += 1
            steps_in_epoch += 1

        train_acc  = correct / total
        train_loss = total_loss / max(steps_in_epoch, 1)
        cur_lr     = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:03d} | loss={train_loss:.4f}  acc={train_acc:.4f}  "
            f"lr={cur_lr:.2e}  time={epoch_time:.1f}s"
        )

        # ── Evaluate EMA model on val ─────────────────────────────────────────
        val_metrics = evaluate_metrics(
            ema.shadow, val_loader, device,
            run_dir=run_dir, split="val", epoch=epoch, logger=logger
        )
        per_language_breakdown(
            val_metrics["labels"],
            val_metrics["preds"],
            run_dir=run_dir, split="val", epoch=epoch, logger=logger
        )

        val_f1 = val_metrics["macro_f1"]

        # ── CSV row ───────────────────────────────────────────────────────────
        csv_log.log({
            "epoch":       epoch,
            "global_step": global_step,
            "train_loss":  round(train_loss, 6),
            "train_acc":   round(train_acc,  4),
            "val_acc":     round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_f1,               4),
            "val_mcc":     round(val_metrics["mcc"],     4),
            "val_kappa":   round(val_metrics["kappa"],   4),
            "lr":          round(cur_lr, 8),
        })

        # ── Checkpoint: last ──────────────────────────────────────────────────
        torch.save({
            "model":       model.state_dict(),
            "ema":         ema.state_dict(),
            "optimizer":   opt.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "epoch":       epoch,
            "global_step": global_step,
            "best_val_f1": best_val_f1,      # FIX: save current best, not stale
        }, last_path)

        # ── Checkpoint: best (on macro F1) ────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model":       model.state_dict(),
                "ema":         ema.state_dict(),
                "optimizer":   opt.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "epoch":       epoch,
                "global_step": global_step,
                "best_val_f1": best_val_f1,
            }, best_path)
            logger.info(f"New best val macro-F1={best_val_f1:.4f}  → saved best.pt")

            # FIX: run test OUTSIDE the main model's state —
            #      load a fresh copy so training loop is unaffected
            eval_model = deepcopy(ema.shadow).to(device)
            test_metrics = evaluate_metrics(
                eval_model, test_loader, device,
                run_dir=run_dir, split="test", epoch=epoch,
                logger=logger, bootstrap=True
            )
            per_language_breakdown(
                test_metrics["labels"],
                test_metrics["preds"],
                run_dir=run_dir, split="test", epoch=epoch, logger=logger
            )
            del eval_model

        # ── Early stopping ────────────────────────────────────────────────────
        if es.step(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch} "
                        f"(patience={args.patience})")
            break

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL TEST  on best checkpoint
    # ══════════════════════════════════════════════════════════════════════════

    logger.info("=== Final test on best checkpoint ===")
    best_ckpt = torch.load(best_path, map_location=device)

    final_model = MaskedTransformerClassifier(
        input_dim=input_dim, num_classes=NUM_CLASSES,
        num_layers=args.layers, max_len=args.max_frames
    ).to(device)
    final_model.load_state_dict(best_ckpt["ema"])   # use EMA weights
    final_model.eval()

    test_metrics = evaluate_metrics(
        final_model, test_loader, device,
        run_dir=run_dir, split="final_test", logger=logger, bootstrap=True
    )

    # ── Attention rollout on the pre-collected rollout_samples buffer ───────────
    logger.info(f"Running attention rollout on {len(rollout_samples)} val samples …")
    for i, sample in enumerate(rollout_samples):
        attention_rollout(
            final_model, sample, device,
            run_dir=run_dir, sample_id=f"val_{i}"
        )

    logger.info(f"All outputs saved to: {run_dir}")


# ── helper: collect predictions without touching model state ──────────────────
@torch.no_grad()
def _collect_preds(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        out.append(model(b["x"].to(device), b["mask"].to(device)).argmax(-1).cpu().numpy())
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════════
# TEST-ONLY MODE
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline(args):
    """
    FIX: --mode test was accepted by argparse but silently did nothing.
    Now loads a checkpoint and runs full evaluation + interpretability.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(args.log_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    logger = build_logger(run_dir, "test_run")

    test_loader, test_size, test_shards = build_shard_loader(
        args.test_shards, args.max_frames, args.batch_size,
        args.num_workers, shuffle=False
    )
    ckpt      = torch.load(args.resume, map_location=device)
    input_dim = 152    # fixed by convert_to_shards.py

    model = MaskedTransformerClassifier(
        input_dim=input_dim, num_classes=NUM_CLASSES,
        num_layers=args.layers, max_len=args.max_frames
    ).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()

    evaluate_metrics(model, test_loader, device,
                     run_dir=run_dir, split="test",
                     logger=logger, bootstrap=True)

    # Collect 5 samples from test loader for rollout
    test_rollout_samples = []
    for sample in test_loader:
        for i in range(sample["x"].shape[0]):
            test_rollout_samples.append({
                "x": sample["x"][i], "mask": sample["mask"][i], "y": sample["y"][i]
            })
            if len(test_rollout_samples) >= 5:
                break
        if len(test_rollout_samples) >= 5:
            break

    logger.info(f"Running attention rollout on {len(test_rollout_samples)} samples …")
    for i, sample in enumerate(test_rollout_samples):
        attention_rollout(model, sample, device,
                          run_dir=run_dir, sample_id=f"test_{i}")

    logger.info(f"Done. Outputs: {run_dir}")




# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "test"], required=True)

    # Data  (pass the shard directory for each split)
    parser.add_argument("--train_shards", default="/DATA7/dattatreya/identificationv3/train",help="Dir containing train shard-*.tar files")
    parser.add_argument("--val_shards",  default="/DATA7/dattatreya/identificationv3/val",  help="Dir containing val   shard-*.tar files")
    parser.add_argument("--test_shards",  default="/DATA7/dattatreya/identificationv3/test", help="Dir containing test  shard-*.tar files")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="Override steps/epoch (auto-detected from size.txt if absent)")
    parser.add_argument("--shuffle_buffer", type=int, default=2000,
                        help="WebDataset shuffle buffer size for train (default: 2000)")

    # Training
    parser.add_argument("--batch_size",       type=int,   default=1024)
    parser.add_argument("--epochs",           type=int,   default=150)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--weight_decay",     type=float, default=1e-2)
    parser.add_argument("--max_frames",       type=int,   default=80)
    parser.add_argument("--warmup_frac",      type=float, default=0.05,
                        help="Fraction of total steps used for linear warmup")
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--label_smoothing",  type=float, default=0.1)
    parser.add_argument("--ema_decay",        type=float, default=0.9999)
    parser.add_argument("--patience",         type=int,   default=7,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--num_workers",      type=int,   default=4)

    # Model
    parser.add_argument("--layers",  type=int,   default=4)
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--resume",  default=None)

    # Output
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--log_dir",  default="logs")

    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline(args)
    elif args.mode == "test":
        if args.resume is None:
            raise ValueError("--mode test requires --resume <checkpoint_path>")
        test_pipeline(args)


if __name__ == "__main__":
    main()
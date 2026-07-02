"""
Self-supervised pretraining of a pose-sequence encoder via momentum contrast.

This script trains a Transformer-based encoder over 522-dimensional skeletal
pose sequences (body, hand, and face landmarks) using a MoCo-style contrastive
objective combined with an auxiliary frame-level reconstruction objective.
The resulting backbone is intended for downstream use in topic-modeling /
representation-learning experiments over sign-language pose data.

Pipeline overview
------------------
1. ``ParagraphDataset`` loads pre-extracted per-paragraph pose tensors and
   applies a stack of temporal/spatial augmentations (cropping, speed
   perturbation, frame dropout, reversal, and regional keypoint masking) to
   produce two independently augmented views of each sequence.
2. ``PoseEncoder`` (a Transformer encoder with attention-based temporal
   pooling) maps a padded pose sequence to a fixed-size embedding, and can
   additionally decode per-frame reconstructions from its hidden states.
3. ``MoCoContrastive`` wraps a query/key pair of encoders (the key encoder
   is a momentum-averaged copy of the query encoder) and maintains a queue
   of negative embeddings for the InfoNCE contrastive loss.
4. ``Trainer`` owns the end-to-end training loop: data loading, the
   combined contrastive + reconstruction loss, checkpointing, early
   stopping, and (optionally) resuming a previous run.

Usage
-----
Adjust the paths and hyperparameters in ``Config`` below, then run this
file directly. To resume from a saved checkpoint, set
``cfg.resume_from_checkpoint`` to either ``"latest"`` or an explicit path
to a checkpoint file before instantiating the ``Trainer``.
"""

import copy
import glob
import math
import os
import random
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# ==================== Configuration ====================

class Config:
    """Central configuration for data paths, model architecture, and
    optimization hyperparameters used throughout the training pipeline."""

    # ---- Paths ----
    preprocessed_dir = "/data/dept_share/sanjeet/Khushi/topic_modeling/new03/video_level_pose"
    output_dir       = "/data/dept_share/sanjeet/Khushi/topic_modeling/new03/train_encoder_output03"
    checkpoint_dir   = os.path.join(output_dir, "checkpoints")
    log_dir          = os.path.join(output_dir, "logs")
    embedding_dir    = os.path.join(output_dir, "embeddings")

    # ---- Resume ----
    # None             -> start a fresh training run
    # "latest"         -> auto-load <checkpoint_dir>/latest.pt
    # "/full/path.pt"  -> load a specific checkpoint file
    resume_from_checkpoint = None

    # ---- Model ----
    pose_dim                 = 522
    d_model                  = 512
    nhead                    = 8
    num_encoder_layers       = 6
    d_feedforward             = 2048
    dropout                  = 0.1
    output_embed_dim         = 256
    proj_dim                 = 128
    max_position_embeddings  = 4096

    # ---- Training ----
    batch_size    = 16
    num_workers   = 2
    learning_rate = 3e-4
    weight_decay  = 1e-4
    max_epochs    = 100
    warmup_epochs = 5

    # ---- MoCo ----
    use_contrastive    = True
    contrastive_weight = 1.0
    temperature        = 0.07
    queue_size          = 4096
    momentum            = 0.999

    # ---- Reconstruction ----
    use_reconstruction    = True
    reconstruction_weight = 0.05

    # ---- Augmentation ----
    augment             = True
    crop_ratio_min      = 0.4
    crop_ratio_max      = 1.0
    speed_factor_min    = 0.75
    speed_factor_max    = 1.25
    frame_drop_keep_min = 0.7
    frame_drop_keep_max = 1.0
    reverse_prob        = 0.2
    kp_mask_prob        = 0.5
    body_regions        = [(0, 174), (174, 348), (348, 522)]

    # ---- Logging & checkpointing ----
    log_every                = 100
    save_every               = 1
    validate_every           = 1
    patience                 = 15
    keep_last_n_checkpoints  = 1000

    # ---- System ----
    seed            = 42
    device          = "cuda"
    mixed_precision = True
    pin_memory      = True

    def setup(self):
        """Create the output/checkpoint/log/embedding directories if they
        do not already exist."""
        for d in (self.output_dir, self.checkpoint_dir,
                  self.log_dir, self.embedding_dir):
            os.makedirs(d, exist_ok=True)


cfg = Config()
cfg.setup()


# ==================== Dataset ====================

class ParagraphDataset(Dataset):
    """Loads paragraph-level pose sequences and (optionally) produces two
    augmented views per sample for contrastive training.

    Each item on disk stores a padded pose tensor ``X`` of shape
    ``(L, 522)``, a boolean validity mask ``M`` of the same shape, and the
    true (unpadded) sequence length ``L``. When ``augment=True``, two
    independently augmented copies (``Xq``/``Mq``/``Lq`` and
    ``Xk``/``Mk``/``Lk``) are generated per call to ``__getitem__`` for use
    as the query/key views in the MoCo objective.
    """

    def __init__(self, preprocessed_dir, augment=False,
                 max_seq_len=4096, config=None):
        self.preprocessed_dir = preprocessed_dir
        self.augment          = augment
        self.max_seq_len      = max_seq_len
        self.cfg              = config or cfg

        self.metadata = pd.read_csv(os.path.join(preprocessed_dir, "metadata.csv"))
        print(f"[Dataset] {len(self.metadata)} paragraphs  augment={augment}")
        print(f"  avg_len={self.metadata['length'].mean():.0f}  "
              f"max_len={self.metadata['length'].max()}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row  = self.metadata.iloc[idx]
        data = torch.load(os.path.join(self.preprocessed_dir, row["file"]),
                          map_location="cpu", weights_only=False)
        X = data["X"]
        M = data["M"]
        L = int(data["L"])

        # Defensive truncation in case a stored sequence exceeds the
        # configured maximum length.
        if L > self.max_seq_len:
            X, M, L = X[:self.max_seq_len], M[:self.max_seq_len], self.max_seq_len

        assert X.shape == (L, 522) and M.shape == (L, 522)

        result = {
            "X": X, "M": M, "L": L,
            "uid":      row["uid"],
            "para_idx": int(row["para_idx"]) if "para_idx" in row else -1,
        }

        if self.augment:
            Xq, Mq, Lq = self._augment(X, M, L)
            Xk, Mk, Lk = self._augment(X, M, L)
            result.update({"Xq": Xq, "Mq": Mq, "Lq": Lq,
                           "Xk": Xk, "Mk": Mk, "Lk": Lk})
        return result

    def _augment(self, X, M, L):
        """Apply a randomized stack of temporal and spatial perturbations
        to a single pose sequence and return the transformed
        ``(X, M, L)`` triple."""
        X, M = X.clone(), M.clone()

        # 1. Temporal crop: sample a contiguous sub-sequence.
        crop_len = max(10, int(L * random.uniform(self.cfg.crop_ratio_min,
                                                   self.cfg.crop_ratio_max)))
        if crop_len < L:
            s = random.randint(0, L - crop_len)
            X, M, L = X[s:s+crop_len], M[s:s+crop_len], crop_len

        # 2. Speed perturbation: resample the timeline to a new length.
        if random.random() < 0.5:
            new_len = max(10, min(int(L * random.uniform(self.cfg.speed_factor_min,
                                                          self.cfg.speed_factor_max)),
                                  self.max_seq_len))
            if new_len != L:
                idx = torch.clamp(torch.linspace(0, L-1, new_len).long(), 0, L-1)
                X, M, L = X[idx], M[idx], new_len

        # 3. Frame dropout: randomly discard a subset of frames.
        if random.random() < 0.5:
            n_keep = max(10, int(L * random.uniform(self.cfg.frame_drop_keep_min,
                                                     self.cfg.frame_drop_keep_max)))
            idx = torch.sort(torch.randperm(L)[:n_keep]).values
            X, M, L = X[idx], M[idx], n_keep

        # 4. Frame reversal: play the sequence backwards.
        if random.random() < self.cfg.reverse_prob:
            X, M = X.flip(0), M.flip(0)

        # 5. Regional keypoint masking: zero out one body region
        #    (body / hands / face) across all frames.
        if random.random() < self.cfg.kp_mask_prob:
            a, b = random.choice(self.cfg.body_regions)
            X[:, a:b] = 0.0
            M[:, a:b] = False

        return X, M, L


# ==================== Collate ====================

def collate_fn(batch):
    """Pad a list of variable-length pose sequences into a single batch,
    building both the per-keypoint validity mask and a per-frame mask
    (a frame is considered valid if any of its keypoints are valid).

    When the batch contains augmented query/key views, those are padded
    and stacked in the same way and included under the ``_q``/``_k``
    suffixes.
    """
    B, pose_dim = len(batch), 522

    def _pad(xs, ms, ls):
        T   = max(ls)
        Xp  = torch.zeros(B, T, pose_dim)
        Mp  = torch.zeros(B, T, pose_dim, dtype=torch.bool)
        Mfp = torch.zeros(B, T,           dtype=torch.bool)
        for i, (x, m, l) in enumerate(zip(xs, ms, ls)):
            Xp[i, :l] = x; Mp[i, :l] = m; Mfp[i, :l] = m.any(dim=1)
        return Xp, Mp, Mfp, torch.tensor(ls, dtype=torch.long)

    Xp, Mp, Mfp, Lp = _pad([s["X"] for s in batch],
                             [s["M"] for s in batch],
                             [s["L"] for s in batch])
    result = {"X": Xp, "M": Mp, "M_frame": Mfp, "L": Lp,
              "uid": [s["uid"] for s in batch],
              "para_idx": [s["para_idx"] for s in batch]}

    if "Xq" in batch[0]:
        Xq, Mq, Mfq, Lq = _pad([s["Xq"] for s in batch],
                                 [s["Mq"] for s in batch],
                                 [s["Lq"] for s in batch])
        Xk, Mk, Mfk, Lk = _pad([s["Xk"] for s in batch],
                                 [s["Mk"] for s in batch],
                                 [s["Lk"] for s in batch])
        result.update({"Xq": Xq, "Mq": Mq, "M_frame_q": Mfq, "Lq": Lq,
                       "Xk": Xk, "Mk": Mk, "M_frame_k": Mfk, "Lk": Lk})
    return result


# ==================== Model ====================

class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding, added to the input
    embeddings before they are passed through the Transformer encoder."""

    def __init__(self, d_model, max_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalPooling(nn.Module):
    """Attention-based pooling over the time dimension: learns a scalar
    attention logit per frame and returns the resulting weighted sum of
    frame representations, ignoring padded (invalid) frames."""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x, mask_frame=None):
        logits = self.attn(x).squeeze(-1)
        if mask_frame is not None:
            # Guard against fully-padded sequences: if a sequence has no
            # valid frames at all, fall back to attending over every frame
            # rather than producing an all -inf softmax.
            safe   = mask_frame | ~mask_frame.any(dim=1, keepdim=True)
            logits = logits.masked_fill(~safe, float("-inf"))
        return (x * F.softmax(logits, dim=1).unsqueeze(-1)).sum(dim=1)


class PoseEncoder(nn.Module):
    """Transformer-based encoder for pose sequences.

    Maps a padded ``(B, T, pose_dim)`` sequence to a fixed-size embedding
    via input projection, sinusoidal positional encoding, a stack of
    pre-norm Transformer encoder layers, and attention-based temporal
    pooling. A lightweight linear reconstruction head can additionally
    decode per-frame pose vectors from the encoder's hidden states, used
    for the auxiliary reconstruction loss.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg        = config
        self.input_proj = nn.Linear(config.pose_dim, config.d_model)
        self.pos_enc    = PositionalEncoding(config.d_model,
                                             config.max_position_embeddings,
                                             config.dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.nhead,
                dim_feedforward=config.d_feedforward, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True),
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model))
        self.pooling     = TemporalPooling(config.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_embed_dim))
        self.recon_head  = nn.Linear(config.d_model, config.pose_dim)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def encode(self, x, mask_frame=None, return_sequence=False):
        """Run the Transformer backbone and temporal pooling. Optionally
        also returns the per-frame hidden-state sequence (prior to
        pooling), e.g. for reconstruction or other sequence-level use."""
        x   = self.pos_enc(self.input_proj(x))
        seq = self.transformer(x, src_key_padding_mask=(
            ~mask_frame if mask_frame is not None else None))
        emb = self.output_proj(self.pooling(seq, mask_frame))
        return (emb, seq) if return_sequence else emb

    def forward(self, x, mask_frame=None):
        return self.encode(x, mask_frame)

    def reconstruct(self, x, mask_frame=None):
        """Decode a per-frame pose reconstruction from the encoder's
        hidden-state sequence."""
        _, seq = self.encode(x, mask_frame, return_sequence=True)
        return self.recon_head(seq)


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (with BatchNorm) mapping encoder
    embeddings into the lower-dimensional space used for the contrastive
    loss, following standard MoCo/SimCLR-style projection-head design."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim))

    def forward(self, x): return self.net(x)


# ==================== MoCo ====================

class MoCoContrastive(nn.Module):
    """Momentum Contrast (MoCo) wrapper around a query/key encoder pair.

    The key encoder and its projection head are momentum-updated copies
    of the query network (no gradients flow into them directly). A fixed-
    size queue of past key embeddings is maintained as the negative set
    for the InfoNCE loss, and is updated after every forward pass.
    """

    def __init__(self, encoder, proj, config):
        super().__init__()
        self.cfg       = config
        self.encoder_q = encoder
        self.proj_q    = proj
        self.encoder_k = copy.deepcopy(encoder)
        self.proj_k    = copy.deepcopy(proj)
        for p in (*self.encoder_k.parameters(), *self.proj_k.parameters()):
            p.requires_grad_(False)
        self.register_buffer("queue",
            F.normalize(torch.randn(config.proj_dim, config.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """Exponential moving average update of the key encoder/projection
        parameters from the corresponding query-network parameters."""
        m = self.cfg.momentum
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=1.0 - m)
        for pq, pk in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=1.0 - m)

    @torch.no_grad()
    def _enqueue(self, keys):
        """Insert a batch of key embeddings into the circular negative
        queue, wrapping around when the batch crosses the queue boundary."""
        B, ptr = keys.shape[0], int(self.queue_ptr)
        end    = ptr + B
        if end <= self.cfg.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            over = end - self.cfg.queue_size
            self.queue[:, ptr:] = keys[:B-over].T
            self.queue[:, :over] = keys[B-over:].T
        self.queue_ptr[0] = end % self.cfg.queue_size

    def forward(self, Xq, Mfq, Xk, Mfk):
        """Compute the InfoNCE contrastive loss between a query view and a
        key view of the same underlying sequence, using the maintained
        negative queue, and return the loss along with the query
        embedding (pre-projection) for potential downstream use."""
        eq = self.encoder_q(Xq, Mfq)
        zq = F.normalize(self.proj_q(eq), dim=1)
        with torch.no_grad():
            self._momentum_update()
            zk = F.normalize(self.proj_k(self.encoder_k(Xk, Mfk)), dim=1)
        logits = torch.cat([(zq * zk).sum(1, keepdim=True),
                             zq @ self.queue.clone().detach()], dim=1)
        logits /= self.cfg.temperature
        loss = F.cross_entropy(logits,
                               torch.zeros(logits.shape[0], dtype=torch.long,
                                           device=logits.device))
        self._enqueue(zk)
        return loss, eq


# ==================== Trainer ====================

class Trainer:
    """Owns the full training lifecycle: model/optimizer construction,
    optional checkpoint resumption, data-loader and scheduler setup, the
    per-epoch train/validate loop, checkpointing, and early stopping."""

    def __init__(self, config):
        self.cfg = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu")

        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_fh = open(os.path.join(config.log_dir, f"train_{ts}.log"), "w")
        self._log(f"Device: {self.device}")

        backbone  = PoseEncoder(config).to(self.device)
        proj_head = ProjectionHead(config.output_embed_dim,
                                   config.proj_dim).to(self.device)
        self.moco = MoCoContrastive(backbone, proj_head, config).to(self.device)

        self._log(f"Backbone parameters: "
                  f"{sum(p.numel() for p in backbone.parameters() if p.requires_grad):,}")
        self._log(f"Queue size: {config.queue_size} negatives/step")

        self.optimizer = optim.AdamW(
            list(self.moco.encoder_q.parameters()) +
            list(self.moco.proj_q.parameters()),
            lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = (GradScaler()
                       if config.mixed_precision and torch.cuda.is_available()
                       else None)

        # Training state below is overwritten by _load_checkpoint() when
        # resuming from a previous run.
        self.start_epoch      = 1
        self.global_step      = 0
        self.best_val_loss    = float("inf")
        self.patience_counter = 0
        self._scheduler_state = None   # populated by _load_checkpoint
        self.train_loader = self.val_loader = self.scheduler = None

    # ------------------------------------------------------------------ #
    def _log(self, msg):
        """Write a line to both the run's log file and stdout."""
        self.log_fh.write(msg + "\n")
        self.log_fh.flush()
        print(msg)

    # ================================================================== #
    # Resume logic
    # ================================================================== #

    def _resolve_checkpoint_path(self) -> Optional[str]:
        """Resolve ``cfg.resume_from_checkpoint`` into a concrete file
        path, handling the ``None`` / ``"latest"`` / explicit-path cases."""
        r = self.cfg.resume_from_checkpoint
        if r is None:
            return None
        if r == "latest":
            p = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"resume_from_checkpoint='latest' but {p} not found.")
            return p
        if not os.path.isfile(r):
            raise FileNotFoundError(
                f"Checkpoint not found: {r}")
        return r

    def _warm_queue(self, n_batches: int = 32):
            """Refill the MoCo queue after resuming so val loss is not corrupted."""
            self._log(f"Warming queue with {n_batches} batches...")
            self.moco.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.train_loader):
                    if i >= n_batches:
                        break
                    Xk  = batch["Xq"].to(self.device)   # use either view
                    Mfk = batch["M_frame_q"].to(self.device)
                    with torch.autocast(device_type=self.device.type,
                                        enabled=self.cfg.mixed_precision):
                        ek = self.moco.encoder_k(Xk, Mfk)
                        zk = F.normalize(self.moco.proj_k(ek), dim=1)
                    self.moco._enqueue(zk)
            self._log("Queue warmed.")
            self.moco.train()

    def _load_checkpoint(self, path: str):
        """Restore all training state from a checkpoint file. Must be
        called before setup_data(), since the scheduler is only
        constructed there and its state is applied afterward."""
        self._log(f"=" * 60)
        self._log(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Model weights
        self.moco.encoder_q.load_state_dict(ckpt["encoder_q_state"])
        self.moco.proj_q.load_state_dict(ckpt["proj_q_state"])
        self.moco.encoder_k.load_state_dict(ckpt["encoder_k_state"])
        self.moco.proj_k.load_state_dict(ckpt["proj_k_state"])

        # MoCo queue
        self.moco.queue.copy_(ckpt["queue"])
        if "queue_ptr" in ckpt:
            self.moco.queue_ptr.copy_(ckpt["queue_ptr"])

        # Optimizer & scaler
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scaler and "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])

        # Scheduler state is saved here; applied after setup_data() builds it
        self._scheduler_state = ckpt.get("scheduler_state_dict", None)

        # Training bookkeeping
        saved_epoch           = int(ckpt["epoch"])
        self.start_epoch      = saved_epoch + 1
        self.global_step      = ckpt.get("global_step", 0)
        self.best_val_loss    = ckpt.get("best_val_loss", float("inf"))
        self.patience_counter = ckpt.get("patience_counter", 0)

        self._log(
            f"Checkpoint loaded  |  saved_epoch={saved_epoch}  "
            f"next_epoch={self.start_epoch}  "
            f"global_step={self.global_step}  "
            f"best_val_loss={self.best_val_loss:.4f}  "
            f"patience={self.patience_counter}/{self.cfg.patience}"
        )

    # ================================================================== #
    # Data setup
    # ================================================================== #

    def setup_data(self):
        """Construct the train/validation datasets and loaders, and build
        the learning-rate scheduler (a linear warmup followed by cosine
        decay over the full training schedule)."""
        self._log("=" * 60)
        self._log("Setting up data...")

        full_aug = ParagraphDataset(self.cfg.preprocessed_dir, augment=True,
                                    max_seq_len=self.cfg.max_position_embeddings,
                                    config=self.cfg)
        full_noa = ParagraphDataset(self.cfg.preprocessed_dir, augment=False,
                                    max_seq_len=self.cfg.max_position_embeddings,
                                    config=self.cfg)

        n, n_train = len(full_aug), int(0.9 * len(full_aug))
        idxs       = list(range(n))
        random.shuffle(idxs)
        train_ds = Subset(full_aug, idxs[:n_train])
        val_ds   = Subset(full_noa, idxs[n_train:])
        self._log(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

        kw = dict(batch_size=self.cfg.batch_size,
                  num_workers=self.cfg.num_workers,
                  collate_fn=collate_fn,
                  pin_memory=self.cfg.pin_memory,
                  persistent_workers=self.cfg.num_workers > 0)
        self.train_loader = DataLoader(train_ds, shuffle=True,  **kw)
        self.val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

        total_steps  = len(self.train_loader) * self.cfg.max_epochs
        warmup_steps = len(self.train_loader) * self.cfg.warmup_epochs

        def lr_lambda(step):
            """Linear warmup for `warmup_steps`, then cosine decay to 0
            over the remainder of training."""
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Restore scheduler state now that it exists
        if self._scheduler_state is not None:
            self.scheduler.load_state_dict(self._scheduler_state)
            self._log("Scheduler state restored.")

    # ================================================================== #
    # Loss
    # ================================================================== #

    def _compute_loss(self, batch):
        """Compute the combined training objective for a batch: the MoCo
        contrastive loss over the augmented query/key views, plus the
        auxiliary masked reconstruction loss on the un-augmented view."""
        X, M, M_frame = (batch["X"].to(self.device),
                         batch["M"].to(self.device),
                         batch["M_frame"].to(self.device))
        total, d = torch.tensor(0.0, device=self.device), {}

        if self.cfg.use_contrastive and "Xq" in batch:
            cl, _ = self.moco(batch["Xq"].to(self.device),
                              batch["M_frame_q"].to(self.device),
                              batch["Xk"].to(self.device),
                              batch["M_frame_k"].to(self.device))
            total += self.cfg.contrastive_weight * cl
            d["contrastive"] = cl.item()

        if self.cfg.use_reconstruction:
            recon = self.moco.encoder_q.reconstruct(X, M_frame)
            rl    = F.mse_loss(recon[M], X[M]) if M.any() else F.mse_loss(recon, X)
            total += self.cfg.reconstruction_weight * rl
            d["reconstruction"] = rl.item()

        d["total"] = total.item()
        return total, d

    # ================================================================== #
    # Train / validate
    # ================================================================== #

    def train_epoch(self, epoch):
        """Run a single training epoch: forward/backward pass with mixed
        precision, gradient clipping, optimizer/scheduler steps, and
        periodic logging of running losses."""
        self.moco.train()
        acc  = dict(total=0.0, cont=0.0, recon=0.0, n=0)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        for bi, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type,
                                enabled=self.cfg.mixed_precision):
                loss, ld = self._compute_loss(batch)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.moco.encoder_q.parameters()) +
                    list(self.moco.proj_q.parameters()), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.moco.encoder_q.parameters()) +
                    list(self.moco.proj_q.parameters()), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            for k in ("total", "cont", "recon"):
                acc[k] += ld.get({"total":"total","cont":"contrastive",
                                   "recon":"reconstruction"}[k], 0.0)
            acc["n"] += 1

            lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{ld['total']:.4f}", lr=f"{lr:.2e}")

            if bi % self.cfg.log_every == 0:
                self._log(f"Epoch {epoch} | Step {bi} | "
                          f"loss={ld['total']:.4f}  "
                          f"cont={ld.get('contrastive',0):.4f}  "
                          f"recon={ld.get('reconstruction',0):.4f}  "
                          f"lr={lr:.2e}")

        n = max(acc["n"], 1)
        return {k: acc[k]/n for k in ("total", "cont", "recon")}

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate on the held-out validation split.

        Before scoring, the negative queue is rebuilt from scratch using
        the eval-mode key encoder so that validation query embeddings
        (computed in eval-mode BatchNorm) and the queue's key embeddings
        live in the same representation space as one another, avoiding a
        train/eval BatchNorm mismatch in the reported contrastive loss.
        """
        self.moco.eval()

        # Re-fill queue with eval-mode encoder_k projections so that
        # val zq (eval BN) and queue keys (eval BN) are in the same space.
        tmp_queue = torch.zeros_like(self.moco.queue)  # [128, 4096]
        ptr = 0
        for batch in self.train_loader:
            if ptr >= self.cfg.queue_size:
                break
            Xk  = batch["X"].to(self.device)
            Mfk = batch["M_frame"].to(self.device)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.cfg.mixed_precision):
                zk = F.normalize(self.moco.proj_k(self.moco.encoder_k(Xk, Mfk)), dim=1)
            B = zk.shape[0]
            end = min(ptr + B, self.cfg.queue_size)
            tmp_queue[:, ptr:end] = zk[:end-ptr].T
            ptr = end
        # Normalise any unfilled slots
        tmp_queue = F.normalize(tmp_queue, dim=0)

        acc = dict(cont=0.0, recon=0.0, n=0)
        for batch in self.val_loader:
            X, M, Mf = (batch["X"].to(self.device),
                        batch["M"].to(self.device),
                        batch["M_frame"].to(self.device))
            with torch.autocast(device_type=self.device.type,
                                enabled=self.cfg.mixed_precision):
                zq = F.normalize(self.moco.proj_q(self.moco.encoder_q(X, Mf)), dim=1)
                zk = F.normalize(self.moco.proj_k(self.moco.encoder_k(X, Mf)), dim=1)
                logits = torch.cat([(zq*zk).sum(1, keepdim=True),
                                    zq @ tmp_queue.detach()], dim=1)
                logits /= self.cfg.temperature
                cl = F.cross_entropy(logits,
                                    torch.zeros(logits.shape[0], dtype=torch.long,
                                                device=self.device))
                recon = self.moco.encoder_q.reconstruct(X, Mf)
                rl    = F.mse_loss(recon[M], X[M]) if M.any() else F.mse_loss(recon, X)

            acc["cont"]  += cl.item()
            acc["recon"] += rl.item()
            acc["n"]     += 1

        n = max(acc["n"], 1)
        return {k: acc[k]/n for k in ("cont", "recon")}

    # ================================================================== #
    # Checkpoint saving
    # ================================================================== #

    def save_checkpoint(self, epoch, is_best=False):
        """Save full training state (model, optimizer, scheduler, MoCo
        queue, and bookkeeping) to disk. Always updates ``latest.pt``;
        additionally writes ``best_model.pt`` / ``best_backbone.pt`` when
        ``is_best`` is set, and prunes old epoch checkpoints beyond the
        configured retention limit."""
        ckpt = {
            "epoch"               : epoch,
            "global_step"         : self.global_step,
            "encoder_q_state"     : self.moco.encoder_q.state_dict(),
            "proj_q_state"        : self.moco.proj_q.state_dict(),
            "encoder_k_state"     : self.moco.encoder_k.state_dict(),
            "proj_k_state"        : self.moco.proj_k.state_dict(),
            "queue"               : self.moco.queue,
            "queue_ptr"           : self.moco.queue_ptr,        # needed for resume
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss"       : self.best_val_loss,
            "patience_counter"    : self.patience_counter,      # needed for resume
        }
        if self.scaler:
            ckpt["scaler_state"] = self.scaler.state_dict()

        path = os.path.join(self.cfg.checkpoint_dir,
                            f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(ckpt, path)
        torch.save(ckpt, os.path.join(self.cfg.checkpoint_dir, "latest.pt"))

        if is_best:
            torch.save(ckpt, os.path.join(self.cfg.checkpoint_dir, "best_model.pt"))
            torch.save(self.moco.encoder_q.state_dict(),
                       os.path.join(self.cfg.checkpoint_dir, "best_backbone.pt"))
            self._log(f"Best model saved  (val_cont={self.best_val_loss:.4f})")

        all_ckpts = sorted(glob.glob(
            os.path.join(self.cfg.checkpoint_dir, "checkpoint_epoch_*.pt")))
        for old in all_ckpts[:-self.cfg.keep_last_n_checkpoints]:
            os.remove(old)

    # ================================================================== #
    # Main loop
    # ================================================================== #

    def train(self):
        """Top-level entry point: optionally resumes from a checkpoint,
        builds the data pipeline, then runs the epoch loop with
        validation, checkpointing, and early stopping."""
        try:
            # 1. Optional resume — must run before setup_data() so that
            #    _scheduler_state is available when the scheduler is built.
            ckpt_path = self._resolve_checkpoint_path()
            if ckpt_path:
                self._load_checkpoint(ckpt_path)

            # 2. Build data loaders and scheduler (restores scheduler state
            #    from checkpoint if present).
            self.setup_data()
            if ckpt_path:
                self._warm_queue(n_batches=256)  # 256 x 16 = 4096 -> full queue replacement

            self._log("=" * 60)
            if ckpt_path:
                self._log(f"Resuming from epoch {self.start_epoch}  "
                          f"(best so far: {self.best_val_loss:.4f}  "
                          f"patience: {self.patience_counter}/{self.cfg.patience})")
            else:
                self._log("Starting training  (v2 - MoCo + stronger augmentations)")

            for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
                train_m = self.train_epoch(epoch)

                if epoch % self.cfg.validate_every == 0:
                    val_m = self.validate(epoch)
                    self._log(
                        f"\nEpoch {epoch:3d} | "
                        f"train total={train_m['total']:.4f}  "
                        f"cont={train_m['cont']:.4f}  "
                        f"recon={train_m['recon']:.4f} | "
                        f"val cont={val_m['cont']:.4f}  "
                        f"recon={val_m['recon']:.4f}"
                    )
                    if val_m["cont"] < self.best_val_loss:
                        self.best_val_loss    = val_m["cont"]
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, is_best=True)
                    else:
                        self.patience_counter += 1
                        self._log(f"  patience {self.patience_counter}"
                                  f"/{self.cfg.patience}")
                else:
                    self._log(f"Epoch {epoch} | train_loss={train_m['total']:.4f}")

                if epoch % self.cfg.save_every == 0:
                    self.save_checkpoint(epoch)

                if self.patience_counter >= self.cfg.patience:
                    self._log(f"Early stopping at epoch {epoch}")
                    break

            self._log("=" * 60)
            self._log("Training complete!")
        finally:
            self.log_fh.close()


# ==================== Downstream helper ====================

def load_backbone(checkpoint_path, config=None, device="cuda"):
    """Load a trained backbone (``best_backbone.pt``) for downstream use,
    e.g. as a frozen feature extractor in topic-modeling experiments."""
    cfg_ = config or Config()
    enc  = PoseEncoder(cfg_).to(device)
    enc.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                   weights_only=True))
    enc.eval()
    return enc


# ==================== Main ====================

def main():
    

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
# finetune_coreference_from_translation.py
#
# Sign Language Coreference Resolution — fine-tuned from translation encoder
#
# Given two sign language video clips (uid_A, uid_B), predicts whether
# they are coreferential (refer to same entity via spatial loci/pointing/role shift)
#
# Architecture:
#   Shared Encoder (pretrained from translation):
#     FeatureProjection + TemporalPE + BertEncoder
#   Applied independently to uid_A and uid_B → feat_A, feat_B
#   Fusion: [a, b, |a-b|, a*b] → classification head → YES/NO
#
# USAGE:
#   python finetune_coreference_from_translation.py \
#     --translation_ckpt /path/to/best_model.pth \
#     --train_csv /path/to/train.csv \
#     --val_csv   /path/to/val.csv   \
#     --test_csv  /path/to/test.csv  \
#     --shard_dir /path/to/coreference_shards \
#     --freeze_epochs 10 --lr 1e-4 >> coref_run1.txt 2>&1

import os
import math
import glob
import csv
import json
import random
import logging
import time
import argparse
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import webdataset as wds

from transformers import BertConfig, BertModel

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)
from sklearn.utils import resample

# ── Label map ─────────────────────────────────────────────────────────────────

IDX_TO_LABEL = {0: "no_coref", 1: "coref"}
NUM_CLASSES  = 2

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Exact copies from translation script ─────────────────────────────────────

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )
    def forward(self, x):
        return self.net(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=80):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── Shared encoder ────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    def __init__(self,
                 feature_projection: FeatureProjection,
                 temporal_pe:        TemporalPositionalEncoding,
                 encoder:            BertModel):
        super().__init__()
        self.feature_projection = feature_projection
        self.temporal_pe        = temporal_pe
        self.encoder            = encoder

    def forward(self, x, mask):
        proj    = self.feature_projection(x)
        proj    = self.temporal_pe(proj)
        out     = self.encoder(
            inputs_embeds  = proj,
            attention_mask = mask.long(),
        )
        hidden  = out.last_hidden_state          # (B, T, 512)
        mask_3d = mask.unsqueeze(-1)
        feat    = (hidden * mask_3d).sum(1) / mask_3d.sum(1).clamp(min=1)
        return feat                              # (B, 512)


# ── Coreference model ─────────────────────────────────────────────────────────

class CoreferenceModel(nn.Module):
    """
    Shared encoder applied to both uid_A and uid_B.
    Fusion: [a, b, |a-b|, a*b] → binary classifier.
    """
    def __init__(self,
                 shared_encoder: SharedEncoder,
                 d_model:        int   = 512,
                 num_classes:    int   = 2,
                 dropout:        float = 0.1):
        super().__init__()
        self.encoder    = shared_encoder
        fusion_dim      = d_model * 4

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_a, mask_a, x_b, mask_b):
        a     = self.encoder(x_a, mask_a)                              # (B, 512)
        b     = self.encoder(x_b, mask_b)                              # (B, 512)
        fused = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)    # (B, 2048)
        return self.classifier(fused)


# ── Load pretrained components ────────────────────────────────────────────────

def load_pretrained_components(ckpt_path: str, device: torch.device):
    print(f"Loading translation checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    feat_proj = FeatureProjection(input_dim=152, output_dim=512)
    feat_proj.load_state_dict(ckpt["proj_state_dict"])
    print("  ✓ feature_projection loaded")

    temp_pe = TemporalPositionalEncoding(d_model=512, max_len=80)
    temp_pe.load_state_dict(ckpt["temporal_pe_state_dict"])
    print("  ✓ temporal_pe loaded")

    encoder = BertModel(BertConfig(
        hidden_size                  = 512,
        num_hidden_layers            = 4,
        num_attention_heads          = 8,
        hidden_dropout_prob          = 0.15,
        attention_probs_dropout_prob = 0.15,
    ))
    full_sd    = ckpt["model_state_dict"]
    encoder_sd = {
        k[len("encoder."):]: v
        for k, v in full_sd.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_sd, strict=True)
    print(f"  ✓ encoder loaded ({len(encoder_sd)} keys)")

    return SharedEncoder(
        feat_proj.to(device),
        temp_pe.to(device),
        encoder.to(device),
    )


# ── In-memory shard loader ────────────────────────────────────────────────────

def load_shards_to_memory(base_dir: str, max_frames: int, desc: str) -> dict:
    """
    Loads shards from train/ val/ test/ subdirectories into one dict.
    Deduplicates by uid — same video appearing in multiple splits loaded once.
    Both uid_A and uid_B are in the same shard dir for coreference.
    """
    store = {}
    for split in ["train", "val", "test"]:
        split_dir   = os.path.join(base_dir, split)
        shard_files = sorted(glob.glob(os.path.join(split_dir, "shard-*.tar")))
        if not shard_files:
            print(f"  Warning: no shards in {split_dir}, skipping")
            continue

        dataset = wds.WebDataset(
            shard_files,
            shardshuffle=False,
            handler=wds.warn_and_continue
        ).decode()

        for sample in tqdm(dataset, desc=f"Loading {desc}/{split}"):
            uid = sample["__key__"]
            if uid in store:
                continue

            feats = sample["data.npy"]
            T, D  = feats.shape

            if T > max_frames:
                indices = np.linspace(0, T - 1, max_frames).astype(int)
                feats   = feats[indices]
                T       = max_frames

            mask = np.ones(T, dtype=np.float32)
            if T < max_frames:
                pad   = max_frames - T
                feats = np.vstack([feats, np.zeros((pad, D), dtype=np.float32)])
                mask  = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])

            store[uid] = {
                "x":    torch.tensor(feats, dtype=torch.float32),
                "mask": torch.tensor(mask,  dtype=torch.float32),
            }

    print(f"  Loaded {len(store)} unique {desc} samples into memory")
    return store


# ── Dataset ───────────────────────────────────────────────────────────────────

class CoreferenceDataset(Dataset):
    """
    Each CSV row is one pair: (uid_A, uid_B, has_coreference).
    Both UIDs looked up from the same in-memory store.
    Rows where either UID is missing from shards are skipped.
    """
    def __init__(self, csv_path: str, store: dict):
        df = pd.read_csv(csv_path)
        df["has_coreference"] = df["has_coreference"].astype(int)

        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            uid_a = str(row["uid_A"])
            uid_b = str(row["uid_B"])
            if uid_a not in store or uid_b not in store:
                skipped += 1
                continue
            self.samples.append((uid_a, uid_b, int(row["has_coreference"])))

        if skipped:
            print(f"  Skipped {skipped} rows — UID not found in shards")
        print(f"  Dataset size: {len(self.samples)} pairs")

        self.store = store

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid_a, uid_b, label = self.samples[idx]
        a = self.store[uid_a]
        b = self.store[uid_b]
        return {
            "x_a":    a["x"],
            "mask_a": a["mask"],
            "x_b":    b["x"],
            "mask_b": b["mask"],
            "label":  torch.tensor(label, dtype=torch.long),
        }


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):          return self.shadow.state_dict()
    def load_state_dict(self, sd): self.shadow.load_state_dict(sd)


# ── LR schedule ───────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Metrics ───────────────────────────────────────────────────────────────────

def bootstrap_ci(labels, preds, metric_fn, n=1000, ci=0.95):
    scores = []
    for _ in range(n):
        idx = resample(range(len(labels)), replace=True)
        try:
            scores.append(metric_fn(labels[idx], preds[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    a = (1 - ci) / 2
    return (float(np.percentile(scores, 100 * a)),
            float(np.percentile(scores, 100 * (1 - a))))


@torch.no_grad()
def evaluate(model, loader, device, run_dir=None,
             split="val", epoch=None, logger=None, bootstrap=False):
    model.eval()
    preds_list, labels_list = [], []

    for b in loader:
        xa   = b["x_a"].to(device)
        ma   = b["mask_a"].to(device)
        xb   = b["x_b"].to(device)
        mb   = b["mask_b"].to(device)
        y    = b["label"].to(device)
        out  = model(xa, ma, xb, mb).argmax(-1)
        preds_list.append(out.cpu().numpy())
        labels_list.append(y.cpu().numpy())

    preds  = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro",    zero_division=0)
    wt_f1    = f1_score(labels, preds, average="weighted", zero_division=0)
    bal_acc  = balanced_accuracy_score(labels, preds)
    kappa    = cohen_kappa_score(labels, preds)
    mcc      = matthews_corrcoef(labels, preds)
    report   = classification_report(
        labels, preds,
        target_names=["no_coref", "coref"],
        digits=4, zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    result = dict(
        accuracy=acc, macro_f1=macro_f1, weighted_f1=wt_f1,
        balanced_accuracy=bal_acc, kappa=kappa, mcc=mcc,
        report=report, cm=cm, preds=preds, labels=labels
    )

    if bootstrap:
        result["acc_ci"] = bootstrap_ci(labels, preds, accuracy_score)
        result["f1_ci"]  = bootstrap_ci(
            labels, preds,
            lambda l, p: f1_score(l, p, average="macro", zero_division=0)
        )

    if run_dir:
        tag = f"_e{epoch:03d}" if epoch else ""
        with open(os.path.join(run_dir, f"{split}_report{tag}.txt"), "w") as f:
            f.write(report)
            if bootstrap and "acc_ci" in result:
                f.write(f"\nAcc 95% CI : {result['acc_ci']}")
                f.write(f"\nF1  95% CI : {result['f1_ci']}")

        pd.DataFrame(
            cm,
            index   = ["True_no_coref",  "True_coref"],
            columns = ["Pred_no_coref",  "Pred_coref"],
        ).to_csv(os.path.join(run_dir, f"{split}_cm{tag}.csv"))

    if logger:
        logger.info(
            f"[{split.upper()}] Acc={acc:.4f}  MacroF1={macro_f1:.4f}  "
            f"MCC={mcc:.4f}  Kappa={kappa:.4f}"
        )
        logger.debug(f"\n{report}")
        if bootstrap and "acc_ci" in result:
            logger.info(
                f"[{split.upper()}] Acc CI={result['acc_ci']}  "
                f"F1 CI={result['f1_ci']}"
            )

    return result


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = -float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = os.path.join(args.log_dir,  run_name)
    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(run_dir, f"{run_name}.log")),
        ]
    )
    logger = logging.getLogger(run_name)
    logger.info(f"Run: {run_name}  |  Task: Sign Language Coreference Resolution")
    logger.info(f"Args: {vars(args)}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load all shards into memory (single store for both uid_A and uid_B) ───
    logger.info("Loading shards into memory...")
    store = load_shards_to_memory(args.shard_dir, args.max_frames, "coref")

    # ── Datasets and loaders ──────────────────────────────────────────────────
    train_ds = CoreferenceDataset(args.train_csv, store)
    val_ds   = CoreferenceDataset(args.val_csv,   store)
    test_ds  = CoreferenceDataset(args.test_csv,  store)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    logger.info(
        f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}"
    )

    # ── Load pretrained shared encoder ────────────────────────────────────────
    shared_encoder = load_pretrained_components(args.translation_ckpt, device)

    model = CoreferenceModel(
        shared_encoder = shared_encoder,
        d_model        = 512,
        num_classes    = NUM_CLASSES,
        dropout        = 0.1,
    ).to(device)

    # ── Freeze/unfreeze helpers ───────────────────────────────────────────────
    def set_encoder_grad(requires_grad: bool):
        for p in model.encoder.feature_projection.parameters(): p.requires_grad_(requires_grad)
        for p in model.encoder.temporal_pe.parameters():        p.requires_grad_(requires_grad)
        for p in model.encoder.encoder.parameters():            p.requires_grad_(requires_grad)

    set_encoder_grad(False)
    logger.info(f"Encoder FROZEN for first {args.freeze_epochs} epochs — head only")

    ema = EMA(model, decay=args.ema_decay)
    es  = EarlyStopping(patience=args.patience)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = int(total_steps * args.warmup_frac)
    logger.info(
        f"Steps/epoch: {steps_per_epoch}  |  "
        f"Total: {total_steps}  |  Warmup: {warmup_steps}"
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def make_optimizer(encoder_unfrozen=False):
        if not encoder_unfrozen:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay
            )
        return torch.optim.AdamW([
            {"params": model.classifier.parameters(),
             "lr": args.lr},
            {"params": model.encoder.feature_projection.parameters(),
             "lr": args.lr * 0.1},
            {"params": model.encoder.temporal_pe.parameters(),
             "lr": args.lr * 0.1},
            {"params": model.encoder.encoder.parameters(),
             "lr": args.lr * 0.1},
        ], weight_decay=args.weight_decay)

    opt       = make_optimizer(encoder_unfrozen=False)
    scheduler = build_scheduler(opt, total_steps, warmup_steps)
    use_amp   = device.type == "cuda"
    scaler    = GradScaler(device.type) if use_amp else None

    best_val_f1 = -1.0
    global_step = 0
    best_path   = os.path.join(ckpt_dir, "best.pt")
    last_path   = os.path.join(ckpt_dir, "last.pt")
    csv_path    = os.path.join(run_dir,  "epoch_metrics.csv")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        if epoch == args.freeze_epochs + 1:
            set_encoder_grad(True)
            remaining_steps = (args.epochs - epoch) * steps_per_epoch
            opt       = make_optimizer(encoder_unfrozen=True)
            scheduler = build_scheduler(opt, remaining_steps, warmup_steps=0)
            logger.info(
                f"Epoch {epoch}: Encoder UNFROZEN — "
                f"head lr={args.lr}, encoder lr={args.lr * 0.1}"
            )

        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        steps      = 0

        for b in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                      total=steps_per_epoch, leave=False):
            xa = b["x_a"].to(device)
            ma = b["mask_a"].to(device)
            xb = b["x_b"].to(device)
            mb = b["mask_b"].to(device)
            y  = b["label"].to(device)

            opt.zero_grad()
            if use_amp:
                with autocast("cuda"):
                    out  = model(xa, ma, xb, mb)
                    loss = loss_fn(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                out  = model(xa, ma, xb, mb)
                loss = loss_fn(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            scheduler.step()
            ema.update(model)
            total_loss  += loss.item()
            correct     += (out.argmax(-1) == y).sum().item()
            total       += y.size(0)
            global_step += 1
            steps       += 1

        train_acc  = correct / max(total, 1)
        train_loss = total_loss / max(steps, 1)
        cur_lr     = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:03d} | loss={train_loss:.4f}  "
            f"acc={train_acc:.4f}  lr={cur_lr:.2e}"
        )

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate(
            ema.shadow, val_loader, device,
            run_dir=run_dir, split="val", epoch=epoch, logger=logger
        )
        val_f1 = val_metrics["macro_f1"]

        # CSV log
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc",
                "val_acc", "val_macro_f1", "val_mcc", "val_kappa", "lr"
            ])
            if write_header: w.writeheader()
            w.writerow(dict(
                epoch        = epoch,
                train_loss   = round(train_loss, 6),
                train_acc    = round(train_acc,  4),
                val_acc      = round(val_metrics["accuracy"], 4),
                val_macro_f1 = round(val_f1, 4),
                val_mcc      = round(val_metrics["mcc"], 4),
                val_kappa    = round(val_metrics["kappa"], 4),
                lr           = round(cur_lr, 8),
            ))

        # Last checkpoint
        torch.save(dict(
            model       = model.state_dict(),
            ema         = ema.state_dict(),
            optimizer   = opt.state_dict(),
            scheduler   = scheduler.state_dict(),
            epoch       = epoch,
            global_step = global_step,
            best_val_f1 = best_val_f1,
        ), last_path)

        # Best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(dict(
                model       = model.state_dict(),
                ema         = ema.state_dict(),
                epoch       = epoch,
                global_step = global_step,
                best_val_f1 = best_val_f1,
            ), best_path)
            logger.info(f"New best val macro-F1={best_val_f1:.4f} → best.pt saved")

            eval_model = deepcopy(ema.shadow).to(device)
            evaluate(eval_model, test_loader, device,
                     run_dir=run_dir, split="test",
                     epoch=epoch, logger=logger, bootstrap=True)
            del eval_model

        if es.step(val_f1):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Final test on best checkpoint ─────────────────────────────────────────
    logger.info("=== Final test on best checkpoint ===")
    best_ckpt      = torch.load(best_path, map_location=device, weights_only=False)
    shared_encoder2 = load_pretrained_components(args.translation_ckpt, device)
    final_model    = CoreferenceModel(
        shared_encoder2, d_model=512, num_classes=NUM_CLASSES
    ).to(device)
    final_model.load_state_dict(best_ckpt["ema"])
    final_model.eval()

    evaluate(final_model, test_loader, device,
             run_dir=run_dir, split="final_test",
             logger=logger, bootstrap=True)

    logger.info(f"Done. All outputs saved to: {run_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Sign Language Coreference Resolution — fine-tuned from translation encoder"
    )
    p.add_argument("--translation_ckpt", required=True)
    p.add_argument("--train_csv",        required=False
                   ,default =  "/data/dept_share/sanjeet/Khushi/coreference/datasets/train.csv")
    p.add_argument("--val_csv",          required=False,
                   default =  "/data/dept_share/sanjeet/Khushi/coreference/datasets/val.csv")
    p.add_argument("--test_csv",         required=False,
                   default =  "/data/dept_share/sanjeet/Khushi/coreference/datasets/test.csv")
    p.add_argument("--shard_dir",        required=False,
                   default = "/indian-slp/dattatreya/coreference/shards/coreference_shards",
                   help="Base dir containing train/ val/ test/ shard subdirs")
    p.add_argument("--freeze_epochs",    type=int,   default=10)
    p.add_argument("--batch_size",       type=int,   default=512)
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-2)
    p.add_argument("--max_frames",       type=int,   default=80)
    p.add_argument("--warmup_frac",      type=float, default=0.1)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--label_smoothing",  type=float, default=0.1)
    p.add_argument("--ema_decay",        type=float, default=0.9999)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--device",           default="cuda:3")
    p.add_argument("--ckpt_dir",         default="checkpoints_coref")
    p.add_argument("--log_dir",          default="logs_coref")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
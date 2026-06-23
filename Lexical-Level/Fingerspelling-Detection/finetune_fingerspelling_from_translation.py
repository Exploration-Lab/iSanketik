# finetune_from_translation.py
#
# Reuses FeatureProjection + TemporalPositionalEncoding + BertEncoder
# from the translation checkpoint for binary YES/NO classification.

import os, math, glob, csv, json, random, logging, time, argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.amp import autocast, GradScaler
import webdataset as wds

from transformers import BertConfig, BertModel
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from sklearn.utils import resample

# ── Label map ────────────────────────────────────────────────────────────────
LABEL_MAP    = {"NO": 0, "YES": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES  = 2

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ── Copy exact classes from translation script ────────────────────────────────

class FeatureProjection(nn.Module):
    """Identical to translation script — weights will be loaded directly."""
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
    """Identical to translation script — weights will be loaded directly."""
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


# ── Classifier that wraps the pretrained components ──────────────────────────

class TranslationEncoderClassifier(nn.Module):
    """
    Uses the pretrained FeatureProjection + TemporalPE + BertEncoder
    from the translation model. Adds a new classification head on top.

    Forward:
        x    : (B, T, 152)  raw pose features
        mask : (B, T)       1=real frame, 0=padding
    """
    def __init__(self,
                 feature_projection: FeatureProjection,
                 temporal_pe:        TemporalPositionalEncoding,
                 encoder:            BertModel,
                 d_model:            int = 512,
                 num_classes:        int = 2,
                 dropout:            float = 0.1):
        super().__init__()
        self.feature_projection = feature_projection
        self.temporal_pe        = temporal_pe
        self.encoder            = encoder

        # Fresh classification head — trained from scratch
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, mask, return_attn=False):
        # x: (B, T, 152) → project → (B, T, 512)
        proj = self.feature_projection(x)
        proj = self.temporal_pe(proj)

        # BertModel expects attention_mask: 1=attend, 0=ignore
        out  = self.encoder(
            inputs_embeds   = proj,
            attention_mask  = mask.long(),
        )

        # Masked mean pool over real frames
        hidden   = out.last_hidden_state            # (B, T, 512)
        mask_3d  = mask.unsqueeze(-1)               # (B, T, 1)
        feat     = (hidden * mask_3d).sum(1) / mask_3d.sum(1).clamp(min=1)

        return self.classifier(feat)


# ── Load pretrained components from translation checkpoint ───────────────────

def load_pretrained_components(ckpt_path: str, device: torch.device):
    """
    Extracts feature_projection, temporal_pe, and encoder weights
    from the translation checkpoint and returns ready-to-use modules.
    """
    print(f"Loading translation checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1. Feature projection
    feat_proj = FeatureProjection(input_dim=152, output_dim=512)
    feat_proj.load_state_dict(ckpt["proj_state_dict"])
    print("  ✓ feature_projection loaded")

    # 2. Temporal positional encoding
    temp_pe = TemporalPositionalEncoding(d_model=512, max_len=80)
    temp_pe.load_state_dict(ckpt["temporal_pe_state_dict"])
    print("  ✓ temporal_pe loaded")

    # 3. BERT encoder — extract encoder.* keys from EncoderDecoderModel state dict
    encoder = BertModel(BertConfig(
        hidden_size               = 512,
        num_hidden_layers         = 4,
        num_attention_heads       = 8,
        hidden_dropout_prob       = 0.15,
        attention_probs_dropout_prob = 0.15,
    ))

    full_sd     = ckpt["model_state_dict"]
    # EncoderDecoderModel stores encoder weights as "encoder.*"
    encoder_sd  = {
        k[len("encoder."):]: v
        for k, v in full_sd.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_sd, strict=True)
    print(f"  ✓ encoder loaded ({len(encoder_sd)} keys)")

    return (
        feat_proj.to(device),
        temp_pe.to(device),
        encoder.to(device),
    )


# ── EMA ──────────────────────────────────────────────────────────────────────

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

    def state_dict(self):        return self.shadow.state_dict()
    def load_state_dict(self, sd): self.shadow.load_state_dict(sd)


# ── LR schedule ──────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Data loader (same as original script) ────────────────────────────────────

def _decode_sample(sample, max_frames):
    feats = sample["data.npy"]
    label = sample["label.txt"].strip().upper()
    if label not in LABEL_MAP:
        return None

    y    = LABEL_MAP[label]
    T, D = feats.shape

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


def build_shard_loader(shard_dir, max_frames, batch_size,
                       num_workers, shuffle=False, shuffle_buffer=2000):
    shard_files  = sorted(glob.glob(os.path.join(shard_dir, "shard-*.tar")))
    if not shard_files:
        raise FileNotFoundError(f"No shards at {shard_dir}")

    size_file    = os.path.join(shard_dir, "size.txt")
    dataset_size = int(open(size_file).read().strip()) if os.path.exists(size_file) else None

    dataset = (
        wds.WebDataset(shard_files, shardshuffle=shuffle,
                       nodesplitter=wds.split_by_node)
        .decode()
        .map(lambda s: _decode_sample(s, max_frames))
        .select(lambda s: s is not None)
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batched(
        batch_size, partial=True,
        collation_fn=torch.utils.data.default_collate
    )
    loader = wds.WebLoader(dataset, batch_size=None,
                           num_workers=num_workers, pin_memory=True)
    return loader, dataset_size, len(shard_files)


# ── Metrics ───────────────────────────────────────────────────────────────────

def bootstrap_ci(labels, preds, metric_fn, n=1000, ci=0.95):
    scores = []
    for _ in range(n):
        idx = resample(range(len(labels)), replace=True)
        try: scores.append(metric_fn(labels[idx], preds[idx]))
        except: pass
    scores = np.array(scores)
    a = (1 - ci) / 2
    return float(np.percentile(scores, 100*a)), float(np.percentile(scores, 100*(1-a)))


@torch.no_grad()
def evaluate(model, loader, device, run_dir=None,
             split="val", epoch=None, logger=None, bootstrap=False):
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
    mcc      = matthews_corrcoef(labels, preds)
    kappa    = cohen_kappa_score(labels, preds)
    report   = classification_report(labels, preds,
                                     target_names=list(LABEL_MAP.keys()), digits=4)
    cm       = confusion_matrix(labels, preds)

    result = dict(accuracy=acc, macro_f1=macro_f1, mcc=mcc,
                  kappa=kappa, report=report, cm=cm,
                  preds=preds, labels=labels)

    if bootstrap:
        result["acc_ci"] = bootstrap_ci(labels, preds, accuracy_score)
        result["f1_ci"]  = bootstrap_ci(
            labels, preds, lambda l, p: f1_score(l, p, average="macro"))

    if run_dir:
        tag = f"_e{epoch:03d}" if epoch else ""
        with open(os.path.join(run_dir, f"{split}_report{tag}.txt"), "w") as f:
            f.write(report)
            if bootstrap and "acc_ci" in result:
                f.write(f"\nAcc 95% CI : {result['acc_ci']}")
                f.write(f"\nF1  95% CI : {result['f1_ci']}")
        pd.DataFrame(
            cm,
            index  =[f"True_{k}" for k in LABEL_MAP],
            columns=[f"Pred_{k}" for k in LABEL_MAP],
        ).to_csv(os.path.join(run_dir, f"{split}_cm{tag}.csv"))

    if logger:
        logger.info(f"[{split.upper()}] Acc={acc:.4f}  MacroF1={macro_f1:.4f}  "
                    f"MCC={mcc:.4f}  Kappa={kappa:.4f}")
        logger.debug(f"\n{report}")

    return result


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = -float("inf"); self.counter = 0; self.should_stop = False

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric; self.counter = 0
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
    logger.info(f"Run: {run_name}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load pretrained components ────────────────────────────────────────────
    feat_proj, temp_pe, encoder = load_pretrained_components(
        args.translation_ckpt, device
    )

    model = TranslationEncoderClassifier(
        feature_projection = feat_proj,
        temporal_pe        = temp_pe,
        encoder            = encoder,
        d_model            = 512,
        num_classes        = NUM_CLASSES,
        dropout            = 0.1,
    ).to(device)

    # ── Stage 1: freeze encoder, train head only ──────────────────────────────
    def set_encoder_grad(requires_grad: bool):
        for p in model.feature_projection.parameters(): p.requires_grad_(requires_grad)
        for p in model.temporal_pe.parameters():        p.requires_grad_(requires_grad)
        for p in model.encoder.parameters():            p.requires_grad_(requires_grad)

    set_encoder_grad(False)   # start frozen
    logger.info(f"Encoder FROZEN for first {args.freeze_epochs} epochs")

    ema = EMA(model, decay=args.ema_decay)
    es  = EarlyStopping(patience=args.patience)

    train_loader, train_size, _ = build_shard_loader(
        args.train_shards, args.max_frames, args.batch_size,
        args.num_workers, shuffle=True)
    val_loader,   val_size,   _ = build_shard_loader(
        args.val_shards,   args.max_frames, args.batch_size,
        args.num_workers, shuffle=False)
    test_loader,  test_size,  _ = build_shard_loader(
        args.test_shards,  args.max_frames, args.batch_size,
        args.num_workers, shuffle=False)

    steps_per_epoch = args.steps_per_epoch or math.ceil((train_size or 1) / args.batch_size)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = int(total_steps * args.warmup_frac)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def make_optimizer():
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

    opt       = make_optimizer()
    scheduler = build_scheduler(opt, total_steps, warmup_steps)
    use_amp   = device.type == "cuda"
    scaler    = GradScaler(device.type) if use_amp else None

    best_val_f1 = -1.0
    global_step = 0
    best_path   = os.path.join(ckpt_dir, "best.pt")
    last_path   = os.path.join(ckpt_dir, "last.pt")

    csv_path = os.path.join(run_dir, "epoch_metrics.csv")

    for epoch in range(1, args.epochs + 1):

        # ── Unfreeze encoder after freeze_epochs ──────────────────────────────
        if epoch == args.freeze_epochs + 1:
            set_encoder_grad(True)
            # Rebuild optimizer with all params + lower LR for pretrained parts
            opt = torch.optim.AdamW([
                {"params": model.classifier.parameters(),          "lr": args.lr},
                {"params": model.feature_projection.parameters(),  "lr": args.lr * 0.1},
                {"params": model.temporal_pe.parameters(),         "lr": args.lr * 0.1},
                {"params": model.encoder.parameters(),             "lr": args.lr * 0.1},
            ], weight_decay=args.weight_decay)
            scheduler = build_scheduler(opt, total_steps - epoch * steps_per_epoch,
                                        warmup_steps=0)
            logger.info(f"Epoch {epoch}: Encoder UNFROZEN — head lr={args.lr}, encoder lr={args.lr*0.1}")

        model.train()
        total_loss = 0.0; correct = 0; total = 0; steps = 0

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
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                out  = model(x, m)
                loss = loss_fn(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            scheduler.step()
            ema.update(model)
            total_loss += loss.item()
            correct    += (out.argmax(-1) == y).sum().item()
            total      += y.size(0)
            global_step += 1
            steps       += 1

        train_acc  = correct / total
        train_loss = total_loss / max(steps, 1)
        cur_lr     = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch:03d} | loss={train_loss:.4f}  acc={train_acc:.4f}  lr={cur_lr:.2e}")

        val_metrics = evaluate(ema.shadow, val_loader, device,
                               run_dir=run_dir, split="val",
                               epoch=epoch, logger=logger)
        val_f1 = val_metrics["macro_f1"]

        # CSV log
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc",
                "val_acc", "val_macro_f1", "val_mcc", "lr"])
            if write_header: w.writeheader()
            w.writerow(dict(epoch=epoch, train_loss=round(train_loss, 6),
                            train_acc=round(train_acc, 4),
                            val_acc=round(val_metrics["accuracy"], 4),
                            val_macro_f1=round(val_f1, 4),
                            val_mcc=round(val_metrics["mcc"], 4),
                            lr=round(cur_lr, 8)))

        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                        optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch, best_val_f1=best_val_f1), last_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                            epoch=epoch, best_val_f1=best_val_f1), best_path)
            logger.info(f"New best val macro-F1={best_val_f1:.4f} → best.pt saved")

            eval_model = deepcopy(ema.shadow).to(device)
            evaluate(eval_model, test_loader, device,
                     run_dir=run_dir, split="test",
                     epoch=epoch, logger=logger, bootstrap=True)
            del eval_model

        if es.step(val_f1):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Final test ────────────────────────────────────────────────────────────
    logger.info("=== Final test on best checkpoint ===")
    best_ckpt   = torch.load(best_path, map_location=device)
    feat_proj2, temp_pe2, encoder2 = load_pretrained_components(
        args.translation_ckpt, device)
    final_model = TranslationEncoderClassifier(
        feat_proj2, temp_pe2, encoder2, d_model=512,
        num_classes=NUM_CLASSES).to(device)
    final_model.load_state_dict(best_ckpt["ema"])
    evaluate(final_model, test_loader, device,
             run_dir=run_dir, split="final_test",
             logger=logger, bootstrap=True)
    logger.info(f"Done. Outputs: {run_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--translation_ckpt", required=True,
                   help="Path to best_model.pth from translation training")
    p.add_argument("--train_shards", default="/indian-slp/dattatreya/fingerspelled/train")
    p.add_argument("--val_shards",   default="/indian-slp/dattatreya/fingerspelled/val")
    p.add_argument("--test_shards",  default="/indian-slp/dattatreya/fingerspelled/test")
    p.add_argument("--steps_per_epoch", type=int, default=None)
    p.add_argument("--freeze_epochs",   type=int, default=10,
                   help="Epochs to train only the head before unfreezing encoder")
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--max_frames",      type=int,   default=80)
    p.add_argument("--warmup_frac",     type=float, default=0.1)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--ema_decay",       type=float, default=0.9999)
    p.add_argument("--patience",        type=int,   default=15)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--device",          default="cuda:3")
    p.add_argument("--ckpt_dir",        default="checkpoints_finetune")
    p.add_argument("--log_dir",         default="logs_finetune")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
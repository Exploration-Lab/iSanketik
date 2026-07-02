"""
Train the pose VAE (Phase 1).

Usage:
  python phase1.py --config configs/vae_config.yaml

or with explicit args:
  python phase1.py \
    --train_shard_dir <path_to_train_shards> \
    --val_shard_dir   <path_to_val_shards> \
    --train_csv <path_to_train_csv>
    --val_csv   <path_to_val_csv> \
    --output_dir ./checkpoints/vae
"""

import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# make the project root importable
sys.path.insert(0, str(Path(__file__).parent))

from helpers.data import get_dataloader
from helpers.pose_vae import PoseVAE, vae_loss


log = logging.getLogger(__name__)


# beta annealing schedule for the KL term

def get_beta(step: int, warmup_steps: int = 5000, beta_max: float = 1e-4) -> float:
    """Linear warmup of beta from ~0 to beta_max over warmup_steps.

    Keeping beta small early on lets the reconstruction term dominate
    while the encoder/decoder are still finding their feet.
    """
    if step < warmup_steps:
        return 1e-6 + (beta_max - 1e-6) * (step / warmup_steps)
    return beta_max


# validation pass

@torch.no_grad()
def validate(model, val_loader, device, beta=1e-5):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    n_batches = 0
    val_bar = tqdm(val_loader, desc="  Validating", leave=False, dynamic_ncols=True)

    for batch in val_bar:
        pose = batch["pose"].to(device)   # (B, T, 190)
        mask = batch["mask"].to(device)   # (B, T)

        x_hat, mu, logvar = model(pose, mask)
        loss, metrics = vae_loss(x_hat, pose, mu, logvar, mask, beta=beta)

        total_loss += loss.item()
        total_recon += metrics["recon"]
        n_batches += 1
        val_bar.set_postfix({"val_recon": f"{total_recon / n_batches:.4f}"})

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_recon": total_recon / max(n_batches, 1),
    }


# main training loop

def phase1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "phase1.log")
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # file only, no stdout
        ],
    )

    log.info(f"Device: {device}")
    log.info(f"PyTorch: {torch.__version__}")

    # dataloaders
    log.info("Building dataloaders...")
    train_loader, train_ds = get_dataloader(
        shard_dir=args.train_shard_dir,
        index_path=args.train_index,
        csv_path=args.train_csv,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name="google/flan-t5-base",   # only needed for uid mapping here, unused by the VAE itself
        max_pose_len=args.max_pose_len,
        augment=True,
        length_aware=False,
    )
    val_loader, _ = get_dataloader(
        shard_dir=args.val_shard_dir,
        index_path=args.val_index,
        csv_path=args.val_csv,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name="google/flan-t5-base",
        max_pose_len=args.max_pose_len,
        augment=False,
        length_aware=False,
    )
    log.info(f"Train: {len(train_ds)} samples | Val: {len(val_loader.dataset)} samples")

    # model
    model = PoseVAE(
        pose_dim=190,
        base_ch=args.base_ch,
        latent_dim=args.latent_dim,
        n_res=args.n_res,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"VAE parameters: {n_params:,}")

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    scaler = GradScaler(enabled=args.fp16)

    # resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    best_val_recon = float("inf")

    ckpt_path = os.path.join(args.output_dir, "latest.pt")
    if os.path.exists(ckpt_path) and args.resume:
        log.info(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_recon = ckpt.get("best_val_recon", float("inf"))
        log.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    # training
    log.info(f"Starting VAE training for {args.epochs} epochs...")
    log.info(f"Steps per epoch: {len(train_loader)}")

    history = {"train_loss": [], "val_loss": [], "val_recon": []}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        t0 = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)  # zero once before epoch

        for step, batch in enumerate(train_bar):
            pose = batch["pose"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            beta = get_beta(global_step, args.beta_warmup, args.beta_max)

            with autocast(enabled=args.fp16):
                x_hat, mu, logvar = model(pose, mask)
                # skip the batch outright if the forward pass already produced NaNs/Infs
                if torch.isnan(x_hat).any() or torch.isinf(x_hat).any() or \
                    torch.isnan(mu).any()   or torch.isinf(mu).any():

                    log.warning(f"Step {global_step}: NaN in model output, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss, metrics = vae_loss(
                    x_hat, pose, mu, logvar, mask,
                    beta=beta,
                    vel_w=args.vel_weight,
                    bone_w=args.bone_weight,
                )
                loss = loss / args.grad_accum  # normalize before backward

            # bail out on NaN/Inf loss too, in case it slipped past the earlier guard
            if torch.isnan(loss) or torch.isinf(loss):
                log.warning(f"Step {global_step}: NaN/Inf loss (recon={metrics['recon']:.4f} "
                            f"kl={metrics['kl']:.4f}), skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            # only step the optimizer every grad_accum batches
            is_last_batch = (step + 1 == len(train_loader))
            if (step + 1) % args.grad_accum == 0 or is_last_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1  # only counts actual optimizer steps
                if global_step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    log.info(
                        f"Epoch {epoch} | Step {step} | "
                        f"loss={loss.item()*args.grad_accum:.4f} recon={metrics['recon']:.4f} "
                        f"vel={metrics['vel']:.4f} bone={metrics['bone']:.4f} "
                        f"kl={metrics['kl']:.4f} β={beta:.2e} lr={lr_now:.2e}"
                    )

            # log unscaled metrics (undo the grad_accum division for reporting)
            epoch_loss += loss.item() * args.grad_accum
            epoch_recon += metrics["recon"]
            epoch_kl += metrics["kl"]

            train_bar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum:.4f}",
                "recon": f"{metrics['recon']:.4f}",
                "kl": f"{metrics['kl']:.4f}",
                "β": f"{beta:.2e}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # epoch summary
        n_steps = len(train_loader)
        avg_loss = epoch_loss / n_steps
        avg_recon = epoch_recon / n_steps
        avg_kl = epoch_kl / n_steps
        elapsed = time.time() - t0

        # run validation
        val_metrics = validate(model, val_loader, device, beta=args.beta_max)

        log.info(
            f"  Epoch {epoch} \n"
            f"  Train  loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}\n"
            f"  Val    loss={val_metrics['val_loss']:.4f} recon={val_metrics['val_recon']:.4f}\n"
            f"  Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_recon"].append(val_metrics["val_recon"])

        # checkpointing
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_recon": best_val_recon,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_path)

        if val_metrics["val_recon"] < best_val_recon:
            best_val_recon = val_metrics["val_recon"]
            best_path = os.path.join(args.output_dir, "best_vae.pt")
            torch.save(ckpt, best_path)
            log.info(f"  ★ New best val_recon={best_val_recon:.4f} → saved {best_path}")

        # periodic epoch checkpoint
        if (epoch + 1) % args.save_every == 0:
            ep_path = os.path.join(args.output_dir, f"vae_epoch_{epoch:03d}.pt")
            torch.save(ckpt, ep_path)

        # persist training history
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # early stopping
        if avg_recon < args.early_stop_threshold:
            log.info(f"Reconstruction loss {avg_recon:.4f} < threshold {args.early_stop_threshold}. "
                     f"Stopping early.")
            break

    log.info("VAE training complete.")
    log.info(f"Best val recon: {best_val_recon:.6f}")
    log.info(f"Checkpoints saved to: {args.output_dir}")


# argument parser

def parse_args():
    p = argparse.ArgumentParser(description="Train ISL Pose VAE")

    # data
    p.add_argument("--train_shard_dir", type=str, default=None)
    p.add_argument("--val_shard_dir",   type=str, default=None)
    p.add_argument("--train_csv",       type=str, default=None)
    p.add_argument("--val_csv",         type=str, default=None)
    p.add_argument("--train_index",     type=str, default=None)
    p.add_argument("--val_index",       type=str, default=None)
    p.add_argument("--max_pose_len",    type=int, default=512)

    # model
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--base_ch",    type=int, default=256)
    p.add_argument("--n_res",      type=int, default=2)

    # training
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--fp16",         action="store_true", default=True)
    p.add_argument("--grad_accum",   type=int,   default=4)

    # loss weights
    p.add_argument("--beta_max",    type=float, default=1e-4)
    p.add_argument("--beta_warmup", type=int,   default=2000)
    p.add_argument("--vel_weight",  type=float, default=0.1)
    p.add_argument("--bone_weight", type=float, default=0.0)

    # early stopping
    p.add_argument("--early_stop_threshold", type=float, default=0.005,
                   help="Stop if train recon loss falls below this value")

    # logging / checkpointing
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--log_every",  type=int, default=200)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--resume",     action="store_true", default=False)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    phase1(args)
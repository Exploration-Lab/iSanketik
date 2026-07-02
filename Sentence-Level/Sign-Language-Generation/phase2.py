"""
Phase 2 Training: Conditional Flow Matching with DiT

Trains a DiT velocity predictor over frozen VAE latents using rectified
flow matching, conditioned on Flan-T5 text embeddings. Jointly trains a
length predictor (log-duration regression from pooled text) that is
warmed up alone for the first LENGTH_WARMUP_STEPS steps before the flow
loss is added in. Text conditioning uses classifier-free guidance dropout
during training. Tracks an EMA copy of the DiT for validation and
inference, and validates generation quality via DTW against ground-truth
poses on the val split.

Run:
  python phase2.py --vae_ckpt <path_to_best_vae.pt> --output_dir ./checkpoints/flow

See parse_args() for the full list of configurable data, model, and
training arguments.
"""
import os
import sys
import time
import argparse
import logging
import json
import math
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from helpers.data import get_dataloader
from helpers.pose_vae import PoseVAE
from helpers.text_encoder import FlanT5TextEncoder
from helpers.dit import DiTFlowModel, LengthPredictor
from helpers.flow_matching import RectifiedFlow


log = logging.getLogger(__name__)


# EMA

class EMA:
    """Exponential moving average of DiT weights. Always use the EMA weights for inference."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# Build latent mask from actual encoded shape

def build_lat_mask_from_encoded(
    z0: torch.Tensor, lengths: torch.Tensor, downsample: int = 4
) -> torch.Tensor:
    """
    Build a boolean mask for z0 based on the original pose lengths.

    Uses the actual z0.shape[1] (the floor-division result from strided
    convolutions), not ceil(T/4), since the two can differ by 1 for odd T.

    z0      : (B, T_lat, D)
    lengths : (B,), original pose frame counts
    Returns : (B, T_lat) float mask
    """
    B, T_lat, _ = z0.shape
    lat_mask = torch.zeros(B, T_lat, device=z0.device)
    for i in range(B):
        actual_lat_len = min(
            (lengths[i].item() + downsample - 1) // downsample, T_lat
        )
        lat_mask[i, :actual_lat_len] = 1.0
    return lat_mask


# DTW (fast, body joints only)

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    DTW between two pose sequences, body joints only (first 66 dims),
    normalized by path length. seq1, seq2: (T1, D), (T2, D).
    """
    s1 = seq1[:, :66]
    s2 = seq2[:, :66]
    T1, T2 = len(s1), len(s2)
    cost = np.full((T1, T2), np.inf, dtype=np.float32)
    cost[0, 0] = np.linalg.norm(s1[0] - s2[0])
    for i in range(1, T1):
        cost[i, 0] = cost[i-1, 0] + np.linalg.norm(s1[i] - s2[0])
    for j in range(1, T2):
        cost[0, j] = cost[0, j-1] + np.linalg.norm(s1[0] - s2[j])
    for i in range(1, T1):
        for j in range(1, T2):
            cost[i, j] = np.linalg.norm(s1[i] - s2[j]) + min(
                cost[i-1, j], cost[i, j-1], cost[i-1, j-1]
            )
    return float(cost[T1-1, T2-1]) / (T1 + T2)


# Validation

@torch.no_grad()
def validate(
    dit: nn.Module,
    text_encoder: nn.Module,
    length_predictor: nn.Module,
    vae: nn.Module,
    flow: RectifiedFlow,
    val_loader,
    device: torch.device,
    args,
    n_samples: int = 100,
    guidance_scale: float = 3.0,
) -> tuple:
    """
    Generate poses for n_samples from the val set and compute DTW.

    Length prediction uses LOG_BIAS_FOR_VAL so predicted lengths are
    corrected for the train-to-val duration shift before generation. A
    length calibration summary is logged each call to track whether the
    predictor is converging to the right distribution.

    GT poses come directly from the dataloader (normalized space), and
    generated poses from vae.decode() are also in normalized space, so DTW
    is always computed normalized vs normalized.
    """
    dit.eval()
    text_encoder.eval()
    length_predictor.eval()

    # Unconditional embedding: zero token ids through the encoder
    dummy_ids = torch.zeros(1, args.max_text_len, dtype=torch.long, device=device)
    dummy_mask = torch.zeros(1, args.max_text_len, dtype=torch.float, device=device)
    uncond_emb, uncond_mask = text_encoder(dummy_ids, dummy_mask)

    dtw_scores = []
    pred_lens = []  # for length calibration diagnostics
    gt_lens = []
    count = 0

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        if count >= n_samples:
            break

        pose_gt = batch["pose"].to(device)
        lengths = batch["lengths"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        B = pose_gt.shape[0]

        text_emb, text_mask = text_encoder(ids, attn)

        # Length prediction with val bias correction.
        # LOG_BIAS_FOR_VAL = log(6.60 / 6.22) ~ +0.060, shifting predicted
        # lengths upward to match the val distribution.
        lat_lens, pose_lens = length_predictor.predict_length(
            text_emb,
            text_mask,
            downsample=4,
            log_bias=LengthPredictor.LOG_BIAS_FOR_VAL,
        )

        for i in range(min(B, n_samples - count)):
            ll = max(lat_lens[i].item(), 8)  # safety floor

            z0 = flow.sample_euler(
                model_fn=dit,
                shape=(1, ll, args.latent_dim),
                text_emb=text_emb[i:i+1],
                text_mask=text_mask[i:i+1],
                n_steps=args.val_ddim_steps,
                guidance_scale=guidance_scale,
                uncond_emb=uncond_emb,
                uncond_mask=uncond_mask,
                device=str(device),
                use_midpoint=True,
            )  # (1, ll, latent_dim)

            # Denormalize latent before VAE decode
            z0_denorm = z0 * (args.lat_std + 1e-6) + args.lat_mean
            pose_pred = vae.decode(z0_denorm).squeeze(0).cpu().numpy()  # (T_pred, D)

            gt_len = lengths[i].item()
            pose_gt_i = pose_gt[i, :gt_len, :].cpu().numpy()

            dtw = dtw_distance(pose_pred, pose_gt_i)
            dtw_scores.append(dtw)

            pred_lens.append(pose_lens[i].item())
            gt_lens.append(gt_len)

            count += 1

    dit.train()
    text_encoder.train()
    length_predictor.train()

    # Length calibration diagnostic
    if pred_lens:
        pred_arr = np.array(pred_lens, dtype=np.float32)
        gt_arr = np.array(gt_lens, dtype=np.float32)
        log.info(
            f"  [Length calibration] "
            f"pred mean={pred_arr.mean():.1f} std={pred_arr.std():.1f} "
            f"median={np.median(pred_arr):.1f} | "
            f"gt mean={gt_arr.mean():.1f} std={gt_arr.std():.1f} "
            f"median={np.median(gt_arr):.1f} | "
            f"bias=+{LengthPredictor.LOG_BIAS_FOR_VAL:.4f} (log-space)"
        )

    if not dtw_scores:
        return float("inf"), 0.0, float("inf"), float("inf")
    pred_mean_val = float(pred_arr.mean()) if pred_lens else float("inf")
    gt_mean_val = float(gt_arr.mean()) if gt_lens else float("inf")
    return float(np.mean(dtw_scores)), float(np.std(dtw_scores)), pred_mean_val, gt_mean_val


# Main training loop

def train_flow(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "phase2.log")
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="a")],
    )
    log.info(f"Device: {device} | PyTorch: {torch.__version__}")
    log.info(
        f"Length-aware sampler: pivot={args.sampler_pivot} alpha={args.sampler_alpha} "
        f"vel_aug=[{args.vel_aug_min}, {args.vel_aug_max}]"
    )

    # Dataloaders
    train_loader, train_ds = get_dataloader(
        shard_dir=args.train_shard_dir,
        csv_path=args.train_csv,
        index_path=args.train_index,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer_name="google/flan-t5-base",
        max_text_len=args.max_text_len,
        max_pose_len=args.max_pose_len,
        augment=True,
        length_aware=args.length_aware,
        sampler_pivot=args.sampler_pivot,
        sampler_alpha=args.sampler_alpha,
        vel_aug_min=args.vel_aug_min,
        vel_aug_max=args.vel_aug_max,
    )
    val_loader, _ = get_dataloader(
        shard_dir=args.val_shard_dir,
        csv_path=args.val_csv,
        index_path=args.val_index,
        split="val",
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        tokenizer_name="google/flan-t5-base",
        max_text_len=args.max_text_len,
        max_pose_len=args.max_pose_len,
        augment=False,
    )
    log.info(f"Train: {len(train_ds)} | Val: {len(val_loader.dataset)}")

    # VAE (frozen)
    log.info(f"Loading VAE from {args.vae_ckpt}")
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_saved_args = vae_ckpt.get("args", {})
    vae = PoseVAE(
        pose_dim=vae_saved_args.get("pose_dim", 190),
        base_ch=vae_saved_args.get("base_ch", 256),
        latent_dim=vae_saved_args.get("latent_dim", 128),
    ).to(device)
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    log.info("VAE loaded and frozen.")

    # Text encoder
    text_encoder = FlanT5TextEncoder(
        fine_tune_layers=args.text_finetune_layers,
    ).to(device)

    # DiT flow model
    dit = DiTFlowModel(
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        text_dim=768,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_seq_len=args.max_pose_len // 4 + 8,
    ).to(device)

    n_dit = sum(p.numel() for p in dit.parameters())
    n_text = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    log.info(f"DiT params: {n_dit:,} | Text encoder trainable: {n_text:,}")

    # Length predictor.
    # log_bias=0.0 at construction: the raw network output has no bias.
    # The bias is applied only inside validate() via predict_length(log_bias=...).
    length_predictor = LengthPredictor(text_dim=768, hidden=512, log_bias=0.0).to(device)

    # Flow matching
    flow = RectifiedFlow(use_logit_normal=True)

    # EMA
    ema = EMA(dit, decay=args.ema_decay)

    # Optimizer.
    # The length predictor uses a lower LR than the DiT and text encoder
    # since it converges much faster and is prone to overfitting early.
    param_groups = [
        {"params": dit.parameters(), "lr": args.lr},
        {"params": length_predictor.parameters(), "lr": args.lr * 0.05},
        {
            "params": [p for p in text_encoder.parameters() if p.requires_grad],
            "lr": args.lr * 0.1,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    # Resume
    start_epoch = 0
    global_step = 0
    best_dtw = float("inf")

    ckpt_path = os.path.join(args.output_dir, "flow_epoch_031.pt")
    if os.path.exists(ckpt_path) and args.resume:
        log.info(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        dit.load_state_dict(ckpt["dit"])
        text_encoder.load_state_dict(ckpt["text_encoder"])
        length_predictor.load_state_dict(ckpt["length_predictor"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_dtw = ckpt.get("best_dtw", float("inf"))
        log.info(f"Resumed at epoch {start_epoch}, step {global_step}, best_dtw={best_dtw:.4f}")

    # Latent stats (always loaded, never recomputed)
    log.info(f"Loading latent stats from {args.latent_stats_path}")
    stats = torch.load(args.latent_stats_path, weights_only=False)
    lat_mean = stats["mean"]
    lat_std = stats["std"]
    args.lat_mean = lat_mean
    args.lat_std = lat_std
    log.info(f"Latent stats: mean={lat_mean:.4f} std={lat_std:.4f}")

    # Length loss criterion
    len_criterion = nn.SmoothL1Loss()

    # History
    history_path = os.path.join(args.output_dir, "history.json")
    if args.resume and os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        log.info("Loaded existing history.")
    else:
        history = {
            "train_flow_loss": [],
            "train_len_loss": [],
            "val_dtw_mean": [],
            "val_dtw_std": [],
            "length_calib_error": [],
        }

    log.info(f"Starting flow training for {args.epochs} epochs.")
    log.info(f"Steps per epoch: {len(train_loader)}")

    # Number of initial steps where only the length predictor is optimized,
    # before the flow loss contributes to the total loss.
    LENGTH_WARMUP_STEPS = 2000

    for epoch in range(start_epoch, args.epochs):
        dit.train()
        text_encoder.train()
        length_predictor.train()

        epoch_flow_loss = 0.0
        epoch_len_loss = 0.0
        t0 = time.time()

        bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, unit="batch")

        for step, batch in enumerate(bar):
            pose = batch["pose"].to(device, non_blocking=True)      # (B, T, 190)
            mask = batch["mask"].to(device, non_blocking=True)      # (B, T)
            lengths = batch["lengths"].to(device, non_blocking=True)  # (B,)
            ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            B = pose.shape[0]

            optimizer.zero_grad(set_to_none=True)
            pose = pose.clamp(-10.0, 10.0)  # guard degenerate inputs

            with torch.amp.autocast('cuda', enabled=args.fp16):

                # 1. Encode pose to latent
                with torch.no_grad():
                    mu, _ = vae.encode(pose, mask)
                    z0 = mu  # (B, T_lat, latent_dim)

                if not torch.isfinite(z0).all():
                    log.warning(
                        f"Step {global_step}: NaN/Inf from VAE encoder, skipping batch."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Normalize latent to approximately N(0, 1)
                z0 = (z0 - lat_mean) / (lat_std + 1e-6)
                z0 = z0.clamp(-5.0, 5.0)

                # Build latent mask from actual encoded shape
                lat_mask = build_lat_mask_from_encoded(z0, lengths, downsample=4)

                # 2. Text encode with CFG dropout.
                # Unconditional: replace input_ids and mask with zeros so the
                # T5 encoder processes "nothing" rather than receiving broken input.
                cfg_drop = torch.rand(B, device=device) < args.cfg_drop_prob
                ids_in = ids.clone()
                attn_in = attn.clone()
                ids_in[cfg_drop] = 0
                attn_in[cfg_drop] = 0

                text_emb, text_mask = text_encoder(ids_in, attn_in)  # (B, L, 768)

                # 3. Sample flow time t ~ logit-normal
                t = flow.sample_t(B, device)  # (B,) in [0, 1]

                # 4. Forward process: x_t and v_target
                noise = torch.randn_like(z0)
                x_t, v_target, _ = flow.forward_process(z0, t, noise)

                # 5. DiT velocity prediction
                v_pred = dit(x_t, t, text_emb, text_mask)  # (B, T_lat, latent_dim)

                # 6. Flow loss (masked MSE on velocity)
                flow_loss = flow.compute_loss(v_pred, v_target, lat_mask)

                # 7. Length prediction loss.
                # Re-encode with the real text (no CFG drop) for the length
                # predictor, kept separate from the flow path.
                text_emb_real, text_mask_real = text_encoder(ids, attn)
                log_len_pred = length_predictor(text_emb_real, text_mask_real)
                log_len_gt = torch.log(lengths.float().clamp(min=1.0)).unsqueeze(1)
                len_loss = len_criterion(log_len_pred, log_len_gt)

                # For the first LENGTH_WARMUP_STEPS steps, train only the
                # length predictor and hold the flow loss out of the total.
                if global_step < LENGTH_WARMUP_STEPS:
                    total_loss = args.len_loss_weight * len_loss
                else:
                    total_loss = flow_loss + args.len_loss_weight * len_loss

            # NaN / exploding loss guard
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 10.0:
                log.warning(
                    f"Step {global_step}: bad loss={total_loss.item():.4f} "
                    f"z0_max={z0.abs().max():.2f} "
                    f"v_target_max={v_target.abs().max():.2f} "
                    f"t=[{t.min():.3f},{t.max():.3f}], skipping."
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(dit.parameters()) +
                list(text_encoder.parameters()) +
                list(length_predictor.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(dit)

            epoch_flow_loss += flow_loss.item()
            epoch_len_loss += len_loss.item()
            global_step += 1

            bar.set_postfix({
                "flow": f"{flow_loss.item():.4f}",
                "len": f"{len_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if global_step % args.log_every == 0:
                log.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"flow={flow_loss.item():.4f} len={len_loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Epoch summary
        n = max(len(train_loader), 1)
        avg_flow = epoch_flow_loss / n
        avg_len = epoch_len_loss / n
        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch} summary: "
            f"avg_flow={avg_flow:.4f} avg_len={avg_len:.4f} "
            f"time={elapsed:.0f}s"
        )

        history["train_flow_loss"].append(avg_flow)
        history["train_len_loss"].append(avg_len)

        # Validation
        if (epoch + 1) % args.val_every == 0:
            log.info("Running DTW validation with EMA model...")
            dtw_mean, dtw_std, pred_arr, gt_arr = validate(
                dit=ema.shadow,
                text_encoder=text_encoder,
                length_predictor=length_predictor,
                vae=vae,
                flow=flow,
                val_loader=val_loader,
                device=device,
                args=args,
                n_samples=args.val_n_samples,
                guidance_scale=args.guidance_scale,
            )
            log.info(f"  Val DTW: mean={dtw_mean:.4f} std={dtw_std:.4f}")
            history["val_dtw_mean"].append(dtw_mean)
            history["val_dtw_std"].append(dtw_std)
            calib_error = abs(pred_arr - gt_arr)

            recent_calib_errors = history["length_calib_error"][-3:]  # last 3 val runs
            avg_calib_error = np.mean(recent_calib_errors) if recent_calib_errors else float("inf")

            if dtw_mean < best_dtw and avg_calib_error < 30.0:
                best_dtw = dtw_mean
                best_path = os.path.join(args.output_dir, "best_flow.pt")
                torch.save({
                    "dit": dit.state_dict(),
                    "ema_dit": ema.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "length_predictor": length_predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_dtw": best_dtw,
                    "args": vars(args),
                }, best_path)
                log.info(f"  New best DTW={best_dtw:.4f}, saved to {best_path}")

        # Checkpoint
        ckpt = {
            "dit": dit.state_dict(),
            "ema_dit": ema.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "length_predictor": length_predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_dtw": best_dtw,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_path)

        if (epoch + 1) % args.save_every == 0:
            ep_path = os.path.join(args.output_dir, f"flow_epoch_{epoch:03d}.pt")
            torch.save({
                "dit": dit.state_dict(),
                "ema_dit": ema.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "length_predictor": length_predictor.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ep_path)

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    log.info(f"Training complete. Best DTW: {best_dtw:.4f}")


# Args

def parse_args():
    p = argparse.ArgumentParser(description="Train ISL Flow Matching with DiT")

    # Data
    p.add_argument("--train_shard_dir", type=str,
        default=None)
    p.add_argument("--val_shard_dir", type=str,
        default=None)
    p.add_argument("--train_csv", type=str,
        default=None)
    p.add_argument("--val_csv", type=str,
        default=None)
    p.add_argument("--train_index", type=str,
        default=None)
    p.add_argument("--val_index", type=str,
        default=None)
    p.add_argument("--latent_stats_path", type=str,
        default=None)
    p.add_argument("--max_pose_len", type=int, default=512)
    p.add_argument("--max_text_len", type=int, default=64)

    # Length-aware sampler.
    # Use --no_length_aware to revert to the original ShardAwareSampler (for ablation).
    p.add_argument("--length_aware", action="store_true", default=True,
        help="Use LengthAwareShardSampler to upweight longer sentences (default: on)")
    p.add_argument("--no_length_aware", dest="length_aware", action="store_false",
        help="Disable length-aware sampler (ablation)")
    p.add_argument("--sampler_pivot", type=float, default=7.0,
        help="Word count below which sample weight = 1.0 (default: 7)")
    p.add_argument("--sampler_alpha", type=float, default=1.5,
        help="Upweight strength above pivot (default: 1.5, ~2.5x at 14 words)")

    # Velocity augment range.
    # Asymmetric and right-skewed toward longer poses to match the val/test
    # duration distribution. The original range was symmetric +/-20%
    # ([0.80, 1.20]); the new default [0.85, 1.40] produces more
    # long-duration samples. To restore the original: --vel_aug_min 0.8 --vel_aug_max 1.2
    p.add_argument("--vel_aug_min", type=float, default=0.85,
        help="Lower bound of velocity augment speed factor (default: 0.85)")
    p.add_argument("--vel_aug_max", type=float, default=1.40,
        help="Upper bound of velocity augment speed factor (default: 1.40)")

    # VAE
    p.add_argument("--vae_ckpt", type=str,
        default=None)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--vae_base_ch", type=int, default=256)

    # DiT model
    p.add_argument("--model_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pose_dim", type=int, default=190)

    # Text encoder
    p.add_argument("--text_finetune_layers", type=int, default=4)

    # Flow matching
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--cfg_drop_prob", type=float, default=0.25)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--len_loss_weight", type=float, default=0.5)

    # Validation
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--val_n_samples", type=int, default=500)
    p.add_argument("--val_ddim_steps", type=int, default=50)

    # Logging / saving
    p.add_argument("--output_dir", type=str,
        default=None)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--resume", action="store_true", default=False)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_flow(args)
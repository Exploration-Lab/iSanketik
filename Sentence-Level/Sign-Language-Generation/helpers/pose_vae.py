"""
Phase 1: Pose VAE

Encodes  (B, T, 190)  ->  (B, T//4, 128)  continuous latent z
Decodes  (B, T//4, 128) -> (B, T, 190)

Architecture:
  Encoder: stacked 1D-conv residual blocks with stride-2 downsampling x2 (net 4x)
  Decoder: stacked 1D-convtranspose residual blocks with stride-2 upsampling x2
  Latent:  reparameterize -> mu, log sigma^2 (dim=128)

Losses:
  L_recon = MSE(X_hat, X) weighted by confidence mask
  L_vel   = MSE(dX_hat/dt, dX/dt), smooth motion
  L_bone  = MSE(bone_lengths_hat, bone_lengths), structural consistency
  L_KL    = beta * KL(q||p), beta starts tiny and anneals

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Building blocks

class SafeGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x):
        return self.gn(x.float()).to(x.dtype)

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            SafeGroupNorm(min(8, channels), channels, eps=1e-3),  # larger eps
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            SafeGroupNorm(min(8, channels), channels, eps=1e-3),
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    """Residual block + stride-2 downsampling."""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.res = nn.Sequential(*[ResBlock1D(out_ch) for _ in range(n_res)])
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.res(x)
        return self.down(x)


class UpBlock(nn.Module):
    """Stride-2 upsampling + residual block."""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = nn.Sequential(*[ResBlock1D(out_ch) for _ in range(n_res)])

    def forward(self, x):
        x = self.up(x)
        return self.res(x)


# Encoder

class PoseEncoder(nn.Module):
    """
    (B, T, 190) -> (B, T//4, latent_dim*2), mean and logvar concatenated.
    Two stride-2 down-blocks: T -> T//2 -> T//4.
    """
    def __init__(
        self,
        pose_dim: int = 190,
        base_ch: int = 256,
        latent_dim: int = 128,
        n_res: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(pose_dim, base_ch, kernel_size=1)

        self.down1 = DownBlock(base_ch, base_ch * 2, n_res)      # T -> T//2
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, n_res)  # T//2 -> T//4

        # Bottleneck residual
        self.bottleneck = nn.Sequential(
            ResBlock1D(base_ch * 4),
            ResBlock1D(base_ch * 4),
        )

        # Project to latent mean + logvar
        self.to_mu = nn.Conv1d(base_ch * 4, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(base_ch * 4, latent_dim, kernel_size=1)

        nn.init.zeros_(self.to_logvar.weight)
        nn.init.zeros_(self.to_logvar.bias)

    def forward(self, x, mask=None):
        """
        x    : (B, T, 190)
        mask : (B, T), optional, not used in encoder directly
        Returns mu (B, T//4, D), logvar (B, T//4, D)
        """
        x = x.permute(0, 2, 1)    # (B, 190, T)
        x = self.input_proj(x)    # (B, base_ch, T)
        x = self.down1(x)         # (B, base_ch*2, T//2)
        x = self.down2(x)         # (B, base_ch*4, T//4)
        x = self.bottleneck(x)    # (B, base_ch*4, T//4)

        mu = self.to_mu(x).permute(0, 2, 1)  # (B, T//4, D)
        logvar = self.to_logvar(x).permute(0, 2, 1).clamp(-6.0, 2.0)
        
        return mu, logvar


# Decoder

class PoseDecoder(nn.Module):
    """
    (B, T//4, latent_dim) -> (B, T, 190)
    Two stride-2 up-blocks: T//4 -> T//2 -> T.
    """
    def __init__(
        self,
        pose_dim: int = 190,
        base_ch: int = 256,
        latent_dim: int = 128,
        n_res: int = 2,
    ):
        super().__init__()
        self.latent_proj = nn.Conv1d(latent_dim, base_ch * 4, kernel_size=1)

        self.bottleneck = nn.Sequential(
            ResBlock1D(base_ch * 4),
            ResBlock1D(base_ch * 4),
        )

        self.up1 = UpBlock(base_ch * 4, base_ch * 2, n_res)  # T//4 -> T//2
        self.up2 = UpBlock(base_ch * 2, base_ch, n_res)      # T//2 -> T

        self.output_proj = nn.Conv1d(base_ch, pose_dim, kernel_size=1)

    def forward(self, z):
        """
        z : (B, T//4, latent_dim)
        Returns x_hat : (B, T, 190)
        """
        x = z.permute(0, 2, 1)    # (B, D, T//4)
        x = self.latent_proj(x)   # (B, base_ch*4, T//4)
        x = self.bottleneck(x)
        x = self.up1(x)           # (B, base_ch*2, T//2)
        x = self.up2(x)           # (B, base_ch, T)
        x = self.output_proj(x)   # (B, 190, T)
        return x.permute(0, 2, 1)  # (B, T, 190)


# Full VAE

class PoseVAE(nn.Module):
    def __init__(
        self,
        pose_dim: int = 190,
        base_ch: int = 256,
        latent_dim: int = 128,
        n_res: int = 2,
        downsample: int = 4,  
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.downsample = downsample

        self.encoder = PoseEncoder(pose_dim, base_ch, latent_dim, n_res)
        self.decoder = PoseDecoder(pose_dim, base_ch, latent_dim, n_res)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  

    def encode(self, x, mask=None):
        """Returns mu, logvar of shape (B, T//4, latent_dim)."""
        return self.encoder(x, mask)

    def decode(self, z, target_T=None):
        """
        z        : (B, T//4, latent_dim)
        target_T : if given, trim/pad decoded output to exactly target_T frames
        """
        x_hat = self.decoder(z)  # (B, T_dec, 190)
        if target_T is not None:
            T_dec = x_hat.shape[1]
            if T_dec > target_T:
                x_hat = x_hat[:, :target_T, :]
            elif T_dec < target_T:
               
                pad = x_hat[:, -1:, :].expand(-1, target_T - T_dec, -1)
                x_hat = torch.cat([x_hat, pad], dim=1)
        return x_hat

    def forward(self, x, mask=None):
        """
        x    : (B, T, 190)
        mask : (B, T)
        Returns: x_hat, mu, logvar
        """
        T = x.shape[1]
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, target_T=T)
        return x_hat, mu, logvar


#
BODY_CONNECTIONS = [
    (11, 12), 
    (11, 13),  
    (13, 15),  
    (12, 14),  
    (14, 16),

    (15, 17),  
    (15, 19),  
    (15, 21),  
    (16, 18),  
    (16, 20),  
    (16, 22),  
]

_BODY_JOINTS = 23
_BODY_DIM = _BODY_JOINTS * 2  # 46

def extract_body_joints(pose):
    """
    pose : (B, T, 190)
    Returns body xy : (B, T, 23, 2)
    """
    body = pose[:, :, :_BODY_DIM]  # (B, T, 46)
    return body.reshape(body.shape[0], body.shape[1], _BODY_JOINTS, 2)

def bone_length_loss(pose_hat, pose_gt):
    """MSE between corresponding bone lengths in predicted vs GT pose. Only called when bone_w > 0."""
    bj_hat = extract_body_joints(pose_hat)  # (B, T, 23, 2)
    bj_gt = extract_body_joints(pose_gt)
    losses = []
    for (i, j) in BODY_CONNECTIONS:
        bone_hat = torch.norm(bj_hat[:, :, i, :] - bj_hat[:, :, j, :], dim=-1)
        bone_gt = torch.norm(bj_gt[:, :, i, :] - bj_gt[:, :, j, :], dim=-1)
        losses.append(F.mse_loss(bone_hat, bone_gt))
    return torch.stack(losses).mean()


# VAE loss

def vae_loss(
    x_hat: torch.Tensor,   # (B, T, 190)
    x_gt: torch.Tensor,    # (B, T, 190)
    mu: torch.Tensor,      # (B, T//4, D)
    logvar: torch.Tensor,  # (B, T//4, D)
    mask: torch.Tensor,    # (B, T), 1=valid 0=padding
    beta: float = 1e-5,
    vel_w: float = 0.1,
    bone_w: float = 0.0,
):
    """
    Total VAE loss combining masked reconstruction MSE, a velocity
    (temporal smoothness) term, an optional bone-length consistency
    term (skipped when bone_w == 0), and beta-weighted KL divergence.
    """
    B, T, D = x_gt.shape

    # 1. Reconstruction loss (masked)
    mask_3d = mask.unsqueeze(-1).expand_as(x_gt)  # (B, T, 190)
    n_valid = mask_3d.sum().clamp(min=1)
    l_recon = ((x_hat - x_gt) ** 2 * mask_3d).sum() / n_valid

    # 2. Velocity loss
    if T > 1:
        vel_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]  # (B, T-1, 190)
        vel_gt = x_gt[:, 1:, :] - x_gt[:, :-1, :]
        vel_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1).expand_as(vel_gt)
        nv = vel_mask.sum().clamp(min=1)
        l_vel = ((vel_hat - vel_gt) ** 2 * vel_mask).sum() / nv
    else:
        l_vel = torch.tensor(0.0, device=x_gt.device)

    # 3. Bone-length loss (guarded, skipped entirely when bone_w == 0.0)
    if bone_w > 0.0:
        l_bone = bone_length_loss(x_hat, x_gt)
    else:
        l_bone = torch.tensor(0.0, device=x_gt.device)

    # 4. KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  
    mu_f = mu.float()
    logvar_f = logvar.float()
    FREE_BITS = 0.5  # nats per latent dimension (typical range 0.25-1.0)
    kl_per_dim = -0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp())
    l_kl = kl_per_dim.clamp(min=FREE_BITS).mean()

    total = l_recon + vel_w * l_vel + bone_w * l_bone + beta * l_kl

    return total, {
        "recon": l_recon.item(),
        "vel": l_vel.item(),
        "bone": l_bone.item(),
        "kl": l_kl.item(),
    }
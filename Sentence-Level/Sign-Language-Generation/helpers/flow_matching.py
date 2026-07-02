"""
Rectified flow (flow matching).

"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Callable, Optional


# time-sampling strategies

def sample_logit_normal_times(
    B: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """
    Sample t from a logit-normal distribution: u ~ Normal(mean, std),
    t = sigmoid(u). Concentrates samples near t=0.5 (medium noise level),
    which we found trains better than plain uniform sampling.
    """
    u = torch.randn(B, device=device) * std + mean
    return torch.sigmoid(u)   # (B,) in (0, 1)


def sample_uniform_times(B: int, device: torch.device) -> torch.Tensor:
    """Plain uniform time sampling, kept around as a baseline/fallback."""
    return torch.rand(B, device=device)


# rectified flow scheduler

class RectifiedFlow(nn.Module):
    

    def __init__(self, use_logit_normal: bool = True):
        super().__init__()
        self.use_logit_normal = use_logit_normal

    def sample_t(self, B: int, device: torch.device) -> torch.Tensor:
        if self.use_logit_normal:
            return sample_logit_normal_times(B, device)
        t = sample_logit_normal_times(B, device)
        return t.clamp(0.02, 0.98)  # keep away from the extreme timesteps
        # return sample_uniform_times(B, device)

    def forward_process(
        self,
        x0: torch.Tensor,             # (B, T, D) clean latent
        t: torch.Tensor,              # (B,) in [0, 1]
        noise: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Compute the noisy x_t and the target velocity v*.
        Returns (x_t, v_target, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # broadcast t from (B,) to (B, 1, 1) to line up with (B, T, D)
        t_b = t.view(-1, 1, 1)

        x_t = (1.0 - t_b) * noise + t_b * x0   # linear interpolation
        v_target = x0 - noise                        # target velocity, constant along the path

        return x_t, v_target, noise

    def compute_loss(
        self,
        v_pred: torch.Tensor,     # (B, T, D) model output
        v_target: torch.Tensor,   # (B, T, D)
        lat_mask: torch.Tensor,   # (B, T)   1=valid 0=padding
        t: Optional[torch.Tensor] = None,   # (B,) reserved for SNR weighting, unused by default
    ) -> torch.Tensor:
        """
        Masked MSE loss between predicted and target velocity, so that
        padded latent frames don't contribute to the gradient.
        """
        mask_3d = lat_mask.unsqueeze(-1).expand_as(v_pred)   # (B, T, D)
        n_valid = mask_3d.sum().clamp(min=1.0)
        loss = ((v_pred - v_target) ** 2 * mask_3d).sum() / n_valid
        return loss

    @torch.no_grad()
    def sample_euler(
        self,
        model_fn: Callable,        # (x_t, t, text_emb, text_mask) -> v_pred
        shape: tuple,               # (B, T_lat, latent_dim)
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        n_steps: int = 50,
        guidance_scale: float = 3.0,
        uncond_emb: Optional[torch.Tensor] = None,
        uncond_mask: Optional[torch.Tensor] = None,
        device: str = "cuda",
        use_midpoint: bool = True,    # 2nd-order midpoint integration
    ) -> torch.Tensor:
        """
        Solve the flow ODE from t=0 (noise) to t=1 (data).

        Defaults to the midpoint method, which costs the same two forward
        passes per step as naive RK2 but is noticeably more accurate than
        plain Euler at matched step counts:
          k1 = v(x_t, t)
          k2 = v(x_t + k1*dt/2, t + dt/2)
          x_{t+1} = x_t + k2 * dt

        With classifier-free guidance:
          v_guided = v_uncond + w * (v_cond - v_uncond)
        """
        B = shape[0]
        x = torch.randn(shape, device=device)

        timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_now = timesteps[i]
            t_mid = t_now + dt * 0.5
            t_batch = torch.full((B,), t_now.item(), device=device)

            # guided velocity at t_now
            v_now = self._cfg_forward(
                model_fn, x, t_batch,
                text_emb, text_mask,
                uncond_emb, uncond_mask,
                guidance_scale,
            )

            if use_midpoint:
                # midpoint estimate
                x_mid = x + v_now * (dt * 0.5)
                t_mid_b = torch.full((B,), t_mid.item(), device=device)
                v_mid = self._cfg_forward(
                    model_fn, x_mid, t_mid_b,
                    text_emb, text_mask,
                    uncond_emb, uncond_mask,
                    guidance_scale,
                )
                x = x + v_mid * dt
            else:
                x = x + v_now * dt

        return x   # (B, T_lat, latent_dim), denoised latent at t=1

    def _cfg_forward(
        self,
        model_fn: Callable,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        uncond_emb: Optional[torch.Tensor],
        uncond_mask: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the model with CFG, batching cond + uncond into one forward pass."""
        if guidance_scale == 1.0 or uncond_emb is None:
            return model_fn(x, t_batch, text_emb, text_mask)

        # single forward pass over the doubled batch
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t_batch, t_batch], dim=0)
        emb_double = torch.cat([text_emb, uncond_emb], dim=0)
        mask_double = torch.cat([text_mask, uncond_mask], dim=0)

        v_double = model_fn(x_double, t_double, emb_double, mask_double)
        v_cond, v_uncond = v_double.chunk(2, dim=0)

        return v_uncond + guidance_scale * (v_cond - v_uncond)
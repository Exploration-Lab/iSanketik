"""
DiT (Diffusion Transformer) Denoiser for Rectified Flow

Architecture: flat Transformer=

The VAE latents are (B, T_lat, 128) with T_lat around 44 frames on average.
A flat Transformer instead attends over
all T_lat frames at every layer, preserving temporal structure.

Input:
  z_t       : (B, T_lat, latent_dim)  noisy latent at flow time t
  t         : (B,)                    continuous time in [0, 1]
  text_emb  : (B, L, text_dim)        token embeddings from T5
  text_mask : (B, L)                  attention mask

Output:
  v_pred    : (B, T_lat, latent_dim)  predicted velocity (data - noise)

Conditioning:
  - Timestep via sinusoidal embedding -> MLP -> AdaLN-Zero per block
  - Text via (a) cross-attention at every block (token-level) and
             (b) mean-pooled text added to the timestep embedding for AdaLN
             (global sentence context modulates every norm layer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal time embedding

class TimestepEmbedding(nn.Module):
    """Embeds scalar flow time t in [0, 1] via sinusoidal features and an MLP."""

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SiLU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SiLU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0, 1]. Returns (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)


# Rotary position embedding (RoPE)

def build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple:
    """Precompute sin/cos for RoPE. Returns (cos, sin), each (seq_len, head_dim)."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)  # (T, head_dim//2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension in half and rotate."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: (B, H, T, head_dim); cos, sin: (T, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# Cross-attention: pose latent attends to text tokens

class CrossAttention(nn.Module):
    """Pose latent (B, T, dim) attends to text (B, L, text_dim). Pre-norm on the query side only."""

    def __init__(self, dim: int, text_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(text_dim, dim, bias=False)
        self.v = nn.Linear(text_dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.out.weight)  # zero-init output for stability

    def forward(
        self,
        x: torch.Tensor,         # (B, T, dim)
        context: torch.Tensor,   # (B, L, text_dim)
        ctx_mask: torch.Tensor,  # (B, L), 1=valid 0=pad
    ) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        Q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if ctx_mask is not None:
            pad = (1.0 - ctx_mask).bool().unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard against all-masked rows
        attn = self.drop(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, -1)
        return residual + self.out(out)


# AdaLN-Zero modulation

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift. scale, shift: (B, dim)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Single DiT block

class DiTBlock(nn.Module):
    """
    One DiT block: AdaLN-Zero self-attention with RoPE, cross-attention to
    text, then AdaLN-Zero feed-forward.

    The conditioning vector c = timestep_emb + pooled_text_emb (dim=model_dim)
    is projected through a zero-initialized linear layer to 6 * dim:
    (shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff). Gates start
    near zero so the block behaves close to identity at initialization.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        text_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Self-attention
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False)
        self.q_sa = nn.Linear(dim, dim, bias=False)
        self.k_sa = nn.Linear(dim, dim, bias=False)
        self.v_sa = nn.Linear(dim, dim, bias=False)
        self.out_sa = nn.Linear(dim, dim, bias=False)
        self.drop_sa = nn.Dropout(dropout)

        # Cross-attention
        self.cross = CrossAttention(dim, text_dim, n_heads, dropout)

        # Feed-forward (SwiGLU)
        ff_hidden = dim * ff_mult
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff_gate = nn.Linear(dim, ff_hidden, bias=False)
        self.ff_up = nn.Linear(dim, ff_hidden, bias=False)
        self.ff_down = nn.Linear(ff_hidden, dim, bias=False)
        self.drop_ff = nn.Dropout(dropout)

        # AdaLN-Zero: projects conditioning to 6 * dim, zero-initialized
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, dim)
        c: torch.Tensor,          # (B, dim), timestep + pooled_text
        text_emb: torch.Tensor,   # (B, L, text_dim)
        text_mask: torch.Tensor,  # (B, L)
        rope_cos: torch.Tensor,   # (T, head_dim)
        rope_sin: torch.Tensor,   # (T, head_dim)
    ) -> torch.Tensor:
        B, T, dim = x.shape

        # AdaLN parameters from conditioning
        ada = self.adaLN(c)  # (B, 6*dim)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = ada.chunk(6, dim=-1)

        # 1. Self-attention
        h = modulate(self.norm_sa(x), shift_sa, scale_sa)
        Q = self.q_sa(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_sa(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_sa(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        Q, K = apply_rope(Q, K, rope_cos, rope_sin)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop_sa(attn)
        sa_out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, dim)
        sa_out = self.out_sa(sa_out)
        x = x + gate_sa.unsqueeze(1) * sa_out

        # 2. Cross-attention
        x = self.cross(x, text_emb, text_mask)

        # 3. Feed-forward (SwiGLU)
        h = modulate(self.norm_ff(x), shift_ff, scale_ff)
        ff_out = self.ff_down(
            self.drop_ff(F.silu(self.ff_gate(h)) * self.ff_up(h))
        )
        x = x + gate_ff.unsqueeze(1) * ff_out

        return x


# Full DiT model

class DiTFlowModel(nn.Module):
    """
    Full DiT model for rectified flow.

    Design:
      - 16 DiT blocks, dim=512, 8 heads (~130M trainable parameters)
      - RoPE positional encoding, no learned position embeddings
      - Dual text conditioning: token-level cross-attention in every block,
        plus mean-pooled text added to the timestep embedding for AdaLN
      - Zero-initialized output projection for a stable training start
      - Predicts velocity v = data - noise (not noise, not x0)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        model_dim: int = 512,
        n_layers: int = 16,
        n_heads: int = 8,
        text_dim: int = 768,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,  # max latent sequence length
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.max_seq_len = max_seq_len

        # Input projection: latent_dim -> model_dim
        self.in_proj = nn.Linear(latent_dim, model_dim, bias=False)

        # Timestep embedding: t in [0, 1] -> model_dim
        self.time_embed = TimestepEmbedding(model_dim)

        # Text pooling projection: text_dim -> model_dim, for AdaLN conditioning
        self.text_pool_proj = nn.Sequential(
            nn.Linear(text_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=model_dim,
                n_heads=n_heads,
                text_dim=text_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output: final norm + zero-initialized projection to latent_dim
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,        # (B, T_lat, latent_dim)
        t: torch.Tensor,          # (B,) float in [0, 1]
        text_emb: torch.Tensor,   # (B, L, text_dim)
        text_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Returns the velocity prediction v: (B, T_lat, latent_dim)."""
        B, T, _ = z_t.shape

        # Input projection
        x = self.in_proj(z_t)  # (B, T, model_dim)

        # Conditioning vector: timestep + pooled text
        t_emb = self.time_embed(t)  # (B, model_dim)

        # Mean-pool text over valid tokens
        mask_f = text_mask.float().unsqueeze(-1)         # (B, L, 1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)          # (B, 1)
        pooled = (text_emb * mask_f).sum(dim=1) / denom   # (B, text_dim)
        text_c = self.text_pool_proj(pooled)               # (B, model_dim)

        c = t_emb + text_c  # (B, model_dim), combined conditioning

        # RoPE cache
        rope_cos, rope_sin = build_rope_cache(T, self.head_dim, z_t.device)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, text_emb, text_mask, rope_cos, rope_sin)

        # Output
        x = self.out_norm(x)
        v = self.out_proj(x)  # (B, T, latent_dim)
        return v


# Length predictor with optional inference-time bias correction
#
# The training split has a shorter mean duration (6.22s) than validation
# (6.60s) and test (7.06s), so a predictor trained purely on train-split
# log-durations will systematically underpredict length at test time. We
# correct for this with an additive log-space bias applied only at
# inference:
#
#   log_len_corrected = log_len_pred + log_bias
#   log_bias = log(target_mean_duration / train_mean_duration)
#
# This does not affect the training loss or network weights; the bias is
# a scalar added to the raw prediction, chosen per evaluation split. The
# predictor also supports temperature scaling on the output:
#
#   pose_len = round(exp(log_len * temperature + log_bias))
#
# with temperature > 1 giving a more spread-out distribution and
# temperature < 1 a more peaked one (default 1.0, no change).

class LengthPredictor(nn.Module):
    """
    Predicts log(pose_length) from a mean-pooled text embedding.

    At inference: pose_len = round(exp(pred + log_bias)), clamped to [16, 512];
    latent_len = ceil(pose_len / 4).

    Args:
        text_dim: dimension of the text encoder output (768 for Flan-T5-base).
        hidden: hidden MLP dimension.
        log_bias: additive log-space correction applied only at inference,
            computed as log(target_mean_duration / train_mean_duration).
            Pass 0.0 to disable (default).
    """

    # Empirical duration statistics in seconds, at 25 fps on the ISL data.
    # Update these if the underlying dataset changes.
    TRAIN_MEAN_DURATION = 6.22
    VAL_MEAN_DURATION = 6.60
    TEST_MEAN_DURATION = 7.06

    # Precomputed log-biases for convenience.
    LOG_BIAS_FOR_VAL = math.log(VAL_MEAN_DURATION / TRAIN_MEAN_DURATION)   # ~ +0.060
    LOG_BIAS_FOR_TEST = math.log(TEST_MEAN_DURATION / TRAIN_MEAN_DURATION)  # ~ +0.128

    def __init__(self, text_dim: int = 768, hidden: int = 512, log_bias: float = 0.0):
        super().__init__()
        self.log_bias = log_bias
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(155.0))

    def forward(self, text_emb: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """
        text_emb: (B, L, text_dim); text_mask: (B, L).
        Returns (B, 1) raw log predicted pose length, with no bias applied
        (the bias is applied only in predict_length, so the raw logit is
        always what's used for the training loss).
        """
        mask_f = text_mask.float().unsqueeze(-1)
        pooled = (text_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.net(pooled)  # (B, 1)

    def predict_length(
        self,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        downsample: int = 4,
        log_bias: float = None,   # overrides self.log_bias if provided
        temperature: float = 1.0,  # >1 more spread, <1 more peaked
    ) -> tuple:
        """
        Returns (lat_len, pose_len) as LongTensors of shape (B,), where
        lat_len = ceil(pose_len / downsample) and
        pose_len = round(exp(raw_log_len * temperature + log_bias)), clamped
        to [16, 512]. log_bias priority is argument over self.log_bias.

        Typical call at test time:
            lat_len, pose_len = predictor.predict_length(
                text_emb, text_mask,
                log_bias=LengthPredictor.LOG_BIAS_FOR_TEST,
            )
        """
        bias = log_bias if log_bias is not None else self.log_bias

        raw = self.forward(text_emb, text_mask)          # (B, 1)
        log_len = raw.squeeze(-1) * temperature + bias    # (B,)
        pose_len = torch.exp(log_len).round().long().clamp(min=16, max=512)
        lat_len = (pose_len + downsample - 1) // downsample
        return lat_len, pose_len

    @classmethod
    def log_bias_for_split(cls, split: str) -> float:
        """
        Returns the appropriate log_bias for a split name.

        Usage:
            bias = LengthPredictor.log_bias_for_split("test")
            lat_len, pose_len = predictor.predict_length(
                text_emb, text_mask, log_bias=bias
            )
        """
        split = split.lower().strip()
        if split == "test":
            return cls.LOG_BIAS_FOR_TEST
        elif split == "val":
            return cls.LOG_BIAS_FOR_VAL
        else:
            return 0.0  # no correction for train
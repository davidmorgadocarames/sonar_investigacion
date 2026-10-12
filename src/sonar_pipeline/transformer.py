"""Transformer (F5-T04) and hybrid CNN + transformer (F5-T05) of round 3 (D-024).

Both read the same inputs as the CNN, lofar (B, frames, bins) and demon (B, bins), and turn them
into a sequence of tokens:

- SonarTransformer: the LOFARgram is cut into 40 Hz bands (20 bins of 2 Hz, the last bin at
  2000 Hz is dropped): 50 tokens, each with its 11 frames x 20 bins, projected by a linear layer.
- SonarHybrid: the first two blocks of the CNN run over the LOFARgram; after the mean over time,
  each of the 62 frequency positions (32 channels) is a token, projected by a linear layer.

Both add one DEMON token (a linear layer in the transformer, the 1D branch of the CNN in the
hybrid) and a learnt class token. The tokens go through pre-norm blocks:

    x = x + Dropout(Attention(RMSNorm(x)))      attention with RoPE and GQA
    x = x + SwiGLU(RMSNorm(x))                  hidden size about 8/3 of the dimension

and the class token, after a final RMSNorm, is classified by a linear layer.

- RMSNorm divides every token by the root mean square of its values and multiplies by a learnt
  gain (no mean subtraction, no bias).
- RoPE rotates pairs of query and key coordinates by an angle proportional to the token's
  frequency position, so the attention score between two bands depends on their distance in
  frequency. The DEMON and class tokens have no frequency position and are not rotated.
- GQA (grouped-query attention): the heads share keys and values in groups; with as many groups
  as heads it is the usual multi-head attention.
- SwiGLU: Linear(SiLU(x W_gate) * (x W_up)) W_down.

The attention weights can be kept for the interpretability maps (keep_attention). Everything is
built from matmuls, softmax and expand, which are deterministic on CUDA.
"""

import math

import numpy as np
import torch
from torch import nn

from sonar_pipeline.cnn import conv_block_1d, conv_block_2d, pool_to
from sonar_pipeline.dataset import CLASSES

TOKEN_BINS = 20  # LOFAR bins per transformer token: 40 Hz with 2 Hz bins
ROPE_BASE = 10_000.0
RMS_EPS = 1e-6
HYBRID_CHANNELS = (16, 32)  # the first two LOFAR blocks of the CNN (D-018)
HYBRID_DEMON_CHANNELS = (16, 32)
DEMON_POOLED = 8  # DEMON positions kept by the hybrid's 1D branch, as in the CNN
INIT_STD = 0.02
N_CLASSES = len(CLASSES)


def swiglu_hidden(dim: int) -> int:
    """About 8/3 of the dimension, rounded up to a multiple of 8."""
    return 8 * math.ceil(8 * dim / 3 / 8)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


def rope_tables(
    positions: torch.Tensor,
    rotate: torch.Tensor,
    head_dim: int,
    base: float = ROPE_BASE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin (T, head_dim / 2) of every token; tokens with rotate False get angle 0."""
    if head_dim % 2:
        raise ValueError("RoPE needs an even head dimension")
    steps = torch.arange(0, head_dim, 2, dtype=torch.float64, device=positions.device)
    inverse = base ** (-steps / head_dim)
    angles = positions.double()[:, None] * inverse[None, :] * rotate.double()[:, None]
    return angles.cos().float(), angles.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the coordinate pairs (0, 1), (2, 3)... of x (..., T, head_dim)."""
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, groups: int, dropout: float):
        super().__init__()
        if dim % heads or heads % groups:
            raise ValueError(
                f"dim {dim}, heads {heads} and groups {groups} do not divide"
            )
        self.heads, self.groups = heads, groups
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, groups * self.head_dim, bias=False)
        self.v = nn.Linear(dim, groups * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.keep_attention = False
        self.attention: torch.Tensor | None = None  # (B, heads, T, T) when kept

    def shared(self, x: torch.Tensor) -> torch.Tensor:
        """(B, groups, T, d) -> (B, heads, T, d): each group serves heads / groups heads."""
        b, _, t, d = x.shape
        repeat = self.heads // self.groups
        return x[:, :, None].expand(b, self.groups, repeat, t, d).reshape(b, -1, t, d)

    def forward(self, x, cos=None, sin=None):
        b, t, dim = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.groups, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.groups, self.head_dim).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = self.shared(k), self.shared(v)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
        self.attention = weights.detach() if self.keep_attention else None
        out = (self.dropout(weights) @ v).transpose(1, 2).reshape(b, t, dim)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        hidden = swiglu_hidden(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(nn.functional.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, groups: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, heads, groups, dropout)
        self.feed_forward = SwiGLU(dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        x = x + self.dropout(self.attention(self.norm1(x), cos, sin))
        return x + self.feed_forward(self.norm2(x))


class TokenClassifier(nn.Module):
    """Shared part: class token, blocks, final norm and head; subclasses make the tokens.

    Sequence order: [class, DEMON, frequency tokens...]; only the frequency tokens are rotated.
    """

    def __init__(self, dim, layers, heads, groups, dropout, n_classes=N_CLASSES):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * INIT_STD)
        self.blocks = nn.ModuleList(
            Block(dim, heads, groups, dropout) for _ in range(layers)
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.head_dim = dim // heads
        for name in ("lofar_mean", "demon_mean"):
            self.register_buffer(name, torch.zeros(()))
        for name in ("lofar_std", "demon_std"):
            self.register_buffer(name, torch.ones(()))

    def set_normalisation(self, lofar: np.ndarray, demon: np.ndarray) -> None:
        self.lofar_mean.fill_(float(lofar.mean()))
        self.lofar_std.fill_(float(lofar.std()))
        self.demon_mean.fill_(float(demon.mean()))
        self.demon_std.fill_(float(demon.std()))

    def tokens(self, lofar, demon) -> tuple[torch.Tensor, torch.Tensor]:
        """(frequency tokens (B, F, dim), DEMON token (B, 1, dim)) of normalised inputs."""
        raise NotImplementedError

    def forward(self, lofar: torch.Tensor, demon: torch.Tensor) -> torch.Tensor:
        lofar = (lofar - self.lofar_mean) / self.lofar_std
        demon = (demon - self.demon_mean) / self.demon_std
        frequency, demon_token = self.tokens(lofar, demon)
        b, n_frequency = frequency.shape[:2]
        x = torch.cat([self.cls_token.expand(b, -1, -1), demon_token, frequency], dim=1)
        positions = torch.cat([torch.zeros(2), torch.arange(n_frequency)]).to(x.device)
        rotate = torch.cat([torch.zeros(2), torch.ones(n_frequency)]).to(x.device)
        cos, sin = rope_tables(positions, rotate, self.head_dim)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm(x[:, 0]))

    def keep_attention(self, keep: bool = True) -> None:
        for block in self.blocks:
            block.attention.keep_attention = keep

    def attention_maps(self) -> list[torch.Tensor]:
        """Attention weights (B, heads, T, T) of every layer in the last forward pass."""
        return [block.attention.attention for block in self.blocks]

    @property
    def tokens_per_segment(self) -> int:
        return 2 + self.n_frequency_tokens


class SonarTransformer(TokenClassifier):
    def __init__(
        self,
        dim=32,
        layers=1,
        heads=2,
        groups=1,
        dropout=0.3,
        lofar_frames=11,
        lofar_bins=1001,
        demon_bins=301,
        n_classes=N_CLASSES,
    ):
        super().__init__(dim, layers, heads, groups, dropout, n_classes)
        self.n_frequency_tokens = lofar_bins // TOKEN_BINS
        self.lofar_embed = nn.Linear(lofar_frames * TOKEN_BINS, dim)
        self.demon_embed = nn.Linear(demon_bins, dim)

    def tokens(self, lofar, demon):
        b, frames, _ = lofar.shape
        n = self.n_frequency_tokens
        bands = lofar[..., : n * TOKEN_BINS].reshape(b, frames, n, TOKEN_BINS)
        bands = bands.permute(0, 2, 1, 3).reshape(b, n, frames * TOKEN_BINS)
        return self.lofar_embed(bands), self.demon_embed(demon)[:, None]


class SonarHybrid(TokenClassifier):
    def __init__(
        self,
        dim=32,
        layers=1,
        heads=2,
        groups=1,
        dropout=0.3,
        lofar_frames=11,
        lofar_bins=1001,
        demon_bins=301,
        n_classes=N_CLASSES,
    ):
        super().__init__(dim, layers, heads, groups, dropout, n_classes)
        c1, c2 = HYBRID_CHANNELS
        d1, d2 = HYBRID_DEMON_CHANNELS
        self.lofar = nn.Sequential(
            conv_block_2d(1, c1, (3, 7), 4), conv_block_2d(c1, c2, (3, 5), 4)
        )
        self.demon = nn.Sequential(
            conv_block_1d(1, d1, 7, 4), conv_block_1d(d1, d2, 5, 4)
        )
        self.n_frequency_tokens = lofar_bins // 4 // 4
        self.lofar_embed = nn.Linear(c2, dim)
        self.demon_embed = nn.Linear(d2 * DEMON_POOLED, dim)

    def tokens(self, lofar, demon):
        a = self.lofar(lofar.unsqueeze(1)).mean(dim=2)  # (B, channels, positions)
        d = pool_to(self.demon(demon.unsqueeze(1)), DEMON_POOLED).flatten(1)
        return self.lofar_embed(a.transpose(1, 2)), self.demon_embed(d)[:, None]

"""
KARL (Kolmogorov-Approximating Representation Learning) — MLX Replication
=========================================================================
Faithful port of the PyTorch reference in karl/ to Apple MLX.

Two training stages:
  Stage 1 — Latent-distillation pretrain  (NLL + code MSE, base tokenizer frozen)
  Stage 2 — Full finetuning               (L1 + LPIPS + GAN, base tokenizer unfrozen)

Usage:
  python karl_mlx.py --stage pretrain  --data_path /path/to/imagenet100
  python karl_mlx.py --stage finetune  --data_path /path/to/imagenet100 \
                     --finetune checkpoint_pretrain.safetensors
"""

import argparse, math, os, time, random, struct
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KARL_CONFIGS = {
    "karl_small": dict(
        encoder_width=1024, encoder_num_layers=8, encoder_num_heads=8,
        decoder_width=1024, decoder_num_layers=8, decoder_num_heads=8,
    ),
    "karl_tiny": dict(
        encoder_width=512, encoder_num_layers=8, encoder_num_heads=8,
        decoder_width=512, decoder_num_layers=8, decoder_num_heads=8,
    ),
}

VQ_DEFAULTS = dict(
    codebook_size=4096, token_dim=12, commitment_cost=0.25, use_l2_norm=True,
)

# Discretized reconstruction-loss bins for conditioning the encoder
REC_LOSS_BINS = mx.array([
    0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
    0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.38,
])
NUM_BINS = 19

ALL_TOKEN_COUNTS = list(range(16, 256 + 128, 16))  # [16, 32, ..., 384]

# VQGAN config from base_tokenizers/configs/vqgan.yaml
VQGAN_CONFIG = dict(
    embed_dim=256, n_embed=1024,
    ddconfig=dict(
        double_z=False, z_channels=256, resolution=256,
        in_channels=3, out_ch=3, ch=128,
        ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
        attn_resolutions=[16], dropout=0.0,
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float = 0.0) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def trunc_normal_(shape, std=0.02):
    """Sample from truncated normal (clipped to ±2*std)."""
    x = mx.random.normal(shape) * std
    return mx.clip(x, -2 * std, 2 * std)


def discretize_loss(loss_vals: mx.array, bins: mx.array, bin_embeds: mx.array,
                    token_add_count=None):
    """Map continuous loss values → nearest-greater bin index, return (binned_loss, binned_embed).
    Includes noise injection matching the reference implementation."""
    loss_exp = mx.expand_dims(mx.stop_gradient(loss_vals), axis=-1)  # (B, S, 1)
    bins_exp = bins.reshape(1, 1, -1)                                # (1, 1, N)
    diffs = bins_exp - loss_exp                                      # (B, S, N)
    diffs = mx.where(diffs >= 0, diffs, mx.array(1e9))
    indices = mx.argmin(diffs, axis=-1)                              # (B, S)

    # Fallback to last bin if no valid bin
    all_invalid = mx.all(diffs >= 1e8, axis=-1)
    indices = mx.where(all_invalid, mx.array(NUM_BINS - 1), indices)

    # Noise injection (reference: 45% chance at boundary conditions)
    if token_add_count is not None and token_add_count == 0:
        if random.random() > 0.55:
            rand = mx.random.uniform(shape=indices.shape)
            indices = 1 + (rand * indices.astype(mx.float32)).astype(mx.int32)
    elif token_add_count is not None and token_add_count == 256 - 32:
        if random.random() > 0.55:
            target = NUM_BINS - 1
            rand = mx.random.uniform(shape=indices.shape)
            new_idx = indices.astype(mx.float32) + rand * (target - indices.astype(mx.float32))
            indices = mx.minimum(new_idx.astype(mx.int32), mx.array(target))

    flat_idx = indices.reshape(-1)
    binned_loss = bins[flat_idx].reshape(indices.shape)
    binned_embed = bin_embeds[flat_idx].reshape(*indices.shape, -1)
    return binned_loss, binned_embed


# ===================================================================
# Vector Quantizer (fixed: codebook receives gradients)
# ===================================================================

class VectorQuantizer(nn.Module):
    """Codebook quantisation with straight-through estimator.
    
    Matches reference: commitment_loss = β * mean((z_q.detach() - z)²)
                       codebook_loss   = mean((z_q - z.detach())²)
    z_q is NOT detached in codebook_loss so gradients flow to embedding.
    """

    def __init__(self, codebook_size: int = 4096, token_dim: int = 12,
                 commitment_cost: float = 0.25, use_l2_norm: bool = True):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        scale = 1.0 / codebook_size
        self.embedding = mx.random.uniform(-scale, scale, (codebook_size, token_dim))

    def __call__(self, z: mx.array):
        B, S, D = z.shape
        z_flat = z.reshape(B * S, D)

        if self.use_l2_norm:
            z_norm = z / (mx.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            z_flat_n = z_norm.reshape(B * S, D)
            emb_n = self.embedding / (mx.linalg.norm(self.embedding, axis=-1, keepdims=True) + 1e-8)
        else:
            z_norm = z
            z_flat_n = z_flat
            emb_n = self.embedding

        # Nearest-neighbour lookup (detach inputs for argmin only)
        z_det = mx.stop_gradient(z_flat_n)
        emb_det = mx.stop_gradient(emb_n)
        d = (mx.sum(z_det ** 2, axis=1, keepdims=True)
             + mx.sum(emb_det ** 2, axis=1).reshape(1, -1)
             - 2.0 * z_det @ emb_det.T)
        indices = mx.argmin(d, axis=1)

        # z_q with gradients flowing to self.embedding (NOT stop_gradient)
        z_q = emb_n[indices].reshape(B, S, D)

        # Losses: commitment pulls encoder toward codebook, codebook pulls codebook toward encoder
        commitment = self.commitment_cost * mx.mean((mx.stop_gradient(z_q) - z_norm) ** 2)
        codebook = mx.mean((z_q - mx.stop_gradient(z_norm)) ** 2)
        loss = commitment + codebook

        # Straight-through: gradient flows through z_norm, not through the lookup
        z_q_st = z_norm + mx.stop_gradient(z_q - z_norm)
        return z_q_st, loss, indices.reshape(B, S)


# ===================================================================
# Transformer building blocks
# ===================================================================

class ResidualAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN block."""

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_width),
            nn.GELU(),
            nn.Linear(mlp_width, d_model),
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        h = self.ln_1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x


class LatentDistillationEncoder(nn.Module):
    def __init__(self, width: int, num_layers: int, num_heads: int):
        super().__init__()
        self.layers = [ResidualAttentionBlock(width, num_heads) for _ in range(num_layers)]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class LatentDistillationDecoder(nn.Module):
    def __init__(self, width: int, num_layers: int, num_heads: int, output_dim: int):
        super().__init__()
        self.ln_pre = nn.LayerNorm(width)
        self.layers = [ResidualAttentionBlock(width, num_heads) for _ in range(num_layers)]
        self.ln_post = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.Tanh(), nn.Linear(2 * width, output_dim),
        )

    def __call__(self, latent_1d: mx.array, masked_2d: mx.array,
                 pos_embed: mx.array, mask: Optional[mx.array] = None):
        latent_1d = latent_1d + mx.expand_dims(pos_embed, 0)
        x = mx.concatenate([masked_2d, latent_1d], axis=1)
        x = self.ln_pre(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        out_2d = x[:, :masked_2d.shape[1]]
        out_2d = self.ffn(self.ln_post(out_2d))
        return out_2d


# ===================================================================
# VQGAN Base Tokenizer (ported from taming-transformers)
# ===================================================================
# Architecture: Encoder/Decoder with ResNet blocks, GroupNorm, attention
# at resolution 16. Config: ch=128, ch_mult=[1,1,2,2,4], 2 res blocks,
# z_channels=256, n_embed=1024, embed_dim=256.

def _swish(x):
    return x * mx.sigmoid(x)


class GroupNorm32(nn.Module):
    """GroupNorm with 32 groups, matching taming's Normalize()."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, num_channels)
    def __call__(self, x):
        return self.gn(x)


class VQGANResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = GroupNorm32(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        else:
            self.nin_shortcut = None

    def __call__(self, x):
        h = _swish(self.norm1(x))
        h = self.conv1(h)
        h = _swish(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class VQGANAttnBlock(nn.Module):
    """Spatial self-attention block used at attn_resolutions (e.g. 16)."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = GroupNorm32(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def __call__(self, x):
        h = self.norm(x)
        B, H, W, C = h.shape
        q = self.q(h).reshape(B, H * W, C)
        k = self.k(h).reshape(B, H * W, C)
        v = self.v(h).reshape(B, H * W, C)
        attn = (q @ mx.transpose(k, (0, 2, 1))) * (C ** -0.5)
        attn = mx.softmax(attn, axis=-1)
        h = (attn @ v).reshape(B, H, W, C)
        h = self.proj_out(h)
        return x + h


class VQGANDownsample(nn.Module):
    """Downsample with stride-2 conv and asymmetric padding."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def __call__(self, x):
        # Asymmetric padding: pad right and bottom by 1
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class VQGANUpsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def __call__(self, x):
        B, H, W, C = x.shape
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(x)


class VQGANEncoder(nn.Module):
    """VQGAN encoder: ch=128, ch_mult=[1,1,2,2,4], 2 res blocks, attn@16, avg_pool downsample."""
    def __init__(self, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 z_channels=256, in_channels=3, double_z=False, dropout=0.0,
                 attn_resolutions=(16,), resolution=256, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1, bias=False)

        self.down = []
        block_in = ch
        curr_res = resolution
        for i_level in range(self.num_resolutions):
            blocks = []
            attn_blocks = []
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                blocks.append(VQGANResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_blocks.append(VQGANAttnBlock(block_in))
            level = {"block": blocks, "attn": attn_blocks}
            if i_level != self.num_resolutions - 1:
                level["downsample"] = True  # use avg_pool
                curr_res = curr_res // 2
            self.down.append(level)

        self.mid_block_1 = VQGANResnetBlock(block_in, block_in, dropout=dropout)
        self.mid_attn = VQGANAttnBlock(block_in)
        self.mid_block_2 = VQGANResnetBlock(block_in, block_in, dropout=dropout)
        self.norm_out = GroupNorm32(block_in)
        out_ch = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

    def __call__(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level]["block"][i_block](h)
                if i_block < len(self.down[i_level]["attn"]):
                    h = self.down[i_level]["attn"][i_block](h)
            if self.down[i_level].get("downsample"):
                # avg_pool2d with kernel_size=2, stride=2 (reference: resamp_with_conv=False)
                B, H, W, C = h.shape
                h = h.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)
        h = _swish(self.norm_out(h))
        h = self.conv_out(h)
        return h


class VQGANDecoder(nn.Module):
    """VQGAN decoder: mirrors encoder with upsampling + attn@16."""
    def __init__(self, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 z_channels=256, out_ch=3, dropout=0.0,
                 attn_resolutions=(16,), resolution=256, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)
        self.mid_block_1 = VQGANResnetBlock(block_in, block_in, dropout=dropout)
        self.mid_attn = VQGANAttnBlock(block_in)
        self.mid_block_2 = VQGANResnetBlock(block_in, block_in, dropout=dropout)

        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            blocks = []
            attn_blocks = []
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                blocks.append(VQGANResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_blocks.append(VQGANAttnBlock(block_in))
            level = {"block": blocks, "attn": attn_blocks}
            if i_level != 0:
                level["upsample"] = VQGANUpsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, level)

        self.norm_out = GroupNorm32(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

    def __call__(self, z):
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level]["block"][i_block](h)
                if i_block < len(self.up[i_level]["attn"]):
                    h = self.up[i_level]["attn"][i_block](h)
            if "upsample" in self.up[i_level]:
                h = self.up[i_level]["upsample"](h)
        h = _swish(self.norm_out(h))
        h = self.conv_out(h)
        return h


class VQGANQuantize(nn.Module):
    """VQGAN codebook (separate from KARL's VQ). n_embed=1024, embed_dim=256."""
    def __init__(self, n_embed: int = 1024, embed_dim: int = 256, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_embed, embed_dim)

    def __call__(self, z):
        # z: (B, H, W, embed_dim) in MLX channel-last
        B, H, W, C = z.shape
        z_flat = z.reshape(-1, C)
        d = (mx.sum(z_flat ** 2, axis=1, keepdims=True)
             + mx.sum(self.embedding.weight ** 2, axis=1).reshape(1, -1)
             - 2.0 * z_flat @ self.embedding.weight.T)
        indices = mx.argmin(d, axis=1)
        z_q = self.embedding(indices).reshape(B, H, W, C)
        loss = self.beta * mx.mean((mx.stop_gradient(z_q) - z) ** 2) + mx.mean((z_q - mx.stop_gradient(z)) ** 2)
        z_q_st = z + mx.stop_gradient(z_q - z)
        return z_q_st, loss, indices.reshape(B, H * W)


class VQGANBaseTokenizer(nn.Module):
    """Full VQGAN base tokenizer wrapping encoder + quantizer + decoder.
    
    In pretrain stage: frozen (no gradients).
    In finetune stage: unfrozen.
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config or VQGAN_CONFIG
        dd = cfg["ddconfig"]
        self.embed_dim = cfg["embed_dim"]       # 256
        self.codebook_size = cfg["n_embed"]      # 1024
        self.encoder = VQGANEncoder(**dd)
        self.decoder = VQGANDecoder(**dd)
        self.quantize = VQGANQuantize(cfg["n_embed"], cfg["embed_dim"])

    def encode(self, x):
        """Returns (z_pre_quant, z_quantized, quant_loss, indices)."""
        z = self.encoder(x)
        z_q, loss, indices = self.quantize(z)
        return z, z_q, loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def get_img_tokens(self, imgs):
        """Returns (vqgan_tokens (B,H,W,embed_dim), gt_indices (B, HW))."""
        z, z_q, _, indices = self.encode(imgs)
        return z, indices


# ===================================================================
# PatchGAN Discriminator (from modules/losses/discriminator.py)
# ===================================================================

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator: input_nc=3, ndf=64, n_layers=3, BatchNorm."""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw, padw = 4, 1
        layers = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=True),
                nn.BatchNorm(ndf * nf_mult),
                nn.LeakyReLU(0.2),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=True),
            nn.BatchNorm(ndf * nf_mult),
            nn.LeakyReLU(0.2),
        ]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*layers)

    def __call__(self, x):
        return self.main(x)


# ===================================================================
# LPIPS Perceptual Loss (simplified VGG-based)
# ===================================================================
# For a full-fidelity LPIPS, load pretrained VGG16 weights.
# This provides the architecture; weights can be loaded from the
# reference checkpoint or torchvision export.

class VGG16Features(nn.Module):
    """VGG16 feature extractor for LPIPS. 5 slices matching the reference."""
    def __init__(self):
        super().__init__()
        # Slice 1: conv1_1, relu, conv1_2, relu (64 channels)
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Slice 2: pool, conv2_1, relu, conv2_2, relu (128 channels)
        self.slice2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        # Slice 3: pool, conv3_1-3, relu (256 channels)
        self.slice3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        # Slice 4: pool, conv4_1-3, relu (512 channels)
        self.slice4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )
        # Slice 5: pool, conv5_1-3, relu (512 channels)
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )

    def __call__(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1):
        super().__init__()
        self.model = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(chn_in, chn_out, 1, padding=0, bias=False))
    def __call__(self, x):
        return self.model(x)


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity."""
    def __init__(self):
        super().__init__()
        self.net = VGG16Features()
        chns = [64, 128, 256, 512, 512]
        self.lins = [NetLinLayer(c) for c in chns]
        # Scaling layer (ImageNet normalization)
        self.shift = mx.array([-0.030, -0.088, -0.188]).reshape(1, 1, 1, 3)
        self.scale = mx.array([0.458, 0.448, 0.450]).reshape(1, 1, 1, 3)

    def __call__(self, x, y):
        x = (x - self.shift) / self.scale
        y = (y - self.shift) / self.scale
        feats_x = self.net(x)
        feats_y = self.net(y)
        loss = mx.array(0.0)
        for i, lin in enumerate(self.lins):
            fx = feats_x[i] / (mx.linalg.norm(feats_x[i], axis=-1, keepdims=True) + 1e-10)
            fy = feats_y[i] / (mx.linalg.norm(feats_y[i], axis=-1, keepdims=True) + 1e-10)
            diff = (fx - fy) ** 2
            val = lin(diff)
            loss = loss + mx.mean(val)
        return loss


# ===================================================================
# Label-Smoothing Cross-Entropy (from modules/losses/nll.py)
# ===================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def __call__(self, logits: mx.array, targets: mx.array):
        """logits: (N, C), targets: (N,) long indices. Returns (nll_loss, smooth_loss) per sample."""
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
        # Gather NLL
        nll = -mx.take_along_axis(log_probs, mx.expand_dims(targets, 1), axis=1).squeeze(1)
        smooth = -mx.mean(log_probs, axis=-1)
        return nll, smooth


# ===================================================================
# KARL Tokenizer (main model)
# ===================================================================

class KARLTokenizer(nn.Module):
    """
    Single-pass adaptive image tokenizer.

    Forward pass (training) executes two phases per iteration:
      1. Estimate Image Complexity (EIC) — sample ε from decaying curriculum,
         attempt compression with random token budget T.
      2. Learn to Tokenize Complexity (LTC) — given T2=256 tokens conditioned
         on the EIC reconstruction error, learn to halt the extra ΔT tokens.
    """

    def __init__(self, cfg_name: str = "karl_small",
                 quantize_latent: bool = True,
                 patch_size: int = 16,
                 max_grid: int = 64,
                 max_latent_tokens: int = 512,
                 vq_codebook_size: int = 4096,
                 vq_token_dim: int = 12,
                 vq_commitment_cost: float = 0.25,
                 vq_use_l2_norm: bool = True,
                 train_stage: str = "pretrain"):
        super().__init__()
        cfg = KARL_CONFIGS[cfg_name]
        ew = cfg["encoder_width"]
        dw = cfg["decoder_width"]
        self.encoder_width = ew
        self.decoder_width = dw
        self.quantize_latent = quantize_latent
        self.vq_token_dim = vq_token_dim
        self.train_stage = train_stage
        scale_e = ew ** -0.5
        scale_d = dw ** -0.5

        # --- Base 2D tokenizer (VQGAN) ---
        self.base_tokenizer = VQGANBaseTokenizer()
        self.base_dim = self.base_tokenizer.embed_dim  # 256

        # --- Patch embedding (extra channels beyond base_dim) ---
        self.patch_embed = nn.Conv2d(3, ew - self.base_dim, patch_size, stride=patch_size)

        # --- Positional embeddings ---
        self.enc_pos = trunc_normal_((ew, max_grid, max_grid), std=scale_e)
        self.dec_pos = trunc_normal_((dw, max_grid, max_grid), std=scale_d)

        # --- Learnable tokens ---
        self.latent_tokens = trunc_normal_((max_latent_tokens, ew), std=scale_e)
        self.dec_mask_token = trunc_normal_((1, 1, dw), std=scale_d)
        self.dec_timestep_embed = trunc_normal_((max_latent_tokens, dw), std=scale_d)

        # --- Encoder pre/post norms ---
        self.enc_ln_pre = nn.LayerNorm(ew)
        self.enc_ln_post = nn.LayerNorm(ew)
        self.enc_ln_post_halt = nn.LayerNorm(ew)

        # --- Halting MLP ---
        self.halt_mlp = nn.Sequential(nn.Linear(ew, 512), nn.Tanh(), nn.Linear(512, 1))

        # --- Pre-quantizer projection ---
        self.pre_quant = nn.Linear(ew, vq_token_dim)

        # --- Decoder input projection ---
        self.dec_embed = nn.Linear(vq_token_dim, dw)

        # --- Encoder / Decoder transformers ---
        self.encoder = LatentDistillationEncoder(ew, cfg["encoder_num_layers"], cfg["encoder_num_heads"])
        # Decoder output_dim = base_tokenizer.codebook_size (1024 for VQGAN)
        self.decoder = LatentDistillationDecoder(dw, cfg["decoder_num_layers"], cfg["decoder_num_heads"],
                                                  output_dim=self.base_tokenizer.codebook_size)

        # --- VQ ---
        if quantize_latent:
            self.vq = VectorQuantizer(vq_codebook_size, vq_token_dim, vq_commitment_cost, vq_use_l2_norm)

        # --- Loss-conditioning embeddings (learnable parameter) ---
        # Reference: nn.Parameter(rec_losses[:, None].repeat(1, encoder_width), requires_grad=True)
        self.rec_loss_embeds = REC_LOSS_BINS[:, None] * mx.ones((1, ew))

        # --- ε-sampling curriculum for EIC ---
        self._eps_prob = np.zeros(NUM_BINS, dtype=np.float64)
        self._eps_prob[REC_LOSS_BINS.tolist() <= np.array([0.04])] = 1.0
        if self._eps_prob.sum() > 0:
            self._eps_prob /= self._eps_prob.sum()

        # --- Stage-specific losses ---
        if train_stage == "pretrain":
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        elif train_stage == "finetune":
            self.lpips = LPIPSLoss()
            # 8 PatchGAN discriminators indexed by token count
            self.discriminators = [NLayerDiscriminator() for _ in range(8)]

        # --- Apply weight initialization ---
        self._init_all_weights()

    def _init_all_weights(self):
        """Truncated normal init for Linear/Conv, ones/zeros for LayerNorm."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight = trunc_normal_(module.weight.shape, std=0.02)
                if "bias" in module:
                    module.bias = mx.zeros(module.bias.shape)
            elif isinstance(module, nn.Conv2d):
                module.weight = trunc_normal_(module.weight.shape, std=0.02)
                if "bias" in module:
                    module.bias = mx.zeros(module.bias.shape)
            elif isinstance(module, nn.LayerNorm):
                module.weight = mx.ones(module.weight.shape)
                module.bias = mx.zeros(module.bias.shape)

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _sample_pos_2d(self, pos: mx.array, grid_h: int, grid_w: int) -> mx.array:
        """Bilinear-interpolate 2D positional embedding to (grid_h, grid_w)."""
        # pos shape: (C, Hmax, Wmax)
        # For default 256x256 with patch_size=16, grid is 16x16 which fits in max_grid=64
        p = pos[:, :grid_h, :grid_w]
        return p.reshape(p.shape[0], -1).T.reshape(1, -1, p.shape[0])

    def _get_2d_tokens(self, imgs: mx.array):
        """Return (img_tokens (B, HW, ew), gt_indices (B, HW), grid_h, grid_w)."""
        # Get VQGAN tokens
        vqgan_z, gt_indices = self.base_tokenizer.get_img_tokens(imgs)
        # vqgan_z: (B, H, W, embed_dim=256)
        patch_out = self.patch_embed(imgs)  # (B, H, W, ew-base_dim)
        grid_h, grid_w = vqgan_z.shape[1], vqgan_z.shape[2]
        vqgan_flat = vqgan_z.reshape(vqgan_z.shape[0], -1, self.base_dim)
        patch_flat = patch_out.reshape(patch_out.shape[0], -1, self.encoder_width - self.base_dim)
        img_tokens = mx.concatenate([patch_flat, vqgan_flat], axis=-1)
        return img_tokens, gt_indices, grid_h, grid_w

    def _get_masked_2d(self, B: int, num_2d: int, grid_h: int, grid_w: int) -> mx.array:
        mask = mx.broadcast_to(self.dec_mask_token, (B, num_2d, self.decoder_width))
        pos = self._sample_pos_2d(self.dec_pos, grid_h, grid_w)
        return mask + pos

    def _encode_phase(self, imgs, img_tokens, grid_h, grid_w, num_tokens, loss_embed):
        """Run encoder. Returns (latent_factorized, halt_logits, halt_prob)."""
        B = imgs.shape[0]
        enc_pos = self._sample_pos_2d(self.enc_pos, grid_h, grid_w)
        x = img_tokens + enc_pos
        lat = mx.broadcast_to(self.latent_tokens[:num_tokens], (B, num_tokens, self.encoder_width))
        x = mx.concatenate([x, lat], axis=1)
        cond = mx.expand_dims(loss_embed, 1)
        x = mx.concatenate([cond, x], axis=1)
        x = self.enc_ln_pre(x)
        x = self.encoder(x)
        latent_out = x[:, 1 + img_tokens.shape[1]:]
        halt_logits = self.halt_mlp(self.enc_ln_post_halt(latent_out))
        halt_prob = mx.sigmoid(halt_logits)
        latent_fact = self.pre_quant(self.enc_ln_post(latent_out))
        return latent_fact, halt_logits, halt_prob

    def _build_decoder_attn_mask(self, halt_prob, num_2d_tokens, num_latent_tokens, halting_threshold=0.75):
        """Build symmetric attention mask for decoder where halted tokens can't attend.
        
        Decoder sequence = [masked_2d (HW tokens), latent_1d (T tokens)].
        Image tokens (2D) are never masked. Halted latent tokens are masked.
        Returns additive float mask: 0 = attend, -inf = block.
        """
        B = halt_prob.shape[0]
        num_all = num_2d_tokens + num_latent_tokens
        # 2D tokens are never masked
        img_mask = mx.zeros((B, num_2d_tokens), dtype=mx.bool_)
        lat_mask = (halt_prob[..., 0] > halting_threshold)  # (B, num_latent)
        combined = mx.concatenate([img_mask, lat_mask], axis=1)  # (B, num_all)
        # Symmetric: if either token is halted, mask the pair
        blocked = mx.expand_dims(combined, 2) | mx.expand_dims(combined, 1)  # (B, num_all, num_all)
        # Diagonal always unmasked
        diag = mx.eye(num_all, dtype=mx.bool_)
        blocked = blocked & ~diag
        # Convert to additive mask: 0 where allowed, -1e9 where blocked
        # Shape (B, 1, num_all, num_all) to broadcast over heads
        attn_mask = mx.where(blocked, mx.array(-1e9), mx.array(0.0))
        return mx.expand_dims(attn_mask, axis=1)

    def _decode_phase(self, latent_q, masked_2d, num_tokens, mask=None):
        """Run decoder, return decoded logits (B, HW, codebook_size)."""
        dec_in = self.dec_embed(latent_q)
        pos = self.dec_timestep_embed[:num_tokens]
        return self.decoder(dec_in, masked_2d, pos, mask=mask)

    def _decode_to_image(self, decoded_logits, grid_h, grid_w):
        """Softmax logits → weighted sum of VQGAN codebook → VQGAN decode → image."""
        probs = mx.softmax(decoded_logits, axis=-1)  # (B, HW, codebook_size)
        # Weighted sum of codebook embeddings
        codebook_weights = mx.stop_gradient(self.base_tokenizer.quantize.embedding.weight)  # (1024, 256)
        decoded_code = probs @ codebook_weights  # (B, HW, 256)
        B = decoded_code.shape[0]
        code_spatial = decoded_code.reshape(B, grid_h, grid_w, self.base_dim)
        return self.base_tokenizer.decode(code_spatial), decoded_code

    # ---------------------------------------------------------------
    # Forward (training): EIC + LTC
    # ---------------------------------------------------------------

    def __call__(self, imgs: mx.array, epoch: int = 0, gan_optimizer_idx: int = 0,
                 gan_loss_weight: float = 0.0):
        """Full training forward: returns scalar loss and a dict of logged metrics."""
        B = imgs.shape[0]
        img_tokens, gt_indices, grid_h, grid_w = self._get_2d_tokens(imgs)
        masked_2d = self._get_masked_2d(B, img_tokens.shape[1], grid_h, grid_w)
        logs = {}
        total_loss = mx.array(0.0)

        # --- Phase 1: Estimate Image Complexity (EIC) ---
        T = random.choice(ALL_TOKEN_COUNTS)
        T = min(T, 256)

        # ε-sampling curriculum: sample from decaying distribution
        if self._eps_prob.sum() > 0:
            eps_idx = np.random.choice(NUM_BINS, p=self._eps_prob)
        else:
            eps_idx = 0
        eps_embed = mx.broadcast_to(
            mx.expand_dims(self.rec_loss_embeds[eps_idx], 0), (B, self.encoder_width)
        )

        lat_f1, halt_log1, halt_p1 = self._encode_phase(
            imgs, img_tokens, grid_h, grid_w, T, eps_embed)

        if self.quantize_latent:
            lat_q1, vq_loss1, vq_indices1 = self.vq(lat_f1)
        else:
            lat_q1, vq_loss1, vq_indices1 = lat_f1, mx.array(0.0), None

        # EIC: no halting mask (all tokens active)
        decoded_logits1 = self._decode_phase(lat_q1, masked_2d, T)

        # Halting: all tokens should be active → target 0
        halt_active1 = nn.losses.binary_cross_entropy(
            halt_log1, mx.zeros_like(halt_p1), reduction="mean")
        total_loss = total_loss + halt_active1

        # --- Reconstruction losses ---
        if self.train_stage == "pretrain":
            nll1, smooth1 = self.criterion(
                decoded_logits1[:, :, :self.base_tokenizer.codebook_size].reshape(-1, self.base_tokenizer.codebook_size),
                gt_indices.reshape(-1))
            nll_loss1 = mx.mean(nll1)

            # Decoded code MSE
            probs1 = mx.softmax(decoded_logits1, axis=-1)
            decoded_code1 = probs1 @ self.base_tokenizer.quantize.embedding.weight
            gt_code1 = self.base_tokenizer.quantize.embedding.weight[gt_indices.reshape(-1)].reshape(B, -1, self.base_dim)
            code_loss1 = mx.mean((gt_code1 - decoded_code1) ** 2)

            total_loss = total_loss + nll_loss1 + code_loss1
            logs["eic_nll"] = nll_loss1.item()
            logs["eic_code"] = code_loss1.item()

        elif self.train_stage == "finetune":
            recon_img1, decoded_code1 = self._decode_to_image(decoded_logits1, grid_h, grid_w)
            l1_loss1 = mx.mean(mx.abs(imgs - recon_img1))
            lpips_loss1 = self.lpips(imgs, recon_img1)
            total_loss = total_loss + l1_loss1 + lpips_loss1
            logs["eic_l1"] = l1_loss1.item()

        if self.quantize_latent:
            total_loss = total_loss + vq_loss1
            logs["vq_loss1"] = vq_loss1.item()

        # Compute pixel-level L1 for conditioning phase 2 (detached)
        if self.train_stage == "pretrain":
            probs1_det = mx.softmax(mx.stop_gradient(decoded_logits1), axis=-1)
            dec_code1_det = probs1_det @ mx.stop_gradient(self.base_tokenizer.quantize.embedding.weight)
            code_spatial = dec_code1_det.reshape(B, grid_h, grid_w, self.base_dim)
            recon_img1_det = mx.stop_gradient(self.base_tokenizer.decode(code_spatial))
        else:
            recon_img1_det = mx.stop_gradient(recon_img1)
        # Per-image L1 (B,) — reference: reshape(B, -1).mean(dim=-1)
        per_img_l1 = mx.stop_gradient(
            mx.mean(mx.abs(imgs - recon_img1_det).reshape(B, -1), axis=-1))  # (B,)

        # Update ε-sampling curriculum using min across batch (reference: iter_rec_loss.min())
        eps0_min = mx.min(per_img_l1).item()
        rec_losses_np = np.array(REC_LOSS_BINS.tolist())
        self._eps_prob[rec_losses_np >= eps0_min] *= 0.99
        if self._eps_prob.sum() > 0:
            self._eps_prob /= self._eps_prob.sum()

        # --- Phase 2: Learn to Tokenize Complexity (LTC) ---
        T2 = 256
        delta_T = max(0, T2 - T)

        # Condition per-image on its own reconstruction error from phase 1
        eps0_for_bin = mx.expand_dims(per_img_l1, 1)  # (B, 1)
        _, cond_embed = discretize_loss(eps0_for_bin, REC_LOSS_BINS, self.rec_loss_embeds,
                                        token_add_count=delta_T)
        cond_embed = cond_embed[:, 0, :]

        lat_f2, halt_log2, halt_p2 = self._encode_phase(
            imgs, img_tokens, grid_h, grid_w, T2, cond_embed)

        if self.quantize_latent:
            lat_q2, vq_loss2, vq_indices2 = self.vq(lat_f2)
        else:
            lat_q2, vq_loss2, vq_indices2 = lat_f2, mx.array(0.0), None

        # Build decoder attention mask for halted tokens
        decoder_mask = self._build_decoder_attn_mask(halt_p2, num_2d_tokens=img_tokens.shape[1],
                                                      num_latent_tokens=T2)
        decoded_logits2 = self._decode_phase(lat_q2, masked_2d, T2, mask=decoder_mask)

        # Halting loss: first T tokens → keep (0), last delta_T → halt (1)
        if delta_T > 0:
            halt_active2 = nn.losses.binary_cross_entropy(
                halt_log2[:, :T, :], mx.zeros_like(halt_p2[:, :T, :]), reduction="mean")
            halt_stop2 = nn.losses.binary_cross_entropy(
                halt_log2[:, T:, :], mx.ones_like(halt_p2[:, T:, :]), reduction="mean")
            halt_loss2 = halt_active2 + halt_stop2
        else:
            halt_loss2 = nn.losses.binary_cross_entropy(
                halt_log2, mx.zeros_like(halt_p2), reduction="mean")
        total_loss = total_loss + halt_loss2

        # --- LTC Reconstruction losses ---
        if self.train_stage == "pretrain":
            nll2, _ = self.criterion(
                decoded_logits2[:, :, :self.base_tokenizer.codebook_size].reshape(-1, self.base_tokenizer.codebook_size),
                gt_indices.reshape(-1))
            nll_loss2 = mx.mean(nll2)
            probs2 = mx.softmax(decoded_logits2, axis=-1)
            decoded_code2 = probs2 @ self.base_tokenizer.quantize.embedding.weight
            gt_code2 = self.base_tokenizer.quantize.embedding.weight[gt_indices.reshape(-1)].reshape(B, -1, self.base_dim)
            code_loss2 = mx.mean((gt_code2 - decoded_code2) ** 2)
            total_loss = total_loss + nll_loss2 + code_loss2
            logs["ltc_nll"] = nll_loss2.item()
            logs["ltc_code"] = code_loss2.item()

        elif self.train_stage == "finetune":
            recon_img2, decoded_code2 = self._decode_to_image(decoded_logits2, grid_h, grid_w)
            l1_loss2 = mx.mean(mx.abs(imgs - recon_img2))
            lpips_loss2 = self.lpips(imgs, recon_img2)
            recon_loss2 = l1_loss2 + lpips_loss2

            # GAN loss
            if gan_loss_weight > 0 and gan_optimizer_idx == 0:
                disc_idx = max(0, min(7, (min(256, 256 - delta_T) // 32) - 1))
                logits_fake = self.discriminators[disc_idx](recon_img2)
                g_loss = -mx.mean(logits_fake)
                total_loss = total_loss + recon_loss2 + gan_loss_weight * g_loss
                logs["g_loss"] = g_loss.item()
            elif gan_optimizer_idx == 1:
                # Discriminator update
                disc_idx = max(0, min(7, (min(256, 256 - delta_T) // 32) - 1))
                logits_real = self.discriminators[disc_idx](mx.stop_gradient(imgs))
                logits_fake = self.discriminators[disc_idx](mx.stop_gradient(recon_img2))
                d_real = mx.mean(nn.relu(1.0 - logits_real))
                d_fake = mx.mean(nn.relu(1.0 + logits_fake))
                d_loss = 0.5 * (d_real + d_fake)
                total_loss = total_loss + d_loss
                logs["d_loss"] = d_loss.item()
            else:
                total_loss = total_loss + recon_loss2
            logs["ltc_l1"] = l1_loss2.item()

        if self.quantize_latent:
            total_loss = total_loss + vq_loss2
            logs["vq_loss2"] = vq_loss2.item()

        logs["halt_loss"] = halt_loss2.item()
        logs["eps0"] = eps0_min
        logs["T"] = T
        return total_loss, logs

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    def encode(self, imgs: mx.array, token_budget: int = 256,
               desired_quality: float = 0.05):
        """Encode images with adaptive halting. Returns (reconstruction, active_count)."""
        B = imgs.shape[0]
        img_tokens, gt_indices, grid_h, grid_w = self._get_2d_tokens(imgs)
        masked_2d = self._get_masked_2d(B, img_tokens.shape[1], grid_h, grid_w)

        quality_arr = mx.full((B, 1), desired_quality)
        _, cond_embed = discretize_loss(quality_arr, REC_LOSS_BINS, self.rec_loss_embeds)
        cond_embed = cond_embed[:, 0, :]

        lat_f, _, halt_p = self._encode_phase(
            imgs, img_tokens, grid_h, grid_w, token_budget, cond_embed)

        if self.quantize_latent:
            lat_q, _, _ = self.vq(lat_f)
        else:
            lat_q = lat_f

        # Build decoder mask for halted tokens
        decoder_mask = self._build_decoder_attn_mask(halt_p, num_2d_tokens=img_tokens.shape[1],
                                                      num_latent_tokens=token_budget)
        decoded_logits = self._decode_phase(lat_q, masked_2d, token_budget, mask=decoder_mask)
        recon_img, _ = self._decode_to_image(decoded_logits, grid_h, grid_w)
        active = mx.sum((halt_p < 0.75).astype(mx.float32), axis=(1, 2))
        return recon_img, active


# ===================================================================
# Data loading with augmentation (RandomResizedCrop + HorizontalFlip)
# ===================================================================

def load_image_augmented(path: str, size: int = 256, augment: bool = True) -> Optional[np.ndarray]:
    """Load, augment (RandomResizedCrop + HorizontalFlip), return (H, W, 3) float32 in [0, 1]."""
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB")
        img.load()
    except Exception:
        return None

    if augment:
        # RandomResizedCrop(size, scale=(0.8, 1.0))
        w, h = img.size
        scale = random.uniform(0.8, 1.0)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        img = img.resize((size, size), Image.BILINEAR)
        # RandomHorizontalFlip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        img = img.resize((size, size), Image.BILINEAR)

    return np.asarray(img, dtype=np.float32) / 255.0


def image_folder_paths(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if Path(f).suffix in exts:
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths


def make_batch(paths: List[str], batch_size: int, size: int = 256, augment: bool = True) -> mx.array:
    chosen = random.sample(paths, min(batch_size, len(paths)))
    imgs = [load_image_augmented(p, size, augment=augment) for p in chosen]
    imgs = [x for x in imgs if x is not None]
    if not imgs:
        raise RuntimeError("All images in batch were unreadable")
    return mx.array(np.stack(imgs))


# ===================================================================
# Weight decay exclusion (matches reference add_weight_decay)
# ===================================================================

def split_params_for_weight_decay(model, weight_decay):
    """Exclude 1D params (bias, norm), and diffloss from weight decay."""
    decay_params = {}
    no_decay_params = {}
    for name, param in model.trainable_parameters():
        flat_name = name.replace(".", "/")
        if len(param.shape) == 1 or "bias" in name or "diffloss" in name:
            no_decay_params[flat_name] = param
        else:
            decay_params[flat_name] = param
    return decay_params, no_decay_params


# ===================================================================
# VQGAN weight conversion (PyTorch → MLX)
# ===================================================================

def convert_vqgan_weights(pt_path: str):
    """Convert a PyTorch VQGAN checkpoint to MLX-compatible weight dict.
    
    Handles channel-first (PyTorch) → channel-last (MLX) for Conv2d weights,
    and remaps key names from taming-transformers to our MLX architecture.
    Returns a dict suitable for model.base_tokenizer.load_weights(strict=False).
    """
    try:
        import torch
        sd = torch.load(pt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
    except ImportError:
        print("[KARL-MLX] PyTorch not available for weight conversion. Using random VQGAN weights.")
        return None

    # Filter out keys we don't need
    skip_prefixes = ("loss.", "quant_conv.", "post_quant_conv.")
    
    mlx_weights = {}
    for key, val in sd.items():
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        # Encoder uses avg_pool, no downsample conv in MLX model
        if "encoder.down" in key and "downsample" in key:
            continue
        
        # Remap mid block names: mid.block_1 -> mid_block_1, mid.attn_1 -> mid_attn, mid.block_2 -> mid_block_2
        new_key = key
        new_key = new_key.replace(".mid.block_1.", ".mid_block_1.")
        new_key = new_key.replace(".mid.block_2.", ".mid_block_2.")
        new_key = new_key.replace(".mid.attn_1.", ".mid_attn.")
        
        # Remap GroupNorm: norm1.weight -> norm1.gn.weight, norm_out.weight -> norm_out.gn.weight
        import re
        new_key = re.sub(r'\.(norm\w*)\.(weight|bias)$', r'.\1.gn.\2', new_key)
        
        # Skip bias keys for conv layers that use bias=False in MLX
        # (conv_in, conv1, conv2, nin_shortcut in ResnetBlocks, downsample convs)
        # But keep bias for attn convs (q, k, v, proj_out) and decoder conv_in/conv_out
        np_val = val.numpy()
        
        # Conv2d weights: PyTorch (out, in, H, W) → MLX (out, H, W, in)
        if "weight" in new_key and len(np_val.shape) == 4:
            np_val = np.transpose(np_val, (0, 2, 3, 1))
        
        mlx_weights[new_key] = mx.array(np_val)
    
    return mlx_weights


# ===================================================================
# Training loop
# ===================================================================

def train(args):
    print(f"[KARL-MLX] Stage: {args.stage} | Model: {args.model}")
    print(f"[KARL-MLX] Data:  {args.data_path}")

    # --- Build model ---
    model = KARLTokenizer(
        cfg_name=args.model,
        quantize_latent=args.quantize_latent,
        train_stage=args.stage,
    )

    # --- Load VQGAN pretrained weights if available ---
    vqgan_ckpt = args.vqgan_ckpt
    if not vqgan_ckpt:
        vqgan_ckpt = "vqgan_imagenet_f16_1024.ckpt"
    _min_vqgan_size = 300_000_000  # ~300MB expected
    if os.path.exists(vqgan_ckpt) and os.path.getsize(vqgan_ckpt) < _min_vqgan_size:
        print(f"[KARL-MLX] VQGAN checkpoint appears corrupted ({os.path.getsize(vqgan_ckpt)} bytes), re-downloading...")
        os.remove(vqgan_ckpt)
    if not os.path.exists(vqgan_ckpt):
        print(f"[KARL-MLX] VQGAN checkpoint not found at {vqgan_ckpt}, downloading...")
        import urllib.request
        url = "https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1"
        response = urllib.request.urlopen(url)
        total = int(response.headers.get("Content-Length", 0))
        with open(vqgan_ckpt, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="VQGAN download") as pbar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
        print(f"[KARL-MLX] Downloaded VQGAN checkpoint to {vqgan_ckpt}")
    print(f"[KARL-MLX] Loading VQGAN weights from {vqgan_ckpt}")
    weights = convert_vqgan_weights(vqgan_ckpt)
    if weights is not None:
        mapped = {}
        for k, v in weights.items():
            mapped[f"base_tokenizer.{k}"] = v
        model.load_weights(list(mapped.items()), strict=False)
        print("[KARL-MLX] VQGAN weights loaded.")

    # --- Freeze/unfreeze base tokenizer per stage ---
    if args.stage == "pretrain":
        model.base_tokenizer.freeze()
        print("[KARL-MLX] Base tokenizer FROZEN (pretrain stage).")
    else:
        model.base_tokenizer.unfreeze()
        print("[KARL-MLX] Base tokenizer UNFROZEN (finetune stage).")

    # --- Load KARL checkpoint for finetuning ---
    if args.finetune and os.path.exists(args.finetune):
        print(f"[KARL-MLX] Loading KARL weights from {args.finetune}")
        model.load_weights(args.finetune)

    # --- Collect image paths ---
    train_dir = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_dir):
        train_dir = args.data_path
    all_paths = image_folder_paths(train_dir)
    print(f"[KARL-MLX] Found {len(all_paths)} training images")
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found under {train_dir}")

    # --- Optimizer: AdamW betas=(0.9, 0.95), LR scaling ---
    steps_per_epoch = max(1, len(all_paths) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # LR scaling: lr = blr * eff_batch_size / 256
    eff_batch_size = args.batch_size
    lr = args.blr * eff_batch_size / 256.0
    print(f"[KARL-MLX] Base LR: {args.blr:.2e}, Effective LR: {lr:.2e}")

    optimizer = optim.AdamW(learning_rate=lr, betas=[0.9, 0.95], weight_decay=args.weight_decay)

    # Separate discriminator optimizer for finetune
    if args.stage == "finetune":
        disc_optimizer = optim.AdamW(learning_rate=lr, betas=[0.9, 0.95], weight_decay=args.weight_decay)

    loss_and_grad = nn.value_and_grad(model, model.__call__)

    os.makedirs(args.output_dir, exist_ok=True)
    step = 0

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        random.shuffle(all_paths)
        num_batches = max(1, len(all_paths) // args.batch_size)

        # GAN weight schedule (finetune only)
        if args.stage == "finetune":
            if epoch <= 20:
                gan_loss_weight = 0.0
            elif epoch <= 100:
                gan_loss_weight = 0.2
            else:
                gan_loss_weight = 0.8
        else:
            gan_loss_weight = 0.0

        max_iters = 8  # 1 disc update per 9 gen steps

        batch_bar = tqdm(range(num_batches), desc=f"  Epoch {epoch}", unit="batch", leave=False)
        for bi in batch_bar:
            # LR schedule
            cur_lr = cosine_lr(step, warmup_steps, total_steps, lr)
            optimizer.learning_rate = mx.array(cur_lr)

            batch_paths = all_paths[bi * args.batch_size : (bi + 1) * args.batch_size]
            imgs = make_batch(batch_paths, args.batch_size, augment=True)

            # Determine GAN optimizer idx for finetune
            if args.stage == "finetune" and gan_loss_weight > 0:
                gan_optimizer_idx = 1 - int(((bi + 1) % (max_iters + 1)) == 0)
                if gan_loss_weight == 0:
                    gan_optimizer_idx = 0
            else:
                gan_optimizer_idx = 0

            (loss, logs), grads = loss_and_grad(imgs, epoch, gan_optimizer_idx, gan_loss_weight)

            grads, _ = optim.clip_grad_norm(grads, max_norm=args.grad_clip)

            if args.stage == "finetune" and gan_optimizer_idx == 1:
                # Discriminator step: zero out all non-discriminator gradients
                import mlx.utils
                filtered = mlx.utils.tree_map_with_path(
                    lambda p, g: g if "discriminators" in p else mx.zeros_like(g), grads)
                optimizer.apply_gradients(filtered, model)
            elif args.stage == "finetune" and gan_loss_weight > 0:
                # Generator step: zero out discriminator gradients
                import mlx.utils
                filtered = mlx.utils.tree_map_with_path(
                    lambda p, g: mx.zeros_like(g) if "discriminators" in p else g, grads)
                optimizer.apply_gradients(filtered, model)
            else:
                optimizer.apply_gradients(grads, model)
            mx.eval(model.parameters(), optimizer.state, loss)

            step += 1
            log_str = {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in logs.items()}
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", **{k: log_str[k] for k in list(log_str)[:3]})

        epoch_bar.set_postfix(loss=f"{loss.item():.4f}", step=step)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt = os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.safetensors")
            model.save_weights(ckpt)
            print(f"  → saved {ckpt}")

    final = os.path.join(args.output_dir, "checkpoint_last.safetensors")
    model.save_weights(final)
    print(f"[KARL-MLX] Training complete. Final checkpoint: {final}")


# ===================================================================
# CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="KARL — MLX replication")
    p.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--model", default="karl_small", choices=list(KARL_CONFIGS.keys()))
    p.add_argument("--data_path", required=True, help="ImageFolder-style dataset root")
    p.add_argument("--output_dir", default="./output_karl_mlx")
    p.add_argument("--finetune", default="", help="KARL checkpoint to load for finetuning")
    p.add_argument("--vqgan_ckpt", default="", help="Pretrained VQGAN checkpoint (.pth/.ckpt)")
    p.add_argument("--quantize_latent", action="store_true", default=True)
    p.add_argument("--no_quantize_latent", dest="quantize_latent", action="store_false")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--blr", type=float, default=1e-3, help="Base learning rate (scaled by batch_size/256)")
    p.add_argument("--lr", type=float, default=None, help="Override absolute LR (skip scaling)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=3.0)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--save_every", type=int, default=20)
    args = p.parse_args()

    # If absolute LR provided, override blr scaling
    if args.lr is not None:
        args.blr = args.lr * 256.0 / args.batch_size

    train(args)


if __name__ == "__main__":
    main()

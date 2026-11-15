"""Vision Transformer (ViT), written from the paper
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021).

Author: Krunal Vaghela

The four equations of the paper map one-to-one onto the modules below:

    (1)  z0  = [x_class ; x_p^1 E ; ... ; x_p^N E] + E_pos          -> PatchEmbedding + class token + PositionEmbedding
    (2)  z'l = MSA(LN(z_{l-1})) + z_{l-1}                           -> EncoderBlock (attention half)
    (3)  zl  = MLP(LN(z'l)) + z'l                                   -> EncoderBlock (MLP half)
    (4)  y   = LN(z_L^0)                                            -> final LayerNorm on the class token, then the head
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 43            # GTSRB has 43 traffic-sign classes
    dim: int = 768                   # D, hidden size        (ViT-Base, Table 1 of the paper)
    depth: int = 12                  # L, encoder layers
    heads: int = 12                  # attention heads
    mlp_dim: int = 3072              # MLP size
    dropout: float = 0.1
    attn_dropout: float = 0.0

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2


def vit_base(**kw):
    return ViTConfig(**kw)


def vit_tiny(**kw):
    """A small ViT for low-resolution inputs such as 64x64 traffic-sign crops."""
    base = dict(image_size=64, patch_size=8, dim=192, depth=6, heads=3, mlp_dim=768)
    base.update(kw)
    return ViTConfig(**base)


class PatchEmbedding(nn.Module):
    """Split the image into non-overlapping P x P patches and linearly project each one to D numbers.

    A convolution with kernel = stride = P is exactly "flatten every patch, multiply by E": each of its D
    filters sees one whole patch, and the filters do not overlap.
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_channels, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x):                       # (B, C, H, W)
        x = self.proj(x)                        # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)     # (B, N, D): a sequence of N patch tokens


class MultiHeadSelfAttention(nn.Module):
    """softmax(Q K^T / sqrt(d)) V, split across heads. Returns the attention weights too, for visualisation."""

    def __init__(self, dim, heads, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):                                                    # (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                     # (B, h, T, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale                        # every token scores every token
        attn = attn.softmax(dim=-1)
        y = self.attn_drop(attn) @ v                                         # weighted sum of the values
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out_drop(self.out(y)), attn


class MLP(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer block: equations (2) and (3)."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(cfg.dim, cfg.heads, cfg.attn_dropout, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, cfg.mlp_dim, cfg.dropout)

    def forward(self, x):
        a, w = self.attn(self.ln1(x))
        x = x + a                       # (2) residual around attention
        x = x + self.mlp(self.ln2(x))   # (3) residual around the MLP
        return x, w


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.dim))   # learned 1-D positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)                     # zero-initialised head, as in the paper's fine-tuning

    def forward(self, x, return_attention=False):
        x = self.patch_embed(x)                                              # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed                      # (1) prepend [class], add positions
        x = self.drop(x)
        maps = []
        for blk in self.blocks:
            x, w = blk(x)
            maps.append(w)
        y = self.norm(x[:, 0])                                               # (4) the [class] token's final state
        logits = self.head(y)
        return (logits, maps) if return_attention else logits


def attention_rollout(maps, discard_ratio=0.0):
    """Abnar & Zuidema (2020): multiply the head-averaged attention of every layer (plus identity for the residual)
    to see which patches the [class] token ultimately draws its information from."""
    with torch.no_grad():
        result = torch.eye(maps[0].shape[-1], device=maps[0].device)
        for a in maps:
            a = a.mean(1)[0]                                                 # average heads, first image
            if discard_ratio:
                flat = a.flatten(); k = int(flat.numel() * discard_ratio)
                flat[flat.topk(k, largest=False).indices] = 0
            a = a + torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(-1, keepdim=True)
            result = a @ result
        mask = result[0, 1:]
        side = int(mask.numel() ** 0.5)
        return (mask / mask.max()).reshape(side, side)


if __name__ == "__main__":
    for name, cfg in (("ViT-Base/16", vit_base()), ("ViT-Tiny/8 (64x64)", vit_tiny())):
        m = VisionTransformer(cfg)
        x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
        logits, maps = m(x, return_attention=True)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:20s} params {n / 1e6:6.1f} M   logits {tuple(logits.shape)}   attention {tuple(maps[0].shape)}")

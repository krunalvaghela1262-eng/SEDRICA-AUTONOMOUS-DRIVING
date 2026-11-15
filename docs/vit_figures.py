"""Animations explaining the Vision Transformer on a traffic sign (drawn here, so no image licensing issues).

    python docs/vit_figures.py

1. patch_embedding.gif : image -> 16x16 patches -> sequence -> flatten -> linear projection -> + [class] + positions
2. attention.gif       : how one patch attends to all others, for a "local" head and a "global" head
The embeddings in (1) are computed with the PatchEmbedding module from vision_transformer/vit/model.py.
The attention weights in (2) are illustrative (built from patch colour/texture similarity and distance);
a trained ViT learns its own query/key projections.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision_transformer"))
from vit.model import PatchEmbedding, ViTConfig   # noqa: E402

OUT = ROOT / "docs" / "figures"
BG, FG, MUTED, ACC, ACC2 = "#05070a", "#f3f4f6", "#9ca3af", "#22d3ee", "#f59e0b"
S, P = 224, 16
G = S // P                                  # 14 x 14 patches


def make_sign():
    """A 50 km/h speed-limit sign on a pole against sky and road."""
    yy = np.linspace(0, 1, S)[:, None]
    sky = np.array([110, 165, 220]) * (1 - yy) + np.array([190, 215, 235]) * yy
    img = np.repeat(sky[:, None, :], S, axis=1)
    img[int(S * 0.78):] = [92, 96, 104]
    img += np.random.default_rng(0).normal(0, 4, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([107, 150, 117, S], fill=(120, 124, 130))
    c, r = (112, 92), 70
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(214, 30, 38))
    d.ellipse([c[0] - r + 15, c[1] - r + 15, c[0] + r - 15, c[1] + r - 15], fill=(250, 250, 250))
    font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), 64)
    d.text(c, "50", fill=(20, 20, 20), font=font, anchor="mm")
    return np.asarray(im)


def grab(fig, frames):
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=255))


def patches_of(img):
    return img.reshape(G, P, G, P, 3).transpose(0, 2, 1, 3, 4).reshape(G * G, P, P, 3)


# ====================================================================================== 1. patch embedding
def patch_embedding_gif(img):
    torch.manual_seed(0)
    cfg = ViTConfig(image_size=S, patch_size=P, dim=768)
    emb = PatchEmbedding(cfg)
    x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)[None]
    with torch.no_grad():
        tokens = emb(x)[0].numpy()                                         # (196, 768): x_p E for every patch
    # 2-D sin-cos position code for display (ViT learns E_pos; trained ones look like this)
    pos = np.zeros((G * G, 768))
    rr, cc = np.divmod(np.arange(G * G), G)
    for k in range(192):
        f = 1 / (100 ** (k / 192))
        pos[:, 4 * k], pos[:, 4 * k + 1] = np.sin(rr * f), np.cos(rr * f)
        pos[:, 4 * k + 2], pos[:, 4 * k + 3] = np.sin(cc * f), np.cos(cc * f)
    patches = patches_of(img)
    fig = plt.figure(figsize=(12, 6.75), dpi=80, facecolor=BG)
    frames, dur = [], []

    def title(step, text):
        fig.text(0.03, 0.93, f"Vision Transformer · step {step}", color=ACC, fontsize=12, weight="bold")
        fig.text(0.03, 0.885, text, color=FG, fontsize=15, weight="bold")

    # --- step 1: the grid
    for k in range(0, G + 1, 2):
        fig.clf(); fig.set_facecolor(BG)
        title(1, "Cut the 224×224 image into a 14×14 grid of 16×16-pixel patches")
        ax = fig.add_axes([0.03, 0.06, 0.45, 0.75]); ax.imshow(img); ax.axis("off")
        for i in range(k + 1):
            ax.axhline(i * P - 0.5, color="white", lw=1); ax.axvline(i * P - 0.5, color="white", lw=1)
        fig.text(0.53, 0.5, f"196 patches\n= 196 \"words\"\nfor the Transformer", color=FG, fontsize=18, va="center")
        grab(fig, frames); dur.append(180)
    dur[-1] = 900
    # --- step 2: patches fly into a sequence (raster order)
    rows, cols = 6, 33
    src = np.c_[(np.arange(196) % G) * P, (np.arange(196) // G) * P].astype(float)          # x, y in image px
    for t in np.linspace(0, 1, 16) ** 0.7:
        fig.clf(); fig.set_facecolor(BG)
        title(2, "Line the patches up into a sequence, left-to-right, top-to-bottom")
        ax = fig.add_axes([0, 0.02, 1, 0.82]); ax.set_xlim(0, 1200); ax.set_ylim(560, 0); ax.axis("off")
        ax.set_facecolor(BG)
        for i in range(196):
            sx, sy = 40 + src[i, 0] * 1.9, 40 + src[i, 1] * 1.9
            dx, dy = 360 + (i % cols) * 25, 120 + (i // cols) * 34
            px, py = sx + (dx - sx) * t, sy + (dy - sy) * t
            sz = 30 * (1 - t) + 22 * t
            ax.imshow(patches[i], extent=(px, px + sz, py + sz, py), interpolation="nearest")
        if t > 0.95:
            ax.text(360, 360, "1   2   3   …   the sequence carries no notion of where each patch came from:\n"
                    "that is what the position embedding adds in step 4", color=MUTED, fontsize=12)
        grab(fig, frames); dur.append(120)
    dur[-1] = 1400
    # --- step 3: one patch -> 768 numbers -> D-dim token
    f = patches.astype(float)
    redfrac = ((f[..., 0] > 150) & (f[..., 1] < 90)).mean((1, 2))
    k = int(np.argmin(np.abs(redfrac - 0.5)))                               # half red ring, half sky / white
    flat = patches[k].reshape(-1, 3)                                         # 256 pixels x RGB
    for a in np.linspace(0, 1, 8):
        fig.clf(); fig.set_facecolor(BG)
        title(3, "Flatten each patch (16×16×3 = 768 numbers) and multiply by a learned matrix E")
        ax = fig.add_axes([0.04, 0.35, 0.18, 0.35]); ax.imshow(patches[k], interpolation="nearest"); ax.axis("off")
        ax.set_title(f"patch #{k + 1}", color=FG, fontsize=12)
        n_show = int(256 * a) + 1
        b = fig.add_axes([0.27, 0.48, 0.3, 0.07]); b.imshow(flat[None, :n_show], aspect="auto",
                                                             extent=(0, n_show, 0, 1), interpolation="nearest")
        b.set_xlim(0, 256); b.set_yticks([]); b.tick_params(colors=MUTED, labelsize=8)
        b.set_title("flattened: 256 pixels × 3 colours = 768 values", color=MUTED, fontsize=10, loc="left")
        fig.text(0.59, 0.5, "×  E", color=FG, fontsize=22, weight="bold", va="center")
        if a == 1:
            e = fig.add_axes([0.67, 0.48, 0.3, 0.07])
            e.imshow(tokens[k][None, :], aspect="auto", cmap="coolwarm"); e.set_yticks([])
            e.tick_params(colors=MUTED, labelsize=8)
            e.set_title("token x_p E: a 768-D embedding", color=MUTED, fontsize=10, loc="left")
            fig.text(0.27, 0.25, "In code this is one Conv2d with kernel = stride = 16: each of its 768 filters\n"
                     "looks at exactly one whole patch, which is the same as flatten-then-multiply.",
                     color=MUTED, fontsize=11.5)
        grab(fig, frames); dur.append(160)
    dur[-1] = 2200
    # --- step 4: + [class] token + positions -> z0
    Z = np.r_[np.zeros((1, 768)), tokens]
    E_pos = np.r_[np.zeros((1, 768)), pos]
    for stage in range(4):
        fig.clf(); fig.set_facecolor(BG)
        title(4, "Prepend a learnable [class] token and add position embeddings  →  z₀")
        axs = [fig.add_axes([0.05 + i * 0.31, 0.1, 0.26, 0.66]) for i in range(3)]
        tok = tokens / (np.abs(tokens).max() + 1e-9)
        mats = [(np.r_[np.full((1, 768), 0.9), tok], "patch tokens  x_p E\n(+ [class] on top)"),
                (E_pos, "position embedding  E_pos"),
                (np.r_[np.full((1, 768), 0.9), tok] + 0.6 * E_pos, "z₀  (197 × 768)")]
        for i, (ax, (m, lab)) in enumerate(zip(axs, mats)):
            ax.set_facecolor(BG)
            if i <= stage:
                ax.imshow(m[:, :64], aspect="auto", cmap="coolwarm", vmin=-1.2, vmax=1.2, interpolation="nearest")
                ax.add_patch(Rectangle((-0.5, -0.5), 64, 1, fill=False, ec=ACC2, lw=2))
            ax.set_title(lab, color=FG, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([0, 196], ["[class]", "patch 196"], color=MUTED, fontsize=9)
            for sp in ax.spines.values():
                sp.set_color("#374151")
        fig.text(0.355, 0.45, "+", color=FG, fontsize=26, weight="bold")
        fig.text(0.665, 0.45, "=", color=FG, fontsize=26, weight="bold")
        fig.text(0.05, 0.03, "z₀ = [x_class ; x_p¹E ; … ; x_p¹⁹⁶E] + E_pos   (eq. 1 of the paper; first 64 of 768 "
                 "dimensions shown)  →  12 Transformer encoder layers  →  class token  →  sign class",
                 color=MUTED, fontsize=11)
        grab(fig, frames); dur.append(900)
    dur[-1] = 3000
    frames[0].save(OUT / "vit_patch_embedding.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0)
    plt.close(fig)


# ====================================================================================== 2. attention
def attention_gif(img):
    patches = patches_of(img).astype(float) / 255
    mean = patches.mean((1, 2))                                           # colour
    tex = patches.std((1, 2)).mean(1, keepdims=True) * 3                  # texture / edges
    feat = np.c_[mean, tex]
    rr, cc = np.divmod(np.arange(G * G), G)
    pos = np.c_[rr, cc].astype(float)
    dfeat = ((feat[:, None] - feat[None]) ** 2).sum(-1)
    dpos = ((pos[:, None] - pos[None]) ** 2).sum(-1)

    def attn(q, tau_c, tau_p):
        s = -dfeat[q] / tau_c - dpos[q] / tau_p
        w = np.exp(s - s.max()); return w / w.sum()

    heads = (("head A · local: nearby patches", 1e3, 2.5), ("head B · global: similar content anywhere", 0.004, 2000.0))
    red = (mean[:, 0] > 0.6) & (mean[:, 1] < 0.4)
    ring = np.nonzero(red)[0]
    queries = [ring[0], 6 * G + 5, ring[len(ring) // 2], 12 * G + 3, 1 * G + 1]
    fig = plt.figure(figsize=(12, 6.75), dpi=80, facecolor=BG)
    frames, dur = [], []
    for q in queries:
        for a in (0.35, 0.7, 1.0, 1.0):
            fig.clf(); fig.set_facecolor(BG)
            fig.text(0.03, 0.93, "Self-attention · every patch looks at every other patch", color=FG, fontsize=15,
                     weight="bold")
            fig.text(0.03, 0.885, "Attention(Q, K, V) = softmax(QKᵀ / √d) V      the highlighted patch is the query; "
                     "brightness = how much it attends to each patch", color=MUTED, fontsize=11)
            for h, (name, tc, tp) in enumerate(heads):
                w = attn(q, tc, tp)
                ax = fig.add_axes([0.03 + h * 0.49, 0.07, 0.44, 0.74]); ax.axis("off")
                ax.imshow(img * 0.45 / 255)
                ax.imshow(w.reshape(G, G) / w.max(), cmap="inferno", alpha=0.75 * a, extent=(0, S, S, 0),
                          interpolation="nearest", vmin=0, vmax=1)
                qr, qc = divmod(q, G)
                for j in np.argsort(w)[-10:]:
                    jr, jc = divmod(j, G)
                    ax.plot([qc * P + 8, jc * P + 8], [qr * P + 8, jr * P + 8], color=ACC, lw=0.6 + 3 * w[j] / w.max(),
                            alpha=0.8 * a)
                ax.add_patch(Rectangle((qc * P, qr * P), P, P, fill=False, ec=ACC, lw=2.5))
                spread = np.sqrt(w @ dpos[q]) * P
                ax.set_title(f"{name}\nmean attention distance {spread:4.0f} px", color=FG, fontsize=11.5)
            grab(fig, frames); dur.append(260 if a < 1 else 650)
    fig.text(0.03, 0.02, "", color=MUTED)
    frames[0].save(OUT / "vit_attention.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    img = make_sign()
    Image.fromarray(img).save(OUT / "vit_sign.png")
    patch_embedding_gif(img)
    attention_gif(img)
    for f in sorted(OUT.glob("vit_*")):
        print(f"{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")

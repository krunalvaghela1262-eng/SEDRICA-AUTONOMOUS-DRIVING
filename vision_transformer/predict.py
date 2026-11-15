"""Classify one traffic-sign image with a trained checkpoint and save its attention-rollout map.

Author: Krunal Vaghela

    python predict.py checkpoints/vit_best.pt sign.png
"""
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image

from vit.data import build_transforms
from vit.model import ViTConfig, VisionTransformer, attention_rollout


def main(ckpt_path, image_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ViTConfig(**ckpt["config"])
    model = VisionTransformer(cfg); model.load_state_dict(ckpt["state_dict"]); model.eval()
    img = Image.open(image_path).convert("RGB")
    x = build_transforms(cfg.image_size, train=False)(img).unsqueeze(0)
    with torch.no_grad():
        logits, maps = model(x, return_attention=True)
    p = logits.softmax(-1)[0]
    top = p.topk(3)
    for prob, idx in zip(top.values, top.indices):
        print(f"class {ckpt['classes'][idx]:>4s}   {prob:.3f}")
    mask = attention_rollout(maps).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
    ax[0].imshow(img.resize((cfg.image_size, cfg.image_size))); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(img.resize((cfg.image_size, cfg.image_size)))
    ax[1].imshow(mask, cmap="inferno", alpha=0.6, extent=(0, cfg.image_size, cfg.image_size, 0), interpolation="bilinear")
    ax[1].set_title("attention rollout"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig("attention_rollout.png", dpi=120)
    print("saved attention_rollout.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Train the Vision Transformer on traffic signs.

Author: Krunal Vaghela

    python train.py                          # ViT-Tiny/8 on 64x64 GTSRB crops (downloads GTSRB on first run)
    python train.py --model base --image-size 224 --epochs 30
    python train.py --data-dir path/to/signs # your own <root>/{train,test}/<class>/*.png

Recipe from the paper, scaled to a small dataset: AdamW, linear warm-up then cosine decay, label smoothing,
gradient clipping at norm 1, mixed precision on GPU.
"""
import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn

from vit.data import build_loaders
from vit.model import VisionTransformer, vit_base, vit_tiny


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "base"], default="tiny")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, test_dl, classes = build_loaders(a.image_size, a.batch_size, data_dir=a.data_dir)
    cfg = (vit_tiny if a.model == "tiny" else vit_base)(image_size=a.image_size, num_classes=len(classes))
    model = VisionTransformer(cfg).to(device)
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, "
          f"{cfg.num_patches} patches of {cfg.patch_size}x{cfg.patch_size}, {len(classes)} classes, on {device}")

    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos_embed" not in n and "cls_token" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and "pos_embed" not in n and "cls_token" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.999))
    steps = a.epochs * len(train_dl)
    warm = a.warmup_epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    out = Path(a.out); out.mkdir(exist_ok=True)
    best = 0.0
    for epoch in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        acc = evaluate(model, test_dl, device)
        print(f"epoch {epoch + 1:3d}  loss {run / len(train_dl):.3f}  test acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save({"config": cfg.__dict__, "state_dict": model.state_dict(), "classes": classes}, out / "vit_best.pt")
    print(f"best test accuracy {best:.4f}  ->  {out / 'vit_best.pt'}")


if __name__ == "__main__":
    main()

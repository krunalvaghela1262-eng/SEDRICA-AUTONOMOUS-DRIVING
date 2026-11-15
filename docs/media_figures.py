"""Banner and golf-cart video preview.   python docs/media_figures.py   (needs opencv-python)"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
VIDEO = ROOT / "docs" / "media" / "golf_cart_autonomous_drive.mp4"


def frame_at(cap, sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, im = cap.read()
    return Image.fromarray(im[..., ::-1]) if ok else None


def video_preview_gif(width=420, fps=5, clips=((3, 2.6), (15, 2.6), (40, 2.6), (49, 2.6))):
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    for s, d in clips:
        for k in range(int(d * fps)):
            f = frame_at(cap, s + k / fps)
            if f is None:
                break
            f = f.resize((width, int(width * f.height / f.width)), Image.LANCZOS)
            frames.append(f.quantize(colors=255))
    frames[0].save(OUT / "golf_cart_preview.gif", save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def banner():
    cap = cv2.VideoCapture(str(VIDEO))
    cart = np.asarray(frame_at(cap, 44))
    pf = Image.open(OUT / "particle_filter.gif"); pf.seek(pf.n_frames - 1); pf = np.asarray(pf.convert("RGB"))
    att = Image.open(OUT / "vit_attention.gif"); att.seek(7); att = np.asarray(att.convert("RGB"))
    fig = plt.figure(figsize=(16, 4), dpi=100, facecolor="#05070a")
    t = fig.add_axes([0, 0, 0.36, 1]); t.axis("off")
    t.text(0.08, 0.62, "SeDriCa", color="white", fontsize=40, weight="bold", transform=t.transAxes)
    t.text(0.08, 0.47, "Self-Driving Car team · IIT Bombay", color="#e5e7eb", fontsize=15, transform=t.transAxes)
    t.text(0.08, 0.32, "autonomous golf cart · RoboRacer (3rd, Techfest 2025)\nvision transformers for traffic signs",
           color="#9ca3af", fontsize=12.5, transform=t.transAxes, linespacing=1.5)
    for i, (im, crop) in enumerate(((cart, (0.0, 1.0, 0.05, 0.95)), (pf, (0.12, 0.98, 0.18, 0.82)), (att, (0.19, 0.95, 0.52, 0.98)))):
        h, w = im.shape[:2]
        im = im[int(crop[0] * h):int(crop[1] * h), int(crop[2] * w):int(crop[3] * w)]
        ax = fig.add_axes([0.37 + i * 0.21, 0.06, 0.2, 0.88]); ax.imshow(im); ax.axis("off")
    fig.savefig(OUT / "banner.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    # video_preview_gif()  (the README now embeds the video itself)
    banner()
    for f in ("golf_cart_preview.gif", "banner.png"):
        print(f, round((OUT / f).stat().st_size / 1e6, 2), "MB")

# Vision Transformer (PyTorch), written from the paper

*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., ICLR 2021). Author of this implementation: Krunal Vaghela.

| File | What it does |
|---|---|
| `vit/model.py` | `PatchEmbedding`, hand-written `MultiHeadSelfAttention`, pre-norm `EncoderBlock`, `VisionTransformer`, `attention_rollout`; configs `vit_base()` (85.8 M parameters) and `vit_tiny()` (2.7 M, for 64×64 inputs) |
| `vit/data.py` | traffic-sign data pipeline: GTSRB through torchvision, or your own `train/` and `test/` class folders; augmentation without horizontal flips (a mirrored "keep left" is "keep right") |
| `train.py` | AdamW with linear warm-up and cosine decay, label smoothing, gradient clipping, mixed precision; saves the best checkpoint |
| `predict.py` | top-3 classes for one image, plus its attention-rollout map |

```bash
pip install -r requirements.txt
python -m vit.model
python train.py --model tiny --image-size 64 --epochs 20
python predict.py checkpoints/vit_best.pt sign.png
```

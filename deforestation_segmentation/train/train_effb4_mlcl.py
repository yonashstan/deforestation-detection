#!/usr/bin/env python
"""Train UNet++ (EfficientNet-B4) for deforestation segmentation with
Multi-Level Contrastive Learning (MLCL-LC).

The loss is
    L = L_seg + λ_p L_pixel + λ_r L_region + λ_g L_global
where the three contrastive terms follow the WACV-23 MLCL-LC idea.

Usage (default hyper-params work on an RTX 3050 Ti):

python scripts/train_effb4_mlcl.py \
    --data-dir data/synth_mixed \
    --out-dir models/effb4_mlcl \
    --epochs 30 \
    --batch-size 8
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys, os, time
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.dataset import DeforestationDataset  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
import pandas as pd  # added near imports

# -------------------- Augmentations --------------------------------------- #

def get_contrastive_augs(img_size: Tuple[int, int] = (512, 512)):
    return A.Compose([
        A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.2), ratio=(0.75, 1.33), p=1.0),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ColorJitter(p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_aug(img_size: Tuple[int, int] = (512, 512)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(),
        ToTensorV2(),
    ])

# -------------------- Dataset with 2 views -------------------------------- #

class DeforestationContrastiveDataset(Dataset):
    """Return two random augmented views of each image + a mask for seg loss."""

    def __init__(self, images_dir: Path, masks_dir: Path, img_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.image_paths = sorted(self.images_dir.glob("*.png"))
        self.mask_paths = sorted(self.masks_dir.glob("*.png"))
        assert len(self.image_paths) == len(self.mask_paths) != 0, "Image/Mask count mismatch or empty dataset"

        self.view_aug = get_contrastive_augs(self.img_size)

    def __len__(self):
        return len(self.image_paths)

    def _read_pair(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk_arr = np.asarray(cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE))
        msk = (msk_arr > 127).astype(np.float32)  # type: ignore[operator]
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, self.img_size[::-1], interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        return img, msk

    def __getitem__(self, idx):
        img, msk = self._read_pair(idx)
        v1 = self.view_aug(image=img, mask=msk)
        v2 = self.view_aug(image=img, mask=msk)
        # mask only needed once for seg loss
        return v1["image"], v2["image"], v1["mask"].unsqueeze(0).float()

# -------------------- Contrastive utilities ------------------------------- #

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Normalized-temperature cross-entropy loss (SimCLR). z1,z2 (N,D)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # 2N,D
    similarity = torch.matmul(representations, representations.T) / temperature  # 2N,2N
    # mask for positives (i <-> i+N and vice versa)
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels + N, labels])
    loss = F.cross_entropy(similarity, labels)
    return loss

# -------------------- Model with Projection Heads ------------------------- #

class UNetEffB4MLCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
        )
        enc_channels = self.backbone.encoder.out_channels  # list len=5
        # Projection heads: global, region, pixel
        self.proj_global = nn.Sequential(
            nn.Linear(enc_channels[-1], 256), nn.ReLU(inplace=True), nn.Linear(256, 128)
        )
        self.proj_region = nn.Sequential(
            nn.Conv2d(enc_channels[-2], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
        )
        self.proj_pixel = nn.Sequential(
            nn.Conv2d(enc_channels[0], 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
        )

    def forward(self, x):
        # overriding to get encoder feats
        feats = self.backbone.encoder(x)  # list of 5 feature maps
        dec = self.backbone.decoder(*feats)
        logits = self.backbone.segmentation_head(dec)
        # Global vector
        glob = torch.flatten(F.adaptive_avg_pool2d(feats[-1], 1), 1)
        glob = self.proj_global(glob)
        # Region features (coarse 4×4 grid)
        region_feat = F.adaptive_avg_pool2d(feats[-2], output_size=(4, 4))
        region_feat = self.proj_region(region_feat)  # B,128,4,4
        # Pixel-level (1/4 resolution)
        pix_feat = self.proj_pixel(feats[0])  # B,128,H',W'
        return logits, glob, region_feat, pix_feat

# -------------------- Loss wrappers --------------------------------------- #

class MultiLevelLoss(nn.Module):
    def __init__(self, seg_loss: nn.Module, lambda_p=1.0, lambda_r=0.5, lambda_g=0.5, temperature=0.1):
        super().__init__()
        self.seg_loss = seg_loss
        self.lambda_p = lambda_p
        self.lambda_r = lambda_r
        self.lambda_g = lambda_g
        self.temp = temperature

    def forward(self, outputs1, outputs2, mask, weight=None):
        # outputs*: (logits, glob, region, pixel)
        logits1, g1, r1, p1 = outputs1
        _, g2, r2, p2 = outputs2
        seg = self.seg_loss(logits1, mask, weight)
        # contrastive losses
        loss_g = nt_xent(g1, g2, self.temp)
        # Region: flatten
        r1_f = r1.flatten(2).permute(0, 2, 1).reshape(-1, r1.size(1))  # (B*16,128)
        r2_f = r2.flatten(2).permute(0, 2, 1).reshape(-1, r2.size(1))
        loss_r = nt_xent(r1_f, r2_f, self.temp)
        # Pixel: downsample to 64×64 for memory, then random sample 1024
        p1_ds = F.adaptive_avg_pool2d(p1, output_size=(64, 64)).flatten(2).permute(0, 2, 1)
        p2_ds = F.adaptive_avg_pool2d(p2, output_size=(64, 64)).flatten(2).permute(0, 2, 1)
        # sample
        idx = torch.randperm(p1_ds.size(1), device=p1.device)[:1024]
        p1_f = p1_ds[:, idx, :].reshape(-1, p1.size(1))
        p2_f = p2_ds[:, idx, :].reshape(-1, p2.size(1))
        loss_p = nt_xent(p1_f, p2_f, self.temp)
        total = seg + self.lambda_p * loss_p + self.lambda_r * loss_r + self.lambda_g * loss_g
        return total, seg.item(), loss_p.item(), loss_r.item(), loss_g.item()

# -------------------- IoU metric ------------------------------------------ #

def iou_score(pred: torch.Tensor, gt: torch.Tensor, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * gt).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

# -------------------- Water/Sky ignore mask ------------------------------ #

def water_sky_weight(imgs: torch.Tensor) -> torch.Tensor:
    """Return 1 for pixels to *keep* in loss, 0 for water/sky/cloud.

    Expects imgs in range [0,1] and channel order RGB (as produced by
    Albumentations' Normalize() with default mean/std).
    """
    R, G, B = imgs[:, 0], imgs[:, 1], imgs[:, 2]
    white = (R > 0.9) & (G > 0.9) & (B > 0.9)
    blue = (B > 0.55) & (B > R + 0.1) & (B > G + 0.05)
    keep = ~(white | blue)
    return keep.float()

# -------------------- Data split helper ----------------------------------- #

def split_dataset(root: Path, train_ratio=0.8, seed=42):
    train_d, val_d = root / "train", root / "val"
    if (train_d / "images").exists():
        return train_d, val_d
    (train_d / "images").mkdir(parents=True, exist_ok=True)
    (train_d / "masks").mkdir(exist_ok=True)
    (val_d / "images").mkdir(parents=True, exist_ok=True)
    (val_d / "masks").mkdir(exist_ok=True)
    imgs = sorted((root / "images").glob("*.png"))
    random.Random(seed).shuffle(imgs)
    split = int(len(imgs) * train_ratio)
    for subset, subset_imgs in (("train", imgs[:split]), ("val", imgs[split:])):
        tgt = train_d if subset == "train" else val_d
        for img_p in subset_imgs:
            msk_p = root / "masks" / img_p.name
            shutil.copy2(img_p, tgt / "images" / img_p.name)
            shutil.copy2(msk_p, tgt / "masks" / img_p.name)
    return train_d, val_d

# -------------------- Main ------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synth_mixed", type=str)
    parser.add_argument("--out-dir", default="models/effb4_mlcl", type=str)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-ignore", action="store_true", help="Ignore water/sky pixels in the segmentation loss")
    parser.add_argument("--img-size", type=int, default=512, help="Square resolution of training images")
    parser.add_argument("--encoder", type=str, default="efficientnet-b2", help="Backbone encoder for UNet++")
    parser.add_argument("--single-view", action="store_true", help="Use only one augmented view (no contrastive loss) for faster training")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed);
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.data_dir)
    img_size = (args.img_size, args.img_size)
    train_dir, val_dir = split_dataset(root)

    train_ds = DeforestationContrastiveDataset(train_dir / "images", train_dir / "masks", img_size=img_size)
    val_ds = DeforestationDataset(val_dir / "images", val_dir / "masks", transform=get_val_aug(img_size), img_size=img_size)

    nw = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    class UNetCustom(nn.Module):
        def __init__(self, encoder_name: str):
            super().__init__()
            self.backbone = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
                decoder_attention_type="scse",
            )
            enc_channels = self.backbone.encoder.out_channels
            self.proj_global = nn.Sequential(nn.Linear(enc_channels[-1], 256), nn.ReLU(inplace=True), nn.Linear(256, 128))
            self.proj_region = nn.Sequential(nn.Conv2d(enc_channels[-2], 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 1))
            self.proj_pixel = nn.Sequential(nn.Conv2d(enc_channels[0], 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 1))

        def forward(self, x):
            feats = self.backbone.encoder(x)
            dec = self.backbone.decoder(*feats)
            logits = self.backbone.segmentation_head(dec)
            glob = self.proj_global(torch.flatten(F.adaptive_avg_pool2d(feats[-1], 1), 1))
            region_feat = self.proj_region(F.adaptive_avg_pool2d(feats[-2], output_size=(4, 4)))
            pix_feat = self.proj_pixel(feats[0])
            return logits, glob, region_feat, pix_feat

    model = UNetCustom(args.encoder).to(device)

    # segmentation loss (weighted focal + lovasz)
    focal = smp.losses.FocalLoss("binary", alpha=0.5, gamma=2)
    lovasz = smp.losses.LovaszLoss("binary")
    def seg_loss_fn(logits, target, weight=None):
        if weight is None:
            return 0.5 * focal(logits, target) + 0.5 * lovasz(logits, target)
        # weighted focal implementation
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_term = (1 - p_t) ** 2 * bce  # gamma=2
        focal_w = (focal_term * weight).sum() / (weight.sum() + 1e-6)
        lovasz_w = lovasz(logits * weight, target * weight)
        return 0.5 * focal_w + 0.5 * lovasz_w

    class _SegLoss(nn.Module):
        """Wrapper so type checkers see a proper nn.Module."""

        def forward(self, logits: torch.Tensor, target: torch.Tensor, weight=None):  # type: ignore[override]
            return seg_loss_fn(logits, target, weight)

    criterion = MultiLevelLoss(_SegLoss())
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 20,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=20,
        final_div_factor=100,
    )
    scaler = GradScaler()

    out_dir = Path(args.out_dir);
    out_dir.mkdir(parents=True, exist_ok=True)

    best_iou, patience, patience_lim = 0.0, 0, 10
    history = {k: [] for k in ("train_loss", "val_loss", "train_iou", "val_iou")}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img1, img2, mask in pbar:
            img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out1 = model(img1)
                if args.single_view:
                    loss = seg_loss_fn(out1[0], mask)
                    seg_l = loss.item(); lp = lr = lg = 0.0
                else:
                    out2 = model(img2)
                    loss, seg_l, lp, lr, lg = criterion(out1, out2, mask, water_sky_weight(img1) if args.color_ignore else None)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "seg": f"{seg_l:.3f}"})
        history["train_loss"].append(epoch_loss / len(train_loader))

        # ---- validation
        model.eval(); val_loss = 0.0; val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits, *_ = model(imgs)
                loss = seg_loss_fn(logits, masks, water_sky_weight(imgs) if args.color_ignore else None)
                iou = iou_score(torch.sigmoid(logits), masks)
                val_loss += loss.item(); val_iou += iou
        val_loss /= len(val_loader); val_iou /= len(val_loader)
        history["val_loss"].append(val_loss); history["val_iou"].append(val_iou)
        # ---- ETA ---------------------------------------------------------
        elapsed = time.time() - start_time
        avg_epoch = elapsed / epoch
        remaining = avg_epoch * (args.epochs - epoch)
        print(f"Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | ETA: {remaining/60:.1f} min")

        # save
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        }
        torch.save(ckpt, out_dir / "latest.pth")
        if val_iou > best_iou:
            best_iou = val_iou; patience = 0
            torch.save(ckpt, out_dir / "best.pth")
            print(f"[+] New best IoU={best_iou:.4f}")
        else:
            patience += 1
            if patience >= patience_lim:
                print("Early stopping")
                break

    # plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend(); plt.title("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(history["val_iou"], label="IoU")
    plt.legend(); plt.title("Val IoU")
    plt.tight_layout(); plt.savefig(out_dir / "curves.png")
    total_time = (time.time() - start_time) / 60
    print(f"Training complete in {total_time:.1f} min. Artifacts in {out_dir}")

    # ---- save history as CSV and a Markdown table ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index_label="epoch")

    best_epoch = int(np.argmax(history["val_iou"])) + 1
    summary_md = (
        f"| Metric | Train (epoch {best_epoch}) | Val (epoch {best_epoch}) |\n"
        f"|--------|--------------------------|-------------------------|\n"
        f"| Loss   | {history['train_loss'][best_epoch-1]:.4f} | {history['val_loss'][best_epoch-1]:.4f} |\n"
        f"| IoU    | {history['val_iou'][best_epoch-1]:.4f} | {history['val_iou'][best_epoch-1]:.4f} |\n"
    )
    with open(out_dir / "results_table.md", "w") as f:
        f.write(summary_md)
    print("Saved training_history.csv and results_table.md")

if __name__ == "__main__":
    main() 
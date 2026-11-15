#!/usr/bin/env python3
"""Train a conservative ResNet50-encoder U-Net for binary/multiclass semantic segmentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2] / "tools" / "vision_compare"))
sys.path.append(str(Path(__file__).resolve().parent))
from common import get_prepared_paths, load_config, resolve_run_dir, seed_everything, setup_logger, torch_load
from datasets import SemanticSegDataset


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class ResNetUNet(nn.Module):
    def __init__(self, num_classes: int, pretrained_resnet50: str | None = None):
        super().__init__()
        from torchvision.models import resnet50
        try:
            from torchvision.models import ResNet50_Weights
            backbone = resnet50(weights=None)
        except Exception:
            backbone = resnet50(pretrained=False)
        if pretrained_resnet50 and Path(pretrained_resnet50).exists():
            state = torch_load(pretrained_resnet50, map_location="cpu")
            backbone.load_state_dict(state, strict=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up4 = UpBlock(2048, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        s0 = self.stem(x)
        x = self.pool(s0)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        x = self.layer4(s3)
        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = self.head(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


def mean_iou_binary(logits, masks):
    preds = logits.argmax(dim=1)
    ious = []
    num_classes = logits.shape[1]
    for c in range(1, num_classes):
        pred = preds == c
        gt = masks == c
        union = (pred | gt).sum().item()
        if union == 0:
            continue
        inter = (pred & gt).sum().item()
        ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


def run_epoch(model, loader, optimizer, device, scaler=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    ce = nn.CrossEntropyLoss()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(images)
                loss = ce(logits, masks)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_iou += mean_iou_binary(logits.detach(), masks) * bs
        count += bs
    return total_loss / max(1, count), total_iou / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["models"]["unet"]
    do_resume = args.resume or bool(cfg["training"].get("resume", False))
    seed_everything(int(cfg["split"]["seed"]))
    run_dir = resolve_run_dir(cfg, "unet", allow_existing_for_resume=do_resume)
    logger = setup_logger("train_unet", Path(cfg["logs_dir"]) / "train_unet.log")
    paths = get_prepared_paths(cfg)
    device = torch.device(cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg.get("classes", ["background", "defect"]))

    train_ds = SemanticSegDataset(str(paths["semantic_root"]), "train", cfg["image"]["img_size"])
    val_ds = SemanticSegDataset(str(paths["semantic_root"]), "val", cfg["image"]["img_size"])
    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"], pin_memory=True)

    model = ResNetUNet(num_classes, model_cfg.get("pretrained_resnet50")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg.get("lr", cfg["training"]["learning_rate"])), weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    scaler = torch.cuda.amp.GradScaler() if cfg["training"].get("amp", True) and device.type == "cuda" else None
    start_epoch = 1
    best_iou = -1.0
    last_path = run_dir / model_cfg["last_name"]
    best_path = run_dir / model_cfg["best_name"]
    if do_resume and last_path.exists():
        ckpt = torch_load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_iou = float(ckpt.get("best_iou", -1.0))
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)

    for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
        tr_loss, tr_iou = run_epoch(model, train_loader, optimizer, device, scaler, train=True)
        va_loss, va_iou = run_epoch(model, val_loader, optimizer, device, None, train=False)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_iou": best_iou, "classes": cfg["classes"]}
        torch.save(ckpt, last_path)
        if va_iou > best_iou:
            best_iou = va_iou
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, best_path)
        logger.info("epoch=%03d train_loss=%.6f train_iou=%.4f val_loss=%.6f val_iou=%.4f best_iou=%.4f", epoch, tr_loss, tr_iou, va_loss, va_iou, best_iou)
    logger.info("Done. best=%s last=%s", best_path, last_path)


if __name__ == "__main__":
    main()

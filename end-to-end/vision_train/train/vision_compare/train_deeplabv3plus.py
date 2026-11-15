#!/usr/bin/env python3
"""Train a lightweight DeepLabV3+ style semantic segmentation model.

This implementation uses only torch/torchvision and local ResNet50 weights when available.
It does not require segmentation_models_pytorch, mmsegmentation, timm, or any extra package.
"""
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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        modules += [ASPPConv(in_channels, out_channels, r) for r in rates]
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))


class DeepLabV3PlusResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained_resnet50: str | None = None):
        super().__init__()
        from torchvision.models import resnet50
        try:
            backbone = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
        except TypeError:
            backbone = resnet50(pretrained=False)
        if pretrained_resnet50 and Path(pretrained_resnet50).exists():
            state = torch_load(pretrained_resnet50, map_location="cpu")
            backbone.load_state_dict(state, strict=False)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # low-level, 256 channels
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # high-level, 2048 channels
        self.aspp = ASPP(2048, 256)
        self.low_project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        x = self.decoder(torch.cat([x, low], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


def build_model(num_classes: int, cfg_model: dict, logger=None):
    # Prefer local ResNet50 ImageNet backbone because it is architecture-compatible and avoids head mismatch.
    backbone_path = cfg_model.get("pretrained_resnet50")
    model = DeepLabV3PlusResNet50(num_classes, backbone_path if backbone_path and Path(backbone_path).exists() else None)
    ckpt_path = cfg_model.get("pretrained_checkpoint")
    if ckpt_path and Path(ckpt_path).exists() and logger:
        logger.info("DeepLabV3+ VOC checkpoint found at %s but not loaded directly because checkpoint key shapes may differ; using local ResNet50 backbone when available.", ckpt_path)
    return model


def mean_iou(logits, masks):
    preds = logits.argmax(dim=1)
    ious = []
    for c in range(1, logits.shape[1]):
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
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = model(images)
                loss = ce(out, masks)
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
        total_iou += mean_iou(out.detach(), masks) * bs
        count += bs
    return total_loss / max(1, count), total_iou / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["models"]["deeplabv3plus"]
    do_resume = args.resume or bool(cfg["training"].get("resume", False))
    seed_everything(int(cfg["split"]["seed"]))
    run_dir = resolve_run_dir(cfg, "deeplabv3plus", allow_existing_for_resume=do_resume)
    logger = setup_logger("train_deeplabv3plus", Path(cfg["logs_dir"]) / "train_deeplabv3plus.log")
    paths = get_prepared_paths(cfg)
    device = torch.device(cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg.get("classes", ["background", "defect"]))

    train_ds = SemanticSegDataset(str(paths["semantic_root"]), "train", cfg["image"]["img_size"])
    val_ds = SemanticSegDataset(str(paths["semantic_root"]), "val", cfg["image"]["img_size"])
    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"], pin_memory=True)

    model = build_model(num_classes, model_cfg, logger).to(device)
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

#!/usr/bin/env python3
"""Train Mask R-CNN for instance segmentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2] / "tools" / "vision_compare"))
sys.path.append(str(Path(__file__).resolve().parent))
from common import get_prepared_paths, load_config, resolve_run_dir, seed_everything, setup_logger, torch_load
from datasets import MaskRCNNDataset, detection_collate


def build_model(num_classes: int, pretrained_checkpoint: str | None = None):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    except TypeError:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
        state = torch_load(pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


@torch.no_grad()
def evaluate_loss(model, loader, device, max_batches=None):
    # Torchvision detection models return losses in train mode when targets are supplied.
    was_training = model.training
    model.train()
    total = 0.0
    count = 0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        total += float(loss.item())
        count += 1
        if max_batches and count >= max_batches:
            break
    model.train(was_training)
    return total / max(1, count)


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total = 0.0
    count = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["models"]["mask_rcnn"]
    do_resume = args.resume or bool(cfg["training"].get("resume", False))
    seed_everything(int(cfg["split"]["seed"]))
    run_dir = resolve_run_dir(cfg, "mask_rcnn", allow_existing_for_resume=do_resume)
    logger = setup_logger("train_mask_rcnn", Path(cfg["logs_dir"]) / "train_mask_rcnn.log")
    paths = get_prepared_paths(cfg)
    device = torch.device(cfg["training"].get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg.get("classes", ["background", "defect"]))

    train_ds = MaskRCNNDataset(str(paths["mask_rcnn_root"]), "train", cfg["image"].get("img_size"))
    val_ds = MaskRCNNDataset(str(paths["mask_rcnn_root"]), "val", cfg["image"].get("img_size"))
    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=True, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"], pin_memory=True, collate_fn=detection_collate)

    model = build_model(num_classes, model_cfg.get("pretrained_checkpoint")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg.get("lr", 5e-5)), weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    scaler = torch.cuda.amp.GradScaler() if cfg["training"].get("amp", True) and device.type == "cuda" else None
    start_epoch = 1
    best_val_loss = float("inf")
    last_path = run_dir / model_cfg["last_name"]
    best_path = run_dir / model_cfg["best_name"]
    if do_resume and last_path.exists():
        ckpt = torch_load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        logger.info("Resumed from %s at epoch %d", last_path, start_epoch)

    for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va_loss = evaluate_loss(model, val_loader, device)
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val_loss": best_val_loss, "classes": cfg["classes"]}
        torch.save(ckpt, last_path)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, best_path)
        logger.info("epoch=%03d train_loss=%.6f val_loss=%.6f best_val_loss=%.6f", epoch, tr_loss, va_loss, best_val_loss)
    logger.info("Done. best=%s last=%s", best_path, last_path)


if __name__ == "__main__":
    main()

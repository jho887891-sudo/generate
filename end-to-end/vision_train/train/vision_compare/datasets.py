#!/usr/bin/env python3
"""Datasets shared by PyTorch models in vision_compare."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SemanticSegDataset(Dataset):
    def __init__(self, semantic_root: str, split: str, img_size: int = 640):
        self.root = Path(semantic_root)
        self.split = split
        self.img_size = img_size
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        self.images = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path}: {mask_path}")
        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return image_t, mask_t


class MaskRCNNDataset(Dataset):
    def __init__(self, mask_rcnn_root: str, split: str, img_size: int | None = None):
        self.root = Path(mask_rcnn_root)
        self.split = split
        self.img_size = img_size
        self.image_dir = self.root / "images" / split
        self.ann_dir = self.root / "annotations" / split
        self.images = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        ann_path = self.ann_dir / f"{img_path.stem}.json"
        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        scale_x = scale_y = 1.0
        if self.img_size:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)

        boxes = []
        labels = []
        masks = []
        areas = []
        for inst in ann["instances"]:
            mask = Image.open(inst["mask_path"]).convert("L")
            if self.img_size:
                mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_np = (np.asarray(mask) > 0).astype(np.uint8)
            ys, xs = np.nonzero(mask_np)
            if len(xs) == 0 or len(ys) == 0:
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
            labels.append(int(inst["class_id"]))
            masks.append(mask_np)
            areas.append(float(mask_np.sum()))

        if not boxes:
            # Should not happen after cleaning, but torchvision requires valid tensors.
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]
            masks = [np.zeros((image_t.shape[1], image_t.shape[2]), dtype=np.uint8)]
            masks[0][0:1, 0:1] = 1
            areas = [1.0]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image_t, target


def detection_collate(batch):
    return tuple(zip(*batch))

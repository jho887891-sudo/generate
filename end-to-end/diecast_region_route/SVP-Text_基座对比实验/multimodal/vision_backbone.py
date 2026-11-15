from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn
from torchvision.ops import roi_align
from PIL import Image
from torchvision.transforms import functional as TF

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


@dataclass
class VisionFeatureOutput:
    feature_map: torch.Tensor
    raw_output: object


def _find_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """
    从复杂输出结构里递归找到第一个 tensor。
    """
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = _find_first_tensor(item)
            if tensor is not None:
                return tensor

    if isinstance(obj, dict):
        for item in obj.values():
            tensor = _find_first_tensor(item)
            if tensor is not None:
                return tensor

    return None


class YOLOFeatureBackbone(nn.Module):
    """
    用 Ultralytics YOLO 作为视觉特征骨干：
    - 通过 forward hook 抓取指定层中间特征
    - 只把底层 torch model 注册成 nn.Module 子模块
    - 不把 ultralytics.YOLO 包装器注册成子模块，避免 model.train() 冲突
    """

    def __init__(self, weight_path: str, feature_layer: int = -2, device: str = "cuda"):
        super().__init__()

        if YOLO is None:
            raise ImportError(f"ultralytics 导入失败: {YOLO_IMPORT_ERROR}")

        self.device = device
        self.feature_layer = feature_layer
        self._cached_feature: Optional[torch.Tensor] = None

        # 关键点：
        # 不要直接 self.wrapper = YOLO(...)，否则它会被 nn.Module 注册成子模块
        # 然后 model.train() 时递归调用到 ultralytics.YOLO.train()，造成冲突
        yolo_wrapper = YOLO(weight_path)
        object.__setattr__(self, "_wrapper", yolo_wrapper)

        # 真正注册到 nn.Module 里的，只放底层 torch 模型
        self.model = yolo_wrapper.model.to(device)

        if not hasattr(self.model, "model"):
            raise RuntimeError("当前 YOLO 底层模型未找到 .model 层列表，无法注册 hook。")

        layers = self.model.model
        num_layers = len(layers)

        real_idx = feature_layer if feature_layer >= 0 else num_layers + feature_layer
        if real_idx < 0 or real_idx >= num_layers:
            raise IndexError(
                f"feature_layer={feature_layer} 越界，当前层数={num_layers}"
            )

        target = layers[real_idx]
        self._hook_handle = target.register_forward_hook(self._hook)

        self.model.eval()

    def _hook(self, _module, _inputs, output):
        tensor = _find_first_tensor(output)
        self._cached_feature = tensor

    def forward(self, image_tensor: torch.Tensor) -> VisionFeatureOutput:
        """
        输入:
            image_tensor: [B, 3, H, W]
        输出:
            VisionFeatureOutput(feature_map, raw_output)
        """
        image_tensor = image_tensor.to(self.device)
        self._cached_feature = None

        raw = self.model(image_tensor)

        if self._cached_feature is None:
            raise RuntimeError("未能捕获 YOLO 中间特征，请检查 yolo_feature_layer。")

        return VisionFeatureOutput(
            feature_map=self._cached_feature,
            raw_output=raw,
        )

    def set_trainable(self, trainable: bool):
        """
        控制 YOLO 底层模型是否参与训练。
        第一轮建议 trainable=False。
        """
        for p in self.model.parameters():
            p.requires_grad = trainable

        if trainable:
            self.model.train()
        else:
            self.model.eval()

    def train(self, mode: bool = True):
        """
        覆盖 nn.Module.train，避免递归碰到 ultralytics.YOLO.train()
        """
        self.training = mode
        self.model.train(mode)
        return self

    def eval(self):
        self.training = False
        self.model.eval()
        return self


def preprocess_image(image_path: str, image_size: int) -> torch.Tensor:
    """
    读取单张图片并缩放到固定尺寸，输出 shape = [3, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    image = TF.resize(image, [image_size, image_size])
    return TF.pil_to_tensor(image).float() / 255.0


def roi_pool_from_boxes(
    feature_map: torch.Tensor,
    boxes_xyxy: Sequence[Sequence[float]],
    image_size: int,
    output_size: int = 1,
    max_rois: int = 12,
) -> torch.Tensor:
    """
    从 feature_map 中根据 box 提取 ROI token，并拼一个 global token。

    参数:
        feature_map: [B, C, Hf, Wf]
        boxes_xyxy: [[x1, y1, x2, y2], ...]，坐标基于输入图像尺寸
        image_size: 输入图像尺寸（如 960）
        output_size: roi_align 输出大小
        max_rois: 最多保留多少个框

    返回:
        [B, 1 + N, C]
        第一个 token 是 global token，后面是 ROI token
    """
    device = feature_map.device

    # 没有 box，就只返回 global token
    if not boxes_xyxy:
        global_token = feature_map.mean(dim=(2, 3))  # [B, C]
        return global_token.unsqueeze(1)             # [B, 1, C]

    clipped = []
    for box in boxes_xyxy[:max_rois]:
        x1, y1, x2, y2 = box
        clipped.append([
            max(0.0, min(float(image_size - 1), float(x1))),
            max(0.0, min(float(image_size - 1), float(y1))),
            max(0.0, min(float(image_size - 1), float(x2))),
            max(0.0, min(float(image_size - 1), float(y2))),
        ])

    rois = torch.tensor(
        [[0.0, *b] for b in clipped],
        dtype=feature_map.dtype,
        device=device,
    )

    spatial_scale = feature_map.shape[-1] / float(image_size)

    pooled = roi_align(
        feature_map,
        rois,
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True,
    )  # [N, C, output_size, output_size]

    pooled = pooled.flatten(2).mean(-1)  # [N, C]

    global_token = feature_map.mean(dim=(2, 3))  # [B, C]

    # 当前默认 B=1，所以拼成 [1+N, C] 后再扩成 [1, 1+N, C]
    tokens = torch.cat([global_token, pooled], dim=0).unsqueeze(0)
    return tokens
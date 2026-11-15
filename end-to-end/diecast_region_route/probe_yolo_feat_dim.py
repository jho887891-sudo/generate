from pathlib import Path
from typing import Any, Optional

import torch
from ultralytics import YOLO

from multimodal.config_region_mm import RegionAwareMMConfig


# =========================
# 这里写死你的测试图像路径
# =========================
TEST_IMAGE_PATH = "/home/T7/zyx/finetune/yolo-X-data/38048 OK/2025-09-16 11-06-48-022_1.jpg"


def _find_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """从复杂输出结构中递归找到第一个 tensor。"""
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


def main() -> None:
    cfg = RegionAwareMMConfig()

    model_path = Path(cfg.yolo_weight_path)
    image_path = Path(TEST_IMAGE_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO 权重不存在：{model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"测试图像不存在：{image_path}")

    device = cfg.device
    imgsz = cfg.vision_image_size
    feature_layer_index = cfg.yolo_feature_layer

    print("=" * 80)
    print("开始探测 YOLO 中间特征维度")
    print(f"MODEL_PATH         : {model_path}")
    print(f"TEST_IMAGE_PATH    : {image_path}")
    print(f"FEATURE_LAYER      : {feature_layer_index}")
    print(f"IMGSZ              : {imgsz}")
    print(f"DEVICE             : {device}")
    print("=" * 80)

    yolo = YOLO(str(model_path))
    net = yolo.model.to(device)
    net.eval()

    if not hasattr(net, "model"):
        raise RuntimeError("当前 YOLO 模型结构中未找到 net.model，无法注册 hook。")

    layers = net.model
    num_layers = len(layers)

    real_idx = feature_layer_index if feature_layer_index >= 0 else num_layers + feature_layer_index
    if real_idx < 0 or real_idx >= num_layers:
        raise IndexError(
            f"yolo_feature_layer={feature_layer_index} 越界，当前层数={num_layers}"
        )

    target_layer = layers[real_idx]
    print(f"总层数: {num_layers}")
    print(f"实际 hook 层索引: {real_idx}")
    print(f"hook 层类型: {target_layer.__class__.__name__}")

    captured = {}

    def forward_hook(module, inputs, output):
        tensor = _find_first_tensor(output)
        if tensor is None:
            captured["feat"] = None
        else:
            captured["feat"] = tensor.detach()

    hook_handle = target_layer.register_forward_hook(forward_hook)

    try:
        with torch.no_grad():
            _ = yolo.predict(
                source=str(image_path),
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
    finally:
        hook_handle.remove()

    feat = captured.get("feat", None)
    if feat is None:
        print("\n未能从该层捕获到 tensor 特征。")
        print("建议修改 config_region_mm.py 中的 yolo_feature_layer，例如：-3、-4、-5")
        return

    print("\n成功捕获特征。")
    print(f"feature_map.shape = {tuple(feat.shape)}")

    if feat.ndim == 4:
        vision_feat_dim = feat.shape[1]
        print("检测到 4D 特征图 [B, C, H, W]")
        print(f"建议在 config_region_mm.py 中设置：vision_feat_dim = {vision_feat_dim}")

    elif feat.ndim == 3:
        vision_feat_dim = feat.shape[-1]
        print("检测到 3D 特征 [B, N, C]")
        print(f"建议在 config_region_mm.py 中设置：vision_feat_dim = {vision_feat_dim}")

    elif feat.ndim == 2:
        vision_feat_dim = feat.shape[-1]
        print("检测到 2D 特征 [N, C]")
        print(f"建议在 config_region_mm.py 中设置：vision_feat_dim = {vision_feat_dim}")

    else:
        print(f"当前特征维度为 {feat.ndim}，请手动确认通道维。")

    print("\n如果这层特征不合适，可以继续改 config_region_mm.py 里的：")
    print("yolo_feature_layer = -3")
    print("yolo_feature_layer = -4")
    print("yolo_feature_layer = -5")


if __name__ == "__main__":
    main()
import torch.nn as nn
import math
from ultralytics import YOLO
from pathlib import Path

# 1. 通用 LoRA 基类
class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


# 2. 通用卷积 LoRA支持 Conv1d/2d/3d
class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size,
                 r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self,
                           r=r,
                           lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if r > 0:
            k_h, k_w = kernel_size
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * k_h * k_w, in_channels * k_h * k_w))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * k_h * k_w,
                     r * k_h * k_w)
                )
            )
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False   # 冻结原卷积权重
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                self.conv.weight.data -= (self.lora_B @ self.lora_A) \
                    .view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                self.conv.weight.data += (self.lora_B @ self.lora_A) \
                    .view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            delta = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            return self.conv._conv_forward(x, self.conv.weight + delta, self.conv.bias)
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv2d, *args, **kwargs)



# 3. 一键注入 LoRA 的函数（保持你原来的接口）
def lora_yolo(model, rank=4, alpha=1.0, target_layers=None):
    """
    把 YOLOv8 模型中的 nn.Conv2d 替换为 Conv2d(LoRA)
    :param model: ultralytics YOLO 模型
    :param rank: LoRA 秩
    :param alpha: LoRA 缩放系数
    :param target_layers: 若给定，则只替换包含这些字符串的层
    """
    def replace(module, name):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}"
            if isinstance(child, nn.Conv2d) and \
               (not target_layers or any(t in full_name for t in target_layers)):
                lora_conv = Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=0.0,
                    merge_weights=True
                )
                # 复制原卷积权重
                lora_conv.conv.load_state_dict(child.state_dict())
                setattr(module, child_name, lora_conv)
            else:
                replace(child, full_name)

    replace(model.model, "")
    return model


# 4. 训练入口
if __name__ == '__main__':
    model = YOLO("D:\zxy\pythonProject\yolo11-seg.yaml")
    model.load("test.pt")# 检测 or 分割权重均可
    model = lora_yolo(model, rank=4, alpha=1.0,
                      target_layers=[ "model.16","model.19","model.22","model.23"])

    # 冻结主干
    for name, p in model.model.named_parameters():
        if "lora" not in name:
            p.requires_grad_(False)


    model.train(data="D:\zxy\pythonProject\P1R\small new data\YOLODataset_seg\dataset.yaml",
                epochs=300,
                imgsz=1280,
                batch=8,
                device=0,
                name="add_lora",
                lr0=0.001,
                mosaic=0,
                )
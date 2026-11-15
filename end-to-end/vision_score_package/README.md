# 视觉层模型评分项目包

这个包用于在 `vk3` 人工标注测试集上统一评测全部训练好的分割模型，并调用 `common_coco_eval.py` 输出 COCO 风格的 `bbox` 和 `segm` 指标。

默认服务器路径已写入 `config.yaml`：

- 测试集：`/home/T7/wkq/end-to-end/vk3`
- 训练代码：`/home/T7/wkq/end-to-end/vision_train`
- 评分输出：`/home/T7/wkq/end-to-end/vision_score_outputs`
- 自己的 YOLO 权重：`/home/T7/wkq/end-to-end/yolo-X-data/best_LR_75.pt`

## 一键运行

```bash
cd /home/T7/wkq/end-to-end/vision_score_package
bash scripts/run_all_eval.sh
```

后台运行：

```bash
cd /home/T7/wkq/end-to-end/vision_score_package
nohup bash scripts/run_all_eval.sh > /home/T7/wkq/end-to-end/vision_score_outputs/eval.nohup.log 2>&1 &
```

查看日志：

```bash
tail -f /home/T7/wkq/end-to-end/vision_score_outputs/eval.nohup.log
```

## 输出结果

运行后会生成：

```text
/home/T7/wkq/end-to-end/vision_score_outputs/gt_vk3_coco.json
/home/T7/wkq/end-to-end/vision_score_outputs/predictions/*.json
/home/T7/wkq/end-to-end/vision_score_outputs/metrics/*_metrics.json
/home/T7/wkq/end-to-end/vision_score_outputs/summary.csv
/home/T7/wkq/end-to-end/vision_score_outputs/summary.md
```

`summary.csv` 是最终汇总表，适合复制到表格或论文实验记录中。

## 模型说明

- `unet` 和 `deeplabv3plus` 是语义分割模型，脚本会将前景概率图按连通域转成实例级 COCO prediction。
- `mask_rcnn` 直接使用模型输出的 boxes、scores、masks。
- `yolov8n_seg`、`yolov11n_seg`、`yolo_x_best_lr_75` 使用 Ultralytics 的 segmentation 输出。

## 依赖

服务器环境需要已有：

```text
torch
torchvision
ultralytics
opencv-python
pillow
numpy
pyyaml
pycocotools
```

如果缺少 `pycocotools`，先安装：

```bash
pip install pycocotools
```

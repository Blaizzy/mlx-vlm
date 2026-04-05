# MLX 版本的 RF-DETR

通过 MLX 移植到 Apple Silicon 的实时检测 Transformer ([RF-DETR](https://github.com/roboflow/rf-detr), ICLR 2026)。支持在 COCO 80 类上执行目标检测和实例分割。

## 快速开始

```python
from pathlib import Path
from PIL import Image
from mlx_vlm.utils import load_model
from mlx_vlm.models.rfdetr.processing_rfdetr import RFDETRProcessor
from mlx_vlm.models.rfdetr.generate import RFDETRPredictor

# 加载模型（使用标准的 mlx-vlm 加载器）
model = load_model(Path("rfdetr-base-mlx"))
processor = RFDETRProcessor.from_pretrained("rfdetr-base-mlx")
predictor = RFDETRPredictor(model, processor, score_threshold=0.3, nms_threshold=0.5)

# 运行检测
result = predictor.predict(Image.open("image.jpg"))

for name, score, box in zip(result.class_names, result.scores, result.boxes):
    print(f"{name}: {score:.2f} [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
```

## 转换权重

从官方 Roboflow 检查点进行一次性转换。需要 `torch` 和 `safetensors`（推理时不需要）。

```bash
# 检测
python -m mlx_vlm.models.rfdetr.convert --variant base --output ./rfdetr-base-mlx
python -m mlx_vlm.models.rfdetr.convert --variant small --output ./rfdetr-small-mlx
python -m mlx_vlm.models.rfdetr.convert --variant large --output ./rfdetr-large-mlx

# 分割（检测 + 实例掩码）
python -m mlx_vlm.models.rfdetr.convert --variant seg-small --output ./rfdetr-seg-small-mlx
python -m mlx_vlm.models.rfdetr.convert --variant seg-large --output ./rfdetr-seg-large-mlx
```

每个输出目录包含 `config.json`、`preprocessor_config.json` 和 `model.safetensors`。运行时不需要 PyTorch 或 rfdetr 依赖。

## 可用变体

| 变体 | 任务 | 分辨率 | 参数 | 延迟 (M4 Max) |
|---------|------|-----------|--------|----------|
| `base` | 检测 | 560 | ~32M | ~33ms |
| `small` | 检测 | 512 | ~32M | - |
| `large` | 检测 | 704 | ~128M | - |
| `seg-small` | 检测 + 分割 | 384 | ~34M | ~88ms |
| `seg-large` | 检测 + 分割 | 480 | ~130M | - |

## 分割

分割模型在每个检测旁边输出每实例的二进制掩码：

```python
result = predictor.predict(image)

# result.boxes   - (N, 4) xyxy 像素坐标
# result.scores  - (N,) 置信度分数
# result.labels  - (N,) COCO 类索引
# result.masks   - (N, H, W) 二进制 uint8 掩码（或检测时为））
```

## 过滤

排除不需要的类或调整阈值：

```python
predictor = RFDETRPredictor(
    model, processor,
    score_threshold=0.3,     # 最小置信度
    nms_threshold=0.5,       # NMS IoU 阈值
    exclude_classes=["couch", "potted plant"],  # 按名称过滤
)
```

## 架构

```
图像 (HxW) --> DINOv2-small (窗口注意力, 12 层)
            --> MultiScaleProjector (C2f 块, P4)
            --> 两阶段编码器 (top-K 查询选择)
            --> 解码器 (3-4 层, 可变形交叉注意力)
            --> 检测头 (类 + 边界框)
            --> [分割头] (可选, 深度可分离卷积 + einsum 掩码)
```

## 文件结构

```
mlx_vlm/models/rfdetr/
  config.py               # 数据类配置
  vision.py               # DINOv2 骨干 + C2f 投影器
  transformer.py           # 编码器 (两阶段) + 解码器 (可变形注意力)
  segmentation.py          # SegmentationHead (掩码预测)
  rfdetr.py               # 主模型 + 权重清理
  generate.py             # RFDETRPredictor + 后处理 + NMS
  processing_rfdetr.py    # 图像预处理 + COCO 类名
  convert.py              # PyTorch 检查点转换器
  language.py             # 存根 (框架兼容性)
```

## 参考

- [RF-DETR: 实时检测 Transformer](https://arxiv.org/abs/2511.09554) (ICLR 2026)
- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)

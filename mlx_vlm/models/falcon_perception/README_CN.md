# Falcon-Perception

Falcon-Perception 是来自 TII 的早期融合视觉语言模型系列，用于目标检测和分割。它为与文本查询匹配的对象生成边界框坐标、大小和分割掩码，使用在共享参数空间中处理图像 patch 和文本 token 的单一 Transformer。

## 模型

| 模型 ID | 参数 | 检测 | 分割 |
|----------|-----------|-----------|-------------|
| `tiiuae/Falcon-Perception` | ~0.6B | 是 | 是 |
| `tiiuae/Falcon-Perception-300M` | ~0.3B | 是 | 否 |

### 链接

- [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception) -- 代码和推理引擎
- [tiiuae/Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception) -- HuggingFace 模型卡片

## 安装

```bash
pip install mlx-vlm
```

## Python

### 检测 + 分割

```python
from mlx_vlm import load

model, processor = load("tiiuae/Falcon-Perception")

detections = model.generate_perception(
    processor,
    image="photo.jpg",
    query="cat",
    max_new_tokens=512,
)

for det in detections:
    xy, hw = det["xy"], det["hw"]
    has_mask = "mask" in det
    print(f"Center: ({xy['x']:.3f}, {xy['y']:.3f}), Size: ({hw['h']:.3f}, {hw['w']:.3f}), Mask: {has_mask}")
```

### 仅检测（300M）

```python
from mlx_vlm import load

model, processor = load("tiiuae/Falcon-Perception-300M")

detections = model.generate_perception(
    processor,
    image="photo.jpg",
    query="cats",
    max_new_tokens=512,
)
```

### 绘制检测结果

```python
from mlx_vlm.models.falcon_perception import plot_detections

plot_detections("photo.jpg", detections, save_path="output.png")
```

有关可视化的完整示例，请参阅 [`examples/falcon_perception_demo.ipynb`](../../../examples/falcon_perception_demo.ipynb)。

### 输出格式

每个检测是一个包含以下内容的字典：
- `xy` -- 归一化到 `[0, 1]` 的中心坐标 `{"x": float, "y": float}`
- `hw` -- 作为图像维度分数的边界框大小 `{"h": float, "w": float}`
- `mask` -- `(H, W)` 二进制 `mx.array` 分割掩码（仅在 0.6B 模型上）

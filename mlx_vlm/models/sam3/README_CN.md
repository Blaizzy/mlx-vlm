# MLX 版本的 SAM3 (Segment Anything Model 3)

[Meta's SAM3](https://github.com/facebookresearch/sam3) 的 MLX 移植版本——开放式词汇检测、分割和视频跟踪模型（约 860M 参数）。

> **注意：** SAM3 不是生成式 VLM。它输出边界框和分割掩码，而不是文本。它使用自定义推理管道（`generate.py`）而不是 `mlx_vlm.generate()`。

## 快速开始

```python
from PIL import Image
from mlx_vlm.utils import load_model, get_model_path
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor

# 加载模型（首次运行时下载约 3.4 GB）
model_path = get_model_path("facebook/sam3")
model = load_model(model_path)
processor = Sam3Processor.from_pretrained(str(model_path))

image = Image.open("photo.jpg")
predictor = Sam3Predictor(model, processor, score_threshold=0.3)
```

## 图像：目标检测

通过文本提示词检测对象——返回边界框和置信度分数：

```python
result = predictor.predict(image, text_prompt="a dog")

for i in range(len(result.scores)):
    x1, y1, x2, y2 = result.boxes[i]
    print(f"[{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

## 图像：实例分割

同样的调用——每实例掩码与边界框一起返回：

```python
result = predictor.predict(image, text_prompt="a person")

# result.boxes   -> (N, 4) xyxy 边界框，缩放到图像尺寸
# result.masks   -> (N, H, W) 二进制分割掩码
# result.scores  -> (N,) 置信度分数
```

### 在图像上叠加掩码

```python
import numpy as np

result = predictor.predict(image, text_prompt="a cat")

overlay = np.array(image).copy()
colors = [(30, 120, 255), (255, 80, 30), (30, 200, 30)]

W, H = image.size
for i in range(len(result.scores)):
    mask = result.masks[i]
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H)))
    binary = mask > 0
    color = np.array(colors[i % len(colors)])
    overlay[binary] = (overlay[binary] * 0.5 + color * 0.5).astype(np.uint8)

Image.fromarray(overlay).save("segmentation_output.png")
```

## 图像：框引导检测

传递边界框提示词以将检测引导到特定区域：

```python
import numpy as np

# 覆盖 xyxy 像素坐标中感兴趣区域的框
boxes = np.array([[100, 50, 400, 350]])
result = predictor.predict(image, text_prompt="a cat", boxes=boxes)
```

## 图像：语义分割

访问密集语义分割输出以及实例预测：

```python
import mlx.core as mx

inputs = processor.preprocess_image(image)
text_inputs = processor.preprocess_text("a cat")

outputs = model.detect(
    mx.array(inputs["pixel_values"]),
    mx.array(text_inputs["input_ids"]),
    mx.array(text_inputs["attention_mask"]),
)
mx.eval(outputs)

# 实例掩码：(B, 200, 288, 288)
pred_masks = outputs["pred_masks"]

# 语义分割：(B, 1, 288, 288)
semantic_seg = outputs["semantic_seg"]
```

## 图像：批量推理

在单次前向传递中处理多个图像：

```python
import mlx.core as mx
import numpy as np

images = [Image.open("photo1.jpg"), Image.open("photo2.jpg")]

# 将像素值堆叠到批次中
pixel_values = mx.array(np.stack([
    processor.preprocess_image(img)["pixel_values"][0] for img in images
]))  # (B, 1008, 1008, 3)

# 编码文本一次（在批次间共享）
text_inputs = processor.preprocess_text("a cat")
input_ids = mx.array(np.tile(text_inputs["input_ids"], (len(images), 1)))
attention_mask = mx.array(np.tile(text_inputs["attention_mask"], (len(images),), 1)))

outputs = model.detect(pixel_values, input_ids, attention_mask)
mx.eval(outputs)

# outputs["pred_logits"]: (B, 200)
# outputs["pred_boxes"]:  (B, 200, 4)
# outputs["pred_masks"]:  (B, 200, 288, 288)
for i in range(len(images)):
    scores = 1 / (1 + np.exp(-np.array(outputs["pred_logits"][i])))
    print(f"Image {i}: top score = {scores.max():.2f}")
```


## CLI

所有任务都可以通过单个命令使用：

```bash
python -m mlx_vlm.models.sam3.generate --task <task> --prompt "..." ...
```

### 目标检测（仅边界框）

```bash
python -m mlx_vlm.models.sam3.generate --task detect --image photo.jpg --prompt "a cat"
```

### 实例分割（仅掩码，默认）

```bash
python -m mlx_vlm.models.sam3.generate --image photo.jpg --prompt "a cat"

# 叠加边界框
python -m mlx_vlm.models.sam3.generate --image photo.jpg --prompt "a cat" --show-boxes

# 框引导分割
python -m mlx_vlm.models.sam3.generate --image photo.jpg --prompt "a cat" --boxes "0,50,350,480"

# 多个框提示词
python -m mlx_vlm.models.sam3.generate --image photo.jpg --prompt "a cat" --boxes "0,50,350,480;300,20,640,375"
```

### 视频跟踪（到文件）

```bash
python -m mlx_vlm.models.sam3.generate --task track --video input.mp4 --prompt "a car"

# 仅跟踪区域中的对象
python -m mlx_vlm.models.sam3.generate --task track --video input.mp4 --prompt "a car" --boxes "200,100,1200,900"

# 使用较低分辨率加快速度
python -m mlx_vlm.models.sam3.generate --task track --video input.mp4 --prompt "a car" --resolution 336
```

### 实时摄像头（实时预览）

打开带有实时检测叠加的网络摄像头窗口。按 `q` 退出。

```bash
python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a person" --resolution 224

# 多个对象
python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a person" "a phone" --resolution 224

# 带有边界框和标签
python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a cup" --resolution 224 --show-boxes
```

使用 3 个线程：帧读取器、推理（224x224 时约 11 FPS）和显示器。多个 `--prompt` 值共享 ViT 骨干——每个额外的提示词增加约 30ms，而不是完整的推理传递。

### 背景交换（摄像头）

将背景替换为自定义图像，同时将检测到的对象保持在前景中：

```bash
python -m mlx_vlm.models.sam3.generate --task realtime --prompt "a person" --resolution 224 --bg-image beach.jpg
```

背景图像自动调整为匹配摄像头分辨率。分割掩码决定哪些像素来自实时摄像头（前景）对比背景图像。

### 所有标志

| 标志 | 默认值 | 描述 |
|------|---------|-------------|
| `--task` | `segment` | `detect`、`segment`、`track`、`realtime` |
| `--image` | | 输入图像路径（detect/segment） |
| `--video` | | 输入视频路径（仅 track） |
| `--prompt` | *(必需)* | 文本提示词。多个：`--prompt "a cat" "a dog"` |
| `--boxes` | | 区域过滤器：`"x1,y1,x2,y2"` 或 `"...;..."` 在像素坐标中 |
| `--show-boxes` | 关闭 | 叠加边界框和标签 |
| `--bg-image` | | 摄像头背景交换的背景图像（仅 realtime） |
| `--output` | 自动命名 | 输出文件路径（仅 track） |
| `--model` | `facebook/sam3` | 模型路径或 HF 仓库 |
| `--threshold` | 0.3 / 0.15 | 分数阈值（图像/视频默认） |
| `--nms-thresh` | `0.5` | NMS IoU 阈值 |
| `--every` | `2` | 每 N 帧检测一次（仅 track） |
| `--resolution` | `1008` | 输入分辨率。越低越快：`336` (~8 FPS)、`224` (~11 FPS) |

## 视频：跟踪 (Python)

```python
import cv2
from mlx_vlm.models.sam3.generate import Sam3VideoPredictor

cap = cv2.VideoCapture("video.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
cap.release()

video_predictor = Sam3VideoPredictor(model, processor, score_threshold=0.15)
video_predictor.set_video(frames)
video_predictor.add_text_prompt("a car", frame_idx=0)

results = video_predictor.propagate()
for r in results:
    print(f"Frame {r.frame_idx}: {len(r.object_ids)} objects tracked")
```

### 视频：批量帧处理

为了最大吞吐量，使用低级 API 一次处理多个视频帧：

```python
import cv2
import mlx.core as mx
import numpy as np

cap = cv2.VideoCapture("video.mp4")
batch_size = 4

# 编码文本一次，在所有帧中重用
text_inputs = processor.preprocess_text("a car")
input_ids = mx.array(text_inputs["input_ids"])
attention_mask = mx.array(text_inputs["attention_mask"])
inputs_embeds, attention_mask = model.get_text_features(input_ids, attention_mask)

while cap.isOpened():
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(processor.preprocess_image(pil)["pixel_values"][0])

    if not frames:
        break

    pixel_values = mx.array(np.stack(frames))  # (B, 1008, 1008, 3)
    B = len(frames)

    outputs = model.detect(
        pixel_values,
        attention_mask=mx.tile(attention_mask, (B, 1)),
        inputs_embeds=mx.tile(inputs_embeds, (B, 1, 1)),
    )
    mx.eval(outputs)

    # 处理批次中的每张图像
    for i in range(B):
        scores = 1 / (1 + np.exp(-np.array(outputs["pred_logits"][i])))
        print(f"  {scores[scores > 0.3].shape[0]} detections")

cap.release()
```

> **提示：** 在 CLI 中使用 `--every N` 跳过帧——这提供了真正的 N 倍加速，因为它完全避免了昂贵的 ViT 传递。批处理处理更多帧，但不会提高每帧速度。

有关完整的交互式笔记本演示，请参阅 [`examples/sam3_demo.ipynb`](../../../examples/sam3_demo.ipynb)。

## 架构

```
图像 (1008x1008)
  |
  v
ViT 骨干 (32 层, 1024d, 窗口 + 全局注意力, 2D RoPE)
  |
  v
FPN 颈部 (4 个尺度: 288x288, 144x144, 72x72, 36x36)
  |                                |
  v                                v
DETR 编码器 (6 层)     跟踪器颈部分离 FPN)
  + 文本交叉注意力         |
  |                              v
  v                        SAM2 跟踪器
DETR 解码器 (6 层)      内存注意力
  200 查询                 内存编码器
  框细化                  掩码解码器
  存在 token
  BoxRPB
  |
  +---> DotProductScoring --> pred_logits
  +---> BoxHead           --> pred_boxes (xyxy)
  +---> MaskDecoder/FPN   --> pred_masks (288x288)
```

### 组件

| 组件 | 描述 | 权重前缀 |
|-----------|-------------|---------------|
| 视觉编码器 | ViT-L + FPN | `detector_model.vision_encoder.*` |
| 文本编码器 | CLIP (24L, 1024d) | `detector_model.text_encoder.*` |
| DETR 编码器 | 6 层 pre-norm 变换器，带文本交叉注意力 | `detector_model.detr_encoder.*` |
| DETR 解码器 | 6 层 post-norm，200 查询，框细化，BoxRPB | `detector_model.detr_decoder.*` |
| 几何编码器 | 编码框/点提示词 | `detector_model.geometry_encoder.*` |
| 掩码解码器 | 像素解码器 + 实例投影 | `detector_model.mask_decoder.*` |
| 点积评分 | 文本-查询点积分类器 | `detector_model.dot_product_scoring.*` |
| 跟踪器 | SAM2 风格基于内存的跟踪器 | `tracker_model.*` |
| 跟踪器颈部 | 跟踪器的独立 FPN | `tracker_neck.*` |

## 文件结构

```
mlx_vlm/models/sam3/
├── __init__.py           # 模块导出
├── config.py             # 所有配置数据类
├── sam3.py               # 主模型类 + 清理
├── vision.py             # ViT 骨干 + FPN 颈部
├── text_encoder.py       # CLIP 文本编码器
├── encoder.py            # DETR 变换器编码器 (pre-norm)
├── decoder.py                       # DETR 变换器解码器 (post-norm, BoxRPB)
├── geometry.py           # 几何编码器 (框/点提示词)
├── segmentation.py       # 掩码解码器 + 点积评分
├── position.py           # 正弦 + 2D RoPE 位置编码
├── sam_components.py     # SAM 提示词编码器、掩码解码器、TwoWayTransformer
├── tracker.py            # 内存编码器、内存注意力、跟踪器模型
├── generate.py           # 推理管道 (Sam3Predictor, Sam3VideoPredictor)
└── processing_sam3.py    # 图像/文本预处理
```

## 许可证

原始 SAM3 模型权重由 Meta 在 [**SAM 许可证**](https://huggingface.co/facebook/sam3/blob/main/LICENSE) 下发布，这是一个自定义的宽松许可证，授予非独占的、全球性的、免版税许可，以使用、复制、分发和修改 SAM 材料。关键点：

- 允许商业和研究用途
- 衍生作品必须包含 SAM 许可证的副本和对 Meta 的归属
- 按原样提供，无担保
- 受适用的贸易管制约束

使用它即表示您同意 Meta 的 SAM 许可证条款。有关详细信息，请参阅 [完整许可证文本](https://huggingface.co/facebook/sam3/blob/main/LICENSE)。

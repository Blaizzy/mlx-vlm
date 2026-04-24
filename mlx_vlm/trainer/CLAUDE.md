[根目录](../../CLAUDE.md) > **mlx_vlm** > **trainer**

# MLX-VLM Trainer 模块

> **模块类型**: 训练框架 | **最后更新**: 2026-04-14 | **文档状态**: ✅ 完善

## 模块职责

Trainer 模块提供完整的视觉语言模型微调能力，支持多种训练方法：

- **LoRA/QLoRA 微调**: 参数高效微调（PEFT）
- **SFT (Supervised Fine-Tuning)**: 监督式微调
- **ORPO (Odds Ratio Preference Optimization)**: 偏好优化训练
- **全参数微调**: 完整模型训练
- **数据集支持**: 视觉-语言多模态数据集

## 入口与启动

### CLI 命令

```bash
# LoRA 微调
mlx_vlm.lora --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --train --data /path/to/data.json \
  --iters 100 --batch-size 4 --learning-rate 1e-5

# ORPO 训练
mlx_vlm.orpo --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --train --data /path/to/preference_data.json \
  --iters 200 --beta 0.1
```

### Python API

```python
from mlx_vlm.trainer import train, train_orpo, get_peft_model

# SFT 训练
from mlx_vlm.trainer import TrainingArgs

args = TrainingArgs(
    batch_size=4,
    iters=100,
    learning_rate=1e-5,
    max_seq_length=2048,
    adapter_file="adapters.safetensors"
)

train(model, processor, train_dataset, val_dataset, args)

# LoRA 微调
model = get_peft_model(
    model,
    linear_layers=["q_proj", "v_proj"],
    rank=16,
    alpha=0.1
)
```

## 对外接口

### 主要类和函数

#### 1. LoRaLayer
```python
class LoRaLayer(nn.Module):
    """LoRA 层实现，替换原始 Linear 层"""

    def __init__(
        self,
        linear: Union[nn.Linear, nn.QuantizedLinear],
        rank: int,
        alpha: float = 0.1,
        dropout: float = 0.0
    )
```

#### 2. 训练函数
```python
# SFT 训练
def train(
    model,
    processor,
    train_dataset,
    val_dataset,
    args: TrainingArgs
)

# ORPO 训练
def train_orpo(
    model,
    processor,
    train_dataset,
    val_dataset,
    args: ORPOTrainingArgs
)
```

#### 3. 工具函数
```python
# 应用 LoRA 层
def apply_lora_layers(model, linear_layers, rank=16, alpha=0.1)

# 查找所有线性层名称
def find_all_linear_names(model)

# 打印可训练参数
def print_trainable_parameters(model)

# 保存适配器权重
def save_adapter(model, adapter_file)

# 替换 LoRA 为 Linear 层（合并权重）
def replace_lora_with_linear(model)
```

## 关键依赖与配置

### 依赖项
- **mlx**: 核心计算框架
- **mlx-nn**: 神经网络模块
- **datasets**: HuggingFace 数据集
- **tqdm**: 进度条
- **Pillow**: 图像处理

### 训练配置

#### TrainingArgs (SFT)
```python
@dataclass
class TrainingArgs:
    batch_size: int = 4
    iters: int = 100
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    max_seq_length: int = 2048
    adapter_file: str = "adapters.safetensors"
    grad_checkpoint: bool = False
    learning_rate: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_learning_rate: float = 1e-6
    full_finetune: bool = False
    gradient_accumulation_steps: int = 1
```

#### ORPOTrainingArgs
```python
@dataclass
class ORPOTrainingArgs(TrainingArgs):
    beta: float = 0.1  # ORPO beta 参数
    eps: float = 1e-8  # 数值稳定性常数
```

## 数据模型

### VisionDataset
```python
class VisionDataset:
    """视觉-语言数据集"""

    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_resize_shape=None
    )

    def process(self, item):
        """处理单个数据项"""
        # 支持图像、音频、对话
```

### 数据格式
```json
{
  "images": ["path/to/image.jpg"],
  "messages": [
    {"role": "user", "content": "What is in this image?"},
    {"role": "assistant", "content": "A cat sitting on a couch."}
  ]
}
```

### PreferenceVisionDataset (ORPO)
```python
class PreferenceVisionDataset:
    """偏好数据集，用于 ORPO 训练"""

    def process(self, item):
        # 返回 chosen 和 rejected 样本
```

## 测试与质量

### 测试文件
- **位置**: `mlx_vlm/tests/test_trainer.py`
- **测试内容**: LoRA 层、训练循环、损失函数
- **运行**: `python -m unittest mlx_vlm.tests.test_trainer`

### 支持的操作模式
- **LoRA 微调**: ✅ 支持
- **QLoRA**: ✅ 支持（量化 + LoRA）
- **全参数微调**: ✅ 支持
- **梯度检查点**: ✅ 支持
- **梯度累积**: ✅ 支持
- **混合精度**: ✅ 自动支持

### 不支持的模型
```python
not_supported_for_training = {"gemma3n", "qwen3_omni"}
```

## 常见问题 (FAQ)

### Q1: 如何选择 LoRA rank 和 alpha？
**A**:
- **rank**: 通常 8-32，越大参数越多，性能可能更好但内存占用增加
- **alpha**: 通常 0.1-1.0，控制 LoRA 更新的缩放
- 经验法则: `alpha = rank / 2`

### Q2: 如何处理多模态数据？
**A**:
```python
from mlx_vlm.trainer import VisionDataset

dataset = VisionDataset(
    hf_dataset=raw_dataset,
    config=model.config,
    processor=processor,
    image_resize_shape=(384, 384)
)
```

### Q3: ORPO 训练需要什么样的数据？
**A**:
```json
{
  "images": ["image.jpg"],
  "chosen": [
    {"role": "user", "content": "What is this?"},
    {"role": "assistant", "content": "Good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "What is this?"},
    {"role": "assistant", "content": "Bad answer"}
  ]
}
```

### Q4: 如何合并 LoRA 权重？
**A**:
```python
from mlx_vlm.trainer import replace_lora_with_linear

# 训练完成后
replace_lora_with_linear(model)
# 现在 model.layers 包含合并后的 Linear 层
```

### Q5: 如何减少内存使用？
**A**:
1. 启用梯度检查点: `grad_checkpoint=True`
2. 使用量化模型
3. 减小 batch_size
4. 使用梯度累积: `gradient_accumulation_steps=4`

## 相关文件清单

### 核心文件
- `trainer/__init__.py`: 模块入口，导出所有公共接口
- `trainer/lora.py`: LoRA 层实现和合并逻辑
- `trainer/sft_trainer.py`: SFT 训练循环和损失函数
- `trainer/orpo_trainer.py`: ORPO 训练逻辑
- `trainer/datasets.py`: 数据集类和数据预处理
- `trainer/utils.py`: 工具函数（学习率调度、参数统计等）

### 测试文件
- `mlx_vlm/tests/test_trainer.py`: 训练器单元测试
- `mlx_vlm/tests/test_trainer_utils.py`: 工具函数测试

### 文档
- `docs/usage.md`: 用户指南
- `CONTRIBUTING.md`: 贡献指南

## 变更记录 (Changelog)

### 2026-04-14 - 初始化文档 🚀
- ✅ **创建模块文档**: 完成 trainer 模块 CLAUDE.md
- 📖 **完整接口文档**: LoRA、SFT、ORPO 训练接口
- 🔧 **配置说明**: 训练参数详细说明
- 📊 **数据模型**: VisionDataset、PreferenceVisionDataset
- ❓ **FAQ**: 5 个常见问题解答

---

**相关模块**:
- [mlx_vlm/generate](../generate/) - 推理引擎
- [mlx_vlm/models](../models/) - 模型实现
- [mlx_vlm/tests](../tests/) - 测试套件

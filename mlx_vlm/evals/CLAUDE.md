[根目录](../../CLAUDE.md) > **mlx_vlm** > **evals**

# MLX-VLM Evals 模块

> **模块类型**: 基准测试 | **最后更新**: 2026-04-14 | **文档状态**: ✅ 完善

## 模块职责

Evals 模块提供视觉语言模型的标准化基准测试能力，支持多个学术和工业界的评估数据集：

- **MathVista**: 数学视觉推理基准测试
- **MMMU**: 多学科多模态理解
- **MMStar**: 多模态综合评估
- **OCRBench**: OCR 性能基准测试

## 入口与启动

### CLI 命令

```bash
# MathVista 基准测试
python -m mlx_vlm.evals.math_vista \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --split testmini

# MMMU 基准测试
python -m mlx_vlm.evals.mmmu \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit

# MMStar 基准测试
python -m mlx_vlm.evals.mmstar \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit

# OCRBench 基准测试
python -m mlx_vlm.evals.ocrbench \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Python API

```python
from mlx_vlm import load
from mlx_vlm.evals.utils import inference

# 加载模型
model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")

# 运行推理
response = inference(
    model,
    processor,
    question="What is in this image?",
    image="path/to/image.jpg",
    max_tokens=1000,
    temperature=0.0
)
```

## 对外接口

### 核心函数

#### 1. inference (通用推理)
```python
def inference(
    model,
    processor,
    question: str,
    image: Union[str, list],
    max_tokens: int = 3000,
    temperature: float = 0.0,
    resize_shape: Optional[tuple] = None,
    verbose: bool = False
) -> str
```
**功能**: 对单个样本运行推理

#### 2. MathVista 评估
```python
# process_question: 格式化问题（包括多选题选项）
def process_question(sample: dict) -> str

# normalize_answer: 提取模型回答中的答案
def normalize_answer(response: str, problem: dict) -> Optional[str]
```

#### 3. 数据集加载
```python
from datasets import load_dataset

# MathVista
dataset = load_dataset("AI4Math/MathVista", split="testmini")

# MMMU
dataset = load_dataset("MMMU/MMMU", split="test")

# MMStar
dataset = load_dataset("Lin-Chen/MMStar", split="test")

# OCRBench
dataset = load_dataset("pbevan11/OCRBench", split="test")
```

## 关键依赖与配置

### 依赖项
- **mlx-vlm**: 核心库
- **datasets**: HuggingFace 数据集
- **Pillow**: 图像处理
- **tqdm**: 进度条
- **numpy**: 数值计算

### 基准测试配置

#### MathVista
- **数据集**: AI4Math/MathVista
- **Split**: testmini (快速测试) / test (完整测试)
- **问题类型**: multi_choice, free_form, text_only
- **答案类型**: float, int, str, bool
- **评估指标**: 准确率（精确匹配）

#### MMMU
- **数据集**: MMMU/MMMU
- **Split**: validation / test
- **学科**: 11 个学科（艺术、商业、科学等）
- **评估指标**: 准确率

#### MMStar
- **数据集**: Lin-Chen/MMStar
- **Split**: test
- **类别**: 12 个类别
- **评估指标**: 准确率

#### OCRBench
- **数据集**: pbevan11/OCRBench
- **Split**: test
- **任务**: 文本识别、文档理解
- **评估指标**: 准确率

## 数据模型

### MathVista 样本格式
```python
{
    "query": "What is the value of x?",
    "question_type": "multi_choice",  # 或 "free_form", "text_only"
    "answer_type": "int",  # 或 "float", "str", "bool"
    "choices": ["10", "20", "30", "40"],
    "answer": "20",
    "image": "path/to/image.jpg"
}
```

### MMMU 样本格式
```python
{
    "question": "What is shown in this image?",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "image": "path/to/image.jpg",
    "subject": "Art",
    "category": "Fine Arts"
}
```

## 测试与质量

### 评估流程

#### 1. 数据准备
```python
from datasets import load_dataset

dataset = load_dataset("AI4Math/MathVista", split="testmini")
```

#### 2. 批量推理
```python
results = []
for sample in tqdm(dataset):
    question = process_question(sample)
    response = inference(
        model,
        processor,
        question=question,
        image=sample["image"],
        max_tokens=1000
    )
    results.append({
        "question": question,
        "prediction": response,
        "ground_truth": sample["answer"]
    })
```

#### 3. 结果评估
```python
correct = 0
for result in results:
    prediction = normalize_answer(result["prediction"], result)
    if prediction == result["ground_truth"]:
        correct += 1

accuracy = correct / len(results)
print(f"Accuracy: {accuracy:.2%}")
```

### 答案提取策略

#### 多选题 (Multi-choice)
1. 查找 `\boxed{X}` 模式
2. 查找 "故选：X" 或 "answer: X" 模式
3. 查找 "(A)", "A)", "A." 模式
4. 优先匹配最后出现的选项

#### 自由形式 (Free-form)
1. 提取数值答案
2. 支持 float, int, str 类型
3. 数值容差比较

#### 文本问题 (Text-only)
1. 直接文本匹配
2. 支持布尔值答案

## 常见问题 (FAQ)

### Q1: 如何运行快速测试？
**A**:
```bash
# 使用 testmini split（约 1000 样本）
python -m mlx_vlm.evals.math_vista \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --split testmini
```

### Q2: 如何保存评估结果？
**A**:
```python
import json

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Q3: 如何处理多图像样本？
**A**:
```python
response = inference(
    model,
    processor,
    question="Compare these images",
    image=["image1.jpg", "image2.jpg"],  # 列表形式
    max_tokens=1000
)
```

### Q4: 如何提高评估速度？
**A**:
1. 使用量化模型（4-bit）
2. 减小 `max_tokens` 限制
3. 批量推理（如果内存允许）
4. 使用更快的采样策略（temperature=0.0）

### Q5: 如何调试答案提取问题？
**A**:
```python
from mlx_vlm.evals.math_vista import normalize_answer

prediction = model_output
ground_truth = sample["answer"]

extracted = normalize_answer(prediction, sample)
print(f"Raw: {prediction}")
print(f"Extracted: {extracted}")
print(f"Ground Truth: {ground_truth}")
```

## 性能基准

### 预期性能（Qwen2-VL-2B）

| 基准测试 | 准确率 | 推理速度 |
|---------|-------|---------|
| MathVista (testmini) | ~35-40% | ~2-3 samples/s |
| MMMU (validation) | ~30-35% | ~1-2 samples/s |
| MMStar (test) | ~40-45% | ~2-3 samples/s |
| OCRBench (test) | ~60-70% | ~3-5 samples/s |

*注：性能因硬件、量化精度、超参数而异*

## 相关文件清单

### 核心文件
- `evals/__init__.py`: 模块入口（空文件）
- `evals/math_vista.py`: MathVista 基准测试
- `evals/mmmu.py`: MMMU 基准测试
- `evals/mmstar.py`: MMStar 基准测试
- `evals/ocrbench.py`: OCRBench 基准测试
- `evals/utils.py`: 通用推理函数

### 测试文件
- **注**: 目前没有自动化测试（手动测试）

### 文档
- `docs/usage.md`: 用户指南（包含评估部分）

## 变更记录 (Changelog)

### 2026-04-14 - 初始化文档 🚀
- ✅ **创建模块文档**: 完成 evals 模块 CLAUDE.md
- 📖 **完整接口文档**: 4 个基准测试的详细说明
- 🔧 **配置说明**: 数据集配置和评估流程
- 📊 **数据模型**: 各基准测试的数据格式
- ❓ **FAQ**: 5 个常见问题解答
- 📈 **性能基准**: 预期性能参考

### 下一步建议
1. 添加自动化测试（单元测试）
2. 添加更多基准测试（如 MME、SEED-Bench）
3. 实现批量推理优化
4. 添加结果可视化工具
5. 支持分布式评估

---

**相关模块**:
- [mlx_vlm/generate](../generate/) - 推理引擎
- [mlx_vlm/models](../models/) - 模型实现
- [mlx_vlm/trainer](../trainer/) - 训练框架

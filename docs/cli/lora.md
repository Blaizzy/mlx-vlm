# mlx_vlm.lora — fine-tuning

Fine-tune a vision-language model with LoRA/QLoRA adapters or full weight fine-tuning, in SFT or ORPO mode. Requires the `train` extra: `pip install -U 'mlx-vlm[train]'`.

## Synopsis

```
python -m mlx_vlm.lora --model-path <repo_or_path> --dataset <repo_or_path> [OPTIONS]
```

`lora` is not a console script; invoke it with `python -m mlx_vlm.lora`. There is no positional argument — the base model comes from `--model-path` (which has a default), so `--dataset` is the only required flag.

## Examples

LoRA fine-tune (default: rank 8, alpha 16):

```
python -m mlx_vlm.lora \
  --model-path mlx-community/Qwen2-VL-2B-Instruct-bf16 \
  --dataset your-org/your-vlm-dataset \
  --iters 1000 --batch-size 4
```

QLoRA — train LoRA adapters on a 4-bit quantized base by pointing `--model-path` at a 4-bit repo:

```
python -m mlx_vlm.lora \
  --model-path mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --dataset your-org/your-vlm-dataset \
  --lora-rank 16 --lora-alpha 32 --learning-rate 1e-4
```

Full fine-tune (train all language-model weights instead of adapters):

```
python -m mlx_vlm.lora \
  --model-path mlx-community/Qwen2-VL-2B-Instruct-bf16 \
  --dataset your-org/your-vlm-dataset \
  --full-finetune --learning-rate 2e-5
```

Resume from an existing adapter:

```
python -m mlx_vlm.lora \
  --model-path mlx-community/Qwen2-VL-2B-Instruct-bf16 \
  --dataset your-org/your-vlm-dataset \
  --adapter-path ./adapters
```

## Options

### Model

| Flag | Default | Description |
| --- | --- | --- |
| `--model-path` | `mlx-community/Qwen2-VL-2B-Instruct-bf16` | HF repo or local path of the base VLM to fine-tune. |

### Data

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` | *(required)* | HF dataset repo or local path to train on. |
| `--split` | `train` | Dataset split to load. |
| `--dataset-config` | `None` | Dataset configuration/subset name passed to `load_dataset`. |
| `--image-resize-shape` | `None` | Two integers (width height) to resize dataset images to. |
| `--custom-prompt-format` | `None` | Custom JSON prompt template mapping `{image}`/`{question}`/`{answer}` into messages. |
| `--train-on-completions` | `False` | Compute the loss only on assistant/completion tokens. |
| `--assistant-id` | `77091` | Token id marking the assistant turn, used for completion-only loss masking. |

### Adapter / LoRA

| Flag | Default | Description |
| --- | --- | --- |
| `--lora-rank` | `8` | LoRA rank. |
| `--lora-alpha` | `16` | LoRA scaling alpha. |
| `--lora-dropout` | `0.0` | LoRA dropout probability. |
| `--full-finetune` | `False` | Train all language-model weights instead of LoRA adapters. |
| `--train-vision` | `False` | Also unfreeze and train the vision stack (vision tower, projector, connector, etc.). |

### Optimization

| Flag | Default | Description |
| --- | --- | --- |
| `--learning-rate` | `2e-5` | Adam learning rate. |
| `--batch-size` | `4` | Number of samples per training step. |
| `--iters` | `1000` | Number of training iterations (ignored when `--epochs` is set). |
| `--epochs` | `None` | Number of epochs; when set, overrides `--iters` (iters = len(dataset)//batch_size × epochs). |
| `--gradient-accumulation-steps` | `1` | Accumulate gradients over N steps before each optimizer update. |
| `--grad-checkpoint` | `False` | Use gradient checkpointing to reduce memory. |
| `--grad-clip` | `None` | Clip the gradient norm to this value (no clipping when unset). |
| `--max-seq-length` | `2048` | Maximum sequence length. |

### Training mode (SFT / ORPO)

| Flag | Default | Description |
| --- | --- | --- |
| `--train-mode` | `sft` | Training mode: `sft` (default) or `orpo`. |
| `--beta` | `0.1` | ORPO beta (odds-ratio weight). |
| `--eps` | `1e-8` | ORPO numerical stability epsilon. |

### Reporting & checkpoints

| Flag | Default | Description |
| --- | --- | --- |
| `--steps-per-report` | `10` | Report training loss every N steps. |
| `--steps-per-eval` | `200` | Run validation every N steps. |
| `--steps-per-save` | `100` | Save the adapter/checkpoint every N steps. |
| `--val-batches` | `4` | Number of validation batches per evaluation. |

### Output

| Flag | Default | Description |
| --- | --- | --- |
| `--output-path` | `adapters.safetensors` | Where the trained adapters are saved; a directory gets `/adapters.safetensors` appended. |
| `--adapter-path` | `None` | Existing adapter path to resume from (LoRA layers are applied before training). |

## See also

- [Fine-tuning guide](../fine-tuning.md)

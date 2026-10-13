# VLM training

Install the training extra in the environment used by your CLI or notebook:

```sh
python -m pip install -e '.[train]'
```

The CLI is `python -m mlx_vlm.train` (or the installed `mlx_vlm.train`
command). It uses the same `Trainer`, argument dataclasses, preprocessing,
callbacks, and MLX engine as the Python API. VLM SFT, ORPO, and DPO are supported.

## Notebook and Python API

Start with one of the standalone notebooks:

- [SFT](../examples/training/sft_text_only.ipynb): text-only JOSIE instruction training.
- [LaTeX OCR SFT](../examples/training/sft_vison_text.ipynb): map Hugging Face `image`/`text` rows into messages and preprocess with `VisionDataset`.
- [ORPO](../examples/training/orpo.ipynb): chosen/rejected responses without a reference model.
- [DPO](../examples/training/dpo.ipynb): chosen/rejected responses with a separate frozen reference.
- [Custom loss](../examples/training/custom_loss.ipynb): replace the SFT objective with chunked cross-entropy.

The examples include generated color images or a Hub vision dataset, editable model/data/settings, training,
evaluation, saving, and reload instructions. Install `jupyterlab ipykernel`
and select the environment where MLX-VLM is installed. Run from top to bottom
in a fresh kernel; relative output paths use the kernel's current directory.
The toy examples demonstrate usage, so replace them with representative data
before assessing model quality.

```python
from mlx_vlm import load
from mlx_vlm.trainer import Trainer, VLMTrainingArgs, prepare_model_for_training
from mlx_vlm.trainer.datasets import load as load_dataset

model, processor = load("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
model = prepare_model_for_training(model, lora_rank=8, lora_alpha=16)
rows = load_dataset("data")
args = VLMTrainingArgs(
    iters=100,
    batch_size=1,
    gradient_accumulation_steps=4,
    train_on_completions=True,
    adapter_file="adapters/sft/weights.safetensors",
)
trainer = Trainer(
    model, args,
    processing_class=processor,
    train_dataset=rows["train"],
    eval_dataset=rows.get("validation", rows.get("valid")),
)
metrics = trainer.train()
validation = trainer.evaluate()  # Requires an evaluation split.
trainer.save_adapter()
```

Pass one finite, indexable split: a list of row dictionaries or a Hugging Face
`Dataset`. Raw rows require `processing_class`. `prepared=True` accepts
processor outputs directly, including `input_ids` and optional masks/media.
`Trainer.evaluate()` restores the model's previous training/evaluation mode.

`prepare_model_for_training` accepts `train_type="lora"`, `"dora"`, or `"full"`.
Adapter modes freeze the whole base and discover language transformer linear
layers by default. `target_modules` accepts linear suffixes or qualified names,
including explicit vision targets. Optional `quantization_bits=4` or `8`
quantizes compatible layers before attaching adapters. Full training dequantizes
packed weights and unfreezes the whole model, which requires more memory.

For ORPO, pass `ORPOTrainingArgs(beta=0.1)` and preference rows; for DPO, pass
`DPOTrainingArgs(beta=0.1)` and `reference_model=reference`. Argument types select
the algorithm automatically, or set `algorithm="orpo"` / `"dpo"` explicitly.
Load a separate compatible base for DPO. The trainer freezes it and checks its
model type, image token identifiers, and vocabulary size against the policy.

## Training and validation metrics

Every task loss returns a scalar loss and a dictionary of scalar metrics:

```python
def loss(model, batch):
    # Compute the objective and any diagnostics using MLX.
    loss_value = ...
    metrics = {
        "weight": supervised_tokens,
        "num_tokens": supervised_tokens,
        "my_quality_score": quality_score,
    }
    return loss_value, metrics
```

Both training and scheduled validation print every reported metric by iterating
through the dictionary. Metric names flow through to callbacks/W&B and
`trainer.log_history`, with `train_` or `val_` prefixes. `trainer.train()` returns
the last report and final scheduled validation; `trainer.evaluate()` returns
all validation metrics. For example:

```python
metrics = trainer.train()
for name, value in metrics.items():
    print(f"{name}: {value}")
report_history = trainer.log_history  # Reports from the latest train() run.
validation = trainer.evaluate()
```

`weight` controls the loss and diagnostic means (supervised tokens for SFT,
pairs for preference objectives); it defaults to 1 for an empty metrics dict.
Names starting with `num_` are summed counts. Other metrics are weighted means,
with separate denominators for metrics that are omitted in some batches.
Use nonnegative scalar weights and integer counts; loss and metrics must stay
in MLX inside a compiled loss, without `.item()` or Python tensor-dependent
branches. Distributed workers must emit the same metric keys for each batch.
Previous scalar losses and `(loss, weight)` custom callables are accepted for
compatibility. Unreduced tensor helpers such as `token_cross_entropy` continue
to return tensors for composing objectives.

SFT reports `token_accuracy` over supervised next-token targets and `nll`.
Perplexity is `exp(weighted_mean_nll)`, computed after aggregation. DPO reports
reward accuracy/margin, chosen/rejected rewards, and sequence log-probabilities.
ORPO reports preference accuracy, odds margin, chosen NLL, and preference loss.
These are training diagnostics; domain-specific held-out evaluations remain
necessary to assess OCR or instruction-following quality.

The shared engine also reports the actual optimizer learning rate, cumulative
supervised tokens, cumulative processed tokens, sequences, throughput, optimizer
updates, step/data preparation time, progress/ETA, and active/cache/peak MLX
memory in decimal GB. Counts are reduced globally across distributed workers;
timings and memory describe the reporting worker.

`processed_tokens` counts non-padding `input_ids[:, :-1]` positions passed to
the policy, including prompts and media placeholders. `trained_tokens` counts
supervised next-token targets. These counts include repeated training examples
but exclude DPO reference forwards and checkpoint recomputation; DPO separately
reports `num_reference_tokens`. `processed_tokens_per_second` uses synchronized
training compute time (forward, backward, and optimizer updates), excluding data
preparation. `end_to_end_tokens_per_second` also includes batch preparation,
but excludes validation/checkpoint/report overhead. Validation's
`val_processed_tokens_per_second` measures synchronized forward/loss compute
without backward or updates. `elapsed_time` includes validation, reporting,
and checkpoint overhead; ETA estimates from elapsed time per microstep. First
steps can include compilation and warm-up.

`train_num_vision_tokens` counts image/video input positions per reporting
interval; `total_vision_tokens` accumulates them across the run, and
`vision_tokens_per_second` reports their training throughput. Validation exposes
`val_num_vision_tokens` and `val_vision_tokens_per_second`. Counts use the model's
configured image/video token IDs, after processor expansion, and exclude padding
and the final token omitted by causal shifting. For Qwen this counts merged visual
embeddings presented to the language model, not the vision encoder's raw patches
or image boundary markers. Models that inject visual embeddings without expanded
token IDs need a model-specific mask to count those embeddings.

The shared `cross_entropy(..., vision_mask=...)` accepts a boolean mask aligned
with logits [B, T]. This is a forward-input mask, independent of the target loss
mask: prompt images are counted even with completion-only supervision. It adds
`num_vision_tokens` to the returned metrics without changing loss weighting,
accuracy, or gradients. Text-only batches report zero vision tokens.

A custom loss can report additional counts such as `num_processed_tokens`,
`num_padded_tokens`, and `num_sequences` to enable the engine's token throughput
and padding diagnostics. The custom-loss notebook shows this using
`vlm_batch_metrics(batch, model)`; replacing only the loss requires no engine changes.

## CLI

```sh
python -m mlx_vlm.train \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --data data --algorithm sft --train-on-completions \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --iters 100 --learning-rate 1e-4 \
  --adapter-file adapters/sft/weights.safetensors

python -m mlx_vlm.train \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --data preferences --algorithm orpo --beta 0.1 \
  --adapter-file adapters/orpo/weights.safetensors
```

DPO uses `--algorithm dpo` and optionally `--reference-model`; otherwise it
loads a separate copy of `--model`. Use `--train-type dora` or `full` to change
model preparation. `--quantization-bits 4` enables optional base quantization.
`--train-vision` also unfreezes the vision tower/projector for CLI training.

`--data` loads a JSONL file, a folder with `train.jsonl` and optional
`validation.jsonl` / `valid.jsonl`, or a Hub dataset repository containing those
files. Relative image paths resolve against the split folder. For datasets
published in Parquet or other Hugging Face formats, use `--dataset owner/name`
and optionally `--hf-dataset-config name` instead. Select splits with
`--train-split` and `--validation-split`.

The previous CLI's main flag aliases are supported: `--model-path`,
`--train-mode`, `--split`, `--adapter-path`, `--output-path`, and
`--full-finetune`. `--output-path` accepts a weights file or an output folder.
`--dataset-config` now accepts a JSON object of preprocessing options; use
`--hf-dataset-config` for a Hugging Face configuration name.

Settings can be saved in JSON or YAML and loaded with `--config path`:

```yaml
model: mlx-community/Qwen2.5-VL-3B-Instruct-4bit
data: data
algorithm: sft
iters: 100
batch_size: 1
train_on_completions: true
adapter_file: adapters/sft/weights.safetensors
```

Explicit command-line flags override configuration values. `--epochs` counts
complete batches within each media group and cannot be combined with an explicit
`--iters`. `iters` counts microbatches; gradient accumulation reduces optimizer
updates. Incomplete global batches are dropped, so start with batch size 1.
`--steps-per-eval none` disables scheduled evaluation, `--val-batches -1`
evaluates the full prepared split, and `--cache-size` bounds the media cache.
Compilation is opt-in with `--compile`.

## Row formats

SFT rows contain chat messages and optional images:

```json
{"images": ["images/red.png"], "messages": [{"role": "user", "content": "Which color?"}, {"role": "assistant", "content": "Red."}]}
```

`conversations` with `from`/`value` roles and `question`/`answer` rows are also
accepted. In Python, image values can be PIL images or decoded Hugging Face image
records. `train_on_completions=True` derives a mask from the tokenized assistant
prefix. Truncation that would remove image placeholders is rejected.

Preference rows share media and a prompt:

```json
{"images": ["images/red.png"], "prompt": "Which color?", "chosen": "Red.", "rejected": "Blue."}
```

Full chosen/rejected chat histories are supported too. Custom column names can
be passed through `dataset_config`, for example
`{"prompt_feature": "question", "chosen_feature": "good", "rejected_feature": "bad"}`.

## Saving and continuing training

Checkpoints contain trainable weights and `adapter_config.json`, including
adapter targets and base quantization metadata. Keep both together and reload
onto the original base:

```python
model, processor = load(MODEL_ID)
model = prepare_model_for_training(
    model,
    checkpoint_path="adapters/sft/weights.safetensors",
)
model.eval()
```

For full checkpoints, also pass `train_type="full"`. A checkpoint folder can
contain `weights.safetensors` or `adapters.safetensors`; pass a file to select
numbered checkpoints. Set `--resume-adapter-file` in the CLI to continue with
a fresh optimizer. Optimizer and RNG state are not saved.

Callbacks implement `TrainingCallback.on_train_loss_report` and
`on_val_loss_report`. Pass a callback through `training_callback`, or use
`--wandb project` after installing the optional `wandb` package.

## Backend structure and compatibility

`trainer/common` owns configuration, model preparation, the shared engine,
callbacks, and the `TrainingTask` contract. `trainer/datasets/loading.py` owns
JSONL loading and bounded caching. `trainer/vlm` owns VLM preprocessing and SFT,
ORPO, and DPO objectives. `trainer/peft` uses MLX-VLM's local LoRA/DoRA layers.
The registry in `trainer/api.py` selects the supported VLM tasks; a custom
`TrainingTask` can be passed directly to `Trainer`.

Existing functional SFT/ORPO trainers and dataset imports remain available.
They use their original argument classes. The public `ORPOTrainingArgs` now
configures the notebook/shared engine API; import
`mlx_vlm.trainer.vlm.orpo.trainer.ORPOTrainingArgs` (also exposed as
`mlx_vlm.trainer.vlm.LegacyORPOTrainingArgs`) for the older `ORPOTrainer` /
`train_orpo` recipe. `TrainingArgs` continues to configure the older SFT recipe;
use `VLMTrainingArgs` with the new `Trainer`.

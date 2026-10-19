# VLM training notebooks

Install the repository into your notebook environment:

```sh
python -m pip install -e '.[train]' jupyterlab ipykernel
jupyter lab
```

Open a notebook and run its cells from top to bottom in a fresh kernel.
The examples use text conversations, generated images, or a Hugging Face vision dataset,
and the same public API as
`python -m mlx_vlm.train`:

| Notebook | Training objective |
| --- | --- |
| [sft_text_only.ipynb](sft_text_only.ipynb) | Text-only response SFT with JOSIE-v2-Instruct-5K |
| [sft_vison_text.ipynb](sft_vison_text.ipynb) | LaTeX OCR from Hugging Face image/text rows with `.map()` and `VisionDataset` |
| [orpo.ipynb](orpo.ipynb) | Reference-free preference optimization |
| [dpo.ipynb](dpo.ipynb) | Preference optimization with a separate frozen reference |
| [grpo_vision.ipynb](grpo_vision.ipynb) | Qwen3.5-2B-4bit image-conditioned GRPO on LaTeX OCR with custom rewards |
| [grpo_text_only.ipynb](grpo_text_only.ipynb) | Qwen3.5-2B-4bit arithmetic GRPO on simple-grpo with correctness/format rewards |
| [custom_loss.ipynb](custom_loss.ipynb) | Notebook-defined chunked cross-entropy |

Edit the model and training settings near the top. The text-only notebook
loads `mlx-community/JOSIE-v2-Instruct-5K`, filters out rows without assistant targets, preserves retained messages, and holds
out 5% of the rows for validation. The generated-image examples accept a
JSONL folder through `DATA_PATH`. The LaTeX OCR example loads `mlx-community/LaTeX_OCR`,
preserves decoded images, maps `text` into chat messages, and uses a seeded
validation partition from `train`. It starts with 2,048 training samples;
set `MAX_TRAIN_SAMPLES = None` to use the full training partition.
Checkpoints default to a relative `adapters/vlm/...`
folder under the kernel's working directory. First use can download the base
model. DPO loads two models, so allow memory for the policy and reference.

Each notebook constructs an MLX `AdamW` optimizer beside its `Trainer` setup.
Edit learning rate, weight decay, or schedules on the optimizer directly;
training arguments do not contain optimizer settings.

Reports include dynamic loss metrics, learning rate, supervised/processed
token counts and rates, timings, memory, and progress. Inspect
`trainer.log_history` for the report history; each notebook includes a cell for
this. SFT reports token accuracy/perplexity and preference training reports
reward or odds diagnostics.

The GRPO notebook generates groups of responses from image prompts and compares
them against LaTeX targets kept outside the prompt. It includes dense similarity
and exact-match rewards, group reward diagnostics, sample inspection, and saving.
Omit the image column to train text-only GRPO. `beta=0` skips loading a reference;
positive beta enables KL regularization with a separate frozen model.

The text-only GRPO notebook loads `mlx-community/Qwen3.5-2B-4bit` directly
and trains on `mlx-community/simple-grpo` without an additional SFT warm-up.
The checkpoint is already post-trained. It uses DeepSeek-R1-Zero-inspired correctness and reasoning-tag
format rewards, keeps the supplied `train`/`valid`/`test` partitions separate,
and uses the published GRPO training instruction from Table 1 of the DeepSeek-R1
paper. Its map function builds the full prompt, keeping answers separate for
rewards, and disables additional chat formatting through dataset_config.
It saves the plain prompt template alongside the adapters and compares
baseline and final rewards and inspects sampled responses.
Its setup notes explain restarting the kernel after dependency updates and
repairing a mixed Hugging Face Hub installation if imports still fail.

Their code cells are tested with tiny
real MLX models and fixture processors across LoRA, DoRA, and full training,
including evaluation and checkpoint reloads. These checks substitute model
loading and shorten training; the default Hub model runs are not covered.

See [the training guide](../../docs/training.md) for CLI examples, dataset
schemas, checkpoint handling, callbacks, and compatibility details.

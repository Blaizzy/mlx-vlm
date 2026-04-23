"""
Smoke test: real Gemma 4 E2B training with mlx-vlm patches.

Loads the actual model, processes 5 real images, runs 5 training steps,
and checks that loss is finite (not NaN/Inf).

Usage: python test_gemma4_training_smoke.py
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

# Use our patched mlx-vlm
from mlx_vlm.utils import load
from mlx_vlm.trainer.sft_trainer import (
    TrainingArgs,
    vision_language_loss_fn,
    iterate_batches,
)
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import (
    find_all_linear_names,
    get_peft_model,
    print_trainable_parameters,
)

MODEL_PATH = "mlx-community/gemma-4-e2b-it-8bit"
DATASET_PATH = Path("/Users/ukintvs/Documents/projects/gemma4-finetune/security-cam/data/processed/uca_dataset.json")
NUM_SAMPLES = 5
NUM_STEPS = 5


def load_local_dataset(path, n_samples):
    """Load our security cam dataset and convert to HF-like format."""
    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data[:n_samples * 2]:  # load extra in case some have missing images
        if len(samples) >= n_samples:
            break

        messages = item["messages"]
        images = []
        ok = True

        for msg in messages:
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        img_path = part["image"]
                        if not Path(img_path).exists():
                            ok = False
                            break
                        from PIL import Image
                        images.append(Image.open(img_path).convert("RGB"))
            if not ok:
                break

        if ok:
            samples.append({"messages": messages, "images": images})

    return samples


class SimpleDataset:
    """Minimal dataset wrapper for mlx-vlm VisionDataset."""
    def __init__(self, items):
        self.items = items
        self.column_names = list(items[0].keys()) if items else []
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


def main():
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: {mx.metal.is_available()}")
    print(f"Model: {MODEL_PATH}")
    print(f"Samples: {NUM_SAMPLES}, Steps: {NUM_STEPS}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model, processor = load(MODEL_PATH, processor_config={"trust_remote_code": True})
    print(f"  Loaded in {time.time() - t0:.1f}s")

    config = model.config.__dict__

    # Load dataset
    print(f"Loading {NUM_SAMPLES} samples from {DATASET_PATH.name}...")
    raw_samples = load_local_dataset(DATASET_PATH, NUM_SAMPLES)
    print(f"  Loaded {len(raw_samples)} samples")

    hf_dataset = SimpleDataset(raw_samples)
    train_dataset = VisionDataset(hf_dataset, config, processor)

    # Apply LoRA
    print("Applying LoRA...")
    modules = find_all_linear_names(model.language_model)
    model = get_peft_model(model, modules, rank=8, alpha=16, dropout=0.0, verbose=False)
    print_trainable_parameters(model)

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=1e-5)

    # Manual training loop (simpler than full train() for smoke test)
    print(f"\n{'='*50}")
    print(f"TRAINING SMOKE TEST — {NUM_STEPS} steps")
    print(f"{'='*50}")

    loss_value_and_grad = nn.value_and_grad(model, vision_language_loss_fn)
    losses = []
    nan_count = 0

    for step in range(NUM_STEPS):
        t_step = time.time()

        # Get batch
        batch_iter = iterate_batches(train_dataset, batch_size=1, max_seq_length=512, train=True)
        batch = next(batch_iter)

        # Forward + backward
        loss, grads = loss_value_and_grad(model, batch)
        mx.eval(loss)

        loss_val = loss.item()
        is_nan = loss_val != loss_val  # NaN check
        is_inf = abs(loss_val) == float("inf")

        if is_nan:
            nan_count += 1
            status = "NaN ❌"
        elif is_inf:
            nan_count += 1
            status = "Inf ❌"
        else:
            status = f"{loss_val:.4f}"
            # Apply gradients only if loss is finite
            grads = tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        losses.append(loss_val)
        dt = time.time() - t_step
        print(f"  Step {step+1}/{NUM_STEPS}: loss={status}  ({dt:.1f}s)")

    print(f"\n{'='*50}")
    print(f"RESULT")
    print(f"{'='*50}")

    if nan_count == 0:
        print(f"  ALL {NUM_STEPS} STEPS FINITE — patches work!")
        print(f"  Loss range: {min(losses):.4f} — {max(losses):.4f}")
        return 0
    else:
        print(f"  {nan_count}/{NUM_STEPS} steps had NaN/Inf — patches may not be working")
        return 1


if __name__ == "__main__":
    sys.exit(main())

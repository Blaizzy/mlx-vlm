"""DWQ (distilled weight quantization).

Recovers low-bit quality by distilling a round-to-nearest quantized student
toward the full-precision teacher: the packed integer weights are frozen and
only the continuous quantization ``scales``/``biases`` are optimized to match
the teacher's output distribution over a calibration set. The forward is the
model's own path, so for multimodal calibration the distillation signal is
conditioned on the image/audio activations the model will see at inference.
Pure mlx.
"""

import functools
import re
from typing import Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import checkpoint as _mlx_checkpoint
from mlx.utils import tree_flatten, tree_map

# Teacher entries are (probs, indices): indices is None for a full-vocab
# distribution, or the top-k column indices when top-k distillation is used.
TeacherEntry = tuple


def capture_teacher(
    forward: Callable[[object], mx.array],
    inputs: List[object],
    temp: float = 1.0,
    top_k: int = 0,
) -> List[TeacherEntry]:
    """Cache the teacher's softened output distribution for each calibration input.

    With ``top_k > 0`` only the top-k probabilities (renormalized) and their
    column indices are stored, which cuts the teacher cache from ``[T, vocab]``
    to ``[T, k]`` -- large for big vocabularies -- and focuses distillation on
    the tokens that carry the teacher's mass.
    """
    teacher: List[TeacherEntry] = []
    for sample in inputs:
        logits = forward(sample).astype(mx.float32) / temp
        probs = mx.softmax(logits, axis=-1)
        if top_k and top_k < int(probs.shape[-1]):
            idx = mx.argpartition(-probs, top_k - 1, axis=-1)[..., :top_k]
            vals = mx.take_along_axis(probs, idx, axis=-1)
            vals = vals / mx.sum(vals, axis=-1, keepdims=True)
            entry = (vals.astype(mx.float16), idx.astype(mx.int32))
        else:
            entry = (probs.astype(mx.float16), None)
        mx.eval([t for t in entry if t is not None])
        teacher.append(entry)
    return teacher


def _enable_layer_checkpointing(model: nn.Module):
    """Gradient-checkpoint every per-layer block (``...layers.N``) by patching its
    class ``__call__``; returns a restore callable. Trades recompute in the
    backward pass for much lower activation memory, letting longer calibration
    sequences fit."""
    originals: Dict[type, Callable] = {}
    for name, module in model.named_modules():
        if re.search(r"layers\.\d+$", name):
            cls = type(module)
            originals.setdefault(cls, cls.__call__)

    for cls, orig in originals.items():

        def make(orig):
            def ckpt_call(self, *args, **kwargs):
                fn = functools.partial(orig, self)
                return _mlx_checkpoint(self, fn)(*args, **kwargs)

            return ckpt_call

        cls.__call__ = make(orig)

    def restore():
        for cls, orig in originals.items():
            cls.__call__ = orig

    return restore


def apply_dwq(
    model: nn.Module,
    forward: Callable[[object], mx.array],
    inputs: List[object],
    teacher: List[TeacherEntry],
    steps: int = 200,
    lr: float = 1e-6,
    temp: float = 1.0,
    val_inputs: Optional[List[object]] = None,
    val_teacher: Optional[List[TeacherEntry]] = None,
    report_every: int = 25,
    patience: int = 0,
    checkpoint: bool = False,
) -> Dict[str, object]:
    """Distill the quantized model's scales/biases toward cached teacher outputs.

    With a held-out ``val_inputs``/``val_teacher`` set, the mean cross-entropy to
    the teacher is measured before the first step (the round-to-nearest baseline)
    and at every report; the scales/biases from the **best** validation point are
    restored at the end, so distillation can never leave the model worse than
    plain quantization. ``patience`` (>0) stops early after that many reports with
    no validation improvement. ``checkpoint`` trades compute for memory.
    """
    model.freeze()
    model.unfreeze(recurse=True, keys=["scales", "biases"])
    n_trainable = len(tree_flatten(model.trainable_parameters()))
    if n_trainable == 0 or steps <= 0:
        return {
            "trained_tensors": n_trainable,
            "initial_loss": None,
            "final_loss": None,
            "history": [],
        }

    schedule = optim.cosine_decay(lr, steps) if steps > 1 else lr
    opt = optim.Adam(learning_rate=schedule)

    def loss_fn(sample, entry):
        vals, idx = entry
        logits = forward(sample).astype(mx.float32) / temp
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if idx is not None:
            logp = mx.take_along_axis(logp, idx, axis=-1)
        return -mx.mean(mx.sum(vals.astype(mx.float32) * logp, axis=-1))

    step_fn = nn.value_and_grad(model, loss_fn)

    def mean_loss(samples, targets):
        if not samples:
            return None
        total = 0.0
        for sample, entry in zip(samples, targets):
            loss = loss_fn(sample, entry)
            mx.eval(loss)
            total += float(loss.item())
        return total / len(samples)

    def snapshot():
        return tree_map(lambda a: mx.array(a), model.trainable_parameters())

    # Distillation needs gradients, so run in training mode: modules that swap in
    # non-differentiable fast kernels at inference (e.g. the gated-delta Metal
    # kernel) fall back to their differentiable path when training.
    model.train()
    restore_ckpt = _enable_layer_checkpointing(model) if checkpoint else None

    history: List[dict] = []
    initial_loss = None
    final_loss = None
    best_val = None
    best_state = None
    stale = 0
    try:
        initial_loss = mean_loss(val_inputs, val_teacher)
        if initial_loss is not None:
            best_val = initial_loss
            best_state = snapshot()

        running = 0.0
        for step in range(steps):
            idx = step % len(inputs)
            loss, grads = step_fn(inputs[idx], teacher[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            running += float(loss.item())
            if report_every and (step + 1) % report_every == 0:
                train = running / report_every
                running = 0.0
                val = mean_loss(val_inputs, val_teacher)
                history.append({"step": step + 1, "train": train, "val": val})
                print(
                    f"[INFO] DWQ step {step + 1}/{steps} train_ce={train:.4f}"
                    + (f" val_ce={val:.4f}" if val is not None else ""),
                    flush=True,
                )
                if val is not None:
                    if val < best_val - 1e-9:
                        best_val, best_state, stale = val, snapshot(), 0
                    else:
                        stale += 1
                        if patience and stale >= patience:
                            print(
                                f"[INFO] DWQ early stop at step {step + 1} "
                                f"(no val improvement for {patience} reports).",
                                flush=True,
                            )
                            break
    finally:
        if restore_ckpt is not None:
            restore_ckpt()
        model.eval()

    if best_state is not None:
        model.update(best_state)
        final_loss = best_val
    else:
        final_loss = mean_loss(val_inputs, val_teacher)

    model.freeze()
    mx.eval(model.parameters())

    if initial_loss is not None and final_loss is not None:
        if final_loss < initial_loss - 1e-9:
            print(
                f"[INFO] DWQ improved val CE {initial_loss:.4f} -> {final_loss:.4f}.",
                flush=True,
            )
        else:
            print(
                f"[INFO] DWQ did not beat the round-to-nearest baseline "
                f"({initial_loss:.4f}); kept the baseline weights.",
                flush=True,
            )

    return {
        "trained_tensors": n_trainable,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "history": history,
    }

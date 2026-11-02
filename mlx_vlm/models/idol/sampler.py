"""Rectified-flow Euler sampling with Idol's guidance convention."""

import math

import mlx.core as mx
import numpy as np
from tqdm import trange


def validate_sampling(steps, rescale_t, cfg_strength):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(rescale_t) or rescale_t <= 0:
        raise ValueError("rescale_t must be positive and finite")
    if not math.isfinite(cfg_strength) or cfg_strength < 0:
        raise ValueError("cfg_strength must be nonnegative and finite")


def sample_flow(
    model,
    noise,
    cond,
    neg_cond=None,
    steps=50,
    rescale_t=3.0,
    cfg_strength=3.0,
    verbose=False,
    batch_cfg=False,
):
    validate_sampling(steps, rescale_t, cfg_strength)
    if neg_cond is None:
        neg_cond = {key: mx.zeros_like(value) for key, value in cond.items()}
    n = noise["latent"].shape[0]
    sample = noise
    ts = np.linspace(1, 0, steps + 1)
    ts = rescale_t * ts / (1 + (rescale_t - 1) * ts)
    both_cond = (
        {k: mx.concatenate([cond[k], neg_cond[k]], axis=0) for k in cond}
        if batch_cfg and cfg_strength > 0
        else None
    )
    for i in trange(steps, desc="Sampling", disable=not verbose):
        t = mx.full((n,), 1000 * float(ts[i]))
        if both_cond is not None:
            both = model(
                {k: mx.concatenate([v, v], axis=0) for k, v in sample.items()},
                mx.concatenate([t, t]),
                both_cond,
            )
            velocity = {
                k: (1 + cfg_strength) * v[:n] - cfg_strength * v[n:]
                for k, v in both.items()
            }
        else:
            velocity = model(sample, t, cond)
            if cfg_strength > 0:
                negative = model(sample, t, neg_cond)
                velocity = {
                    k: (1 + cfg_strength) * v - cfg_strength * negative[k]
                    for k, v in velocity.items()
                }
        sample = {
            k: v - velocity[k] * float(ts[i] - ts[i + 1]) for k, v in sample.items()
        }
        mx.eval(sample)
    return sample

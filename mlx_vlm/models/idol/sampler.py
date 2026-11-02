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


def _guided_velocity(model, sample, t, cond, negative, strength, both_cond):
    if both_cond is not None:
        n = sample["latent"].shape[0]
        both = model(
            {k: mx.concatenate([v, v], axis=0) for k, v in sample.items()},
            mx.concatenate([t, t]),
            both_cond,
        )
        return {k: (1 + strength) * v[:n] - strength * v[n:] for k, v in both.items()}
    velocity = model(sample, t, cond)
    if strength > 0:
        uncond = model(sample, t, negative)
        velocity = {
            k: (1 + strength) * v - strength * uncond[k] for k, v in velocity.items()
        }
    return velocity


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
        velocity = _guided_velocity(
            model, sample, t, cond, neg_cond, cfg_strength, both_cond
        )
        sample = {
            k: v - velocity[k] * float(ts[i] - ts[i + 1]) for k, v in sample.items()
        }
        mx.eval(sample)
    return sample


def sample_multiview(
    model,
    noise,
    conditions,
    *,
    steps=50,
    rescale_t=3.0,
    cfg_strength=3.0,
    verbose=False,
    batch_cfg=False,
):
    """Experimental equal-weight velocity fusion, with one camera per view.

    Idol is trained on single images. This inference-only extension shares an
    object latent across views; it does not calibrate or geometrically align them.
    """
    validate_sampling(steps, rescale_t, cfg_strength)
    if not conditions:
        raise ValueError("At least one image condition is required")
    if any(value.shape[0] != 1 for value in noise.values()) or any(
        value.shape[0] != 1 for cond in conditions for value in cond.values()
    ):
        raise ValueError(
            "Multi-image sampling expects one object and one image per view"
        )
    negative = [{k: mx.zeros_like(v) for k, v in cond.items()} for cond in conditions]
    both = [
        (
            {k: mx.concatenate([cond[k], neg[k]], axis=0) for k in cond}
            if batch_cfg and cfg_strength > 0
            else None
        )
        for cond, neg in zip(conditions, negative)
    ]
    latent = noise["latent"]
    cameras = [noise["camera"] for _ in conditions]
    ts = np.linspace(1, 0, steps + 1)
    ts = rescale_t * ts / (1 + (rescale_t - 1) * ts)
    for step in trange(steps, desc="Sampling views", disable=not verbose):
        t = mx.full((1,), 1000 * float(ts[step]))
        dt = float(ts[step] - ts[step + 1])
        combined = None
        for index, cond in enumerate(conditions):
            # Every view sees the same pre-update object latent. Its camera
            # evolves using only that view's prediction, never the camera mean.
            velocity = _guided_velocity(
                model,
                {"latent": latent, "camera": cameras[index]},
                t,
                cond,
                negative[index],
                cfg_strength,
                both[index],
            )
            contribution = velocity["latent"] / len(conditions)
            combined = contribution if combined is None else combined + contribution
            cameras[index] = cameras[index] - velocity["camera"] * dt
            # Evaluate each view before building the next denoiser graph.
            mx.eval(combined, cameras[index])
        latent = latent - combined * dt
        mx.eval(latent)
    return {"latent": latent, "camera": cameras[0], "cameras": cameras}

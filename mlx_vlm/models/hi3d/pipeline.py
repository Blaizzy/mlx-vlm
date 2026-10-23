"""Hi3D refinement: image + coarse mesh -> high-resolution mesh."""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from .mesh import decoded_mesh, fill_holes, grid_matrix, postprocess
from .sparse import Grid
from .vision import prepare_image
from .voxelize import adaptive_voxelize

# A decode larger than this cannot seed another pass, as in the reference.
DECODE_FEATURE_CEILING = 12_000_000


def schedule(steps, num_tokens, sigma_min=0.002, time_rescale=2.0):
    """Sigmas and inflated model times of the released flow schedule.

    The trajectory is warped by the token count; times are the sigmas
    inflated by a two-level gamma, without the matching churn noise.
    """
    f32 = np.float32
    warp = (19.0 * num_tokens - 20000.0) / (3.0 * num_tokens + 60000.0)
    warp = max(1.0, min(5.0, warp))
    base = np.linspace(1.0, 0.0, steps + 1, dtype=f32)

    def rescale(t, factor):
        return f32(factor) * t / (f32(1.0) + f32(factor - 1.0) * t)

    warped = rescale(rescale(base, time_rescale), warp)
    sigmas = f32(sigma_min) + f32(1.0 - sigma_min) * warped
    gammas = np.where(warped[:-1] > 0.985, f32(0.04), f32(0.002)).astype(f32)
    return sigmas, (sigmas[:-1] * (f32(1.0) + gammas)).astype(f32)


def guidance_scale(t):
    """Zero-negative guidance: 4 at the first time, ``10 t²`` on ``[0.7, 1)``."""
    if t >= 1.0:
        return 4.0
    return float(np.float32(10.0) * np.float32(t) ** 2)


def velocity(transformer, x, t, cond, grid):
    """Guided flow velocity; guidance runs only from time 0.7 upwards."""
    time_in = mx.array(np.array([1000.0], np.float32) * np.float32(t))
    if t < 0.7:
        return transformer(x, time_in, cond, grid)
    both = transformer(
        mx.concatenate([x, x]),
        mx.concatenate([time_in, time_in]),
        mx.concatenate([cond, mx.zeros_like(cond)]),
        grid,
    )
    positive, negative = both[:1], both[1:]
    return negative + mx.array(guidance_scale(t), mx.float32) * (positive - negative)


def sample(
    transformer, cond, grid, noise, steps, sigma_min, time_rescale, progress=None
):
    """Heun integration of the flow from ``noise`` ``(1, N, C)`` to latents."""
    sigmas, times = schedule(steps, len(grid), sigma_min, time_rescale)
    x = noise
    for i in range(steps):
        t, nxt = float(times[i]), float(sigmas[i + 1])
        step = mx.array(np.array(times[i] - sigmas[i + 1], np.float32))
        v = velocity(transformer, x, t, cond, grid)
        v2 = velocity(transformer, x - step * v, nxt, cond, grid)
        x = x - step * (0.5 * (v + v2))
        mx.eval(x)
        if progress is not None:
            progress(i + 1, steps)
    config = transformer.config
    return x * mx.array(config.latent_std) + mx.array(config.latent_mean)


def decode(vae, latents, grid):
    """Corner sign logits, activated corner features and the decoded grid."""
    out, decoded = vae(latents[0], grid)
    sdf_sign, feats = out[:, :8], mx.tanh(out[:, 8:])
    mx.eval(sdf_sign, feats, decoded.coords)
    return np.array(sdf_sign), np.array(feats), decoded


@dataclass
class Result:
    vertices: np.ndarray
    faces: np.ndarray
    resolution: int
    timings: dict


class Pipeline:
    def __init__(self, model):
        self.model = model
        self.config = model.config

    def generate(
        self,
        image,
        vertices,
        faces,
        resolution: int = 1024,
        steps: Optional[int] = None,
        seed: int = 0,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Refine a coarse mesh of the object in ``image``.

        Args:
            image: PIL image or HWC array of the object; an alpha channel marks it.
            vertices, faces: The coarse triangle mesh.
            resolution: Output resolution, one of ``config.output_resolutions``.
            steps: Heun steps per pass (default ``config.num_inference_steps``).
            seed: Seed of the latent noise.
            progress: Optional callback receiving status lines.

        Returns:
            The refined mesh, centered and scaled to unit radius.
        """
        config = self.config
        if resolution not in config.output_resolutions:
            raise ValueError(f"resolution must be one of {config.output_resolutions}")
        steps = steps or config.num_inference_steps
        report = progress or (lambda message: None)
        target = config.output_resolutions.index(resolution)
        timings = {}
        start = time.perf_counter()
        cond = self.model.conditioner(prepare_image(image))
        mx.eval(cond)
        timings["conditioning"] = time.perf_counter() - start
        key = mx.random.key(seed)
        vertices = np.asarray(vertices, np.float32)
        faces = np.asarray(faces, np.int64)
        coords = adaptive_voxelize(
            vertices, faces, config.structure_resolutions[0], config.pass_budgets[0]
        )
        for index in range(target + 1):
            structure = config.structure_resolutions[index]
            transformer = (
                self.model.transformer_512
                if index == 0
                else self.model.transformer_1024
            )
            grid = Grid(mx.array(coords), structure)
            key, subkey = mx.random.split(key)
            noise = mx.random.normal(
                (1, len(grid), transformer.config.in_channels), key=subkey
            )
            report(f"pass {index + 1}/{target + 1}: {len(grid)} voxels at {structure}³")
            start = time.perf_counter()
            latents = sample(
                transformer,
                cond,
                grid,
                noise,
                steps,
                config.sigma_min,
                config.time_rescale_factor,
                lambda i, n: report(f"  step {i}/{n}"),
            )
            mx.eval(latents)
            timings[f"pass{index + 1}_sampling"] = time.perf_counter() - start
            start = time.perf_counter()
            sdf_sign, feats, decoded = decode(self.model.vae, latents, grid)
            if index < target and len(decoded) > DECODE_FEATURE_CEILING:
                raise RuntimeError(
                    f"decoded {len(decoded)} voxels, above the {DECODE_FEATURE_CEILING} "
                    "that can seed another pass"
                )
            vertices, faces = decoded_mesh(
                sdf_sign,
                feats,
                np.array(decoded.coords),
                decoded.resolution,
                grid_matrix(coords),
            )
            if not len(faces):
                raise RuntimeError("mesh extraction produced an empty isosurface")
            faces = fill_holes(vertices, faces)
            timings[f"pass{index + 1}_decoding"] = time.perf_counter() - start
            mx.clear_cache()
            if index < target:
                coords = adaptive_voxelize(
                    vertices,
                    faces,
                    config.structure_resolutions[index + 1],
                    config.pass_budgets[index + 1],
                )
        start = time.perf_counter()
        vertices, faces = postprocess(
            vertices, faces, config.target_face_counts[target]
        )
        timings["postprocess"] = time.perf_counter() - start
        return Result(vertices, faces, resolution, timings)

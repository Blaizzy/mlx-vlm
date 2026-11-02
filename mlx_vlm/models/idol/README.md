# Idol

Native MLX port of [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk),
using the original [speridlabs/idol](https://huggingface.co/speridlabs/idol) weights.
An object image becomes a standard 3D Gaussian Splatting `.ply` file, a prepared
input PNG, and predicted camera parameters in JSON.

The denoiser, octree decoder, Gaussian decoder, DINOv3, FLUX.2 VAE encoder, and
BiRefNet (including deformable convolutions) run in MLX. NumPy and Pillow handle
image I/O, variable-length octree packing, and PLY export. No PyTorch inference
fallback, CUDA extension, or weight conversion step is required.

## Install and download

From this checkout:

```sh
pip install -e .
python -m mlx_vlm.models.idol --download-only
```

This downloads all four Idol checkpoints (about 3.1 GB) to the Hugging Face
cache and prints their snapshot directory. Pass `--model /path/to/idol` to
use an existing local directory. `--local-files-only` disables downloads.

Image conditioning also requires authorized access to both upstream encoders:

- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)

Accept their terms on Hugging Face and authenticate with `hf auth login`.
The encoders download on the first image request; only the `vae/` component
of FLUX.2 is fetched. Encoder access is not needed to download Idol's weights
or to sample and decode precomputed conditioning features.

## Generate

```sh
python -m mlx_vlm.models.idol \
  --image photo.jpg \
  --mode real \
  --output outputs/object.ply
```

Use `--mode synthetic` (the default) for renders and generated images. A directory
can be passed to `--image` to process multiple images; use an output directory
in that case. A transparent RGBA input skips background removal. To require
transparency, pass `--no-background-removal`.

Defaults match the SDK: 50 Euler steps, guidance strength 3.0, time shift 3.0,
seed 42, and 262,144 Gaussians. `--num-gaussians` accepts multiples of 32 from
32,768 to 1,048,576. `--steps`, `--cfg-strength`, `--rescale-t`, `--crop-margin`,
and `--seed` override these settings. `--batch-cfg` batches the positive and
negative passes at the cost of more memory. MLX and PyTorch use different random
number generators, so matching seeds do not produce identical objects across
the two implementations.

For local encoders:

```sh
python -m mlx_vlm.models.idol --image object.png \
  --dinov3 /path/to/dinov3 \
  --flux2-vae /path/to/FLUX.2-dev/vae \
  --local-files-only
```

DINOv3 defaults to bfloat16 (`--dinov3-dtype fp32` selects full precision).
The generation transformers use float16 linear layers and float32 embeddings,
norm parameters, and learned rotary positions, matching the SDK. BiRefNet uses
float32, and the VAE uses bfloat16.

## Python

```python
from mlx_vlm.models.idol import IdolPipeline

pipeline = IdolPipeline.from_pretrained("speridlabs/idol")
result = pipeline("photo.jpg", mode="real", seed=42)
result.save("outputs/object.ply")  # also writes _input.png and _camera.json

# Or export just the splats:
result.gaussian.save_ply("outputs/splats.ply")
print(result.camera)
```

`pipeline.sample(condition, ...)` accepts `feature1` (DINOv3) and `feature2`
(VAE) MLX arrays, and returns a dictionary with `latent` and `camera` arrays.
At the standard 1024px resolution, conditioning shapes are `[1, 4101, 1280]`
and `[1, 4101, 128]`. `pipeline.decode(latent, num_gaussians=...)` converts a
`[1, 8192, 16]` latent to a `Gaussian` without loading the image encoders.
Gaussians are z-up in memory; PLY export rotates positions and quaternions to y-up.

Idol uses this dedicated image-to-3D API and CLI; it does not implement the
text-generation interface used by `mlx_vlm.generate`.

## Validation

```sh
python -m pytest mlx_vlm/tests/test_idol.py \
  mlx_vlm/tests/test_models.py::TestIdol::test_idol -q
```

Optional reference comparisons need PyTorch, torchvision, diffusers, and the
SDK's test dependencies (`tensordict` and `plyfile`). No model download occurs
during the unit tests. To include SDK parity tests:

```sh
git clone https://github.com/speridlabs/idol-sdk /tmp/idol-sdk
IDOL_REFERENCE_SDK=/tmp/idol-sdk python -m pytest mlx_vlm/tests/test_idol.py -q
```

To include tests with downloaded checkpoints, also set `IDOL_WEIGHTS` to the
Idol snapshot directory. The checkpoint smoke test uses zero conditioning and
checks execution and finite output, not reconstruction quality.
Set `IDOL_TEST_IMAGE` to an input image to enable the image-conditioned
integration test as well. That test uses the cached DINOv3 and FLUX.2 VAE
checkpoints with `local_files_only=True`; it does not download weights.

Initial validation on Apple M3 Ultra:

- Small float32 denoiser and decoder outputs matched the SDK within 5e-6.
- Full trained transformer stacks, tested on shortened sequences in their
  production mixed precision, had relative L2 error below 0.003.
- Native deformable convolution matched torchvision within 1e-6.
- A full-image BiRefNet mask differed from upstream by at most one 8-bit alpha
  level (mean absolute difference 0.0019 out of 255).
- DINOv3 float32 matched Transformers within 1e-6; small bfloat16 VAE tests
  matched within the expected lower-precision tolerance.
- Real Idol weights completed 8192-token sampling, octree decoding, and PLY
  export of 32,768 finite Gaussians. BiRefNet ran at its native 1024px resolution.

Full image-conditioned validation was completed on 2026-11-02 after obtaining
access to both encoder checkpoints:

- The SDK's `demo/examples/toy-castle.webp` completed the standard 50-step run
  with guidance 3.0, seed 42, and 262,144 Gaussians. All exported attributes were
  finite, and front, side, and rear splat renders showed a coherent castle.
- This single validation run took 126 seconds, including image preprocessing
  and encoder loading, and used 9.49 GB peak MLX memory on Apple M3 Ultra.
- On a 256px image, the pretrained bfloat16 DINOv3 output had cosine similarity
  0.9993 to Transformers on CPU. The pretrained VAE posterior mean and log
  variance had relative L2 differences of 0.0255 and 0.0080 against Diffusers
  on MPS. These are numerical comparisons, not reconstruction-quality scores.

## Attribution

Ported from Idol SDK commit `4702a7f00850e34b6e83a2170cc123c3e9570744` and
tested with model revision `2731c2291fc65a797c5cb627a5e5526fc1490902`.
Image-conditioned validation used DINOv3 revision
`c807c9eeea853df70aec4069e6f56b28ddc82acc` and FLUX.2 revision
`26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`.
Adapted SDK code is licensed under Apache-2.0; see [LICENSE](LICENSE).
The original denoiser also credits the Wan and Hugging Face teams.

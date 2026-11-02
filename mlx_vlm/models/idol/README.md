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
```

The first generation downloads all four Idol checkpoints (about 3.1 GB) to the
Hugging Face cache. Pass `--model /path/to/idol` to use a local directory.

Image conditioning also requires authorized access to both upstream encoders:

- [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)

Accept their terms on Hugging Face and authenticate with `hf auth login`.
The encoders download on the first image request; only the `vae/` component
of FLUX.2 is fetched. Encoder access is not needed to download Idol's weights
or to sample and decode precomputed conditioning features.

## Generate

```sh
python -m mlx_vlm.generate \
  --model speridlabs/idol --output-modality 3d \
  --image photo.jpg \
  --output outputs/object.ply \
  --gen-kwargs '{"mode": "real"}'
```

Use `mode="real"` for photos; the default `synthetic` mode is for renders and
generated images. The shared CLI accepts one image path, URL, or data URI.
A transparent input skips background removal. The default output is
`outputs/object-42.ply`, with prepared input and camera files beside it.

Defaults match the SDK: 50 Euler steps, guidance strength 3.0, time shift 3.0,
seed 42, and 262,144 Gaussians. Use the shared `--steps`, `--guidance`, `--seed`,
and `--verbose` flags. Model-specific options go in `--gen-kwargs`, for example
`'{"mode": "real", "num_gaussians": 32768, "batch_cfg": true}'`.
`num_gaussians` accepts multiples of 32 from 32,768 to 1,048,576;
`rescale_t` and `crop_margin` override time shift and framing. `batch_cfg`
batches the positive and negative passes at the cost of more memory.
MLX and PyTorch use different random number generators, so matching seeds
do not produce identical objects across the two implementations.

## Python

```python
from mlx_vlm.generate import generate_3d, load_3d_model

model = load_3d_model("speridlabs/idol")
result = generate_3d(model, "photo.jpg", mode="real", output_path="outputs/object.ply")
print(result.path, result.camera)

# Or export just the splats:
result.gaussian.save_ply("outputs/splats.ply")
```

For local encoders or offline use:

```python
model = load_3d_model(
    "/path/to/idol",
    dinov3="/path/to/dinov3",
    flux2_vae="/path/to/FLUX.2-dev/vae",
    local_files_only=True,
)
```

Loader options also include `dinov3_dtype="fp32"` (default: `"bf16"`) and
`background_removal=False` to require an image with transparency.
The generation transformers use float16 linear layers and float32 embeddings,
norm parameters, and learned rotary positions, matching the SDK. BiRefNet uses
float32, and the VAE uses bfloat16.

`model.sample(condition, ...)` accepts `feature1` (DINOv3) and `feature2`
(VAE) MLX arrays, and returns a dictionary with `latent` and `camera` arrays.
At the standard 1024px resolution, conditioning shapes are `[1, 4101, 1280]`
and `[1, 4101, 128]`. `model.decode(latent, num_gaussians=...)` converts a
`[1, 8192, 16]` latent to a `Gaussian` without loading the image encoders.
Gaussians are z-up in memory; PLY export rotates positions and quaternions to y-up.

`IdolPipeline.from_pretrained(...)` remains available for direct model use.
The shared CLI routes `--output-modality 3d` to this pipeline without loading
a text tokenizer or applying a chat template.

## Validation

```sh
python -m pytest mlx_vlm/tests/test_three_d.py -q
```

The suite covers the shared 3D API and CLI, model-family cases in
`mlx_vlm/tests/three_d_cases.json`, and Idol component and numerical checks.

Optional reference comparisons need PyTorch, torchvision, diffusers, and the
SDK's test dependencies (`tensordict` and `plyfile`). No model download occurs
during the unit tests. To include SDK parity tests:

```sh
git clone https://github.com/speridlabs/idol-sdk /tmp/idol-sdk
IDOL_REFERENCE_SDK=/tmp/idol-sdk python -m pytest mlx_vlm/tests/test_three_d.py -q
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
Adapted SDK code is licensed under Apache-2.0; see the
[upstream license](https://github.com/speridlabs/idol-sdk/blob/4702a7f00850e34b6e83a2170cc123c3e9570744/LICENSE).
The original denoiser also credits the Wan and Hugging Face teams.

# SAM 3D Objects for MLX

Inference-only MLX port of [Meta SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects). The [converted bundle](https://huggingface.co/mlx-community/sam-3d-objects-bf16)
contains weights and metadata.

Install mlx-vlm and download the bundle separately, for example with the
optional Hugging Face CLI:

```sh
hf download mlx-community/sam-3d-objects-bf16 --local-dir models/sam-3d-objects-bf16
```

`Pipeline.from_pretrained` loads a local bundle directory:

```python
from mlx_vlm.models.sam3d_objects.generate import (
    read_image,
    read_mask,
    write_gaussians,
    write_obj,
)
from mlx_vlm.models.sam3d_objects.pipeline import Pipeline

pipeline = Pipeline.from_pretrained("models/sam-3d-objects-bf16")
result = pipeline.generate(read_image("image.png"), read_mask("mask.png"), seed=42)
write_gaussians("object.ply", result["gaussian"])
write_obj("object.obj", result["mesh"])
```

The model also loads through `mlx_vlm.utils.load_model`. The higher-level
`mlx_vlm.load` expects a tokenizer or processor and is not the entry point
for this geometry model.

For file-based inference:

```sh
python -m mlx_vlm.models.sam3d_objects.generate \
  --model models/sam-3d-objects-bf16 \
  --image image.png --mask mask.png --output outputs/object \
  --formats gaussian mesh
```

The CLI writes tensor outputs to `result.safetensors` and selected geometry
to PLY/OBJ files. Use `--help` for sampling and depth options. With `--jsonl`,
it reads requests from stdin and emits progress records to stdout:

```json
{"id":"frame-1","image":"frame-1.png","mask":"mask-1.png","seed":42}
{"id":"frame-2","image":"frame-2.png","mask":"mask-2.png","seed":42}
```

JSONL results go into numbered subdirectories under `--output`. IDs are
preserved as labels, never interpreted as paths. An external point map can
be supplied with `--pointmap` or a JSONL `pointmap` field; both take a
safetensors file containing a `points` tensor.

## Input and output contract

- **Image:** an HWC RGB or RGBA MLX array, either `uint8` or finite floating
  values in `[0, 1]`.
- **Mask:** an HW array matching the image, using booleans, 0/1, or 0/255.
  If omitted, the image must have an alpha channel, which becomes the mask.
  Foreground must span at least two rows and two columns. `read_mask` uses
  alpha for images with transparency and the first channel otherwise,
  treating nonzero values as foreground.
- **Point map:** an optional HWC XYZ array aligned with the image, using
  **+X left, +Y up, +Z forward**, with NaN for invalid points. Preprocessing
  resizes it to the image dimensions, then normalizes it for conditioning
  while retaining the shift and scale needed to decode pose.

When no point map is supplied, the pipeline estimates one if the bundle
contains a depth model and `estimate_depth=True`. The depth adapter uses the
shared [MoGe-3 implementation](../moge3/README.md), converts its camera axes,
and preserves its metric scale. `depth_num_tokens` in the bundle configuration
controls depth inference resolution. Without a supplied or estimated map,
the conditioner uses its trained point-map dropout path; pose then lacks an
observed depth reference. The result's `pointmap_conditioned` flag distinguishes
these cases.

`generate` returns an evaluated dictionary containing:

| Key | Meaning |
| --- | --- |
| `pose` | `rotation` as a WXYZ quaternion, `rotation_matrix`, `translation`, and `scale`. |
| `coords` | Sparse voxel coordinates with columns `(batch, x, y, z)`. |
| `latents` | Denormalized sparse features, one row per coordinate. |
| `pointmap_conditioned` | Whether an external or estimated point map was used. |
| `gaussian`, `gaussian_4`, `mesh` | Decoder outputs requested through `formats`. |

The default formats are `gaussian` and `mesh`; `gaussian_4` selects the
alternative Gaussian decoder. Gaussian dictionaries contain `positions`,
`sh_dc`, `scales`, `rotations`, and `opacities`. Mesh dictionaries contain
`vertices`, zero-based triangle `faces`, and six-channel `vertex_colors`
(RGB followed by learned attributes).

Geometry remains in object-local coordinates, including exported files;
apply `pose` separately for scene placement. `write_gaussians` converts
scales and opacities to the log-scale and logit fields expected by Gaussian
PLY readers. `write_obj` exports RGB vertex colors and converts face indices
to OBJ's one-based convention.

## Inference stages and code map

The two sampling stages are named `ss` (sparse structure and pose) and `slat`
(sparse latent features) throughout the code and configuration.

| Stage | Implementation and responsibility |
| --- | --- |
| Prepare inputs | [depth.py](depth.py) estimates optional point maps; [processing.py](processing.py) builds object crops and full-image views, normalizes points, and decodes pose. Image resizing uses the shared `mlx_vlm.models.interpolate` module. |
| Encode conditions | [vision.py](vision.py) wraps the shared [DINOv2 backbone](../dinov2/README.md) and embeds point maps for structure conditioning. |
| Sample structure and pose | [flow.py](flow.py) defines `StructureFlow`; [pipeline.py](pipeline.py) integrates it, decodes occupancy, and selects the sparse grid. |
| Sample sparse features | `LatentFlow` in [flow.py](flow.py) generates features on that grid; the pipeline denormalizes them before decoding. |
| Decode geometry | [decoders.py](decoders.py) defines occupancy, Gaussian, and mesh heads; [mesh.py](mesh.py) and [mesh_tables.py](mesh_tables.py) extract triangles with FlexiCubes. |

[config.py](config.py) defines architecture and sampling settings.
[sam3d_objects.py](sam3d_objects.py) assembles and loads the model;
[sparse.py](sparse.py) implements grid topology, sparse convolutions, pooling,
and window attention. Start with `Pipeline._run` to follow a complete request.

Work shared across flow steps is prepared once per request: condition
projections, sparse topology, and reusable image features. Identical DINOv2
backbones are shared only after their parameters have been checked for
equality. When changing these paths, preserve cache scope and compare cached
or fused operations against their direct equivalents.

## Streaming and execution

`stream(Request(...))` yields synchronous progress events. `astream` accepts
a synchronous or asynchronous iterable of requests and processes them in
order, one request at a time, with backpressure at each yielded event:

```python
from contextlib import aclosing
from mlx_vlm.models.sam3d_objects.pipeline import Request

async def reconstruct(pipeline, frames):
    async def requests():
        async for request_id, image, mask in frames:
            yield Request(image, mask, request_id=request_id)

    async with aclosing(pipeline.astream(requests())) as events:
        async for event in events:
            if event.stage == "complete":
                yield event.request_id, event.data
```

Each event has `request_id`, `stage`, `step`, `total_steps`, and stage-specific
`data`. Events cover optional depth estimation, conditioning, structure flow,
occupancy, latent flow, selected decoders, and completion. `complete` carries
the same result structure as `generate`.

Streaming schedules arrays with `mx.async_eval`; an event does not imply GPU
completion. Use `mx.eval(event.data)` when completed values are needed, and
include synchronization when measuring latency. Dynamic sparse topology also
requires scalar synchronization to determine output sizes.

Async submissions run on a single worker to keep preprocessing and topology
work off the caller's event loop. Each pipeline permits one active stream.
Close an async stream when stopping early (`aclose()` or `aclosing`); cleanup
stops further submissions and drains submitted work before releasing the
worker. Close synchronous streams with `close()` when stopping early.

## Checkpoint conversion

Use the original SAM 3D Objects inference checkpoints and an MLX MoGe-3
bundle, such as `mlx-community/moge-3-vitl-mlx-fp32`:

```sh
python -m mlx_vlm.models.sam3d_objects.convert \
  --source /path/to/sam-3d-objects/checkpoints \
  --moge-checkpoint models/moge-3-vitl-mlx-fp32 \
  --output models/sam-3d-objects-bf16
```

Omit `--moge-checkpoint` to build a bundle without automatic depth estimation.
See the [MoGe-3 documentation](../moge3/README.md) to convert its weights.

[checkpoint.py](checkpoint.py) reads tensor-only PyTorch ZIP archives through
a restricted unpickler and preserves storage bytes in intermediate
safetensors. Unsupported pickle globals and storage layouts are rejected.
[convert.py](convert.py) maps names and tensor layouts to MLX, casts SAM
floating weights to BF16, and uses the dedicated occupancy checkpoint instead
of the duplicate embedded in the structure checkpoint. Bundled depth weights
keep their stored dtype unless `--moge-dtype` is supplied.

The loader checks parameter names and shapes strictly. When changing module
structure or checkpoint mapping, update conversion and loading together.
`config.json`, `model.safetensors.index.json`, and `conversion.json` describe
the resulting bundle; consult those files for its contents.
[bundle.py](bundle.py) generates the Hub model card from this README and adds
benchmark metadata from `validation.json` when present.

## Validation and maintenance

Run the focused component suite from the repository root:

```sh
python -m pytest mlx_vlm/tests/test_extraction_models.py -q -k TestSAM3DObjects
```

`TestSAM3DObjects` in
[test_extraction_models.py](../../tests/test_extraction_models.py) runs a
tiny random model through bundle loading, input preprocessing, sparse
operators, sampling, mesh extraction, the streaming pipeline end to end, and
shared-model integration. Use it to check local contracts; assess
reconstruction quality separately with representative images and actual
converted weights.

For numerical changes, compare components in FP32 before checking the target
device and weight dtype. Iterative sampling can amplify small rounding
changes into different occupied voxels and geometry. Preprocessing changes
also need checks for mask alignment, invalid-point handling, camera axes,
and pose scale. Record benchmark and parity results with their checkpoint,
input, seed, sampling settings, software versions, and hardware in validation
artifacts rather than embedding measurements in this guide.

## Licenses

This is an unofficial MLX port. SAM-derived code and weights retain the SAM
License, MoGe-3 retains MIT, and FlexiCubes retains Apache-2.0. Preserve source
notices and the `LICENSE` and `LICENSE-APACHE` texts distributed with the
converted weights.

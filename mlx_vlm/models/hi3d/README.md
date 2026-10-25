# Hi3D v1.0 for MLX

Inference-only MLX port of [Hi3D v1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0),
an image-conditioned mesh refiner built on [Sparc3D](https://arxiv.org/abs/2505.14521).
Hi3D takes an image of an object and a coarse mesh of the same object and
returns high-resolution geometry (no texture). The reference implementation
needs Linux, an NVIDIA GPU, spconv, flash-attn and Triton; this port runs on
Apple silicon with MLX and NumPy.

The [converted bundle](https://huggingface.co/mlx-community/Hi3Dv1.0-mlx)
holds the Hi3D weights in their released dtypes together with the DINOv2-L and
CLIP ViT-L/14 image encoders it conditions on.

## Usage

```sh
hf download mlx-community/Hi3Dv1.0-mlx --local-dir models/Hi3Dv1.0-mlx
```

Refine an existing coarse mesh:

```sh
python -m mlx_vlm.models.hi3d.generate \
  --model models/Hi3Dv1.0-mlx \
  --image object.png --mesh coarse.obj \
  --output refined.obj --resolution 1024
```

Or build the coarse mesh from the same image with
[SAM 3D Objects](../sam3d_objects/README.md), so the whole image-to-mesh
path runs in MLX. The image's alpha channel (or `--mask`) selects the object:

```sh
hf download mlx-community/sam-3d-objects-bf16 --local-dir models/sam-3d-objects-bf16
python -m mlx_vlm.models.hi3d.generate \
  --model models/Hi3Dv1.0-mlx \
  --sam3d-model models/sam-3d-objects-bf16 \
  --image object.png --output refined.obj --resolution 1024
```

From Python:

```python
from PIL import Image

from mlx_vlm.models.hi3d import Model
from mlx_vlm.models.hi3d.mesh import read_mesh, write_obj

model = Model.from_pretrained("models/Hi3Dv1.0-mlx")
vertices, faces = read_mesh("coarse.obj")
result = model(Image.open("object.png"), vertices, faces, resolution=1024, seed=0)
write_obj("refined.obj", result.vertices, result.faces)
```

The model also loads through `mlx_vlm.utils.load_model`.

## Inputs and outputs

- **Image:** an RGB or RGBA image. Transparent pixels mark the background; the
  subject is cropped to a square, composited on white and padded so it fills
  `1 / 1.2` of the frame, the framing the conditioner was trained on.
- **Coarse mesh:** any triangle mesh of the same object. Hi3D re-voxelizes it,
  so position and scale do not matter, but orientation does. The reference
  demo feeds Hunyuan3D-2 meshes unchanged and documents no frame; this port
  places the side shown in the image facing +Z with +Y up, which
  `mesh.coarse_from_sam3d` does for a SAM 3D Objects result. If refined
  detail lands on the hidden side, turn the coarse mesh 180 degrees about Y.
  `read_mesh` reads OBJ files directly and other formats through the
  optional `trimesh` package.
- **Resolution:** 512 (one pass), 1024 (two passes) or 1536 (three passes).
  512 and 1024 are tested in this port; 1536 is not. Upstream validated it
  only on a 96 GB GPU.
- **Result:** vertices centered at the origin with unit maximum radius, and
  triangle faces. Small disconnected pieces are removed. When the optional
  `pyfqmr` package is installed the mesh is simplified to the released face
  targets (1M faces at 512 and 1024, 2M at 1536); otherwise it is kept whole.

## How it works

| Stage | Module |
| --- | --- |
| Frame the image; DINOv2-L tokens at 756 px joined with the CLIP-L embedding | [vision.py](vision.py) (reuses `dinov2` and the `llava` CLIP tower) |
| Surface-voxelize the coarse mesh within an occupied-voxel budget | [voxelize.py](voxelize.py) (NumPy port of the Triton kernel) |
| Heun flow sampling with token-warped schedule and late guidance | [pipeline.py](pipeline.py) |
| Sparse latent flow transformer: sparse residual I/O blocks, 24 full-attention blocks with spatial RoPE | [transformer.py](transformer.py), [sparse.py](sparse.py) |
| Point Transformer V3 decoder: serialized window attention, occupancy-pruned 2x upsampling | [decoder.py](decoder.py) |
| Sparcubes vertex averaging, FlexiCubes, hole filling, cleanup | [mesh.py](mesh.py) (NumPy) |

Each pass refines the voxels of the previous pass's mesh, so the 1024 output
runs the 512 transformer on a 64³ grid and then the 1024 transformer on a
128³ grid. The transformer runs in float16 with float32 norms, RoPE and
modulation, as released; the decoder runs in float32 with float16 attention
inputs and float32 softmax, matching the reference's flash attention.
Submanifold convolutions gather each voxel's 27 neighbors into one im2col
row; grid keys are int64 so the 1536³ decode does not collide.

On an M4 Pro with 64 GB, a 512 refinement of the example dragon (12k voxels)
takes about 3.5 minutes at 17 GB peak memory, and a 1024 refinement about 30
minutes at 30 GB, most of it in the second pass's full attention over 60k
voxels.

Latent noise comes from MLX's generator, so a seed does not reproduce a
PyTorch run sample for sample.

## Validation

Components were compared with the original PyTorch code run on CPU, with
the unavailable spconv and flash-attn kernels replaced by reference
implementations:

| Component | Result |
| --- | --- |
| Image conditioning (float32) | max abs error 5.7e-6 |
| Flow schedule | max abs error 6e-8 |
| Transformer velocity, float32 | relative L2 error 2.2e-6 |
| Transformer velocity, float16 | relative L2 error 1.3e-3 against float32 (the reference's own float16 run: 1.4e-3) |
| `transformer_1024` on a 156³ grid (stretched RoPE), float32 / float16 | relative L2 error 2.6e-6 / 1.3e-3 |
| Decoder occupancy, 1.44M voxels | 14 voxels differ at near-zero logits |
| FlexiCubes and hole filling on reference decoder output | identical faces, vertices within 3e-6 |

Run the component suite from the repository root:

```sh
python -m pytest mlx_vlm/tests/test_extraction_models.py -q -k TestHi3D
```

## Licenses

This is an unofficial port. Hi3D code and weights are licensed under the
[Hi3D Open License Agreement 1.0](https://huggingface.co/hi3d-official/Hi3Dv1.0/blob/main/LICENSE);
products that integrate them commercially must display "Powered by Hi3D 1.0"
and larger deployments need a separate license. Portions derived from
Microsoft TRELLIS and Point Transformer V3 are MIT licensed, FlexiCubes is
Apache-2.0, DINOv2 is Apache-2.0 and CLIP is MIT. The bundle carries
`LICENSE`, `NOTICE`, `LICENSE-APACHE`, `LICENSE-CLIP` and
`THIRD_PARTY_LICENSES.md`; keep them with redistributed weights.

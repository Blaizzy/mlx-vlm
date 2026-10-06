# EmbeddingGemma 2

Native MLX embeddings for `nativ-community/embeddinggemma-2`: text, images, audio,
video, and combinations of these modalities in a shared 768-dimensional space.
The bidirectional encoder uses alternating local/full attention, projection-only
per-layer inputs, and the Gemma 4 vision and audio towers.

The model returns projected token representations in `last_hidden_state` and
mask-aware mean-pooled, L2-normalized float32 vectors in `text_embeds`. Prompts
and media tokens participate in pooling; padding does not.

## Setup

The local processor works with Transformers >= 5.14.0 and uses its Gemma 4 image
and audio components.

## Text retrieval and Matryoshka embeddings

```python
import mlx.core as mx

from mlx_vlm import load

model, processor = load("nativ-community/embeddinggemma-2")
inputs = processor.tokenizer(
    [
        "task: search result | query: Which planet is known as the Red Planet?",
        "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
        "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    ],
    padding=True,
    return_tensors="np",
)
embeddings = model(**{key: mx.array(value) for key, value in inputs.items()}).text_embeds
print(embeddings[:1] @ embeddings[1:].T)

# Use the same dimension for queries and documents: 128, 256, 512, or 768.
embeddings = embeddings[:, :256]
embeddings /= mx.linalg.norm(embeddings, axis=-1, keepdims=True)
```

Task prompts are optional. The checkpoint's `config_sentence_transformers.json`
contains the prompt catalog. For native MLX calls, prepend the chosen prompt to
plain text or supply it as a system message to the processor. Sentence
Transformers itself runs the reference PyTorch model.

## Multimodal inputs

`mlx_vlm.load` selects the local `EmbeddingGemma2Processor`. It expands media
placeholders, computes patch positions and audio features, and samples video
frames. Use `return_tensors="mlx"` to receive model-ready MLX arrays:

```python
from mlx_vlm import load

model, processor = load("nativ-community/embeddinggemma-2")
conversations = [
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [
            {"type": "text", "text": "A photo of a cat"},
            {"type": "image", "url": "cat.jpg"},
        ]},
    ],
    [{"role": "user", "content": [{"type": "audio", "url": "speech.wav"}]}],
    [{"role": "user", "content": [{"type": "video", "url": "sample_video.mp4"}]}],
]
inputs = processor.apply_chat_template(
    conversations, tokenize=True, return_dict=True, return_tensors="mlx"
)
embeddings = model(**inputs).text_embeds
print(embeddings.shape)  # (3, 768)
```

The processor preserves content order and supports manual `<|image|>`,
`<|audio|>`, and `<|video|>` placeholders for interleaving. Direct `processor(...)`
calls support media-only and nested per-sample inputs. Pass `max_soft_tokens`
(70, 140, 280, 560, or 1120) to control visual budgets; video also accepts `fps`,
`max_frames`, `overflow_strategy`, and `add_timestamps`, as described in the
[upstream documentation](https://huggingface.co/google/embeddinggemma-2/blob/main/embedding_gemma2_documentation.md).
Mismatched expanded media-token and feature counts raise an error.

## Selective tower loading and conversion

```python
from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.utils import get_model_path, load_config

path = get_model_path("nativ-community/embeddinggemma-2")
config = load_config(path)
config["audio_config"] = None
config["vision_config"] = None  # Omit this assignment to retain images/video.
text_model = load_embedding_model(path, config=config)
```

Unused tower weights are discarded during loading. The complete checkpoint can
also be converted and reloaded with the standard CLI:

```sh
mlx_vlm.convert --hf-path nativ-community/embeddinggemma-2 \
  --mlx-path embeddinggemma2-mlx --dtype bfloat16
```

Both `load_embedding_model` and `mlx_vlm.load` accept the converted directory.
Conversion saves the tokenizer, chat template, image/audio/video settings, and
processor configuration. `mlx_vlm.load` restores the local processor from these
files, including when loading offline.

## Performance and quantization

BF16 and affine 4/6/8-bit were measured on Apple M5 Max. BF16 is fastest for text,
image, and audio in these tests. 8-bit reduces full-model weight storage by
17.1% with minimum embedding cosine 0.99966 against the float32 reference;
plain 4-bit falls to 0.96716 and fails the documented accuracy limits.

## Reference validation

Validation uses checkpoint revision `fc77679a26fcb86250765859d04ce2fcc6cb0b2c`,
extras revision `f6c512df20896fd06f85d39db10c45a0a9849ef8`, and the supplied
Transformers 5.18.0.dev0 reference on CPU. The reference always runs in float32;
MLX runs on an Apple M5 Max with float32 or BF16 weights/activations.

All **20 Python examples pass in both precisions**, with **32 paired forward
comparisons per precision**. This covers both Sentence Transformers and AutoModel
examples: retrieval, prompts, Matryoshka, single/composed modalities, ordering,
manual interleaving, heterogeneous batches, selective towers, image budgets,
video sampling/timestamps, and direct/nested processor calls. Processor-only
examples are additionally forwarded through both models. All 128/256/512 prefixes
are compared after renormalization on every forward pass.

| MLX precision vs float32 reference | Max embedding absolute error | Min embedding cosine | Max token relative L2 error |
| --- | ---: | ---: | ---: |
| Float32 | 5.79e-07 | 0.99999999999 | 3.58e-05 |
| BF16 | 0.00271 | 0.99979058879 | 0.167 |

Unnormalized token states are more sensitive to BF16 rounding than the normalized
sentence embeddings. Float32 comparisons disable MLX's default TF32 matmul on M5
(`MLX_ENABLE_TF32=0`); the library does not change users' precision settings.
BF16 conversion/reload produces bit-identical embeddings for text, image, audio,
and video. A heterogeneous BF16 batch versus individual encoding has minimum
cosine similarity 0.99997.

Acceptance thresholds are cosine >=
0.999999 / 0.999 and embedding max error <= 1e-5 / 0.01 for float32 / BF16,
respectively; token relative L2 bounds are 1e-4 / 0.2. Unit regressions cover
bidirectional attention, the inclusive local-window boundary, explicit positions,
padding, media placement, disabled towers, and checkpoint sanitization/reloading:

```sh
python -m pytest -q mlx_vlm/tests/test_models.py \
  mlx_vlm/tests/test_processors.py -k embedding_gemma2
```

Processor validation covers 49 cases across text, images, audio, video, mixed
and nested batches, visual budgets, frame sampling, timestamps, and invalid
inputs. All numeric outputs match the reference exactly, both before and after
processor save/reload, including with stock Transformers 5.14.0. Full BF16 and
8-bit conversions preserve processor outputs and settings; BF16 embeddings are
bit-identical before and after conversion across all modalities and mixed batches.

# Mistral Large 4

Mistral Large 4 supports text and image inference with expert parallelism through
MLX-VLM's standard loading and generation APIs, following the MiniMax sharding
interface. The model implements latent attention, dynamic sparse attention (DSA),
shared KV and indexers, NoPE, LeRoPE, SSMax, and logit softcapping. Vision reuses
the Pixtral encoder and Mistral projector.

## Loading

Use a model directory with a standard `config.json` (`model_type: mistral_large4`,
`text_config`, and `vision_config`), tokenizer/processor files, and safetensors.
The original Mistral release uses `params.json`; that metadata must be normalized
to the standard configuration before loading. This port does not add a separate
checkpoint loader or staging utility.

The model sanitizer maps native weight names and stacks experts. The shared
quantization converter preserves packed NVFP4 expert weights, block scales, and
FP32 global scales, and expands dense FP8 weights to BF16. Its ModelOpt conversion
also handles the release's FP8_PB_WO 128×128 block scales.

## Distributed inference

Use the existing `examples/sharded_generate.py` with an MLX JACCL hostfile and
the same checkout, environment, model, and image paths on each Mac:

```sh
mlx.launch \
  --hostfile /path/to/hosts.json \
  --cwd /path/to/mlx-vlm \
  --python /path/to/venv/bin/python \
  /path/to/mlx-vlm/examples/sharded_generate.py \
  --backend jaccl \
  --model /path/to/model \
  --prompt 'Describe this image.' \
  --image /path/to/image.png \
  --max-tokens 128
```

The runner uses `sharded_load`, which calls `model.shard(group)`, followed by
`stream_generate`. For deterministic distributed decoding, set `do_sample` to
`false` in each model directory's `generation_config.json`.

Experts are split as `N * rank // size : N * (rank + 1) // size`: 64/64 on two
Macs or 42/43/43 on three Macs. Dense layers and vision are replicated. Expert
outputs are reduced before weighting and accumulation, preserving BF16 operation
order. A directory containing only one rank's experts must declare its
`expert_start` and `expert_end` in `text_config`; sharding validates that range.

## Validation and limitations

The released NVFP4 weights have run on two and three 512 GiB M3 Ultra Macs over
TB5 RDMA with MLX main `1ba7f969ad05d86decfeeb83787cc9b79e440962`. Prepared
rank-local weights occupy approximately 316 GB per Mac for two ranks or
217–222 GB for three ranks. Short-request peak MLX allocation was approximately
334 GB and 234–239 GB per device, respectively.

The unmodified `examples/sharded_generate.py` completed both text and image
requests on two Macs: it answered `4` for `2 + 2` and identified a red image.

Small distributed models matched unsharded logits. A small eight-layer model
matched the Transformers reference within `1.08e-6` maximum hidden-state error.
Model, cache, and quantization coverage lives in the shared test suites.

Long-context performance remains unvalidated: attention expands historical KV
and builds dense DSA selection masks. FP8 activation/KV quantization, DSpark/MTP
speculative decoding, continuous batching, and padded variable-length batches
are not implemented or validated here.

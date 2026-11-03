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

## DSpark speculative decoding

The native checkpoint includes a 4.20 GB BF16 draft head. Load it with the shared
drafter API and pass it to the standard generator on every rank:

```python
from mlx_vlm.speculative import load_drafter

draft, kind = load_drafter(model_path)
for response in stream_generate(
    model, processor, prompt=prompt, max_tokens=128, temperature=0,
    draft_model=draft, draft_kind=kind, draft_block_size=4,
):
    if rank == 0:
        print(response.text, end="", flush=True)
```

The block width of four means one anchor plus three proposed tokens. The head
shares the target embedding and output projection and uses three dense MLA
layers, position-specific masks and a recurrent Markov correction. Its metadata
comes from `text_config.mtp_config` or the native `params.json` sidecar.

For rank-local target shards that omit `mtp_dspark.*`, extract the head once with
`python -m mlx_vlm.split_mtp --model /path/to/full-model --output /path/to/draft`,
copy it to each Mac, and pass that directory to `load_drafter`. Draft weights are
replicated; the target retains its existing expert partition.

Verification preserves absorbed decode attention and native NVFP4 accumulation,
and rolls rejected positions back through the shared cache transaction. Batched
speculation uses uniform acceptance across active requests.

Floating-point KV projections use absorbed sparse attention during decode;
prefill still expands historical KV and builds dense DSA selection masks.
FP8 activation/KV quantization is not implemented here. Native NVFP4 expert
weights are supported with continuous batching; additional KV-cache quantization
is not.

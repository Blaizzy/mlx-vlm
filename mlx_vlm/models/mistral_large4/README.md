# Mistral Large 4 on MLX

This port loads the native `mistralai/Mistral-Large-4-1T-A52B-NVFP4`
checkpoint and supports text and image inference with expert parallelism.
It implements latent attention, dynamic sparse attention (DSA), shared KV and
indexers, NoPE layers, LeRoPE, SSMax, and logit softcapping. Vision uses the
existing Pixtral encoder and Mistral projector.

Expert weights keep their original packed NVFP4 values, E4M3 block scales, and
FP32 global scales. Dense FP8 weights are expanded to BF16 at load time by the shared quantization
converter. Like MiniMax, the model uses the standard `load_model` pipeline,
model sanitization, and shared scaled NVFP4 expert layers.
The expert partition is `N * rank // size : N * (rank + 1) // size`, allowing
128 experts to be split across three devices as 42/43/43. Dense layers and the
vision encoder are replicated. Following the MiniMax distributed implementation,
each route is reduced across devices before weighting and summation, preserving
the order of accumulation.

## Prepare the devices

Use the same checkout and Python environment paths on each Mac. The validated
environment used MLX 0.32.4 on all three Macs, Transformers 5.17.0 on rank 0 and
5.22.1 on ranks 1 and 2, with this repository's dependencies. Enable RDMA and
passwordless SSH between hosts, and describe the
Thunderbolt interfaces in an `mlx.launch` JACCL hostfile. All ranks must use the
same source and MLX versions. Set `PYTHONPATH` in the hostfile to this
checkout so a previously installed `mlx-vlm` does not override it.

Example hostfile for the validated three-device topology (adjust for your Macs):

```json
{
  "backend": "jaccl",
  "envs": ["MLX_METAL_FAST_SYNCH=1", "PYTHONPATH=/path/to/mlx-vlm"],
  "hosts": [
    {"ssh": "user0@host0", "ips": ["169.254.216.98"], "rdma": [null, "rdma_en4", "rdma_en5"]},
    {"ssh": "user1@host1", "ips": ["169.254.239.82"], "rdma": ["rdma_en5", null, "rdma_en4"]},
    {"ssh": "user2@host2", "ips": ["169.254.250.99"], "rdma": ["rdma_en5", "rdma_en4", null]}
  ]
}
```

## Stage the checkpoint

The original snapshot occupies 610.1 GB. To avoid copying every expert to every
device, run the staging server on the Mac holding the original snapshot:

```sh
python examples/mistral_large4_shard.py serve /path/to/original/snapshot \
  --bind 169.254.216.98 --port 8944
```

Then run the following on each device, replacing `RANK` with 0, 1, or 2:

```sh
python examples/mistral_large4_shard.py stage \
  http://169.254.216.98:8944 RANK 3 /path/to/model
```

Use a separate destination from the original snapshot. Each destination needs
about 217–222 GB plus 10 GiB of free space. The script copies only the assigned
experts, replicates dense and vision weights, and omits optional MTP draft weights.
It preserves tensor bytes, rewriting unsupported FP8 header types as U8 for
decoding by the shared loader. `rank.json` records the partition.
`model.shard(group)` rejects a partition that does not match the rank. Stop the temporary HTTP server after staging;
it is intended for a trusted private link.

## Run

```sh
/path/to/venv/bin/mlx.launch \
  --hostfile /path/to/hosts.json \
  --cwd /path/to/mlx-vlm \
  --python /path/to/venv/bin/python \
  /path/to/mlx-vlm/examples/mistral_large4_distributed.py \
  --model /path/to/model \
  --prompt 'What is 2 + 2? Answer briefly.' \
  --max-tokens 128 --output /path/to/results
```

For vision, also pass `--image /path/to/image.png`; the image must exist at that
path on every device. The example uses greedy decoding, low reasoning effort,
MLX's wired-memory context, and checks that every rank produces identical token
IDs. Each rank writes its own JSON to its local output directory. Decode timing
excludes prefill and the first generated token. Image encoding is excluded from
prefill timing.

The example uses the same loading and sharding interface as MiniMax:

```python
from pathlib import Path

import mlx.core as mx
from mlx_vlm.utils import load_model

group = mx.distributed.init(backend="jaccl", strict=True)
model = load_model(Path("/path/to/model"), lazy=True)
model.shard(group)
mx.eval(model.parameters())
```

`load_config` recognizes native `params.json` and reads the expert range from
`rank.json` when staged. The common loader handles quantization conversion and
strict weight loading; the model sanitizer maps names and stacks experts.
A full checkpoint can also be sharded after lazy loading, but staging avoids
keeping the entire 610 GB checkpoint on every machine.

## Validation and limits

On 2026-10-30, three M3 Ultra Mac Studios with 512 GiB each completed inference
over TB5 RDMA using JACCL. The native snapshot revision was
`bb6d00fe249b0e876f0cffc6b85014be02de3e68`.

- Text: answered `4`, 26 prompt tokens and 52 generated tokens including reasoning;
  9.64 decode tokens/second at batch size one, with identical output on all ranks.
- Vision: correctly identified a uniform red image on all three ranks;
  98 prompt tokens, 32 generated tokens, and 9.28 decode tokens/second.
- Peak MLX allocation: approximately 234–239 GB per device on these short requests.
- A small eight-layer model matched the Transformers reference
  (`0460c6d97702c7428460d83e53bcea6b76d110c0`) within `1.08e-6` maximum hidden-state
  error. This exercised sparse selection, shared KV, shared indexers, NoPE,
  LeRoPE, and SSMax. Chunked cached inference matched full-sequence inference.
- A three-rank small-model run matched unsharded logits exactly. A real NVFP4
  expert slice matched independently decoded BF16 projection values exactly.
- All 41 model and utility tests passed; formatting and import checks passed.

The architecture reference is the
[Transformers Mistral Large 4 implementation](https://github.com/huggingface/transformers/tree/0460c6d97702c7428460d83e53bcea6b76d110c0/src/transformers/models/mistral_large4).

These are short-request correctness and throughput checks, not a comprehensive
quality benchmark or validation of the advertised million-token context. The
cache retains compressed KV, but attention currently expands historical KV and
builds dense selection masks. Long-context performance needs further work.
FP8 KV/indexer activation quantization, DSpark/MTP speculative decoding,
continuous batching, and padded variable-length batches are not implemented or
validated here. Native-format loading supports the released NVFP4 variant.

### MLX main comparison

The same three Macs, checkpoint, model source, and prompts were also run with
MLX core built from upstream `main` commit
`1ba7f969ad05d86decfeeb83787cc9b79e440962`, version
`0.32.5.dev20261030+1ba7f96`. One wheel was built with Metal and JACCL enabled
and a macOS deployment target of 26.2; installed binary hashes matched on all
three devices.

| Request | MLX 0.32.4 decode tokens/s | MLX main decode tokens/s |
| --- | ---: | ---: |
| Text | 9.64 | 9.31 |
| Image | 9.28 | 9.27 |

All generated tokens matched the release runs exactly. Peak memory remained
234–239 GB per device, all 41 tests passed, and the small three-rank model again
matched unsharded logits exactly. Each timing is one short request, so these
measurements do not establish a statistically significant performance change.

### Two Macs

The same checkpoint also ran successfully on two of the 512 GiB M3 Ultra Macs
over TB5 RDMA, using the same MLX `main` build. Use a two-host JACCL hostfile
and stage with `SIZE=2`, assigning ranks 0 and 1. This gives 64 experts per Mac
and requires about 316.4 GB of checkpoint storage per device. Stage into new
directories; `model.shard(group)` checks that the partition matches the group.

| Request | Two-Mac decode tokens/s | Three-Mac decode tokens/s | Two-Mac peak memory/device |
| --- | ---: | ---: | ---: |
| Text | 9.44 | 9.31 | 334.4 GB |
| Image | 9.14 | 9.27 | 334.1 GB |

The two-Mac results above use the shared loader and shared NVFP4 expert layers.
Both outputs matched the previous two- and three-Mac token sequences exactly. The small two-rank
model also matched unsharded logits exactly. Text prefill took 3.12 seconds
versus 2.04 seconds on three Macs; image prefill took 3.76 versus 2.93 seconds.
These single short-request measurements show comparable decode speed with more
memory per device and slower prefill; long-context behavior remains unvalidated.

The shared-loader refactor passed 677 local checks covering Mistral loading,
cache behavior, native expert partitions, sanitizer idempotence, save/reload,
MiniMax, shared quantization operations, and sampling defaults. CI now includes
the Mistral test file and its cache contract.

# LFM2 MoE / LFM2.5

The native `lfm2_moe` implementation supports
[`lq-preview/LFM2.5-24B-A2B-Preview`](https://huggingface.co/lq-preview/LFM2.5-24B-A2B-Preview).
It uses sigmoid expert routing, selection-only expert bias, normalized top-k
weights, and the configured routed scaling factor. Both individual expert
weights and Transformers packed expert tensors are accepted.

Convert the checkpoint with the standard converter (authenticated Hugging Face
access is needed for this preview):

```sh
python -m mlx_vlm convert \
  --hf-path lq-preview/LFM2.5-24B-A2B-Preview \
  --revision f772950ce6f17d35af5469d00334b61a9c2fc034 \
  --mlx-path /path/to/LFM2.5-24B-A2B-Preview-4bit \
  -q --q-bits 4 --q-group-size 64 --dtype bfloat16

python -m mlx_vlm generate \
  --model /path/to/LFM2.5-24B-A2B-Preview-4bit \
  --prompt "Explain mixture-of-experts models." --max-tokens 256
```

The model's quantization predicate keeps router matrices at 8 bits. Expert bias
remains float32, while unquantized floating weights use BF16.

For an unquantized BF16 export, omit quantization options:

```sh
python -m mlx_vlm convert \
  --hf-path lq-preview/LFM2.5-24B-A2B-Preview \
  --revision f772950ce6f17d35af5469d00334b61a9c2fc034 \
  --mlx-path /path/to/LFM2.5-24B-A2B-Preview-BF16 \
  --dtype bfloat16
```

The BF16 export contains approximately 47.94 GB of weights; expert biases
remain FP32. Use its path in the same generation command.

The pinned revision supports 131,072 context tokens. The earlier `c715ae25`
revision declared a 128,000-token limit.

## Fused short-convolution decode

On Metal, cached single-token inference automatically fuses the input gate,
depthwise convolution, output gate, and state update into one kernel. Input and
output projections still use the standard linear layers. The shared LFM2 path
supports FP32, FP16, and BF16, including quantized models whose convolution
weights remain floating point. It reads strided cache views directly and keeps
the intermediate rounding of the original operations.

The kernel requires evaluation mode (`model.eval()`), an initialized cache,
matching activation/state/convolution dtypes, and a kernel size of 2–8.
Prefill, training, CPU execution, speculative verification with state capture,
and masked or length-tracked batches use the standard implementation.

# mlx_vlm.generate — run inference

Run text, image, audio, or video understanding on a supported MLX vision-language or omni model. The same command also generates images, video, or speech when a matching diffusion or omni checkpoint is loaded.

## Synopsis

```
mlx_vlm.generate [OPTIONS]
```

The console script `mlx_vlm.generate` (or `python -m mlx_vlm generate`) also works. Note: the source prints a deprecation notice for `python -m mlx_vlm.generate` and recommends `mlx_vlm generate` or `python -m mlx_vlm generate` instead.

## Examples

Text-only prompt:

```
mlx_vlm.generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --prompt "Write a haiku about autumn."
```

Image understanding:

```
mlx_vlm.generate \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --image https://example.com/cat.jpg \
  --prompt "Describe this image in detail."
```

Audio understanding:

```
mlx_vlm.generate \
  --model mlx-community/Qwen2-Audio-7B-Instruct-4bit \
  --audio ./clip.wav \
  --prompt "Transcribe this audio."
```

Speculative decoding with a drafter:

```
mlx_vlm.generate \
  --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --draft-model mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --prompt "Explain how KV caching works." \
  --max-tokens 256
```

## Options

### Model & loading

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `mlx-community/nanoLLaVA-1.5-8bit` | Path to the local model directory or Hugging Face repo. |
| `--adapter-path` | `None` | Path to the adapter (LoRA) weights. |
| `--revision` | `main` | Specific model version to load (branch, tag, or commit). |
| `--force-download` | `False` | Force re-download of the model from Hugging Face. |
| `--trust-remote-code` | `False` | Trust remote code when loading the model. |
| `--processor-kwargs` | `{}` | Extra processor kwargs as a JSON object, e.g. `'{"cropping": false, "max_patches": 3}'`. |
| `--gen-kwargs` | `{}` | Extra generation kwargs as a JSON object, e.g. `'{"custom_arg": true}'`. |

### Inputs

| Flag | Default | Description |
| --- | --- | --- |
| `--prompt` | `What are these?` | Message(s) to be processed by the model. |
| `--system` | `None` | System message for the model. |
| `--image` | `None` | One or more image URLs or paths to process. |
| `--audio` | `None` | One or more audio URLs or paths to process. |
| `--video` | `None` | One or more video URLs or paths to process. |
| `--fps` | `2.0` | Frames per second to sample from `--video`. |
| `--video-num-frames` | `None` | Exact number of frames to sample from `--video`, overriding `--fps`. |
| `--video-min-frames` | `None` | Lower bound on the frames sampled from `--video`. |
| `--video-max-frames` | `None` | Upper bound on the frames sampled from `--video`; falls back to 16 for processors without native video support (sent evenly re-sampled stills). |
| `--resize-shape` | `None` | Resize shape (1 or 2 integers) applied to the input image. |

### Text output & sampling

| Flag | Default | Description |
| --- | --- | --- |
| `--max-tokens` | `2048` | Maximum number of tokens to generate. |
| `--temperature` | `0.0` | Sampling temperature; 0 is greedy, positive values below 0.01 are clamped to 0.01. |
| `--top-p` | `1.0` | Nucleus sampling: keep the smallest set of tokens whose probabilities sum to this; 1.0 disables it. |
| `--top-k` | `0` | Keep only the k most probable tokens; 0 disables it. |
| `--min-p` | `0.0` | Drop tokens below this fraction of the top token's probability; 0 disables it. |
| `--repetition-penalty` | `None` | Penalty factor for previously generated tokens. |
| `--repetition-context-size` | `20` | Number of recent generated tokens used for the repetition penalty. |
| `--presence-penalty` | `None` | Additive penalty for tokens that already appeared. |
| `--presence-context-size` | `20` | Number of recent generated tokens used for the presence penalty. |
| `--frequency-penalty` | `None` | Additive penalty scaled by token frequency. |
| `--frequency-context-size` | `20` | Number of recent generated tokens used for the frequency penalty. |
| `--seed` | `None` | PRNG seed for reproducible sampling and diffusion canvas init (image/video generation default to a random 32-bit seed). |
| `--eos-tokens` | `None` | Additional EOS tokens to register with the tokenizer. |
| `--skip-special-tokens` | `False` | Skip special tokens in the detokenized output. |
| `--verbose`, `--no-verbose` | `False` | Print detailed output, timing, and progress bars; by default only the final result is printed. |
| `--output` | `None` | Output path for image, video, or audio generation (.wav for audio). |

### Thinking

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-thinking` | `False` | Enable thinking in the chat template (templates using `thinking_mode` receive `enabled`). |
| `--thinking-mode` | `None` | Chat-template thinking mode when supported: `enabled`, `disabled`, or `adaptive`. |
| `--thinking-budget` | `None` | Maximum number of thinking tokens before forcing the end-of-thinking token. |
| `--thinking-start-token` | `<think>` | Token that marks the start of a thinking block. |
| `--thinking-end-token` | `</think>` | Token that marks the end of a thinking block. |

### Speculative decoding

| Flag | Default | Description |
| --- | --- | --- |
| `--draft-model` | `None` | Speculative drafter path or HF id (e.g. `z-lab/Qwen3.5-4B-DFlash`). |
| `--draft-kind` | `None` | Drafter family: `dflash`, `eagle3`, or `mtp`; auto-detected from the drafter's model type when omitted. |
| `--draft-block-size` | `None` | Override the drafter's configured block size. |

### KV cache & quantization

| Flag | Default | Description |
| --- | --- | --- |
| `--max-kv-size` | `None` | Maximum KV size for the prompt cache. |
| `--kv-bits` | `None` | Number of bits to quantize the KV cache to. |
| `--kv-key-bits` | `None` | Override the TurboQuant key bit-width (defaults to floor of `--kv-bits`). |
| `--kv-value-bits` | `None` | Override the TurboQuant value bit-width (defaults to ceil of `--kv-bits`). |
| `--kv-key-scheme` | `None` | Override the KV quantization backend for keys only: `uniform` or `turboquant`. |
| `--kv-value-scheme` | `None` | Override the KV quantization backend for values only: `uniform` or `turboquant`. |
| `--kv-quant-scheme` | `uniform` | KV cache quantization backend; fractional `--kv-bits` values use TurboQuant automatically. |
| `--kv-group-size` | `64` | Group size for uniform KV cache quantization. |
| `--quantized-kv-start` | `5000` | Token index at which the KV cache starts being quantized. |
| `--quantize-activations`, `-qa` | `False` | Enable activation quantization for QQLinear layers (only for `nvfp4`/`mxfp8` quantized models). |

### Performance & memory

| Flag | Default | Description |
| --- | --- | --- |
| `--prefill-step-size` | `2048` | Tokens processed per prefill step; lower values cut peak memory (try 512 or 256 on prefill OOM). |
| `--expert-cache-gb` | `None` | For an `mlx_vlm.moe_offload` checkpoint, cap the resident routed-expert set in GB (default: 70% of the GPU working set); ignored otherwise. |

### Chat mode

| Flag | Default | Description |
| --- | --- | --- |
| `--chat` | `False` | Run an interactive multi-turn chat session. |

## Image & media generation

These options apply when `--output-modality` selects `image`, `video`, or `audio`, and the diffusion-specific flags apply to diffusion (block/masked) language models. Set `--output` to write the generated file (`.wav` for audio).

| Flag | Default | Description |
| --- | --- | --- |
| `--output-modality` | `text` | Output type: `text` (VLM), `image`, `video`, or `audio` (omni speech). |
| `--task` | `generate` | Image task when `--output-modality image` is set: `generate` or `edit`. |
| `--size` | `None` | Output size as WIDTHxHEIGHT (images default to 512x512; editing uses the first reference size; video uses the model default). |
| `--steps` | `None` | Number of inference steps (defaults to 4 for images, 30 for videos). |
| `--guidance` | `None` | Classifier-free guidance for image generation/editing. |
| `--num-frames` | `None` | Requested number of generated video frames. |
| `--last-image` | `None` | Last-frame conditioning image for FL2VA video generation. |
| `--reference` | `None` | Ordered Ref2VA reference as `KIND=PATH` (KIND is image, video, or audio); repeat to preserve order. |
| `--ref-audio` | `None` | Reference voice audio for `--output-modality audio`. |
| `--workflow` | `None` | Video-generation workflow (`t2va`, `fl2va`, `ref2va`); inferred from inputs when omitted. |
| `--prompt-expansion-model` | `None` | Text model path or HF repo used to expand plain image prompts into Ideogram 4 JSON captions. |
| `--max-denoising-steps` | `None` | Maximum denoising steps for diffusion generation (checkpoint default is typically 48). |
| `--diffusion-sampler` | `confidence-threshold` | Canvas update sampler: `entropy-bound` (reference-style) or `confidence-threshold` (faster for quantized block-diffusion). |
| `--diffusion-full-canvas` | `False` | Use the full checkpoint canvas length even when `--max-tokens` requests a partial block. |
| `--diffusion-min-canvas-length` | `None` | Minimum active canvas length for diffusion partial blocks (default 64). |
| `--diffusion-max-canvas-length` | `None` | Maximum active canvas length for diffusion generation (default: checkpoint canvas length). |
| `--editing-threshold` | `None` | Confidence threshold for diffusion post-fill token edits. |
| `--max-post-steps` | `None` | Maximum diffusion post-fill editing steps per block. |
| `--stability-steps` | `None` | Stop post-fill refinement after this many stable no-edit steps. |
| `--threshold` | `None` | Token probability threshold for diffusion confidence transfer (default 0.9 for confidence-threshold sampling). |
| `--min-threshold` | `None` | Lowest token probability threshold for masked diffusion transfer. |
| `--block-length` | `None` | Block length for diffusion text generation. |
| `--num-to-transfer` | `None` | Target number of masked tokens to transfer per diffusion denoising step. |
| `--max-transfer-per-step` | `None` | Maximum confident masked tokens to transfer per denoising step. |

Image generation example:

```
mlx_vlm.generate \
  --model mlx-community/FLUX.1-schnell-4bit \
  --output-modality image \
  --prompt "A watercolor fox in a snowy forest" \
  --size 1024x1024 --steps 4 \
  --output fox.png
```

## See also

- [Speculative Decoding](../performance/speculative-decoding.md)
- [KV cache quantization](../performance/kv-cache-quantization.md)
- [Models](../models.md)

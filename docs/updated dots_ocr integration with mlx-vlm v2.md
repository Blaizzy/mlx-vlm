Great—thanks for the peer review doc. I incorporated the useful gaps/recs and tightened the plan so it’s directly actionable in `mlx-vlm` with clear APIs, validation, and tests.

# Updated Integration Plan: `dots.ocr` → `mlx-vlm`

## 0) Scope & Guiding Principles

* **Keep Qwen2 (text) as-is**; add a **new MLX vision tower** + glue.
* **Parity-first**: match HF/PyTorch math (RMSNorm, SwiGLU, rotary, non-causal SDPA, 2×2 PatchMerger).
* **Safety & robustness**: strict config/shape validation, graceful fallbacks, strong tests.
* The updates below fold in the critical gaps highlighted by the second analysis (config, batching, masks, tests, etc.).&#x20;

---

## 1) Project Layout (new files)

```
mlx_vlm/
  models/
    dots_ocr/
      __init__.py
      dots_ocr.py              # Adapter/wrapper (vision + text)
      dots_vision.py           # Vision tower (MLX)
      processor.py             # Preproc (resize, norm, grid_thw)
      registry.py              # Model registration hook
  convert/
    convert_dots_ocr.py        # Safetensors → MLX vision weights
  configs/
    dots_ocr_config_schema.json
  tests/
    test_dots_vision_parity.py
    test_dots_masks_batching.py
    test_dots_end2end.py
docs/
  dots_ocr.md                  # user guide, examples, troubleshooting
examples/
  dots_ocr_infer.py
```

---

## 2) Config & Validation (robust configuration management)

* **Schema** (`configs/dots_ocr_config_schema.json`):

  * `vision_config`: `embed_dim`, `num_layers`, `num_heads`, `mlp_hidden_dim`, `patch_size=14`, `merge_size=2`, `rms_eps`, `rotary_theta=10000.0`, `bias=False`, `post_norm=True`.
  * `text_config_ref`: model id for the Qwen2 MLX checkpoint (avoid mixing LLM weights into this repo initially).
  * `processor`: `min_pixels`, `max_pixels`, `mean`, `std`, `interpolation`, `pad_to_multiple_of=14`.
  * `tokens`: `image_token_id`, `chat_template_path` (optional; defaults to model’s).
* **Loader**:

  ```python
  class DotsOCRConfig:
      def __init__(self, cfg: dict):
          self.vision = self._vision(cfg.get("vision_config", {}))
          self.text = self._text(cfg.get("text_config_ref"))
          self.processor = self._processor(cfg.get("processor", {}))
          self.tokens = self._tokens(cfg.get("tokens", {}))
          self._cross_validate()  # e.g., vision.embed_dim == text.hidden_size
  ```
* **Hard checks**: assert **embed\_dim == text.hidden\_size == 1536**, **bias=False** in all vision linears/conv, **merge\_size==2**, **patch\_size==14**, **num\_heads divides embed\_dim**, **rms\_eps present**. Provide clear error messages + remediation hints.&#x20;

---

## 3) Vision Tower (MLX) — implementation details

* **Layers**

  * `PatchEmbed`: `Conv2d(3→1536, k=stride=14, bias=False)` + `RMSNorm(1536, eps)`. Verify MLX conv weight layout at conversion time (add an assert that running a dummy forward matches PyTorch output to <1e-3 MAE).&#x20;
  * **Blocks × N** (e.g., 42):

    * `x += Attn(RMSNorm(x))`; `x += SwiGLU(RMSNorm(x))`.
    * **Attention**: fused QKV (single Linear), split to `(Q,K,V)` → apply **2D rotary** to `(Q,K)` only → **non-causal SDPA** with **block-diagonal mask** per image → output proj (Linear).
    * **MLP**: SwiGLU (`silu(fc1(x)) * fc3(x)` → `fc2`), all **bias=False**.
  * `Post RMSNorm` (if `post_norm=True`).
  * `PatchMerger (2×2)`: `RMSNorm` → `Linear(4*embed_dim → embed_dim)` → `GELU` → `Linear(embed_dim → embed_dim)`; returns **1536**—no projector.

* **2D rotary (explicit)**

  ```python
  def get_2d_rotary_pos_embed(head_dim_half, H, W, theta=10000.0):
      # build cos,sin for H and W then broadcast/interleave to [H*W, head_dim]
      return cos, sin  # shapes align with [seq, heads, head_dim]
  ```

  * Keep head\_dim/2 rotation; cache per `(H,W)`; ensure dtype matches compute (bf16).

* **Attention mask & cu\_seqlens**

  ```python
  def build_cu_seqlens(hw_list):  # hw_list = [(H_i, W_i), ...]
      # return [0, HW_0, HW_0+HW_1, ...] (int32)
  def block_diag_mask_from_cu(cu):
      # produce an SDPA mask that allows attention within each [HW_i] block only
  ```

  * Support multiple images per batch with **variable `H×W`** sequences; mask build must be O(num\_images) and memory-cheap.&#x20;

* **MLX compilation points**

  * Decorate `VisionBlock.__call__` or `DotsVisionTransformer.__call__` with `@mx.compile` once parity is proven (toggle via env flag). Keep a **non-compiled** path for debugging.&#x20;

---

## 4) Processor (image → patches/tokens)

* Respect min/max pixel budget; resize with aspect-ratio preservation; **pad** to multiples of 14; normalize with provided mean/std.
* Return: `pixels` (NCHW), `grid_thw=[[1,H,W]... ]`, and `cu_seqlens`.
* Add **batcher** for mixed sizes:

  ```python
  class DotsOCRBatchProcessor:
      def process(self, images, max_tokens_per_batch=8192):
          # 1) preprocess -> (pixels, H, W, HW)
          # 2) greedy pack by total HW; spill into multiple micro-batches
          # 3) emit batches with their own cu_seqlens & masks
  ```

  This addresses MLX memory characteristics and variable image sizes.&#x20;

---

## 5) Adapter / Token Integration (vision + text)

* Insert **image tokens** precisely where `<|imgpad|>` (or configured `image_token_id`) appears in input; validate **count matches** number of images.
* Build the **decoder attention mask** and positional info the same way other `mlx-vlm` models do.
* API:

  ```python
  class DotsOCRForCausalLM_MLX(nn.Module):
      def __init__(self, cfg: DotsOCRConfig, qwen2_decoder):
          ...
      def encode_images(self, images) -> List[mx.array]:
          # uses Processor + Vision tower → merged tokens
      def prepare_inputs(self, input_ids, vision_tokens, image_token_id):
          # splice tokens, validate positions, build masks
      def generate(self, prompt, images, **gen_kwargs) -> str:
          ...
  ```
* Add **strict checks**: multiple images per prompt, missing image tokens, overflow of total tokens; raise clear errors.&#x20;

---

## 6) Weight Conversion & Loading (robustness)

* `convert_dots_ocr.py`:

  * Use `safetensors.safe_open` to iterate shards; verify all **expected** keys exist and **no extras** (allow a whitelist of ignorable).
  * **Shape checks** per layer; detailed diff in error messages.
  * **Conv weight layout**: detect MLX layout at runtime (small dummy conv compare) and transpose iff required; unit test covers this.&#x20;
  * Keep **vision weights only** in MLX npz (or similar). The **text model** is loaded from a known MLX Qwen2 checkpoint (configured in `text_config_ref`), checked for hidden size = 1536.&#x20;
  * Optionally support PyTorch `.bin` fallback with an explicit flag.

* **Load-time validation**

  ```python
  def load_weights_with_validation(model, path):
      # 1) load index -> expected keys
      # 2) check missing/extra
      # 3) per-key shape match
      # 4) dtype cast (bf16), device placement
  ```

  * Fail fast with a concise report and a path to a fixer script.&#x20;

---

## 7) Quantization Strategy

* **Phase 1**: **No quant** for vision; use bf16 compute.
* **Phase 2 (optional)**: expose `--llm-quant {int8,int4}` with **group\_size=64** (configurable). After quant, run **quality smoke tests** (CER/WER on a tiny eval set) to ensure minimal OCR degradation. Provide a warning that quantizing vision may harm layout fidelity.&#x20;

---

## 8) Error Handling (comprehensive)

* **Malformed images**: type/shape/NaN checks; clear message.
* **Token overflow**: estimate final seq len before splice; suggest reducing image resolution or batch size.
* **Memory exhaustion**: catch MLX OOM; retry with smaller micro-batches (automatic backoff).
* **Weights incompatibility**: summarize first mismatch; print both shapes and the responsible layer path; suggest running `convert_dots_ocr.py` with `--strict`.
* **Missing HF files**: explicit list of required files and download hints.&#x20;

---

## 9) Testing (complete suite)

* **Unit parity (vision)**

  * Same random seed & image → compare:

    * After PatchEmbed
    * After Block\[0] Attn, Block\[0] MLP
    * After final PostNorm
    * After PatchMerger
  * Tolerance: bf16 MAE < 1e-3.
* **Masking & batching**

  * Variable `(H,W)` images, multiple per batch, verify **block-diagonal** masking and deterministic outputs independent of batch grouping.
* **End-to-end**

  * Single image prompt: compare first-step logits vs PyTorch.
  * Multi-image prompt: positions + counts validated.
* **Stress & edge**

  * Tiny images; max-pixels resize; corrupted inputs; long prompts with multiple `<|imgpad|>`.
* **Memory**

  * With `max_tokens_per_batch` sweep, guarantee no OOM on standard Mac configs; document expected throughput.&#x20;

---

## 10) Model Registration & Discovery

```python
SUPPORTED_MODELS.update({
  "dots_ocr": {
    "model_class": "DotsOCRForCausalLM_MLX",
    "config_class": "DotsOCRConfig",
    "processor_class": "DotsOCRProcessor",
    "converter": "convert_dots_ocr"
  }
})
```

Expose a CLI entry (mirroring other models):

```bash
python -m mlx_vlm.convert.convert_dots_ocr --hf rednote-hilab/dots.ocr --out weights/dots_ocr_vision.npz
python examples/dots_ocr_infer.py --model dots_ocr --text-ckpt mlx-community/Qwen2-1_5B-mlx ...
```

(Ensure tests import via the registry.)&#x20;

---

## 11) Documentation & Examples

* **docs/dots\_ocr.md**

  * Install, conversion steps, example inference (single & multi-image), performance tips, quantization notes, troubleshooting (mask errors, shape mismatches, OOM).
* **examples/dots\_ocr\_infer.py**

  * Minimal chat with 1–2 images, showing `<|imgpad|>` token insertion and processor usage.
* **Benchmarks**

  * Latency (Mac M-series), memory footprint, and a tiny qualitative OCR sample (for parity sanity).&#x20;

---

## 12) Recommended Build Order (phased)

1. **Vision core** (RMSNorm, SwiGLU, fused QKV + 2D rotary, SDPA, PatchMerger) + **unit parity**.
2. **Processor + batching** (grid\_thw, cu\_seqlens, masks) + **mask tests**.
3. **Weight conversion** (strict validation) + **loader**.
4. **Adapter + token splice** into Qwen2 + **E2E logits parity**.
5. **Optimizations** (`@mx.compile`), **quant options** for text.
6. **Docs + examples + CI tests**.&#x20;

---

## 13) Acceptance Criteria (Definition of Done)

* Vision unit tests pass with MAE < 1e-3 vs PyTorch checkpoints.
* Mixed-size multi-image batches produce correct outputs (mask test).
* First-step logits delta vs PyTorch ≤ 1e-3 (bf16) on a reference prompt+image.
* No OOM on a baseline Mac (document the tested device); automatic micro-batch fallback works.
* Weight converter catches missing/extra keys with friendly messages.
* Docs include: install, convert, run, quantify gotchas, troubleshooting.

---

If you want, I can generate the **exact weight-name mapping table** (HF → MLX) and a **mask visualizer** to debug `cu_seqlens` on real pages—both are quick adds and save a ton of time during bring-up.

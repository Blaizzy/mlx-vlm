Perfect—since you’ve already cloned both repos locally, here’s the revised, bite-size task plan using your paths:

* `dots.ocr` at `/Users/garryosborne/dots.ocr`
* `mlx-vlm` at `/Users/garryosborne/mlx-vlm`

I’ll assume you’re on macOS with Apple Silicon. Run commands as the user `garryosborne` in a fresh venv.

---

# Phase 0 — Local setup & branch (tiny)

**T0.1 Create a clean Python env (Apple Silicon)**

* Terminal:

  ```bash
  # user: garryosborne
  cd /Users/garryosborne/mlx-vlm
  python3 -m venv .venv
  source .venv/bin/activate
  python -V
  pip install --upgrade pip
  pip install -e .
  ```
* Quick check: `python -c "import mlx_vlm; print('mlx_vlm ok')"`

**T0.2 Feature branch**

```bash
cd /Users/garryosborne/mlx-vlm
git checkout -b feat/dots-ocr
```

---

# Phase A — Scaffolding (super small commits)

**T1.1 Create folders & stubs**

* Create:

  ```
  /Users/garryosborne/mlx-vlm/mlx_vlm/models/dots_ocr/{__init__.py,dots_ocr.py,dots_vision.py,processor.py,registry.py}
  /Users/garryosborne/mlx-vlm/mlx_vlm/convert/convert_dots_ocr.py
  /Users/garryosborne/mlx-vlm/mlx_vlm/configs/dots_ocr_config_schema.json
  /Users/garryosborne/mlx-vlm/tests/{test_dots_vision_parity.py,test_dots_masks_batching.py,test_dots_end2end.py}
  /Users/garryosborne/mlx-vlm/docs/dots_ocr.md
  /Users/garryosborne/mlx-vlm/examples/dots_ocr_infer.py
  ```
* Minimal `registry.py` with placeholder registration.
* Quick check: `python -c "import mlx_vlm.models.dots_ocr.registry as r; print('registry stub ok')"`
* Commit: `chore(dots-ocr): scaffold module, tests, docs`

**T1.2 Register model key (placeholder)**

* Add `SUPPORTED_MODELS["dots_ocr"] = {...}` in your top-level registry import path.
* Commit: `feat(dots-ocr): add registry placeholder`

---

# Phase B — Config & validation (tiny chunks)

**T2.1 Minimal config class**

* Add `DotsOCRConfig` in `dots_ocr.py` (or `config.py` if you prefer).
* Fields: `vision_config`, `text_config_ref`, `processor`, `tokens`.
* Quick check: `python -c "from mlx_vlm.models.dots_ocr.dots_ocr import DotsOCRConfig; DotsOCRConfig({'vision_config':{'embed_dim':1536,'num_heads':12,'num_layers':42,'patch_size':14,'merge_size':2,'bias':False}})"`
* Commit: `feat(dots-ocr): config class`

**T2.2 Strict validators**

* Assert: `embed_dim==1536`, `num_heads|embed_dim`, `patch_size==14`, `merge_size==2`, `bias=False`.
* Commit: `feat(dots-ocr): strict config validation`

---

# Phase C — Vision tower in MLX (micro-steps)

**T3.1 RMSNorm**

* Implement in `dots_vision.py`.
* Unit micro-test: inline NumPy check.
* Commit: `feat(dots-ocr): RMSNorm`

**T3.2 SwiGLU MLP (fc1, fc3, fc2; bias=False)**

* Implement in `dots_vision.py`.
* Shape check only.
* Commit: `feat(dots-ocr): SwiGLU`

**T3.3 PatchEmbed (Conv2d k=s=14 + RMSNorm)**

* Implement; ensure flatten to `[HW, 1536]`.
* Quick check with dummy `[1,3,672,672]`.
* Commit: `feat(dots-ocr): PatchEmbed`

**T3.4 2D Rotary utils**

* Implement `(cos,sin)` builder for (H,W), apply to Q,K only.
* Commit: `feat(dots-ocr): 2D rotary`

**T3.5 Attention (fused QKV + SDPA, non-causal)**

* Implement eager SDPA; accept `cu_seqlens` & build block-diag mask.
* Commit: `feat(dots-ocr): vision attention`

**T3.6 VisionBlock (pre-norm residual)**

* `x += Attn(Norm(x)) ; x += MLP(Norm(x))`
* Commit: `feat(dots-ocr): vision block`

**T3.7 PatchMerger (2×2) + post\_norm**

* Implement merger MLP: `RMSNorm → Linear(4*emb→emb) → GELU → Linear(emb→emb)`.
* Commit: `feat(dots-ocr): PatchMerger + postnorm`

**T3.8 DotsVisionTransformer\_MLX wrapper**

* Wire PatchEmbed + N blocks + post\_norm + merger.
* Commit: `feat(dots-ocr): vision wrapper`

---

# Phase D — Processor & batching (runs independently after T3.3)

**T4.1 Image preprocessor**

* In `processor.py`: min/max pixels, keep aspect, pad to multiples of 14, mean/std normalize.
* Commit: `feat(dots-ocr): image preprocessor`

**T4.2 grid\_thw + cu\_seqlens**

* Return `grid_thw=[[1,H,W],…]` + `cu_seqlens=[0, HW0, …]`.
* Quick check vs your mask visualizer script.
* Commit: `feat(dots-ocr): grid_thw + cu_seqlens`

**T4.3 Micro-batch packer**

* Pack images by `sum(HW)` under threshold; emit multiple micro-batches as needed.
* Commit: `feat(dots-ocr): micro-batching`

---

# Phase E — Weight conversion (vision only)

**T5.1 Safetensors scan**

* In `/Users/garryosborne/mlx-vlm/mlx_vlm/convert/convert_dots_ocr.py`: iterate shards from HF path (you can temporarily point it to `/Users/garryosborne/dots.ocr` weights or your local cache).
* Verify keys against your **HF→MLX mapping CSV** (copy that CSV into repo as `/Users/garryosborne/mlx-vlm/mlx_vlm/convert/dots_ocr_hf_to_mlx_mapping.csv`).
* Commit: `feat(dots-ocr): converter key scan`

**T5.2 Key rename & shape checks**

* Implement rename using the CSV; validate shapes.
* Conv layout: default to no transpose, but keep a guarded fallback if MLX expects HWIO.
* Commit: `feat(dots-ocr): converter rename+shapes`

**T5.3 Emit MLX vision weights**

* Save to `/Users/garryosborne/mlx-vlm/weights/dots_ocr_vision.npz`.
* Quick load test: assign to model; run a dummy forward.
* Commit: `feat(dots-ocr): write MLX vision checkpoint`

---

# Phase F — Adapter + text decoder hookup

**T6.1 Load MLX Qwen2 text model by reference**

* In `dots_ocr.py`: load MLX Qwen2 with `hidden_size==1536` (e.g., from your local MLX cache or configured model id).
* Assert `vision.embed_dim == text.hidden_size`.
* Commit: `feat(dots-ocr): load qwen2 text`

**T6.2 Image token splice**

* Utilities to locate `<|imgpad|>` (or configured `image_token_id`) and splice `vision_tokens`.
* Validate image count == token placeholders.
* Commit: `feat(dots-ocr): image token splice`

**T6.3 Single-image E2E**

* Add `examples/dots_ocr_infer.py` to run `prompt + 1 image`.
* Commit: `feat(dots-ocr): e2e single image`

**T6.4 Multi-image E2E**

* Ensure masks prevent cross-image attention; test with 2 images.
* Commit: `feat(dots-ocr): e2e multi image`

---

# Phase G — Tests (keep each file small)

**T7.1 PatchEmbed parity (PyTorch vs MLX)**

* Put minimal PyTorch reference in `tests/test_dots_vision_parity.py`.
* Compare MAE < 1e-3 after PatchEmbed.
* Commit: `test(dots-ocr): patch embed parity`

**T7.2 Block\[0] parity**

* Compare after Block\[0] attn & mlp.
* Commit: `test(dots-ocr): block0 parity`

**T7.3 PostNorm + Merger parity**

* Compare final tokens (vision output).
* Commit: `test(dots-ocr): postnorm+merger parity`

**T7.4 Masking tests (variable sizes)**

* Build `(H,W)` lists; assert block-diag mask; compare against your visualizer logic.
* Commit: `test(dots-ocr): masking & cu_seqlens`

**T7.5 First-step logits parity (end-to-end)**

* With one known image and prompt; compare decoder first-step logits to tolerance.
* Commit: `test(dots-ocr): e2e first-step logits`

**T7.6 Negative tests**

* Missing `<|imgpad|>`, extra images, token overflow, malformed image.
* Commit: `test(dots-ocr): robust errors`

---

# Phase H — Performance & quality

**T8.1 Optional `@mx.compile` on vision**

* Toggle with env flag; confirm numerics stable.
* Commit: `perf(dots-ocr): enable mx.compile (optional)`

**T8.2 Auto backoff micro-batching**

* Catch OOM → retry smaller `max_tokens_per_batch`.
* Commit: `feat(dots-ocr): OOM backoff`

**T8.3 Text-only quant flags (optional)**

* `--llm-quant {int8,int4}` pass-through; skip vision quant.
* Commit: `feat(dots-ocr): llm quant options`

---

# Phase I — Docs & examples

**T9.1 Example script**

* Finalize `examples/dots_ocr_infer.py` (CLI: `--images … --prompt … --text-ckpt …`).
* Commit: `docs(dots-ocr): example inference`

**T9.2 User guide**

* Fill `docs/dots_ocr.md` with: install, convert, run, tests, troubleshooting (mask errors, shape mismatch, OOM).
* Commit: `docs(dots-ocr): user guide`

**T9.3 README touch**

* Add a short “dots.ocr (MLX)” section linking to the guide.
* Commit: `docs: add dots.ocr to README`

---

## Where to use your existing helper files

* Copy the **HF→MLX mapping CSV** you generated into:

  ```
  /Users/garryosborne/mlx-vlm/mlx_vlm/convert/dots_ocr_hf_to_mlx_mapping.csv
  ```

  Use it in T5.2.
* Keep the **mask visualizer** in your tools dir (outside repo or inside `scripts/`), e.g.:

  ```
  /Users/garryosborne/mlx-vlm/scripts/dots_ocr_mask_visualizer.py
  ```

  Use it during T4.2/T7.4 to sanity-check `cu_seqlens`.

---

If you want, I can also draft the exact file contents for the **smallest possible** first PR (RMSNorm + PatchEmbed + config + stub tests) so you can commit T3.1–T3.3 + T2.1 in one shot.

# Nemotron 3.5 Parse

Native MLX inference for NVIDIA's `RadioMBartPrefixForCausalLM`. This is a
separate architecture from the Nemotron Parse 1.x/2.0 encoder-decoder.

The C-RADIOv4-SO400M tower uses content-normalized NaViT positions and a
convolutional adapter. The decoder uses grouped-query attention, a fully causal
visual prefix, tied embeddings, and no decoder position embeddings. The native
processor preserves the raw six-token control prompt and packs one page at a
time. Plain autoregressive decoding is supported; MTP acceleration is omitted.

```bash
mlx_vlm.convert --hf-path nvidia/Nemotron-3.5-Parse \
  --revision e7ea79242a87ea0ad3919b672bfedb429a683206 \
  --mlx-path Nemotron-3.5-Parse-bf16 --dtype bfloat16
mlx_vlm.generate --model Nemotron-3.5-Parse-bf16 --image page.png \
  --prompt '</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>' \
  --temperature 0 --max-tokens 8744 \
  --repetition-penalty 1.1 --repetition-context-size 0
```

Use `--quantize --q-bits 8` or `--quantize --q-bits 4` to quantize the language
model. The vision tower remains in the selected floating-point dtype.

The forward-pass port follows NVIDIA's provided implementation and is covered by
the OpenMDW-1.1 license. The NVIDIA model materials, including its architecture
and weights, use the same license; see [LICENSE](LICENSE). The tokenizer is
separately licensed under MIT. Retain the source model's license, third-party
notices, and tokenizer attribution when redistributing converted weights. Local
full-checkpoint fidelity measurements are separate from the small architecture
contract tests. They require model downloads and Apple Silicon; they are not CI
checks.

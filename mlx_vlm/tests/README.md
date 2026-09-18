# Test organization

The suite has **15 test modules plus `conftest.py`: 16 Python files**.
`test_extraction_models.py` remains independent. The manual `test_smoke.py`
download runner has been deleted.

Add tests and reusable helpers to the module covering the same production
behavior. Import shared helpers from that module instead of adding separate
fixture files. Keep dependency, hardware, and checkpoint skips scoped to the
tests that need them.

Run the automated suite from the repository root:

```sh
python -m pytest -q mlx_vlm/tests
```

Use a module path or `-k` to select a smaller group while developing.

| Test module | Scope |
| --- | --- |
| `test_models.py` + `model_cases.json` | Shared language, vision, audio, projector, embedding, position and native forward/cache contracts; checkpoint loading, sanitization, and document layout |
| `test_model_ops.py` | Attention kernels and numerical parity, rotary embeddings, weight quantization, packed Hadamard checkpoints, and format conversion |
| `test_cache.py` | Cache lifecycle, APC lookup and prefix reuse, adapters, memory budgets, disk persistence, TurboQuant, batched masks/attention, and vision-feature LRU behavior |
| `test_processors.py` + `processor_cases.json` | Tokenizers and detokenizers, processor loading and media contracts, image/video utilities, prompt construction, and dynamically discovered tool parsers |
| `test_audio_models.py` | Audio/omni model components, speech generation, streaming, checkpoint conversion, audio loading, downmixing, and resampling |
| `test_video_generation_models.py` + `video_generation_cases.json` | Video model components, conditioning workflows, cached trajectories, numerical references, conversion, discovery, request/result adapters, and audio/video muxing |
| `test_extraction_models.py` | GLiNER candidate pools, span/schema handling, privacy tagging/quantized inference, and Sapiens2 vision extraction |
| `test_image_generation_models.py` + `image_generation_cases.json` | Bonsai, Flux2, Ideogram4, Z-Image, ERNIE Image, and Mage Flow generation/editing, components, loading, and conversion |
| `test_diffusion_models.py` | LLaDA, Nemotron, and DiffusionGemma models/generation, numerical parity, caches, vision, sanitization, and generation-config loading |
| `test_speculative.py` | Drafter loading and compatibility, generation parity, verification, cache transactions, and quantized speculative state |
| `test_generate.py` | Generation, sampling, stopping criteria, EOS reset behavior, structured logits, and thinking-phase state |
| `test_server.py` | Chat/Responses/Anthropic APIs, image endpoints, batching/cancellation, runtime settings, reranking, tool stream state, HTTP audio, and realtime voice sessions |
| `test_cli.py` | Text/image/audio/video CLI routing, arguments, diffusion display/visualizers, detector display options, and CLI/library default parity |
| `test_trainer.py` | Training workflows, adapter loading, MRoPE/gated-delta gradients, and MoE gradient/expert-replacement checks |
| `test_moe_offload.py` | MoE checkpoint repacking, expert offload, output parity, and failure handling |

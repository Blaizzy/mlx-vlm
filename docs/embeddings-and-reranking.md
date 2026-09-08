# Embeddings & Reranking

MLX-VLM serves OpenAI-compatible **embeddings** and **reranking** from native MLX models. Both are exposed by the [server](server.md) — `/v1/embeddings` and `/v1/rerank` — and can be preloaded at startup with `--embedding-model` / `--reranker-model`.

## Embeddings

```sh
curl -X POST "http://localhost:8080/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["The quick brown fox.", "A fast auburn fox."]
  }'
```

Preload a default with `--embedding-model <repo-or-path>`. Supported architectures: BERT, XLM-RoBERTa, ModernBERT, Qwen3-Embedding, EmbeddingGemma (gemma3), LFM2, SigLIP (text), Qwen3-VL-Embedding, and Llama-Nemotron-VL, plus LLM2Vec bidirectional Llama. ColBERT-style multi-vector models (ColIdefics3, ColQwen2.5) are also available for late-interaction use.

## Reranking

```sh
curl -X POST "http://localhost:8080/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Reranker-0.6B-4bit",
    "query": "What is the capital of France?",
    "documents": ["Berlin is in Germany.", "Paris is the capital of France."],
    "top_n": 1,
    "return_documents": true
  }'
```

Preload a default with `--reranker-model <repo-or-path>`. Supported text rerankers include one-label BERT, XLM-RoBERTa, and ModernBERT sequence-classification checkpoints, plus Qwen3 generative rerankers. Qwen3-VL rerankers also accept objects containing `text`, `image`, `image_url`, `video`, or `video_url`. Sequence-classification rerankers accept text pairs and do not support custom instructions.

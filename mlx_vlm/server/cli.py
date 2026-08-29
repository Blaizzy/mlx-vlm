import argparse
import logging
import os

import uvicorn

from ..generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
)
from .generation import (
    DEFAULT_ENABLE_THINKING,
    get_log_progress_interval,
    get_server_max_tokens,
    get_server_thinking_budget,
    get_server_thinking_end_token,
    get_server_thinking_start_token,
)
from .runtime import MODEL_DISCOVERY_ENV, MODEL_DISCOVERY_MODES

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080

logger = logging.getLogger("mlx_vlm.server")


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a language model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help="Pre-load an image generation model at startup.",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default=None,
        help="Pre-load a text-to-speech model at startup.",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=None,
        help="Pre-load a speech-to-text model at startup.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Pre-load a supported reranker model at startup.",
    )
    parser.add_argument(
        "--model-discovery",
        choices=MODEL_DISCOVERY_MODES,
        default=None,
        help=(
            "Models exposed by /v1/models: 'served' lists only models loaded by "
            "this process (default); 'hf-cache' also scans the shared Hugging "
            "Face cache. Maps to MLX_VLM_MODEL_DISCOVERY."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--log-progress-interval",
        type=int,
        default=get_log_progress_interval(),
        help=(
            "Decoded tokens between progress log messages; 0 disables periodic "
            "decode progress (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_server_max_tokens(),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Enable thinking mode by default for requests that do not set "
            "enable_thinking explicitly."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=get_server_thinking_budget(),
        help=(
            "Default maximum number of tokens allowed inside a thinking block. "
            "Requests can override this with thinking_budget."
        ),
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=get_server_thinking_start_token(),
        help=(
            "Default token that opens a thinking block. Requests can override "
            "this with thinking_start_token."
        ),
    )
    parser.add_argument(
        "--thinking-end-token",
        "--thinking-eos-token",
        dest="thinking_end_token",
        type=str,
        default=get_server_thinking_end_token(),
        help=(
            "Default token that closes a thinking block. Requests can override "
            "this with thinking_end_token."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-key-bits",
        type=float,
        default=None,
        help="Override the TurboQuant key bit-width (defaults to floor(--kv-bits)).",
    )
    parser.add_argument(
        "--kv-value-bits",
        type=float,
        default=None,
        help="Override the TurboQuant value bit-width (defaults to ceil(--kv-bits)).",
    )
    parser.add_argument(
        "--kv-key-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=None,
        help="Override the KV quantization backend for keys only.",
    )
    parser.add_argument(
        "--kv-value-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=None,
        help="Override the KV quantization backend for values only.",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--expert-cache-gb",
        type=float,
        default=None,
        help="For an mlx_vlm.moe_offload checkpoint, bound the resident routed-"
        "expert set to this many GB (default: 70%% of the GPU's recommended "
        "working set). Ignored for a normal, non-offloaded checkpoint.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, google/gemma-4-31B-it-assistant)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp"],
        help="Drafter family -- 'dflash', 'eagle3', or 'mtp' (Gemma 4). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Maximum number of sequences decoded concurrently in the continuous "
            "batch. Requests beyond this wait in the queue (backpressure), bounding "
            "peak memory. Default: unbounded. Maps to MLX_VLM_MAX_NUM_SEQS."
        ),
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Optional bearer token required for inference, model discovery, and "
            "management endpoints. "
            "Maps to the MLX_VLM_SERVER_API_KEY env var."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    if args.image_model:
        os.environ["MLX_VLM_PRELOAD_IMAGE_MODEL"] = args.image_model
    if args.tts_model:
        os.environ["MLX_VLM_PRELOAD_TTS_MODEL"] = args.tts_model
    if args.stt_model:
        os.environ["MLX_VLM_PRELOAD_STT_MODEL"] = args.stt_model
    if args.embedding_model:
        os.environ["MLX_VLM_PRELOAD_EMBEDDING_MODEL"] = args.embedding_model
    if args.reranker_model:
        os.environ["MLX_VLM_PRELOAD_RERANKER_MODEL"] = args.reranker_model
    if args.model_discovery:
        os.environ[MODEL_DISCOVERY_ENV] = args.model_discovery
    os.environ["MLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["MLX_VLM_DRAFT_MODEL"] = args.draft_model
    if args.draft_kind is not None:
        os.environ["MLX_VLM_DRAFT_KIND"] = args.draft_kind
    if args.draft_block_size is not None:
        os.environ["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.max_num_seqs is not None:
        os.environ["MLX_VLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["MLX_VLM_LOG_PROGRESS_INTERVAL"] = str(args.log_progress_interval)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["MLX_VLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.thinking_budget is not None:
        os.environ["MLX_VLM_THINKING_BUDGET"] = str(args.thinking_budget)
    if args.thinking_start_token is not None:
        os.environ["MLX_VLM_THINKING_START_TOKEN"] = args.thinking_start_token
    if args.thinking_end_token is not None:
        os.environ["MLX_VLM_THINKING_END_TOKEN"] = args.thinking_end_token
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    if args.kv_key_bits is not None:
        os.environ["KV_KEY_BITS"] = str(args.kv_key_bits)
    if args.kv_value_bits is not None:
        os.environ["KV_VALUE_BITS"] = str(args.kv_value_bits)
    if args.kv_key_scheme is not None:
        os.environ["KV_KEY_SCHEME"] = args.kv_key_scheme
    if args.kv_value_scheme is not None:
        os.environ["KV_VALUE_SCHEME"] = args.kv_value_scheme
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.expert_cache_gb is not None:
        os.environ["EXPERT_CACHE_GB"] = str(args.expert_cache_gb)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)
    if args.api_key:
        os.environ["MLX_VLM_SERVER_API_KEY"] = args.api_key

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )

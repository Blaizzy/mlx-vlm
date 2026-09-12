import argparse
import gc
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
from safetensors import safe_open

from ....utils import _load_safetensors, get_model_path
from ..mtp_split import MTPSplitter


def sanitize_dspark_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Map the ``mtp.<stage>.*`` checkpoint layout onto ``stages.<i>.*``.

    Names carry over verbatim otherwise: converted checkpoints already store
    structural shapes (attention projections, per-expert MoE, hyper-connection
    tensors, markov/confidence heads), so the split is a pure namespace move.
    """
    out = {}
    for key, value in weights.items():
        match = re.match(r"mtp\.(\d+)\.(.*)", key)
        if not match:
            out[key] = value
            continue
        out[f"stages.{match.group(1)}.{match.group(2)}"] = value
    return out


def _dequantize_release(tensors: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Fold the release's native quantization to float.

    DeepSeek-V4.1 ships fp8 weights beside ue8m0 ``.scale`` companions and packs
    the routed experts as fp4. Neither is an MLX layout, so convert here rather
    than leave uint8 for a quantizer that only accepts floats. Checkpoints
    already in an MLX layout carry no ``.scale`` and pass through untouched.
    """
    from ....models.deepseek_v41.dequant import dequant_fp4, dequant_fp8, is_fp4_expert

    out = {}
    for key, value in tensors.items():
        if key.endswith(".scale"):
            continue
        scale = (
            tensors.get(f"{key[: -len('.weight')]}.scale")
            if key.endswith(".weight")
            else None
        )
        if scale is None:
            out[key] = value
        elif is_fp4_expert(key):
            out[key] = dequant_fp4(value, scale)
        else:
            out[key] = dequant_fp8(value, scale)
    return out


class DeepseekV41DsparkSplitter(MTPSplitter):
    """Extract DeepSeek-V4.1's native DSpark head (``mtp.<stage>.*``, stages 0..2)
    into a standalone ``deepseek_v41_dspark`` drafter."""

    output_model_type = "deepseek_v41_dspark"
    draft_model_cls = None
    require_text_config = False
    tie_word_embeddings_default = False
    block_size_extra = 1
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    )

    def __init__(self):
        self._n_mtp_layers = 3

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith("mtp.")

    def load_shard(self, file: Path, keys: List[str]) -> Dict[str, mx.array]:
        try:
            with safe_open(file, framework="mlx") as stream:
                return {key: mx.array(stream.get_tensor(key)) for key in keys}
        except (AttributeError, RuntimeError, TypeError):
            shard = _load_safetensors(str(file))
            return {key: shard[key] for key in keys}

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return sanitize_dspark_weights(_dequantize_release(tensors))

    def should_quantize_key(self, key: str) -> bool:
        """The router gate stays full precision; this family names it ``ffn.gate``."""
        return super().should_quantize_key(key) and not key.endswith("ffn.gate.weight")

    def quantization_from_source(self, tensors, source_config):
        recipe = source_config.get("quantization_config") or {}
        base = {
            "group_size": int(recipe.get("group_size", 64)),
            "bits": int(recipe.get("bits", 8)),
            "mode": recipe.get("mode", "affine"),
        }
        quantization = dict(base)
        for key in tensors:
            if key.endswith(".scales"):
                quantization[key[: -len(".scales")]] = dict(base)
        return quantization if len(quantization) > 3 else None

    def extra_config(self, text_config: dict) -> dict:
        dspark_block = int(text_config.get("dspark_block_size", 0) or 0)
        targets = list(text_config.get("dspark_target_layer_ids", []) or [])
        noise = int(text_config.get("dspark_noise_token_id", 0) or 0)
        rank = int(text_config.get("dspark_markov_rank", 256) or 256)
        return {
            "n_mtp_layers": self._n_mtp_layers,
            "dspark_block_size": dspark_block,
            "dspark_noise_token_id": noise,
            "dspark_target_layer_ids": targets,
            "dspark_markov_rank": rank,
            "target_layer_ids": targets,
            "mask_token_id": noise,
            "markov_rank": rank,
            "block_size": (dspark_block + 1) if dspark_block else 0,
        }

    def split(
        self,
        source: str,
        output: str,
        *,
        revision: Optional[str] = None,
        block_size: Optional[int] = None,
        force_download: bool = False,
        **quant_opts,
    ) -> Path:
        """Stream each native DSpark stage into a separate MLX shard."""
        source_path = get_model_path(
            source, revision=revision, force_download=force_download
        )
        output_path = Path(output)
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError(f"DSpark output is not empty: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        with (source_path / "config.json").open() as stream:
            source_config = json.load(stream)
        text_config = self.read_text_config(source_config)
        stage_sources: dict[int, list[tuple[Path, list[str]]]] = {}
        for file, keys in self.iter_selected(source_path, text_config):
            for key in keys:
                match = re.match(r"mtp\.(\d+)\.", key)
                if match is not None:
                    stage_sources.setdefault(int(match.group(1)), []).append(
                        (file, [key])
                    )
        if not stage_sources:
            raise ValueError(f"No DSpark tensors found in {source_path}.")

        self._n_mtp_layers = max(stage_sources) + 1
        expected_stages = set(range(self._n_mtp_layers))
        if set(stage_sources) != expected_stages:
            raise ValueError(
                "DSpark checkpoint stages must be contiguous from zero; got "
                f"{sorted(stage_sources)}."
            )
        output_map: dict[str, str] = {}
        total_size = 0
        quantization = None
        for output_index, stage_id in enumerate(sorted(stage_sources), start=1):
            selected: dict[str, mx.array] = {}
            by_file: dict[Path, list[str]] = {}
            for file, keys in stage_sources[stage_id]:
                by_file.setdefault(file, []).extend(keys)
            for file, keys in by_file.items():
                selected.update(self.load_shard(file, keys))

            weights = self.transform(selected, text_config, source_is_mlx=False)
            stage_quantization = self.quantization(
                weights, source_config, text_config, quant_opts
            )
            if stage_quantization is not None:
                if quantization is None:
                    quantization = {
                        key: stage_quantization[key]
                        for key in ("group_size", "bits", "mode")
                    }
                quantization.update(stage_quantization)
            filename = (
                f"model-{output_index:05d}-of-{len(stage_sources):05d}.safetensors"
            )
            mx.save_safetensors(
                str(output_path / filename),
                weights,
                metadata={"format": "mlx", "model_type": self.output_model_type},
            )
            for key, value in weights.items():
                if key in output_map:
                    raise ValueError(f"Duplicate converted DSpark tensor: {key}")
                output_map[key] = filename
                total_size += value.nbytes
            del selected, weights
            gc.collect()
            mx.clear_cache()

        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(output_map.items())),
        }
        (output_path / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )

        depth = self.depth(text_config)
        draft_config = {
            "model_type": self.output_model_type,
            "text_config": text_config,
            "block_size": depth + self.block_size_extra,
            "tie_word_embeddings": bool(
                text_config.get("tie_word_embeddings", self.tie_word_embeddings_default)
            ),
        }
        draft_config.update(self.extra_config(text_config))
        if block_size is not None:
            draft_config["block_size"] = int(block_size)
        if quantization is not None:
            draft_config["quantization"] = quantization
            draft_config["quantization_config"] = quantization
        (output_path / "config.json").write_text(
            json.dumps(dict(sorted(draft_config.items())), indent=2) + "\n"
        )

        for name in self.tokenizer_files:
            tokenizer_file = source_path / name
            if tokenizer_file.exists():
                shutil.copy2(tokenizer_file, output_path / name)
        return output_path


def split_deepseek_v41_dspark(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write DeepSeek-V4.1 DSpark MTP tensors into a standalone drafter folder."""
    return DeepseekV41DsparkSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split DeepSeek-V4.1 DSpark MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_deepseek_v41_dspark(**vars(args))
    print(f"Wrote DeepSeek-V4.1 DSpark drafter to {output}")


if __name__ == "__main__":
    main()

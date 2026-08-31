"""Central framework for extracting native MTP tensors into a standalone drafter.

One ``MTPSplitter`` base owns the shared mechanics (shard discovery, selective
load, config assembly, tokenizer copy). A family customizes by subclassing and
overriding the small hooks that vary: ``select_keys`` (which tensors are MTP),
``rename`` / ``on_mlx_source``, ``sanitize_ctx``, ``postprocess``, and
``quantization``. Register each splitter by its base ``model_type`` in
``MTP_SPLITTERS`` (lazy import paths) so ``convert`` and ``split_mtp`` can
dispatch on a source checkpoint.
"""

import glob
import importlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
from safetensors import safe_open

from ...utils import get_model_path


def _safetensor_files(model_path: Path) -> List[Path]:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        return json.load(f).get("weight_map", {})


def _is_mlx_safetensors(file: Path) -> bool:
    with safe_open(file, framework="mlx") as f:
        return (f.metadata() or {}).get("format") == "mlx"


class MTPSplitter:
    # --- declarative per-family config (override in subclass) ---
    output_model_type: str = ""
    draft_model_cls = None  # class with a ``sanitize(self, weights)`` staticlike method
    require_text_config: bool = (
        True  # True: use text_config only; False: fall back to root
    )
    tie_word_embeddings_default: bool = False
    depth_field: str = "num_nextn_predict_layers"
    block_size_extra: int = 1  # block_size default = depth + block_size_extra
    supports_mlx_source: bool = False  # re-split of an already-MLX drafter
    tokenizer_files: Tuple[str, ...] = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    # --- hooks (override the small differences) ---
    def read_text_config(self, source_config: dict) -> dict:
        text_config = dict(
            source_config.get("text_config")
            or ({} if self.require_text_config else source_config)
        )
        if self.require_text_config and not text_config:
            raise ValueError("source config does not contain a text_config.")
        return text_config

    def select_keys(self, key: str, text_config: dict) -> bool:
        raise NotImplementedError

    def load_shard(self, file: Path, keys: List[str]) -> Dict[str, mx.array]:
        try:
            with safe_open(file, framework="mlx") as f:
                return {key: mx.array(f.get_tensor(key)) for key in keys}
        except (AttributeError, RuntimeError, TypeError):
            shard = mx.load(str(file))
            return {key: shard[key] for key in keys}

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return tensors

    def on_mlx_source(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return tensors

    def sanitize_ctx(self, text_config: dict):
        return None

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        if self.draft_model_cls is None:
            return tensors
        return self.draft_model_cls.sanitize(self.sanitize_ctx(text_config), tensors)

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        pass

    def quantization(
        self,
        tensors: Dict[str, mx.array],
        source_config: dict,
        text_config: dict,
        quant_opts: dict,
    ) -> Optional[dict]:
        # (a) source already carries quantized MTP tensors -> record their config
        existing = self.quantization_from_source(tensors, source_config)
        if existing is not None:
            return existing
        # (b) fp drafter + a quantization was requested (e.g. convert --mtp --bits)
        q_bits = quant_opts.get("q_bits")
        if q_bits is None:
            return None
        return self._affine_quantize(
            tensors, q_bits, quant_opts.get("q_group_size", 64)
        )

    def quantization_from_source(
        self, tensors: Dict[str, mx.array], source_config: dict
    ) -> Optional[dict]:
        return None

    def should_quantize_key(self, key: str) -> bool:
        # skip the router gate (kept full precision for routing stability); norms
        # and the fp32 correction bias fall out via the ndim/divisibility checks
        return key.endswith(".weight") and not key.endswith("mlp.gate.weight")

    def _affine_quantize(
        self, weights: Dict[str, mx.array], bits: int, group_size: int
    ) -> Optional[dict]:
        quantized_any = False
        for key in list(weights):
            if not self.should_quantize_key(key):
                continue
            weight = weights[key]
            if weight.ndim < 2 or weight.shape[-1] % group_size != 0:
                continue
            wq, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
            weights[key] = wq
            weights[key[: -len(".weight")] + ".scales"] = scales
            weights[key[: -len(".weight")] + ".biases"] = biases
            quantized_any = True
        if not quantized_any:
            return None
        return {"group_size": group_size, "bits": bits, "mode": "affine"}

    def depth(self, text_config: dict) -> int:
        return int(text_config.get(self.depth_field, 1) or 1)

    def extra_config(self, text_config: dict) -> dict:
        return {}

    # --- shared orchestration (do not override) ---
    def iter_selected(
        self, source_path: Path, text_config: dict
    ) -> Iterable[Tuple[Path, List[str]]]:
        weight_map = _weight_map(source_path)
        if weight_map:
            by_file: Dict[str, List[str]] = {}
            for key, filename in weight_map.items():
                if self.select_keys(key, text_config):
                    by_file.setdefault(filename, []).append(key)
            if by_file:
                for filename, keys in by_file.items():
                    yield source_path / filename, keys
                return

        for file in _safetensor_files(source_path):
            with safe_open(file, framework="mlx") as f:
                keys = [key for key in f.keys() if self.select_keys(key, text_config)]
            if keys:
                yield file, keys

    def transform(
        self, tensors: Dict[str, mx.array], text_config: dict, source_is_mlx: bool
    ) -> Dict[str, mx.array]:
        tensors = self.rename(tensors, text_config)
        if source_is_mlx and self.supports_mlx_source:
            return self.on_mlx_source(tensors, text_config)
        tensors = self.run_sanitize(tensors, text_config)
        self.postprocess(tensors, text_config)
        return tensors

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
        source_path = get_model_path(
            source, revision=revision, force_download=force_download
        )
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(source_path / "config.json") as f:
            source_config = json.load(f)
        text_config = self.read_text_config(source_config)

        selected: Dict[str, mx.array] = {}
        source_is_mlx = False
        for file, keys in self.iter_selected(source_path, text_config):
            if self.supports_mlx_source:
                source_is_mlx = source_is_mlx or _is_mlx_safetensors(file)
            selected.update(self.load_shard(file, keys))
        if not selected:
            raise ValueError(f"No MTP tensors found in {source_path}.")

        weights = self.transform(selected, text_config, source_is_mlx)
        quantization = self.quantization(
            weights, source_config, text_config, quant_opts
        )

        mx.eval(list(weights.values()))
        mx.save_safetensors(
            str(output_path / "model.safetensors"),
            weights,
            metadata={"format": "mlx"},
        )

        depth = self.depth(text_config)
        draft_config = {
            "model_type": self.output_model_type,
            "text_config": text_config,
            "block_size": int(block_size or depth + self.block_size_extra),
            "tie_word_embeddings": bool(
                text_config.get("tie_word_embeddings", self.tie_word_embeddings_default)
            ),
        }
        draft_config.update(self.extra_config(text_config))
        if quantization is not None:
            draft_config["quantization"] = quantization
            draft_config["quantization_config"] = quantization

        with open(output_path / "config.json", "w") as f:
            json.dump(dict(sorted(draft_config.items())), f, indent=2)

        for name in self.tokenizer_files:
            src = source_path / name
            if src.exists():
                shutil.copy(src, output_path / name)

        return output_path


# base model_type -> "module_path:ClassName" (lazy so importing this module is cheap)
MTP_SPLITTERS: Dict[str, str] = {
    "qwen3_5": "mlx_vlm.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",
    "qwen3_5_moe": "mlx_vlm.speculative.drafters.qwen3_5_mtp.split:Qwen3_5MTPSplitter",
    "qwen3_next": "mlx_vlm.speculative.drafters.qwen3_5_mtp.split:Qwen3NextMTPSplitter",
    "qwen4_exp": "mlx_vlm.speculative.drafters.qwen4_exp_mtp.split:Qwen4ExpMTPSplitter",
    "qwen4_exp_text": "mlx_vlm.speculative.drafters.qwen4_exp_mtp.split:Qwen4ExpMTPSplitter",
    "deepseek_v4": "mlx_vlm.speculative.drafters.deepseek_v4_mtp.split:DeepseekV4MTPSplitter",
    "glm4_moe_lite": "mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split:Glm4MoeLiteMTPSplitter",
    "glm5_next": "mlx_vlm.speculative.drafters.glm5_next_mtp.split:Glm5NextMTPSplitter",
    "glm_moe_dsa": "mlx_vlm.speculative.drafters.glm_moe_dsa_mtp.split:GlmMoeDsaMTPSplitter",
    "inkling_mm_model": "mlx_vlm.speculative.drafters.inkling_mtp.split:InklingMTPSplitter",
}


def get_mtp_splitter(base_model_type: str) -> Optional[MTPSplitter]:
    target = MTP_SPLITTERS.get(base_model_type)
    if target is None:
        return None
    module_path, class_name = target.split(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls()


def detect_mtp_splitter(model_path: Path) -> Optional[MTPSplitter]:
    """Return the splitter for a source checkpoint, or None.

    Chooses by base ``model_type`` in config.json, then confirms MTP tensors are
    actually present (config flags alone are unreliable -- some models declare
    MTP but ship no tensors, others ship tensors with no flag).
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        source_config = json.load(f)
    text_config = source_config.get("text_config") or source_config
    # Some checkpoints name the inner text stack separately from the
    # architecture (Apodex 1.1 uses text_config "qwen3_5_moe_text" under a
    # root "qwen3_5_moe"), so fall back to the root type before giving up.
    splitter = None
    for base_model_type in (
        text_config.get("model_type"),
        source_config.get("model_type"),
    ):
        if not base_model_type:
            continue
        splitter = get_mtp_splitter(base_model_type)
        if splitter is not None:
            break
    if splitter is None:
        return None
    tc = splitter.read_text_config(source_config)
    for _ in splitter.iter_selected(model_path, tc):
        return splitter
    return None

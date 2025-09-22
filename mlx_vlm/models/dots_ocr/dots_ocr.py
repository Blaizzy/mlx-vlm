from dataclasses import dataclass

import mlx.core as mx

from .dots_vision import DotsVisionTransformer_MLX
from .tokenizer import SimpleTokenizer, render_chat
from .weight_loader import load_npz_into_vision

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is required for projector loading.
    _np = None

@dataclass
class VisionCfg:
    embed_dim: int = 1536
    num_heads: int = 12
    num_layers: int = 42
    patch_size: int = 14
    merge_size: int = 2
    bias: bool = False
    rms_eps: float = 1e-6
    rotary_theta: float = 10000.0
    post_norm: bool = True

@dataclass
class ProcessorCfg:
    min_pixels: int = 262144
    max_pixels: int = 1048576
    mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    std:  tuple = (0.26862954, 0.26130258, 0.27577711)
    pad_to_multiple_of: int = 14

@dataclass
class TokenCfg:
    image_token_id: int = 151652
    chat_template_path: str | None = None

class DotsOCRConfig:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        v = cfg.get("vision_config", {})
        p = cfg.get("processor", {})
        t = cfg.get("tokens", {})
        self.vision = VisionCfg(**{k: v.get(k, getattr(VisionCfg, k, None)) for k in VisionCfg.__annotations__.keys()})
        self.processor = ProcessorCfg(**{k: p.get(k, getattr(ProcessorCfg, k, None)) for k in ProcessorCfg.__annotations__.keys()})
        self.tokens = TokenCfg(**{k: t.get(k, getattr(TokenCfg, k, None)) for k in TokenCfg.__annotations__.keys()})
        self.text_config_ref = cfg.get("text_config_ref", None)
        self._validate()

    def _validate(self):
        assert self.vision.embed_dim == 1536, "embed_dim must be 1536"
        assert self.vision.embed_dim % self.vision.num_heads == 0, "num_heads must divide embed_dim"
        assert self.vision.patch_size == 14, "patch_size must be 14"
        assert self.vision.merge_size == 2, "merge_size must be 2"
        assert self.vision.bias is False, "vision layers must use bias=False"
        assert self.vision.num_layers > 0, "num_layers > 0"


class DotsOCRForCausalLM_MLX:
    """
    Thin vision adapter that preprocesses PIL images and returns merged tokens.
    """

    def __init__(self, cfg: DotsOCRConfig):
        self.cfg = cfg
        from .processor import DotsOCRProcessor

        self.processor = DotsOCRProcessor(cfg)
        self.vision = DotsVisionTransformer_MLX(cfg)
        self.projector_weight: mx.array | None = None

    def encode_images(self, pil_images, max_tokens_per_batch: int | None = None):
        images = list(pil_images)
        if not images:
            raise ValueError("encode_images requires at least one image")

        processed = self.processor.process(images)
        from .processor import GroupedBatchPacker

        packer = GroupedBatchPacker(max_tokens_per_batch)
        tokens = []
        grids: list[tuple[int, int, int]] = []
        for px_batch, grid_batch in packer.pack(processed):
            vt = self.vision(px_batch, grid_batch)
            tokens.append(vt)
            for gr in grid_batch:
                grids.append(tuple(gr))

        tokens_out = tokens[0] if len(tokens) == 1 else mx.concatenate(tokens, axis=0)
        return tokens_out, grids

    def load_vision_npz(self, npz_path: str):
        """Load converted NPZ weights into the vision tower."""

        return load_npz_into_vision(self.vision, npz_path)

    def load_projector_npz(self, npz_path: str):
        """Load linear projector weights (vision -> text hidden)."""

        if _np is None:
            raise RuntimeError("numpy is required to load projector weights")

        if not npz_path:
            raise ValueError("Projector NPZ path is required")

        try:
            with _np.load(npz_path) as data:
                weight = data.get("projector.proj.weight")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Projector NPZ not found: {npz_path}") from exc

        if weight is None:
            raise ValueError(
                f"projector.proj.weight missing from NPZ {npz_path}"
            )
        if weight.ndim != 2:
            raise ValueError(
                f"Projector weight must be 2D, got shape {weight.shape}"
            )

        vision_dim = self.cfg.vision.embed_dim
        out_dim, in_dim = weight.shape
        if in_dim == vision_dim:
            proj = weight.astype(_np.float32, copy=False)
        elif out_dim == vision_dim:
            proj = weight.T.astype(_np.float32, copy=False)
        else:
            raise ValueError(
                "Projector weight dimensions do not match vision embedding dimension: "
                f"expected one axis to equal {vision_dim}, got {weight.shape}"
            )

        self.projector_weight = mx.array(proj)

        def _project(tokens: mx.array) -> mx.array:
            tokens32 = mx.array(tokens, dtype=mx.float32)
            return mx.matmul(tokens32, self.projector_weight.T)

        self.projector = _project
        return {"shape": tuple(proj.shape)}

    def prepare_inputs(
        self,
        prompt: str,
        tokenizer: SimpleTokenizer,
        vision_tokens,
        image_token_id: int | None = None,
    ):
        """Tokenize prompt and compute fused length after vision splicing."""

        token_id = image_token_id or tokenizer.image_token_id
        text = render_chat(prompt)
        input_ids = tokenizer.encode(text)

        if isinstance(vision_tokens, list):
            positions, fused_len = splice_image_tokens_multi(
                input_ids, token_id, vision_tokens
            )
        else:
            position, fused_len = splice_image_tokens(input_ids, token_id, vision_tokens)
            positions = [position]

        return input_ids, positions, fused_len

    def generate(self, prompt: str, pil_images, npz_path: str | None = None):
        """Vision-only demo that returns splice bookkeeping information."""

        if npz_path:
            self.load_vision_npz(npz_path)

        vision_tokens, grids = self.encode_images(pil_images)

        tok_path = None
        try:
            tok_path = os.environ.get("DOTS_TOKENIZER_DIR")
        except Exception:
            tok_path = None

        if tok_path:
            from .tokenizer import load_qwen_tokenizer

            hf_tok = load_qwen_tokenizer(tok_path)
            text = render_chat(prompt)
            encoded = hf_tok(
                text, add_special_tokens=True, return_tensors="np"
            )
            ids = encoded["input_ids"][0].tolist()
            if "<image>" in hf_tok.get_vocab():
                image_tok_id = hf_tok.convert_tokens_to_ids("<image>")
            else:
                image_tok_id = self.cfg.tokens.image_token_id
            position, fused_len = splice_image_tokens(ids, image_tok_id, vision_tokens)
            positions = [position]
            input_ids = ids
        else:
            tokenizer = SimpleTokenizer(image_token_id=self.cfg.tokens.image_token_id)
            input_ids, positions, fused_len = self.prepare_inputs(
                prompt, tokenizer, vision_tokens
            )
        return {
            "input_len": len(input_ids),
            "image_pos": positions,
            "fused_len": fused_len,
            "tokens_shape": tuple(vision_tokens.shape),
            "grids": grids,
        }


def splice_image_tokens(input_ids, image_token_id: int, vision_tokens):
    """Return the placeholder position and fused length when splicing tokens."""

    ids = [int(tok) for tok in input_ids]
    positions = [idx for idx, tok in enumerate(ids) if tok == image_token_id]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly 1 image token, found {len(positions)}")

    try:
        vision_count = int(vision_tokens.shape[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("vision_tokens must expose shape[0]") from exc

    fused_len = len(ids) - 1 + vision_count
    return positions[0], fused_len


def splice_image_tokens_multi(input_ids, image_token_id: int, vision_token_list):
    """Return placeholder positions and fused length for multiple images."""

    ids = [int(tok) for tok in input_ids]
    positions = [idx for idx, tok in enumerate(ids) if tok == image_token_id]
    if len(positions) != len(vision_token_list):
        raise ValueError(
            "placeholders ({}) != images ({})".format(
                len(positions), len(vision_token_list)
            )
        )

    try:
        vision_total = sum(int(v.shape[0]) for v in vision_token_list)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("Each vision token chunk must expose shape[0]") from exc

    fused_len = len(ids) - len(positions) + vision_total
    return positions, fused_len

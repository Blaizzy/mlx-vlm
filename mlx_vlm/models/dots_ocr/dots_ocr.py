from dataclasses import dataclass

import mlx.core as mx

from .dots_vision import DotsVisionTransformer_MLX
from .tokenizer import SimpleTokenizer, render_chat
from .weight_loader import load_npz_into_vision

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

    def encode_images(self, pil_images):
        images = list(pil_images)
        if not images:
            raise ValueError("encode_images requires at least one image")

        processed = self.processor.process(images)
        tokens = []
        grids = []
        for pixels, grid in processed:
            vt = self.vision(pixels, grid)
            tokens.append(vt)
            grids.append(tuple(grid[0]))
        tokens_out = tokens[0] if len(tokens) == 1 else mx.concatenate(tokens, axis=0)
        return tokens_out, grids

    def load_vision_npz(self, npz_path: str):
        """Load converted NPZ weights into the vision tower."""

        return load_npz_into_vision(self.vision, npz_path)

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

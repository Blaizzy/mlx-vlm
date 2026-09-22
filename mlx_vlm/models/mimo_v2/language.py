import mlx.core as mx

_FP8_BLOCK = 128


def _layer_index(key):
    parts = key.split(".")
    return int(parts[parts.index("layers") + 1])


def _dequant_block_fp8(weight, scale, block=_FP8_BLOCK):
    weight = mx.from_fp8(weight, dtype=mx.float32)
    rows, cols = weight.shape
    pad_rows = block * scale.shape[0] - rows
    pad_cols = block * scale.shape[1] - cols
    weight = mx.pad(weight, ((0, pad_rows), (0, pad_cols)))
    weight = weight.reshape(
        (rows + pad_rows) // block, block, (cols + pad_cols) // block, block
    )
    weight = (weight * scale[:, None, :, None]).reshape(
        rows + pad_rows, cols + pad_cols
    )
    return weight[:rows, :cols].astype(mx.bfloat16)


from ..mimo_v2_flash.language import LanguageModel as MiMoV2FlashLanguageModel
from .config import TextConfig


class LanguageModel(MiMoV2FlashLanguageModel):
    """MiMo-V2.6 text backbone.

    The decoder stack is unchanged from MiMo-V2-Flash. V2.6 differs only in how
    the checkpoint stores its weights: attention projections are fused into a
    single ``qkv_proj``, and the routed experts ship as MXFP4 rather than FP8.
    Both are undone in :meth:`sanitize` so the inherited modules load as-is.
    """

    def __init__(self, config: TextConfig):
        super().__init__(config)

    def _qkv_sizes(self, layer):
        args = self.args
        if args.hybrid_layer_pattern[layer]:
            heads, kv = args.swa_num_attention_heads, args.swa_num_key_value_heads
            head_dim, v_head_dim = args.swa_head_dim, args.swa_v_head_dim
        else:
            heads, kv = args.num_attention_heads, args.num_key_value_heads
            head_dim, v_head_dim = args.head_dim, args.v_head_dim
        return [heads * head_dim, kv * head_dim, kv * v_head_dim]

    def _dequant_fused_qkv(self, weights):
        """Dequantize ``qkv_proj`` one projection at a time.

        The fused weight concatenates q, k and v, but its FP8 scale grid
        concatenates the grids the three projections had while they were still
        separate tensors, and each of those covers its projection zero-padded
        to a multiple of 512 rows. A full attention layer's k is 768 rows but
        carries 8 block-rows (1024 padded), so the fused grid is 96 + 8 + 4 =
        108 for a weight only 106 blocks tall. Walking it contiguously lands v
        two block-rows early. Verified against MiMo-V2-Flash, which ships the
        same projections unfused with the same 8-row grid for its 768-row k.
        """
        out = {}
        for key, value in weights.items():
            fused = key.endswith("self_attn.qkv_proj.weight") and ".mtp." not in key
            if not fused:
                is_scale = (
                    key.endswith("self_attn.qkv_proj.weight_scale_inv")
                    and ".mtp." not in key
                )
                if not is_scale:
                    out[key] = value
                continue
            scale = weights[f"{key}_scale_inv"]
            prefix = key[: -len("qkv_proj.weight")]
            sizes = self._qkv_sizes(_layer_index(key))
            if sum(sizes) != value.shape[0]:
                raise ValueError(
                    f"{key}: expected fused dim {sum(sizes)}, got {value.shape[0]}"
                )
            row = scale_row = 0
            for name, size in zip(("q_proj", "k_proj", "v_proj"), sizes):
                n = -(-size // 512) * 512 // _FP8_BLOCK
                out[f"{prefix}{name}.weight"] = _dequant_block_fp8(
                    value[row : row + size], scale[scale_row : scale_row + n]
                )
                row += size
                scale_row += n
        return out

    def _split_fused_qkv(self, weights):
        """Split ``qkv_proj`` into q/k/v.

        Must run *after* the FP8 dequantization in the parent's ``sanitize``:
        the fused weight and its ``weight_scale_inv`` are a matched pair, and
        the scale grid covers the weight zero-padded up to a multiple of 512
        rows, so the split offsets are only meaningful once it is dense.
        """
        args = self.args
        out = {}
        for k, v in weights.items():
            if not k.endswith("self_attn.qkv_proj.weight"):
                out[k] = v
                continue
            prefix = k[: -len("qkv_proj.weight")]
            layer = int(k.split(".")[2])
            if args.hybrid_layer_pattern[layer]:
                n_heads = args.swa_num_attention_heads
                n_kv_heads = args.swa_num_key_value_heads
                head_dim = args.swa_head_dim
                v_head_dim = args.swa_v_head_dim
            else:
                n_heads = args.num_attention_heads
                n_kv_heads = args.num_key_value_heads
                head_dim = args.head_dim
                v_head_dim = args.v_head_dim
            q_dim = n_heads * head_dim
            k_dim = n_kv_heads * head_dim
            v_dim = n_kv_heads * v_head_dim
            if v.shape[0] != q_dim + k_dim + v_dim:
                raise ValueError(
                    f"{k}: expected fused dim {q_dim + k_dim + v_dim}, got {v.shape[0]}"
                )
            out[f"{prefix}q_proj.weight"] = v[:q_dim]
            out[f"{prefix}k_proj.weight"] = v[q_dim : q_dim + k_dim]
            out[f"{prefix}v_proj.weight"] = v[q_dim + k_dim :]
        return out

    def _dequant_mxfp4_experts(self, weights):
        """Dequantize the routed experts, which ship as MXFP4.

        The checkpoint packs two 4-bit values per ``uint8`` and stores one
        ``uint8`` E8M0 scale per 32 elements, which is byte-for-byte what MLX's
        ``mxfp4`` mode expects once the payload is viewed as ``uint32``.
        """
        out = {}
        for k, v in weights.items():
            if k.endswith(".weight_scale"):
                continue
            scale = weights.get(f"{k}_scale")
            if scale is None:
                out[k] = v
                continue
            out[k] = mx.dequantize(
                v.view(mx.uint32),
                scale,
                group_size=32,
                bits=4,
                mode="mxfp4",
            ).astype(mx.bfloat16)
        return out

    def sanitize(self, weights):
        if any(k.endswith("weight_scale") for k in weights):
            weights = self._dequant_mxfp4_experts(weights)
        if any(k.endswith("qkv_proj.weight_scale_inv") for k in weights):
            weights = self._dequant_fused_qkv(weights)
        weights = super().sanitize(weights)
        return self._split_fused_qkv(weights)

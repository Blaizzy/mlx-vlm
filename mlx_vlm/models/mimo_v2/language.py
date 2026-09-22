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

    def _qkv_heads(self, layer):
        args = self.args
        if args.hybrid_layer_pattern[layer]:
            return (
                args.swa_num_attention_heads,
                args.swa_num_key_value_heads,
                args.swa_head_dim,
                args.swa_v_head_dim,
            )
        return (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
            args.v_head_dim,
        )

    @staticmethod
    def _candidate_degrees(rows, heads, kv_heads, head_dim, v_head_dim, grid_rows):
        """Tensor-parallel degrees consistent with one fused tensor's shapes."""
        found, degree = [], 1
        while degree <= kv_heads:
            if heads % degree == 0 and kv_heads % degree == 0:
                shard = (heads // degree) * head_dim + (kv_heads // degree) * (
                    head_dim + v_head_dim
                )
                if (
                    shard * degree == rows
                    and -(-shard // _FP8_BLOCK) * degree == grid_rows
                ):
                    found.append(degree)
            degree *= 2
        return found

    def _tensor_parallel_degree(self, weights):
        """Recover the degree the checkpoint was sharded with.

        A layer whose fused rows happen to be a whole number of 128-row blocks
        admits several degrees, so the degree is intersected across every fused
        tensor. Full attention layers, where the per-shard padding shows up in
        the grid, are what pin it down.
        """
        common = None
        for key, value in weights.items():
            if not key.endswith("self_attn.qkv_proj.weight") or ".mtp." in key:
                continue
            scale = weights.get(f"{key}_scale_inv")
            if scale is None:
                continue
            heads, kv_heads, head_dim, v_head_dim = self._qkv_heads(_layer_index(key))
            found = set(
                self._candidate_degrees(
                    value.shape[0],
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    scale.shape[0],
                )
            )
            common = found if common is None else common & found
        if not common:
            raise ValueError("no tensor-parallel degree explains the fused qkv shapes")
        return max(common)

    def _unfuse_qkv(self, weights):
        """Split ``qkv_proj`` back into q/k/v.

        The fused weight is not one tensor followed by a single scale grid: it
        is the tensor-parallel shards concatenated, each holding its own q
        heads then k then v, and each carrying its own FP8 scale grid padded up
        to a whole number of 128-row blocks. That is why the grid has 108 rows
        for a 106-block full-attention weight (4 x 27) and 116 for a 116-block
        SWA one (4 x 29) -- read as a single grid neither divides, and walking
        it contiguously misaligns every shard after the first.
        """
        out = {}
        sharded = any(
            k.endswith("self_attn.qkv_proj.weight_scale_inv") and ".mtp." not in k
            for k in weights
        )
        degree = self._tensor_parallel_degree(weights) if sharded else 1
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

            scale = weights.get(f"{key}_scale_inv")
            prefix = key[: -len("qkv_proj.weight")]
            heads, kv_heads, head_dim, v_head_dim = self._qkv_heads(_layer_index(key))
            q_rows = heads // degree * head_dim
            k_rows = kv_heads // degree * head_dim
            v_rows = kv_heads // degree * v_head_dim
            shard = q_rows + k_rows + v_rows
            blocks = -(-shard // _FP8_BLOCK)

            parts = {"q_proj": [], "k_proj": [], "v_proj": []}
            for rank in range(degree):
                rows = value[rank * shard : (rank + 1) * shard]
                dense = (
                    _dequant_block_fp8(rows, scale[rank * blocks : (rank + 1) * blocks])
                    if scale is not None
                    else rows
                )
                row = 0
                for name, size in (
                    ("q_proj", q_rows),
                    ("k_proj", k_rows),
                    ("v_proj", v_rows),
                ):
                    parts[name].append(dense[row : row + size])
                    row += size
            for name, chunks in parts.items():
                out[f"{prefix}{name}.weight"] = mx.concatenate(chunks)
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
        if any(k.endswith("self_attn.qkv_proj.weight") for k in weights):
            weights = self._unfuse_qkv(weights)
        return super().sanitize(weights)

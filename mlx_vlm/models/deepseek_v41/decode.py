"""Fixed-shape decode graphs with explicit position and cache inputs."""

import mlx.core as mx

from .fakequant import fake_quant_fp8_ue8m0
from .hc_norm import fused_hc_pre_norm


class CompiledDecode:
    """Keep compilation local to one block; never capture generation state.

    Module parameters are implicit graph inputs, so weight updates remain visible.
    Positions are array inputs to avoid retracing at every generated token.
    Prefill and partially filled windows retain the ordinary attention path.
    """

    def __init__(self, block):
        from .language import hc_expand

        self.block = block
        attn = block.attn
        self.attn = attn
        self.expand = hc_expand
        # RoPE lazily populates frequency caches; do that outside compilation.
        mx.eval(
            attn.rope._get_freqs(attn.head_dim, False),
            attn.rope._get_freqs(attn.head_dim, True),
        )

        def collapse_norm(h, mix, norm):
            out = fused_hc_pre_norm(h, mix, norm.weight, norm.eps)
            return norm(block.hc_pre(h, mix)) if out is None else out

        def pre(h, pre_mix):
            ap, ao, ac = block.hc_mixes(
                h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
            )
            return collapse_norm(h, pre_mix, block.attn_norm), ap, ao, ac

        def mid(x, residual, ap, ao, ac):
            h = hc_expand(x, residual, ao, ac)
            fp, fo, fc = block.hc_mixes(
                h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
            )
            return collapse_norm(h, ap, block.ffn_norm), h, fp, fo, fc

        def project(x, offset):
            batch, width = x.shape[:2]
            qr = attn.q_norm(attn.wq_a(fake_quant_fp8_ue8m0(x)))
            q = attn.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(
                batch, width, attn.n_heads, attn.head_dim
            )
            q = attn.rope(q.transpose(0, 2, 1, 3), offset)
            kv = attn.kv_norm(attn.wkv(fake_quant_fp8_ue8m0(x)))
            kv = attn.rope(kv[:, None], offset).reshape(batch, width, attn.head_dim)
            return qr, q, fake_quant_fp8_ue8m0(kv)

        def project_out(out, offset):
            batch, _, width, _ = out.shape
            out = attn.rope(out, offset, inverse=True)
            out = out.reshape(batch, attn.o_groups, -1, width, attn.head_dim)
            out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
            out = attn.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
            return attn.wo_b(fake_quant_fp8_ue8m0(out))

        def window(buffer, kv, offset):
            batch = kv.shape[0]
            win = attn.window_size
            history = mx.take(
                buffer[:batch], (mx.arange(win - 1) + offset - win + 1) % win, axis=1
            )
            part = mx.concatenate([history, kv], axis=1)
            buffer[:batch, offset.reshape(1) % win] = kv
            return buffer, part

        self.pre = mx.compile(pre, inputs=block)
        self.mid = mx.compile(mid, inputs=block)
        self.moe = mx.compile(block.ffn.__call__, inputs=block.ffn)
        self.project = mx.compile(project, inputs=attn)
        self.project_out = mx.compile(project_out, inputs=attn)
        self.window = mx.compile(window)
        self.window_mask = mx.ones((1, 1, 1, attn.window_size), mx.bool_)

    def attention(self, cache, x):
        attn = self.attn
        buffer = cache.window[attn.layer_idx]
        if (
            x.shape[1] != 1
            or cache.offset < attn.window_size
            or buffer is None
            or buffer.shape[0] < x.shape[0]
        ):
            return attn._attention(cache, x)
        offset = mx.array(cache.offset, mx.int32)
        qr, q, kv = self.project(x, offset)
        buffer, window = self.window(buffer, kv, offset)
        cache.window[attn.layer_idx] = buffer
        out = None
        if attn.compress_ratio:
            pool, indices = attn._compress_part(x, qr, cache.offset, cache)
            if pool.shape[1]:
                out = attn._sparse_attention(q, window, pool, indices, self.window_mask)
        if out is None:
            kv = window[:, None].astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(
                q,
                kv,
                kv,
                scale=attn.scale,
                mask=self.window_mask,
                sinks=attn.attn_sink.astype(q.dtype),
            )
        return self.project_out(out, offset)

    def __call__(self, h, pre_mix, image_mask, cache):
        x, ap, ao, ac = self.pre(h, pre_mix)
        out = cache.map(self.attention, x)
        out = mx.zeros_like(x) if out is None else out
        x, residual, fp, fo, fc = self.mid(out, h, ap, ao, ac)
        x = self.moe(x, image_mask)
        return self.expand(x, residual, fo, fc), fp

"""Omni encoder (vision + audio paths) of Youtu-Parsing-Omni.

Vision: linear patch embedding -> 28 Qwen3-style bidirectional blocks over packed
segments (3D T/H/W chunked RoPE) -> 2x2 patch merger / projector.
Audio:  3x Conv2d(stride 2) front-end over 100-frame mel chunks -> linear proj ->
the same 28-block encoder (one attention window per audio clip, t = 0..n-1 RoPE)
-> temporal-merge RMSNorm/MLP projector.

Ported from tencent/Youtu-Parsing-Omni (https://huggingface.co/tencent/Youtu-Parsing-Omni).
Copyright (C) 2026 Tencent. The original code is distributed under the Youtu-Parsing
license (https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/main/LICENSE);
modified / re-implemented here for MLX.
"""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import OmniConfig


def split_4d_dims(dim_rot: int, m_dim: int) -> Tuple[int, int, int, int]:
    rest = dim_rot - m_dim
    per = rest // 3
    rem = rest - per * 3
    return m_dim, per, per, per + rem


def build_vision_hw_pos(h: int, w: int, sm: int) -> np.ndarray:
    """(h, w) index of every patch, in the spatial-merge order of the processor."""
    hpos = np.broadcast_to(np.arange(h)[:, None], (h, w))
    hpos = hpos.reshape(h // sm, sm, w // sm, sm).transpose(0, 2, 1, 3).reshape(-1)
    wpos = np.broadcast_to(np.arange(w)[None, :], (h, w))
    wpos = wpos.reshape(h // sm, sm, w // sm, sm).transpose(0, 2, 1, 3).reshape(-1)
    return np.stack([hpos, wpos], axis=-1)


def vision_pos_ids(grid_thw, sm: int, m_id: int = 0) -> np.ndarray:
    """[S, 4] = (m, t, h, w) for every vision token."""
    out = []
    for t, h, w in np.asarray(grid_thw).tolist():
        hw = build_vision_hw_pos(int(h), int(w), sm)
        n = hw.shape[0]
        for ti in range(int(t)):
            out.append(
                np.stack(
                    [np.full(n, m_id), np.full(n, ti), hw[:, 0], hw[:, 1]], axis=-1
                )
            )
    return np.concatenate(out, axis=0).astype(np.int64)


def cu_seqlens_from_grid(grid_thw) -> List[int]:
    lens = []
    for t, h, w in np.asarray(grid_thw).tolist():
        lens += [int(h) * int(w)] * int(t)
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return cu


class ChunkedMTHWRotary:
    """[M | T | H | W] chunked angles; returns raw angles [S, dim_rot] (float32)."""

    def __init__(self, dim_rot, m_dim, theta_m, theta):
        self.m_dim, self.t_dim, self.h_dim, self.w_dim = split_4d_dims(dim_rot, m_dim)
        self.theta_m = theta_m
        self.theta = theta
        self.dim_rot = dim_rot

    @staticmethod
    def _inv(d, base):
        return 1.0 / (base ** (mx.arange(0, d * 2, 2, dtype=mx.float32) / (d * 2)))

    def __call__(self, pos_ids: mx.array) -> mx.array:
        pos = pos_ids.astype(mx.float32)
        segs = []
        if self.m_dim > 0:
            segs.append(mx.outer(pos[:, 0], self._inv(self.m_dim, self.theta_m)))
        segs.append(mx.outer(pos[:, 1], self._inv(self.t_dim, self.theta)))
        segs.append(mx.outer(pos[:, 2], self._inv(self.h_dim, self.theta)))
        segs.append(mx.outer(pos[:, 3], self._inv(self.w_dim, self.theta)))
        return mx.concatenate(segs, axis=-1)


def apply_rotary_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Non-interleaved rotate-half. x [S, H, D]; cos/sin [S, D/2] float32."""
    dt = x.dtype
    x = x.astype(mx.float32)
    d2 = cos.shape[-1]
    c = cos[:, None, :]
    s = sin[:, None, :]
    x1, x2 = x[..., :d2], x[..., d2:]
    out = mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    return out.astype(dt)


class OmniAttention(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, cu_seqlens, cos, sin):
        S = x.shape[0]
        q = self.q_proj(x).reshape(S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(S, self.num_kv_heads, self.head_dim)
        q = apply_rotary_half(self.q_norm(q), cos, sin)
        k = apply_rotary_half(self.k_norm(k), cos, sin)
        outs = []
        # one attention window per segment (image / frame); no dense mask.
        # mx.fast SDPA maps q head i -> kv head i // (Hq / Hkv), as the reference does.
        for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            if b == a:
                continue
            qs = q[a:b].transpose(1, 0, 2)[None]
            ks = k[a:b].transpose(1, 0, 2)[None]
            vs = v[a:b].transpose(1, 0, 2)[None]
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=self.scale)
            outs.append(o[0].transpose(1, 0, 2))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return self.o_proj(o.reshape(S, -1))


class OmniMLP(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class OmniEncoderLayer(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = OmniAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = OmniMLP(config)

    def __call__(self, x, cu_seqlens, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class OmniEncoder(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.layers = [
            OmniEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.rotary_4d = ChunkedMTHWRotary(
            config.head_dim // 2,
            config.rope_m_dim,
            config.rope_theta_m,
            config.rope_theta,
        )

    def __call__(self, x, cu_seqlens, angles, capture=None):
        """cu_seqlens: list[int] shared by all layers, or a per-layer list of such lists.
        angles: [S, 64] raw angles shared by all layers, or a per-layer list of arrays
        (the same array object may be repeated; cos/sin are computed once per object).
        """
        per_layer_cu = len(cu_seqlens) > 0 and isinstance(cu_seqlens[0], (list, tuple))
        per_layer_angles = isinstance(angles, (list, tuple))
        trig = {}

        def cos_sin(a):
            if (
                id(a) not in trig
            ):  # cat(emb, emb) -> cos/sin chunked back to the first half
                trig[id(a)] = (mx.cos(a), mx.sin(a))
            return trig[id(a)]

        for i, layer in enumerate(self.layers):
            cos, sin = cos_sin(angles[i] if per_layer_angles else angles)
            x = layer(x, cu_seqlens[i] if per_layer_cu else cu_seqlens, cos, sin)
            if capture is not None:
                capture[f"layer{i}"] = x
        return x


# ----------------------------------------------------------------------------- video
M_VIDEO_FRAME = 2
M_VIDEO_AUDIO = 3
# audio tokens per second of video (100 Hz mel / 8x conv down-sampling): the per-frame t step
# on the fusion-layer time axis, so that a frame and the audio of the same instant share a t.
VIDEO_FRAME_T_STEP = 100.0 / 8


def fusion_layer_pattern(freq, num_layers: int) -> List[bool]:
    """HF ``_normalize_video_fusion_layer_pattern``."""
    if freq is None:
        return [True] * num_layers
    if isinstance(freq, bool):
        raise ValueError(
            f"video_fusion_layer_freq must be int / list / None, got bool {freq!r}"
        )
    if isinstance(freq, int):
        n = max(int(freq), 1)
        return [(i % n == 0) for i in range(num_layers)]
    if len(freq) != num_layers:
        raise ValueError(
            f"video_fusion_layer_freq length {len(freq)} != num_hidden_layers {num_layers}"
        )
    return [bool(v) for v in freq]


def order_video_events(n_frames, n_chunks, frame_pos, chunk_pos):
    """Temporal order of one video's frames ('I') and audio chunks ('A') as [(kind, local_idx)].

    ``frame_pos`` / ``chunk_pos``: sequence position of the first pad token of every frame / chunk
    (the order the processor wrote them into ``input_ids``). Ties cannot occur."""
    if n_frames and n_chunks:
        if frame_pos is not None and chunk_pos is not None:
            if any(chunk_pos[i] > chunk_pos[i + 1] for i in range(len(chunk_pos) - 1)):
                raise ValueError("video_audio_indices are not in temporal order")
            ev = [(p, 0, k) for k, p in enumerate(frame_pos)] + [
                (p, 1, k) for k, p in enumerate(chunk_pos)
            ]
            return [(m, k) for _, m, k in sorted(ev)]
        # HF fallback without indices: spread audio chunks evenly over the frames
        per, extra = divmod(n_chunks, n_frames)
        out, a = [], 0
        for f in range(n_frames):
            out.append((0, f))
            for _ in range(per + (1 if f < extra else 0)):
                out.append((1, a))
                a += 1
        return out
    return [(0, k) for k in range(n_frames)] + [(1, k) for k in range(n_chunks)]


def group_video_events(events, group_attention: bool):
    """Split a video's events into fusion attention groups (HF ``video_group_attention``): a new
    group starts at every audio -> image boundary; single-modality videos stay in one group.
    """
    kinds = {m for m, _ in events}
    if not (group_attention and kinds == {0, 1}):
        return [events]
    groups, cur, prev = [], [], None
    for m, k in events:
        if m == 0 and prev == 1:
            groups.append(cur)
            cur = []
        cur.append((m, k))
        prev = m
    if cur:
        groups.append(cur)
    return groups


class VideoLayout:
    """Packed-sequence layout of the joint video encoder pass (numpy only).

    ``order``: [(kind, global_idx, start, length)] in packed order, kind 0 = frame, 1 = audio chunk.
    ``cu_fusion`` / ``cu_nofusion``: attention windows of fusion (one per group) / non-fusion
    (one per frame or chunk) layers. ``pos_fusion`` / ``pos_nofusion``: [S, 4] (m, t, h, w).
    """

    def __init__(
        self,
        grid_thw,
        video_split,
        audio_lens,
        frame_seq_pos,
        chunk_seq_pos,
        sm,
        group_attention,
    ):
        grid = np.asarray(grid_thw).reshape(-1, 3)
        split = np.asarray(video_split).reshape(-1, 2)
        self.order, self.cu_fusion, self.cu_nofusion = [], [0], [0]
        pf, pn = [], []
        hw_cache = {}
        f0 = c0 = 0  # global frame / chunk cursors
        tok0 = 0  # cursor into frame_seq_pos (text-side pad tokens)
        start = 0
        for n_img, n_aud in split.tolist():
            tpf = [
                int(t * h // sm) * int(w // sm)
                for t, h, w in grid[f0 : f0 + n_img].tolist()
            ]
            fpos = None
            if frame_seq_pos is not None:
                fpos, cur = [], tok0
                for k in range(n_img):
                    fpos.append(frame_seq_pos[cur])
                    cur += tpf[k]
            tok0 += sum(tpf)
            cpos = (
                None if chunk_seq_pos is None else list(chunk_seq_pos[c0 : c0 + n_aud])
            )
            events = order_video_events(n_img, n_aud, fpos, cpos)
            for group in group_video_events(events, group_attention):
                if not group:
                    continue
                t_v = t_a = 0.0  # independent per-group time cursors
                g_start = start
                for kind, k in group:
                    if kind == 0:
                        t, h, w = (int(v) for v in grid[f0 + k])
                        if (h, w) not in hw_cache:
                            hw_cache[(h, w)] = build_vision_hw_pos(h, w, sm).astype(
                                np.float32
                            )
                        hw = hw_cache[(h, w)]
                        n = hw.shape[0]
                        rows_f, rows_n = [], []
                        for ti in range(t):
                            m_col = np.full(n, float(M_VIDEO_FRAME), np.float32)
                            rows_f.append(
                                np.stack(
                                    [
                                        m_col,
                                        np.full(
                                            n, np.float32(t_v + ti * VIDEO_FRAME_T_STEP)
                                        ),
                                        hw[:, 0],
                                        hw[:, 1],
                                    ],
                                    -1,
                                )
                            )
                            rows_n.append(
                                np.stack(
                                    [
                                        m_col,
                                        np.full(n, np.float32(ti * VIDEO_FRAME_T_STEP)),
                                        hw[:, 0],
                                        hw[:, 1],
                                    ],
                                    -1,
                                )
                            )
                        pf += rows_f
                        pn += rows_n
                        t_v += t * VIDEO_FRAME_T_STEP
                        length, gidx = t * n, f0 + k
                    else:
                        length, gidx = int(audio_lens[c0 + k]), c0 + k
                        z = np.zeros(length, np.float32)
                        m_col = np.full(length, float(M_VIDEO_AUDIO), np.float32)
                        ar = np.arange(length, dtype=np.float32)
                        pf.append(np.stack([m_col, ar + np.float32(t_a), z, z], -1))
                        pn.append(np.stack([m_col, ar, z, z], -1))
                        t_a += length
                    self.order.append((kind, gidx, start, length))
                    start += length
                    self.cu_nofusion.append(start)
                self.cu_fusion.append(start)
            f0 += n_img
            c0 += n_aud
        self.total = start
        self.pos_fusion = np.concatenate(pf, 0) if pf else np.zeros((0, 4), np.float32)
        self.pos_nofusion = (
            np.concatenate(pn, 0) if pn else np.zeros((0, 4), np.float32)
        )


class VisionEmbeddings(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size**2, config.hidden_size, bias=True
        )

    def __call__(self, pixel_values):
        return self.patch_embedding(
            pixel_values.astype(self.patch_embedding.weight.dtype)
        )


class VisionPatchMerger(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.norm = nn.RMSNorm(self.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(
            self.hidden_size, config.merger_hidden_size, bias=False
        )
        self.linear_fc2 = nn.Linear(
            config.merger_hidden_size, config.out_hidden_size, bias=False
        )

    def __call__(self, x):
        x = x.reshape(-1, self.hidden_size)
        # HF: nn.RMSNorm(eps=None) -> torch.finfo(dtype).eps
        eps = float(mx.finfo(x.dtype).eps)
        x = mx.fast.rms_norm(x, self.norm.weight, eps)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))  # exact GELU


def audio_output_length(n: int) -> int:
    """HF ``_get_feat_extract_output_lengths`` (mel frames -> audio tokens)."""
    leave = n % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13


class AudioEmbeddings(nn.Module):
    """Conv2d front-end. Input per chunk is [mel(H), time(W), 1] (NHWC)."""

    def __init__(self, config: OmniConfig):
        super().__init__()
        c = config.downsample_hidden_size
        self.chunk_frames = config.n_window * 2
        self.conv_chunksize = config.conv_chunksize
        self.conv2d1 = nn.Conv2d(1, c, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(c, c, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(c, c, 3, 2, padding=1)
        mel = config.num_mel_bins
        for _ in range(3):
            mel = (mel + 1) // 2
        self.linear_proj = nn.Linear(c * mel, config.hidden_size, bias=False)

    def __call__(self, audios):
        """audios: list of [T_i, mel] arrays -> (hidden [sum(n_i), H], [n_i])."""
        W = self.chunk_frames
        dt = self.conv2d1.weight.dtype
        chunks, valid = (
            [],
            [],
        )  # every chunk zero-padded to W frames (as HF pad_sequence)
        out_lens = []
        for a in audios:
            a = mx.array(a) if not isinstance(a, mx.array) else a
            T = a.shape[0]
            out_lens.append(audio_output_length(T))
            for s in range(0, T, W):
                c = a[s : s + W]
                valid.append(audio_output_length(c.shape[0]))
                if c.shape[0] < W:
                    c = mx.pad(c, [(0, W - c.shape[0]), (0, 0)])
                chunks.append(c)
        x = mx.stack(chunks, axis=0).astype(dt)  # [N, W, mel]
        x = x.transpose(0, 2, 1)[..., None]  # [N, mel(H), W(time), 1]
        outs = []
        for i in range(0, x.shape[0], self.conv_chunksize):
            h = nn.gelu(self.conv2d1(x[i : i + self.conv_chunksize]))
            h = nn.gelu(self.conv2d2(h))
            h = nn.gelu(self.conv2d3(h))  # [n, f, t, c]
            n, f, t, c = h.shape
            outs.append(h.transpose(0, 2, 3, 1).reshape(n, t, c * f))
        h = mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
        h = mx.concatenate([h[i, :v] for i, v in enumerate(valid)], axis=0)
        return self.linear_proj(h), out_lens


class AudioPatchMerger(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.temporal_merge_size = config.temporal_merge_size
        self.hidden_size = config.hidden_size * config.temporal_merge_size
        self.norm = nn.RMSNorm(self.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(
            self.hidden_size, config.merger_hidden_size, bias=False
        )
        self.linear_fc2 = nn.Linear(
            config.merger_hidden_size, config.out_hidden_size, bias=False
        )

    def __call__(self, x):
        """x [n, H] for ONE clip -> [ceil(n/M), out]."""
        M = self.temporal_merge_size
        if M > 1:  # HF pad_and_reshape
            n = x.shape[0]
            pad = (M - n % M) % M
            if pad:
                x = mx.pad(x, [(0, pad), (0, 0)])
            x = x.reshape(-1, M * x.shape[-1])
        eps = float(mx.finfo(x.dtype).eps)  # HF: nn.RMSNorm(eps=None)
        x = mx.fast.rms_norm(x, self.norm.weight, eps)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class OmniModel(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        self.vision_embeddings = VisionEmbeddings(config)
        self.encoder = OmniEncoder(config)
        self.vision_merger = VisionPatchMerger(config)
        self.audio_embeddings = AudioEmbeddings(config)
        self.audio_merger = AudioPatchMerger(config)

    def forward_vision(self, pixel_values, grid_thw, capture=None):
        grid = np.asarray(grid_thw)
        h = self.vision_embeddings(pixel_values)
        if capture is not None:
            capture["vision_embeddings"] = h
        pos_ids = vision_pos_ids(grid, self.config.spatial_merge_size)
        angles = self.encoder.rotary_4d(mx.array(pos_ids))
        if capture is not None:
            capture["pos_ids"] = mx.array(pos_ids)
            capture["rotary"] = angles
        cu = cu_seqlens_from_grid(grid)
        h = self.encoder(h, cu, angles, capture=capture)
        if capture is not None:
            capture["final"] = h
        out = self.vision_merger(h)
        if capture is not None:
            capture["merger"] = out
        return out

    def forward_audio(self, audios, capture=None):
        """audios: list of [T_i, num_mel_bins] log-mel arrays (one attention window each).

        Returns a list of [ceil(n_i / temporal_merge_size), out_hidden] features.
        """
        h, lens = self.audio_embeddings(audios)
        if capture is not None:
            capture["audio_embeddings"] = h
            capture["lens"] = lens
        pos = []
        for n in lens:  # (m, t, h, w) = (M_AUDIO, 0..n-1, 0, 0); t restarts per clip
            t = np.arange(n, dtype=np.int64)
            z = np.zeros_like(t)
            pos.append(np.stack([np.full_like(t, 1), t, z, z], axis=-1))
        pos_ids = np.concatenate(pos, axis=0)
        angles = self.encoder.rotary_4d(mx.array(pos_ids))
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        if capture is not None:
            capture["pos_ids"] = mx.array(pos_ids)
            capture["rotary"] = angles
            capture["cu_seqlens"] = cu
        h = self.encoder(h, cu, angles, capture=capture)
        if capture is not None:
            capture["final"] = h
        outs = []
        for a, b in zip(cu[:-1], cu[1:]):
            outs.append(self.audio_merger(h[a:b]))
        if capture is not None:
            capture["merger"] = outs
        return outs

    def forward_video(
        self,
        video_images,
        video_grid_thw,
        video_audios,
        video_split,
        frame_seq_pos=None,
        chunk_seq_pos=None,
        capture=None,
    ):
        """Joint video pass: frames and audio chunks of one video share the shared encoder.

        Mirrors HF ``YoutuVITAOmniModel.forward_video``: fusion layers (``video_fusion_layer_freq``)
        attend within a group of frames + audio chunks (``video_group_attention``), the other layers
        within each single frame / chunk; the two kinds see different time coordinates.

        Returns ``(frame_features [N_tokens, out], [audio features [n_i, out] per chunk])``.
        """
        cfg = self.config
        sm = cfg.spatial_merge_size
        grid = np.asarray(video_grid_thw).reshape(-1, 3)
        v = self.vision_embeddings(video_images)
        has_audio = video_audios is not None and len(video_audios) > 0
        if has_audio:
            a, a_lens = self.audio_embeddings(list(video_audios))
        else:
            a, a_lens = None, []
        lay = VideoLayout(
            grid,
            video_split,
            a_lens,
            frame_seq_pos,
            chunk_seq_pos,
            sm,
            cfg.video_group_attention,
        )
        pattern = fusion_layer_pattern(
            cfg.video_fusion_layer_freq, cfg.num_hidden_layers
        )

        patch_off = np.concatenate(
            [[0], np.cumsum(grid[:, 0] * grid[:, 1] * grid[:, 2])]
        ).tolist()
        aud_off = np.concatenate([[0], np.cumsum(a_lens)]).astype(int).tolist()
        pieces = []
        for kind, g, _, n in lay.order:
            pieces.append(
                v[patch_off[g] : patch_off[g] + n]
                if kind == 0
                else a[aud_off[g] : aud_off[g] + n]
            )
        h = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0)

        angles_f = angles_n = None
        if any(pattern):
            angles_f = self.encoder.rotary_4d(mx.array(lay.pos_fusion))
        if not all(pattern):
            angles_n = self.encoder.rotary_4d(mx.array(lay.pos_nofusion))
        angles = [angles_f if f else angles_n for f in pattern]
        cu = [lay.cu_fusion if f else lay.cu_nofusion for f in pattern]
        if capture is not None:
            capture.update(
                vision_embeddings=v,
                audio_embeddings=a,
                layout=lay,
                cu_seqlens=cu,
                rotary_fusion=angles_f,
                rotary_nofusion=angles_n,
                packed_in=h,
            )
        h = self.encoder(h, cu, angles, capture=capture)
        if capture is not None:
            capture["final"] = h

        frames, chunks = {}, {}
        for kind, g, s0, n in lay.order:
            (frames if kind == 0 else chunks)[g] = h[s0 : s0 + n]
        vis = [frames[g] for g in sorted(frames)]
        vis = mx.concatenate(vis, axis=0) if len(vis) > 1 else vis[0]
        vis = self.vision_merger(vis)
        auds = [self.audio_merger(chunks[g]) for g in sorted(chunks)]
        if capture is not None:
            capture["merger"] = (vis, auds)
        return vis, auds

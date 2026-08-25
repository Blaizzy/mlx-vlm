import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

MASK_LOGIT = -1e4


def _gather_rows(states, indices):
    return mx.stack([states[i, indices[i]] for i in range(states.shape[0])])


class ResidualSwiGLU(nn.Module):
    def __init__(self, dim, multiplier=2.0):
        super().__init__()
        hidden = max(1, int(dim * multiplier))
        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, hidden * 2)
        self.output_projection = nn.Linear(hidden, dim)

    def __call__(self, states):
        value, gate = mx.split(self.input_projection(self.norm(states)), 2, axis=-1)
        return states + self.output_projection(value * nn.silu(gate))


class BoundaryAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, window=0):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.qkv_projection = nn.Linear(dim, dim * 3)
        self.output_projection = nn.Linear(dim, dim)

    def __call__(self, states, mask):
        batch, length, dim = states.shape
        qkv = self.qkv_projection(self.norm(states)).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = [
            x.squeeze(2).transpose(0, 2, 1, 3) for x in mx.split(qkv, 3, axis=2)
        ]
        allowed = mx.broadcast_to(mask[:, None, None, :], (batch, 1, length, length))
        if self.window > 0:
            positions = mx.arange(length)
            local = mx.abs(positions[:, None] - positions[None, :]) <= self.window
            allowed = allowed & local[None, None]
        allowed = allowed | mx.eye(length, dtype=mx.bool_)[None, None]
        attention_mask = mx.where(allowed, 0.0, MASK_LOGIT).astype(states.dtype)
        attended = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1 / math.sqrt(self.head_dim), mask=attention_mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return (states + self.output_projection(attended)) * mask[..., None]


@dataclass
class BoundaryEncoding:
    states: mx.array
    mask: mx.array


class BoundaryEncoder(nn.Module):
    def __init__(self, hidden_size, settings):
        super().__init__()
        dim = settings.get("boundary_dim", 128)
        self.left_projection = nn.Linear(hidden_size, dim)
        self.right_projection = nn.Linear(hidden_size, dim)
        self.output_projection = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.attention_blocks = [
            BoundaryAttentionBlock(
                dim,
                settings.get("boundary_attention_heads", 4),
                settings.get("boundary_attention_window", 0),
            )
            for _ in range(settings.get("boundary_attention_layers", 0))
        ]
        self.refinement_blocks = [
            ResidualSwiGLU(dim, settings.get("boundary_ffn_multiplier", 2.0))
            for _ in range(settings.get("boundary_refinement_layers", 1))
        ]
        self.bos_state = mx.zeros((hidden_size,))
        self.eos_state = mx.zeros((hidden_size,))

    def __call__(self, text_states, text_mask):
        batch, length, hidden = text_states.shape
        lengths = mx.sum(text_mask, axis=1).astype(mx.int32)
        bos = mx.broadcast_to(self.bos_state, (batch, 1, hidden))
        eos = mx.broadcast_to(self.eos_state, (batch, 1, hidden))
        left = mx.concatenate((bos, text_states), axis=1)
        right = mx.concatenate((text_states, eos), axis=1)
        positions = mx.arange(length + 1)[None, :, None]
        right = mx.where(positions == lengths[:, None, None], eos, right)
        states = self.output_projection(
            mx.concatenate(
                (self.left_projection(left), self.right_projection(right)), axis=-1
            )
        )
        states = self.layer_norm(states)
        mask = mx.arange(length + 1)[None] <= lengths[:, None]
        for block in self.attention_blocks:
            states = block(states, mask)
        for block in self.refinement_blocks:
            states = block(states)
        return BoundaryEncoding(states * mask[..., None], mask)


@dataclass
class Marginals:
    start_logits: mx.array
    end_logits: mx.array
    inside_prefix: mx.array
    inside_mean: mx.array


class BoundaryQueryHead(nn.Module):
    def __init__(self, hidden_size, boundary_dim):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.start_boundary_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(hidden_size, boundary_dim)
        self.end_boundary_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_query_projection = nn.Linear(hidden_size, boundary_dim)
        self.inside_text_projection = nn.Linear(hidden_size, boundary_dim)
        self.inside_query_projection = nn.Linear(hidden_size, boundary_dim)

    def __call__(self, boundaries, text_states, text_mask, queries, query_mask):
        scale = 1 / math.sqrt(self.boundary_dim)
        start = (
            self.start_query_projection(queries)
            @ self.start_boundary_projection(boundaries.states).swapaxes(-1, -2)
        ) * scale
        end = (
            self.end_query_projection(queries)
            @ self.end_boundary_projection(boundaries.states).swapaxes(-1, -2)
        ) * scale
        inside = (
            self.inside_query_projection(queries)
            @ self.inside_text_projection(text_states).swapaxes(-1, -2)
        ) * scale
        boundary_keep = boundaries.mask[:, None] & query_mask[..., None]
        text_keep = text_mask[:, None] & query_mask[..., None]
        start = mx.where(boundary_keep, start, MASK_LOGIT)
        end = mx.where(boundary_keep, end, MASK_LOGIT)
        inside_values = mx.where(text_keep, inside, 0.0).astype(mx.float32)
        count = mx.maximum(mx.sum(text_keep, axis=-1, keepdims=True), 1)
        mean = mx.stop_gradient(mx.sum(inside_values, axis=-1, keepdims=True) / count)
        centered = (inside_values - mean) * text_keep
        prefix = mx.concatenate(
            (mx.zeros((*centered.shape[:-1], 1)), mx.cumsum(centered, axis=-1)),
            axis=-1,
        )
        return Marginals(start, end, prefix, mean)


@dataclass
class PooledCandidates:
    indices: mx.array
    mask: mx.array
    compat_logits: mx.array


class DocumentCandidatePool(nn.Module):
    def __init__(self, boundary_dim, settings):
        super().__init__()
        self.pool_boundary_top_k = settings.get("pool_boundary_top_k", 32)
        self.pool_size = settings.get(
            "pool_size", settings.get("candidate_budget", 192)
        )
        self.min_pool_per_query = settings.get("min_pool_per_query", 8)
        self.start_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_projection = nn.Linear(boundary_dim, boundary_dim)

    def __call__(
        self, boundary_states, boundary_mask, query_mask, start_logits, end_logits
    ):
        batch, boundaries, dim = boundary_states.shape
        projected_start = self.start_projection(boundary_states)
        projected_end = self.end_projection(boundary_states)
        outputs = []
        output_masks = []
        output_compat = []
        for row in range(batch):
            valid_queries = query_mask[row]
            union_start = mx.max(
                mx.where(valid_queries[:, None], start_logits[row], MASK_LOGIT), axis=0
            )
            union_end = mx.max(
                mx.where(valid_queries[:, None], end_logits[row], MASK_LOGIT), axis=0
            )
            union_start = mx.where(boundary_mask[row], union_start, MASK_LOGIT)
            union_end = mx.where(boundary_mask[row], union_end, MASK_LOGIT)
            top_k = min(self.pool_boundary_top_k, boundaries)
            starts = mx.argsort(-union_start)[:top_k]
            ends = mx.argsort(-union_end)[:top_k]
            pair_start = mx.repeat(starts, top_k)
            pair_end = mx.tile(ends, top_k)
            valid = pair_end > pair_start
            compatibility = mx.sum(
                projected_start[row, pair_start] * projected_end[row, pair_end], axis=-1
            ) / math.sqrt(dim)
            global_score = compatibility + union_start[pair_start] + union_end[pair_end]
            # Read the pair grid across once. Ranking used to evaluate a fresh
            # three-term graph inside the sort key, so a single request issued
            # thousands of device synchronizations and the pool -- not the
            # encoder -- dominated end-to-end latency.
            mx.eval(pair_start, pair_end, valid, compatibility, global_score)
            starts_list = pair_start.tolist()
            ends_list = pair_end.tolist()
            valid_list = valid.tolist()
            global_list = global_score.tolist()
            pair_rows = [
                (starts_list[i], ends_list[i], i)
                for i in range(len(valid_list))
                if valid_list[i]
            ]
            priorities = {(start, end): global_list[i] for start, end, i in pair_rows}
            quota = min(self.min_pool_per_query, len(pair_rows))
            if pair_rows and quota:
                # Score every (query, pair) once instead of per comparison.
                # mx.take keeps the query axis first; plain advanced indexing
                # would move the gathered axis to the front.
                pair_index = mx.array([row_[2] for row_ in pair_rows], dtype=mx.int32)
                query_scores = (
                    mx.take(
                        start_logits[row],
                        mx.array([row_[0] for row_ in pair_rows], dtype=mx.int32),
                        axis=1,
                    )
                    + mx.take(
                        end_logits[row],
                        mx.array([row_[1] for row_ in pair_rows], dtype=mx.int32),
                        axis=1,
                    )
                    + compatibility[pair_index][None, :]
                )
                mx.eval(query_scores)
                query_scores = query_scores.tolist()
                order = range(len(pair_rows))
                for query_index, active in enumerate(query_mask[row].tolist()):
                    if not active:
                        continue
                    scores = query_scores[query_index]
                    ranked = sorted(order, key=scores.__getitem__, reverse=True)[:quota]
                    for rank, index in enumerate(ranked):
                        start, end, _ = pair_rows[index]
                        priorities[(start, end)] = max(
                            priorities[(start, end)], 5000.0 + quota - rank
                        )
            selected = sorted(priorities, key=priorities.get, reverse=True)[
                : self.pool_size
            ]
            count = len(selected)
            selected += [(0, 0)] * (self.pool_size - count)
            indices = mx.array(selected, dtype=mx.int32)
            pool_mask = mx.arange(self.pool_size) < count
            compat = mx.sum(
                projected_start[row, indices[:, 0]] * projected_end[row, indices[:, 1]],
                axis=-1,
            ) / math.sqrt(dim)
            outputs.append(indices)
            output_masks.append(pool_mask)
            output_compat.append(mx.where(pool_mask, compat, 0.0))
        return PooledCandidates(
            mx.stack(outputs), mx.stack(output_masks), mx.stack(output_compat)
        )


class SpanContentPooler(nn.Module):
    def __init__(self, hidden_size, content_dim):
        super().__init__()
        self.value_projection = nn.Linear(hidden_size, content_dim)
        self.layer_norm = nn.LayerNorm(content_dim)

    def __call__(self, text_states, text_mask, starts, ends):
        values = self.value_projection(text_states) * text_mask[..., None]
        prefix = mx.concatenate(
            (
                mx.zeros((values.shape[0], 1, values.shape[-1])),
                mx.cumsum(values.astype(mx.float32), axis=1),
            ),
            axis=1,
        )
        end_values = _gather_rows(prefix, ends)
        start_values = _gather_rows(prefix, starts)
        length = mx.maximum(ends - starts, 1)[..., None]
        return self.layer_norm(
            ((end_values - start_values) / length).astype(values.dtype)
        )


class SharedPoolScorer(nn.Module):
    def __init__(self, hidden_size, boundary_dim, settings):
        super().__init__()
        pair_dim = settings.get("pair_dim", 128)
        self.start_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_projection = nn.Linear(boundary_dim, pair_dim)
        self.length_projection = nn.Linear(3, pair_dim)
        self.prior_projection = nn.Linear(1, pair_dim)
        self.content_pooler = SpanContentPooler(
            hidden_size, settings.get("content_dim", 64)
        )
        self.content_projection = nn.Linear(settings.get("content_dim", 64), pair_dim)
        self.candidate_norm = nn.LayerNorm(pair_dim)
        self.query_projection = nn.Linear(hidden_size, pair_dim)
        self.film = nn.Linear(pair_dim, pair_dim * 2)
        self.film_output = [
            nn.Linear(pair_dim, 64),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1),
        ]

    def __call__(
        self,
        boundaries,
        queries,
        query_mask,
        pooled,
        marginals,
        text_states,
        text_mask,
    ):
        starts, ends = pooled.indices[..., 0], pooled.indices[..., 1]
        start_rep = _gather_rows(self.start_projection(boundaries), starts)
        end_rep = _gather_rows(self.end_projection(boundaries), ends)
        lengths = mx.maximum(ends - starts, 1).astype(start_rep.dtype)
        text_lengths = mx.maximum(mx.sum(text_mask, axis=-1), 1)[:, None]
        features = mx.stack(
            (mx.log1p(lengths), lengths / text_lengths, mx.rsqrt(lengths)), axis=-1
        )
        candidate = (
            start_rep
            + end_rep
            + self.length_projection(features)
            + self.prior_projection(pooled.compat_logits[..., None])
            + self.content_projection(
                self.content_pooler(text_states, text_mask, starts, ends)
            )
        )
        candidate = self.candidate_norm(candidate) * pooled.mask[..., None]
        query = self.query_projection(queries)
        score = candidate @ query.swapaxes(-1, -2) / math.sqrt(candidate.shape[-1])
        gamma, beta = mx.split(self.film(query), 2, axis=-1)
        conditioned = candidate[:, :, None] * (1 + gamma[:, None]) + beta[:, None]
        film = conditioned
        for layer in self.film_output:
            film = layer(film)
        score = score + film.squeeze(-1)
        score_rows = []
        for row in range(score.shape[0]):
            # Indexing a (queries, positions) plane with a candidate vector puts
            # the advanced axis first, so every gather below is
            # (candidates, queries) -- the same layout as ``score[row]``.
            row_score = (
                score[row]
                + marginals.start_logits[row, :, starts[row]]
                + marginals.end_logits[row, :, ends[row]]
            )
            inside = (
                marginals.inside_prefix[row, :, ends[row]]
                - marginals.inside_prefix[row, :, starts[row]]
            )
            # (queries, 1) * (1, candidates) -> transpose into candidate-major.
            interval = inside + (marginals.inside_mean[row] * lengths[row][None]).T
            row_score = row_score + interval / mx.sqrt(lengths[row])[:, None]
            score_rows.append(row_score)
        score = mx.stack(score_rows)
        keep = pooled.mask[..., None] & query_mask[:, None]
        return mx.where(keep, score, MASK_LOGIT)


class BoundaryHead(nn.Module):
    def __init__(self, hidden_size, settings):
        super().__init__()
        boundary_dim = settings.get("boundary_dim", 128)
        self.boundary_encoder = BoundaryEncoder(hidden_size, settings)
        self.boundary_query_head = BoundaryQueryHead(hidden_size, boundary_dim)
        self.shared_pool_builder = DocumentCandidatePool(boundary_dim, settings)
        self.shared_pool_scorer = SharedPoolScorer(hidden_size, boundary_dim, settings)
        self.null_projection = nn.Linear(hidden_size, 1)
        self.count_head = nn.Linear(hidden_size, 1)

    def __call__(self, text_states, text_mask, query_states, query_mask):
        encoding = self.boundary_encoder(text_states, text_mask)
        marginals = self.boundary_query_head(
            encoding, text_states, text_mask, query_states, query_mask
        )
        pooled = self.shared_pool_builder(
            encoding.states,
            encoding.mask,
            query_mask,
            marginals.start_logits,
            marginals.end_logits,
        )
        logits = self.shared_pool_scorer(
            encoding.states,
            query_states,
            query_mask,
            pooled,
            marginals,
            text_states,
            text_mask,
        )
        return pooled, logits, self.null_projection(query_states).squeeze(-1)


__all__ = ["BoundaryHead", "PooledCandidates"]

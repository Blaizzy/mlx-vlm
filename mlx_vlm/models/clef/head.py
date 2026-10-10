# Adapted from Cloudflare/clef joint_schema_model.py (Apache-2.0).
# Copyright 2026 Cloudflare. See LICENSE in this directory.
"""MLX implementation of Clef's joint schema head."""

import math

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    """PyTorch MultiheadAttention layout, with fused MLX attention."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, queries, memory):
        wq, wk, wv = mx.split(self.in_proj_weight, 3)
        bq, bk, bv = mx.split(self.in_proj_bias, 3)
        q = mx.addmm(bq, queries, wq.T)
        k = mx.addmm(bk, memory, wk.T)
        v = mx.addmm(bv, memory, wv.T)
        b, length, width = q.shape
        q, k, v = [
            x.reshape(b, -1, self.heads, width // self.heads).transpose(0, 2, 1, 3)
            for x in (q, k, v)
        ]
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(width // self.heads) ** -0.5
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, length, width))


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        # Keep upstream sequential indices (index 2 is inference-time dropout).
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(feedforward, width),
            nn.Identity(),
        )

    def __call__(self, queries, memory):
        queries = queries + self.attention(
            self.query_norm(queries), self.memory_norm(memory)
        )
        return queries + self.feedforward(self.feedforward_norm(queries))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.multihead_attn = Attention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, fields, memory):
        normed = self.norm1(fields)
        fields = fields + self.self_attn(normed, normed)
        fields = fields + self.multihead_attn(self.norm2(fields), memory)
        return fields + self.linear2(nn.gelu(self.linear1(self.norm3(fields))))


def normalize(x, eps=1e-12):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        width,
        routing_layers,
        layers,
        heads,
        feedforward,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [DecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Identity(), nn.Linear(width, 1)
        )
        self.prior_logit_scale = mx.array(0.0)
        self.joint_logit_scale = mx.array(0.0)
        self.residual_gate = mx.array(0.0)

    def __call__(
        self, hidden_states, input_ids, record, output_embeddings, checkpoint=None
    ):
        # A request is one unpadded sequence. Every question shares its backbone pass.
        sequence = self.hidden_norm(hidden_states)[0]
        memory = self.memory_projection(sequence)[None]
        if checkpoint is not None:
            checkpoint(memory)
        global_vector = sequence[-1]
        questions = mx.stack(
            [
                sequence[q.question_span[0] : q.question_span[1]].mean(0)
                for q in record.questions
            ]
        )
        type_ids = mx.array([q.question_type for q in record.questions])
        lexical_options, option_queries, counts = [], [], []
        for index, question in enumerate(record.questions):
            contexts = mx.stack(
                [sequence[start:end].mean(0) for start, end in question.option_spans]
            )
            lexical = mx.stack(
                [
                    output_embeddings(input_ids[0, start:end]).mean(0)
                    for start, end in question.option_spans
                ]
            )
            lexical_options.append(lexical)
            counts.append(len(question.option_spans))
            option_queries.append(
                self.option_context_projection(contexts)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(questions[index])[None]
            )
        routed = mx.concatenate(option_queries)[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
            if checkpoint is not None:
                checkpoint(routed)
        offsets, total = [], 0
        for count in counts[:-1]:
            total += count
            offsets.append(total)
        split_options = mx.split(routed[0], offsets)
        base_fields = self.question_projection(questions)
        summaries = []
        for field, options in zip(base_fields, split_options):
            weights = mx.softmax(
                options @ field / math.sqrt(options.shape[-1]), precise=True
            )
            summaries.append((weights[:, None] * options).sum(0))
        fields = (
            base_fields
            + self.option_summary_norm(mx.stack(summaries))
            + self.global_projection(global_vector)[None]
            + self.type_embedding(type_ids)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
            if checkpoint is not None:
                checkpoint(fields)
        fields = self.field_norm(fields[0])
        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        logits = []
        for index, (field, lexical, routed) in enumerate(
            zip(fields, lexical_options, split_options)
        ):
            anchor = normalize(questions[index] + global_vector)
            prior = prior_scale * (normalize(lexical) @ anchor)
            options = self.option_norm(routed)
            repeated = mx.broadcast_to(field[None], options.shape)
            cosine = (normalize(repeated, 1e-8) * normalize(options, 1e-8)).sum(-1)
            features = mx.concatenate(
                [repeated, options, repeated * options, mx.abs(repeated - options)],
                axis=-1,
            )
            residual = self.residual_scorer(features).squeeze(-1)
            logits.append(
                prior
                + mx.sigmoid(self.residual_gate) * (joint_scale * cosine + residual)
            )
        return logits

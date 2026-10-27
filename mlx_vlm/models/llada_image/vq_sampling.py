# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Adapted from inclusionAI/LLaDA-Image for MLX.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLaDA's block-diffusion sampler for the discrete image prior."""

from __future__ import annotations

import mlx.core as mx

IMAGE_TOKEN_OFFSET = 157184
MASK_TOKEN_ID = 156895
END_OF_IMAGE_ID = 156902


def vq_prompt_ids(tokenizer, prompt: str, height: int, width: int):
    scale = max(max(height, width) / 512, 1.0)
    rows, columns = int(height / scale) // 16, int(width / scale) // 16
    if min(rows, columns) < 1:
        raise ValueError("Image aspect ratio leaves no VQ tokens on the shorter side")
    system = "<role>SYSTEM</role> You are a text-to-image generation assistant. <role>HUMAN</role>"
    image = tokenizer.encode(
        f"<|image|><|reserved_token_{rows}|><|reserved_token_{columns}|><boi><|/image|>"
    )[:-1]
    conditional = tokenizer.encode(system + prompt + "<role>ASSISTANT</role>") + image
    unconditional = (
        tokenizer.encode(system + "<uncondition><role>ASSISTANT</role>") + image
    )
    return conditional, unconditional, rows * columns


def generate_vq_tokens(
    text_encoder,
    lm_head,
    input_ids,
    uncond_ids,
    gen_length,
    *,
    codebook_size=16384,
    block_length=32,
    steps=8,
    cfg_scale=2.0,
    threshold=0.95,
    image_token_offset=IMAGE_TOKEN_OFFSET,
):
    if gen_length < 1 or steps < 1 or steps > block_length:
        raise ValueError("Invalid VQ token count or diffusion step count")
    steps = min(steps, gen_length)
    prompt_length = len(input_ids)
    pad_length = prompt_length - len(uncond_ids)
    if pad_length < 0:
        raise ValueError(
            "The unconditional VQ prompt is longer than the conditional prompt"
        )
    blocks = (prompt_length + gen_length + block_length - 1) // block_length
    length = blocks * block_length
    positions = mx.arange(length)
    block_ids = positions // block_length
    attention = (block_ids[:, None] >= block_ids[None, :])[None, None]
    uncond_attention = attention & (positions >= pad_length)[None, None, None]
    uncond_positions = mx.maximum(positions - pad_length, 0)
    tokens = mx.full((1, length), MASK_TOKEN_ID, dtype=mx.int32)
    tokens[:, :prompt_length] = mx.array(input_ids)
    uncond_prefix = mx.array([MASK_TOKEN_ID] * pad_length + list(uncond_ids))
    transfers = [
        block_length // steps + (index < block_length % steps) for index in range(steps)
    ]
    for block in range(prompt_length // block_length, blocks):
        end = (block + 1) * block_length
        current = tokens[:, :end]
        for count in transfers:
            active = current[:, -block_length:] == MASK_TOKEN_ID
            remaining = int(mx.sum(active).item())
            if not remaining:
                break
            if cfg_scale != 1:
                unconditional = mx.concatenate(
                    [uncond_prefix[None], current[:, prompt_length:]], axis=1
                )
                ids = mx.concatenate([current, unconditional])
                pos = mx.stack([positions[:end], uncond_positions[:end]])
                mask = mx.concatenate(
                    [attention[:, :, :end, :end], uncond_attention[:, :, :end, :end]]
                )
            else:
                ids, pos, mask = (
                    current,
                    positions[None, :end],
                    attention[:, :, :end, :end],
                )
            hidden = text_encoder(ids, mask=mask, position_ids=pos)
            logits = lm_head(hidden[:, -block_length:])
            if cfg_scale != 1:
                logits = logits[1:] + cfg_scale * (logits[:1] - logits[1:])
            predicted = mx.argmax(logits, axis=-1)
            confidence = mx.take_along_axis(
                mx.softmax(logits, axis=-1), predicted[..., None], axis=-1
            )[..., 0]
            scores = mx.where(active, confidence, -float("inf"))
            selected = scores > threshold
            if int(mx.sum(selected).item()) < count:
                indices = mx.argsort(scores, axis=-1)[:, -min(count, remaining) :]
                selected = mx.zeros(active.shape, mx.bool_)
                selected[0, indices[0]] = True
            current[:, -block_length:] = mx.where(
                selected, predicted, current[:, -block_length:]
            )
            mx.eval(current)
            generated = current[0, prompt_length:]
            stops = mx.where(
                generated == END_OF_IMAGE_ID, mx.arange(generated.size), generated.size
            )
            stop = int(mx.min(stops).item())
            if stop < generated.size and not bool(
                mx.any(generated[:stop] == MASK_TOKEN_ID).item()
            ):
                if stop < gen_length:
                    raise ValueError(
                        f"The MLLM ended after {stop} VQ tokens; expected {gen_length}"
                    )
                break
        tokens[:, :end] = current
    result = tokens[:, prompt_length : prompt_length + gen_length] - image_token_offset
    if bool(mx.any((result < 0) | (result >= codebook_size)).item()):
        raise ValueError("The MLLM generated tokens outside the SigVQ codebook")
    return result

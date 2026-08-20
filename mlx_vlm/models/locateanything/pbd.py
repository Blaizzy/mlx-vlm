from typing import Dict, List, Optional

import mlx.core as mx

from .config import ModelConfig
from .language import build_magi_block_mask


def get_token_ids(config: ModelConfig) -> Dict[str, int]:
    text = config.text_config
    eos = config.eos_token_id
    im_end = eos[0] if isinstance(eos, (list, tuple)) and eos else 151645
    return {
        "box_start_token_id": config.box_start_token_id,
        "box_end_token_id": config.box_end_token_id,
        "coord_start_token_id": config.coord_start_token_id,
        "coord_end_token_id": config.coord_end_token_id,
        "ref_start_token_id": config.ref_start_token_id,
        "ref_end_token_id": config.ref_end_token_id,
        "none_token_id": config.none_token_id,
        "null_token_id": text.null_token_id,
        "switch_token_id": text.switch_token_id,
        "default_mask_token_id": text.text_mask_token_id,
        "im_end_token_id": im_end,
    }


def _softmax(logits: mx.array) -> mx.array:
    return mx.softmax(logits.astype(mx.float32), axis=-1)


_materialize = getattr(mx, "eval")


def is_valid_box_frame(
    probs: mx.array,
    token_ids: Dict[str, int],
    start_thresh: float = 0.6,
    end_thresh: float = 0.2,
) -> str:
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    null_id = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    none_id = token_ids["none_token_id"]

    if float(probs[0, box_start]) >= start_thresh:
        if (
            float(probs[1, none_id]) > 0.2
            and float(probs[2, box_end]) > 0.2
            and float(probs[3, null_id]) > 0.1
            and float(probs[4, null_id]) > 0.1
        ):
            return "empty_box"

    p_start = float(probs[0, box_start])
    if p_start < float(probs[0, im_end]) or p_start < float(probs[0, null_id]):
        return "illegal_box"

    end_score = (
        float(probs[5, box_end]) + float(probs[5, null_id]) + float(probs[5, im_end])
    )
    if end_score >= end_thresh:
        return "legal_box"
    return "illegal_box"


def decode_bbox_avg(
    probs: mx.array,
    token_ids: Dict[str, int],
    keep_k: int = 5,
    start_thresh: float = 0.7,
    end_thresh: float = 0.2,
    generation_mode: str = "hybrid",
) -> Optional[List[int]]:
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none_id = token_ids["none_token_id"]
    null_id = token_ids["null_token_id"]

    box_type = is_valid_box_frame(probs, token_ids, start_thresh, end_thresh)
    if box_type == "empty_box":
        return [box_start, none_id, box_end, null_id, null_id, null_id]
    if box_type == "illegal_box":
        return None

    sub = probs[1:5]
    order = mx.argsort(-sub, axis=-1)[:, :keep_k]
    pos_ids = mx.take_along_axis(
        mx.broadcast_to(mx.arange(sub.shape[-1])[None], (4, sub.shape[-1])),
        order,
        axis=-1,
    )
    pos_probs = mx.take_along_axis(sub, order, axis=-1)
    pos_ids = pos_ids.tolist()
    pos_probs = pos_probs.tolist()

    final_coords: List[int] = []
    for i in range(4):
        ids_i = pos_ids[i]
        probs_i = pos_probs[i]
        valid = [
            (cid, p)
            for cid, p in zip(ids_i, probs_i)
            if coord_start <= cid <= coord_end
        ]
        if not valid:
            return None
        first_id, first_p = valid[0]
        if generation_mode == "hybrid":
            valid_ids = [cid for cid, _ in valid]
            is_abnormal = (
                first_p < 0.9
                and len(valid_ids) > 1
                and (max(valid_ids) - min(valid_ids)) > 60
            )
            final_coords.append(0 if is_abnormal else first_id)
        else:
            final_coords.append(first_id)

    return [box_start, *final_coords, box_end]


def decode_ref(
    probs: mx.array,
    token_ids: Dict[str, int],
    keep_k: int = 5,
    start_thresh: float = 0.6,
) -> Optional[List[int]]:
    ref_start = token_ids["ref_start_token_id"]
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]

    if float(probs[0, ref_start]) < start_thresh:
        return None

    sub = probs[1:]
    L = sub.shape[0]
    order = mx.argsort(-sub, axis=-1)[:, :keep_k]
    pos_ids = mx.take_along_axis(
        mx.broadcast_to(mx.arange(sub.shape[-1])[None], (L, sub.shape[-1])),
        order,
        axis=-1,
    ).tolist()

    final_ids: List[int] = []
    for ids_i in pos_ids:
        valid = [cid for cid in ids_i if not (coord_start <= cid <= coord_end)]
        if not valid:
            return None
        final_ids.append(valid[0])

    return [ref_start, *final_ids]


def sample_block(
    block_logits: mx.array,
    token_ids: Dict[str, int],
    generation_mode: str = "hybrid",
    keep_k: int = 5,
) -> List[int]:
    probs = _softmax(block_logits)
    x0 = mx.argmax(probs, axis=-1).tolist()

    box = decode_bbox_avg(
        probs, token_ids, keep_k=keep_k, generation_mode=generation_mode
    )
    if box is not None:
        return box
    ref = decode_ref(probs, token_ids, keep_k=keep_k)
    if ref is not None:
        return ref
    return x0


def handle_pattern(
    x0: List[int], token_ids: Dict[str, int], generation_mode: str = "hybrid"
) -> Dict:
    null_id = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none_id = token_ids["none_token_id"]
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]
    ref_end = token_ids["ref_end_token_id"]

    if x0[0] in (null_id, im_end):
        return {
            "type": "im_end",
            "tokens": [im_end],
            "need_switch_to_ar": False,
            "is_terminal": True,
        }
    if x0[:2] == [box_start, none_id]:
        return {
            "type": "empty_box",
            "tokens": [box_start, none_id, box_end],
            "need_switch_to_ar": False,
            "is_terminal": False,
        }
    if x0[0] == box_start:
        coord_ix = 1
        for coord in x0[1:5]:
            if coord_start <= coord <= coord_end:
                coord_ix += 1
            else:
                break

        if coord_ix == 5 and x0[5] == box_end:
            return {
                "type": "coord_box",
                "tokens": x0,
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        if coord_ix == 3 and x0[3] == box_end:
            return {
                "type": "point_box",
                "tokens": x0[:4],
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        if generation_mode == "fast":
            return {
                "type": "coord_box",
                "tokens": x0,
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        return {
            "type": "error_box",
            "tokens": x0[:coord_ix],
            "need_switch_to_ar": True,
            "is_terminal": False,
        }

    tokens = list(x0)
    for i, token in enumerate(tokens):
        if token == null_id:
            tokens = tokens[:i]
            break
    if len(tokens) >= 2 and tokens[-1] == tokens[-2] == ref_end:
        tokens = tokens[:-1]
    return {
        "type": "ref_object",
        "tokens": tokens,
        "need_switch_to_ar": False,
        "is_terminal": False,
    }


class PBDDecoder:
    def __init__(self, model, generation_mode: str = "hybrid"):
        assert generation_mode in (
            "fast",
            "slow",
            "hybrid",
        ), f"Unsupported generation_mode={generation_mode!r}."
        self.model = model
        self.lm = model.language_model
        self.mode = generation_mode
        self.config = model.config
        self.token_ids = get_token_ids(model.config)
        self.block_size = int(model.config.text_config.block_size)
        assert self.block_size == 6, (
            f"PBD decode utils assume block_size=6, got {self.block_size}; "
            "update handle_pattern/decode_ref before using this checkpoint."
        )
        self.mask_token = self.token_ids["default_mask_token_id"]
        self.im_end = self.token_ids["im_end_token_id"]

    def _forward_mtp(self, generated: List[int], cache) -> mx.array:
        B = self.block_size
        acc = cache[0].offset
        tail = generated[acc:]
        window = tail + [generated[-1]] + [self.mask_token] * (B - 1)
        q_len = len(window)
        kv_len = acc + q_len

        positions = list(range(acc, acc + q_len))
        for i in range(B):
            positions[-(i + 1)] -= 1
        position_ids = mx.array([positions])

        mask = build_magi_block_mask(kv_len, q_len, B)
        inputs = mx.array([window], dtype=mx.int32)
        out = self.lm(inputs, mask=mask, cache=cache, position_ids=position_ids)

        block_logits = out.logits[0, -B:, :]
        _materialize(block_logits)
        for c in cache:
            c.trim(B)

        return block_logits

    def _forward_ar(self, generated: List[int], cache) -> mx.array:
        acc = cache[0].offset
        tail = generated[acc:]
        inputs = mx.array([tail], dtype=mx.int32)
        out = self.lm(inputs, cache=cache)
        return out.logits[0, -1, :]

    def _sample_ar(self, logits: mx.array) -> tuple:
        token = int(mx.argmax(logits).item())
        coord_start = self.token_ids["coord_start_token_id"]
        coord_end = self.token_ids["coord_end_token_id"]
        box_end = self.token_ids["box_end_token_id"]
        none_id = self.token_ids["none_token_id"]

        if self.mode == "hybrid":
            if token == box_end:
                out_type = "box_end_ar"
            elif coord_start <= token <= coord_end or token == none_id:
                out_type = "coord_ar"
            else:
                out_type = "im_end"
        else:
            out_type = "im_end" if token == self.im_end else "continue_ar"
        return out_type, token

    def generate(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        cache,
        max_tokens: int = 2048,
    ) -> List[int]:
        prompt = input_ids[0].tolist()
        generated = list(prompt)
        prompt_len = len(prompt)

        use_mtp = self.mode in ("fast", "hybrid")

        if use_mtp:
            block_logits = self._mtp_prefill(inputs_embeds, cache)
            out_type, tokens = self._consume_block(block_logits)
            generated.extend(tokens)
            if out_type == "im_end":
                return generated[prompt_len : prompt_len + max_tokens]
            if self.mode == "hybrid" and out_type == "error_box":
                use_mtp = False
        else:
            out = self.lm(input_ids, inputs_embeds=inputs_embeds, cache=cache)
            logits = out.logits[0, -1, :]
            out_type, token = self._sample_ar(logits)
            generated.append(token)
            if out_type == "im_end":
                return generated[prompt_len : prompt_len + max_tokens]

        while len(generated) < prompt_len + max_tokens:
            if use_mtp:
                block_logits = self._forward_mtp(generated, cache)
                out_type, tokens = self._consume_block(block_logits)
                generated.extend(tokens)
                if out_type == "im_end":
                    break
                if self.mode == "hybrid" and out_type == "error_box":
                    use_mtp = False
            else:
                logits = self._forward_ar(generated, cache)
                out_type, token = self._sample_ar(logits)
                generated.append(token)
                if out_type == "im_end":
                    break
                if self.mode == "hybrid" and out_type == "box_end_ar":
                    use_mtp = True

        return generated[prompt_len : prompt_len + max_tokens]

    def _mtp_prefill(self, inputs_embeds: mx.array, cache) -> mx.array:
        B = self.block_size
        bridge = inputs_embeds[:, -1:, :]
        mask_embed = self.lm.model.embed_tokens(mx.array([[self.mask_token]]))
        mask_block = mx.broadcast_to(mask_embed, (1, B - 1, inputs_embeds.shape[-1]))
        window = mx.concatenate([inputs_embeds, bridge, mask_block], axis=1)
        q_len = window.shape[1]
        kv_len = q_len

        positions = list(range(q_len))
        for i in range(B):
            positions[-(i + 1)] -= 1
        position_ids = mx.array([positions])

        mask = build_magi_block_mask(kv_len, q_len, B)
        out = self.lm(
            inputs=None,
            inputs_embeds=window,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        block_logits = out.logits[0, -B:, :]
        _materialize(block_logits)
        for c in cache:
            c.trim(B)
        return block_logits

    def _consume_block(self, block_logits: mx.array) -> tuple:
        x0 = sample_block(block_logits, self.token_ids, self.mode)
        pattern = handle_pattern(x0, self.token_ids, self.mode)
        return pattern["type"], pattern["tokens"]

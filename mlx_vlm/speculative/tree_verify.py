"""Tree-attention verification for DDTree (Section 4.4 of the paper).

Overview
--------
Given a ``List[DDTreeNode]`` produced by :func:`build_ddtree`, this module:

1. Flattens the tree into a linear token sequence ``[b, n₀, n₁, …, n_{|T|-1}]``
   where ``b`` is the previously-held bonus (root) and ``nᵢ`` are drafted
   nodes in the order returned by ``build_ddtree``.
2. Builds a 2D ancestor mask ``[Q_len, K_len]`` where each drafted node
   attends only to the committed KV-cache prefix, the root, its ancestors,
   and itself (plus the usual context columns).
3. Builds per-node position ids from tree depth so RoPE lines up.
4. Runs one target-model forward pass and does a greedy walk from the root,
   advancing to whichever child matches the target's argmax at that node.
5. Commits the accepted path via the same snapshot + replay pattern used by
   :mod:`dflash_loop`.

Hybrid-architecture limitation
------------------------------
This verifier is **only correct on pure full-attention target models**. The
ancestor mask presumes per-query causal control, which does not exist inside
the recurrent ``Qwen3_5GatedDeltaNet`` layers that make up 24/32 of
Qwen3.5-4B's decoder. If you invoke this path on a hybrid model, sibling
nodes leak their linear-attention state into each other's state updates and
the verifier's acceptance decisions become incorrect.

For Qwen3.5-4B-DFlash, use :func:`dflash_generate` (Stage 3). Vanilla DFlash
alone is safe on hybrid models because the "tree" is a single trajectory —
no sibling interference in the linear state.

Pure-FA targets (for example the Qwen3-4B-DFlash + Qwen3-4B pair from the
paper) can use this verifier directly.
"""

from typing import Any, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models.qwen3_5_dflash import DFlashDraftModel
from .cache_snapshot import restore_caches, snapshot_caches
from .ddtree import DDTreeNode, build_ddtree


def _flatten_tree(
    tree: List[DDTreeNode], bonus_token: int
) -> Tuple[List[int], List[int], List[int], List[List[int]]]:
    """Return ``(token_ids, depths, parents, children_by_index)`` where all
    indices are into the flat sequence ``[b, n₀, n₁, …]``. Index 0 is the
    root bonus token; indices ``i ≥ 1`` correspond to ``tree[i-1]``.

    ``parents[i]`` is the flat index of ``i``'s parent (root = 0 for depth-1
    nodes). ``children_by_index[i]`` is the list of flat indices of ``i``'s
    direct children in the tree, in the order they were added by Algorithm 1.
    """
    n = 1 + len(tree)
    token_ids = [bonus_token] + [t.token_ids[-1] for t in tree]
    depths = [0] + [t.depth for t in tree]
    parents = [0] * n  # root's own parent is itself for convenience
    for i, node in enumerate(tree, start=1):
        # DDTreeNode.parent indexes tree[]; shift by +1 for flat (because
        # flat[0] is the bonus token). parent == -1 means the root bonus.
        parents[i] = 0 if node.parent == -1 else node.parent + 1
    children: List[List[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        children[parents[i]].append(i)
    return token_ids, depths, parents, children


def _ancestor_mask(parents: List[int], context_len: int) -> mx.array:
    """Build the attention mask for verification.

    Shape: ``[Q_len, K_len]`` with ``Q_len = len(parents)`` (tree nodes
    including the root-bonus slot) and ``K_len = context_len + Q_len`` (past
    KV cache columns + tree columns). ``True`` entries are allowed to attend.

    * Context columns (``K < context_len``) are always allowed.
    * Tree-column ``K == context_len + j`` is allowed by query ``Q = i`` iff
      ``j`` is an ancestor of ``i`` (inclusive).
    """
    q_len = len(parents)
    k_len = context_len + q_len

    # Precompute the ancestor set (as a bitset row) for each node.
    rows = [[False] * q_len for _ in range(q_len)]
    for i in range(q_len):
        j = i
        while True:
            rows[i][j] = True
            if j == 0:
                break
            j = parents[j]

    mask = mx.zeros((q_len, k_len), dtype=mx.bool_)
    if context_len > 0:
        mask = mx.concatenate(
            [mx.ones((q_len, context_len), dtype=mx.bool_), mx.array(rows)],
            axis=1,
        )
    else:
        mask = mx.array(rows)
    return mask


def _position_ids_for_tree(
    depths: List[int], context_len: int
) -> mx.array:
    """Tree position ids = context_len + depth (root bonus sits at
    ``context_len``, depth-1 children at ``context_len + 1``, etc.)."""
    return mx.array(
        [[context_len + d for d in depths]], dtype=mx.int32
    )


def _greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def _walk_tree(
    flat_tokens: List[int],
    children: List[List[int]],
    target_choices: List[int],
) -> Tuple[List[int], List[int], int]:
    """Walk the tree greedily following target's argmax. Returns
    ``(accepted_path_tokens, accepted_flat_indices, next_bonus)``.

    ``accepted_path_tokens`` is the sequence of accepted drafted tokens
    (NOT including the root bonus). ``accepted_flat_indices`` gives the
    flat tree indices of the accepted nodes (useful for cache commit).
    ``next_bonus`` is the first target-argmax token that did NOT appear in
    any explored node's children — this is the token that bridges this
    round to the next.
    """
    accepted_tokens: List[int] = []
    accepted_indices: List[int] = []
    cursor = 0  # root bonus
    while True:
        pred = target_choices[cursor]
        # Look for a child whose token == pred
        matched_child = -1
        for child in children[cursor]:
            if flat_tokens[child] == pred:
                matched_child = child
                break
        if matched_child == -1:
            return accepted_tokens, accepted_indices, pred
        accepted_tokens.append(flat_tokens[matched_child])
        accepted_indices.append(matched_child)
        cursor = matched_child


def ddtree_generate(
    target_model: nn.Module,
    drafter: DFlashDraftModel,
    input_ids: mx.array,
    *,
    max_new_tokens: int = 256,
    node_budget: int = 64,
    block_size: Optional[int] = None,
    target_layer_ids: Optional[List[int]] = None,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Iterator[Tuple[int, int]]:
    """DDTree speculative decoding with tree-attention verification.

    .. warning::
        Correct only on pure full-attention target models. See module
        docstring for details. For Qwen3.5-4B-DFlash (hybrid) use
        :func:`mlx_vlm.speculative.dflash_loop.dflash_generate` instead.
    """
    cfg = drafter.config
    L = block_size if block_size is not None else cfg.block_size
    target_layer_ids = target_layer_ids or cfg.target_layer_ids
    mask_token_id = mask_token_id if mask_token_id is not None else cfg.mask_token_id

    lm = (
        target_model.language_model
        if hasattr(target_model, "language_model")
        else target_model
    )
    embed = lm.model.embed_tokens

    prompt_cache = lm.make_cache()
    prefill = lm(input_ids, cache=prompt_cache, capture_layer_ids=target_layer_ids)
    target_hidden_buffer = mx.concatenate(prefill.hidden_states, axis=-1)
    mx.eval(target_hidden_buffer)

    b = int(_greedy(prefill.logits[:, -1, :]).item())
    yield b, 0
    emitted = 1
    if eos_token_id is not None and b == eos_token_id:
        return

    while emitted < max_new_tokens:
        snap = snapshot_caches(prompt_cache)
        T = target_hidden_buffer.shape[1]

        # --- Drafter forward ---------------------------------------------
        noise_ids = mx.array([[b] + [mask_token_id] * L], dtype=input_ids.dtype)
        noise = embed(noise_ids)
        position_ids = mx.arange(T + L + 1, dtype=mx.int32)[None, :]
        drafter_hidden = drafter(noise, target_hidden_buffer, position_ids)
        drafter_logits = embed.as_linear(drafter_hidden)  # [1, L+1, V]

        # --- DDTree construction -----------------------------------------
        tree_nodes = build_ddtree(drafter_logits, node_budget, slot_offset=1)
        if not tree_nodes:
            # Fall back to the next-bonus token only
            next_bonus = int(_greedy(drafter_logits[:, 1:2, :]).item())
            yield next_bonus, 0
            emitted += 1
            b = next_bonus
            if eos_token_id is not None and next_bonus == eos_token_id:
                return
            continue

        # --- Flatten + verification forward ------------------------------
        flat_tokens, depths, parents, children = _flatten_tree(tree_nodes, b)
        q_len = len(flat_tokens)
        mask = _ancestor_mask(parents, context_len=T)[None, None, :, :]
        pos_ids = _position_ids_for_tree(depths, context_len=T)

        verify_tokens = mx.array([flat_tokens], dtype=input_ids.dtype)
        verify_out = lm(
            verify_tokens,
            cache=prompt_cache,
            mask=mask,
            position_ids=mx.broadcast_to(pos_ids, (3, 1, q_len)),
        )
        verify_logits = verify_out.logits  # [1, q_len, V]
        target_choices = _greedy(verify_logits).reshape(-1).tolist()

        # --- Walk --------------------------------------------------------
        accepted_tokens, accepted_indices, next_bonus = _walk_tree(
            flat_tokens, children, target_choices
        )

        # --- Commit: restore + replay the accepted path ------------------
        restore_caches(prompt_cache, snap)
        commit_tokens = mx.array(
            [[b] + accepted_tokens], dtype=input_ids.dtype
        )
        commit_out = lm(
            commit_tokens,
            cache=prompt_cache,
            capture_layer_ids=target_layer_ids,
        )
        commit_hidden = mx.concatenate(commit_out.hidden_states, axis=-1)
        target_hidden_buffer = mx.concatenate(
            [target_hidden_buffer, commit_hidden], axis=1
        )
        mx.eval(target_hidden_buffer)

        for i, tok in enumerate(accepted_tokens):
            yield tok, i + 1
            emitted += 1
            if emitted >= max_new_tokens:
                return
            if eos_token_id is not None and tok == eos_token_id:
                return

        yield next_bonus, 0
        emitted += 1
        if eos_token_id is not None and next_bonus == eos_token_id:
            return
        b = next_bonus

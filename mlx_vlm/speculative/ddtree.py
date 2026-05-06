"""DDTree best-first draft-tree construction (Algorithm 1 from the paper).

Given per-position token marginals ``q_i(v) = softmax(ℓ_i)_v`` for
``i = 1..L`` from a single block-diffusion drafter forward pass, this module
builds a prefix-closed draft tree of up to ``B`` nodes that maximizes

    E_{y~Q(·|c,b)}[α_T(Y_{1:L})] = Σ_{u ∈ T} q(u|c,b)

where ``q(u|c,b) = ∏_{i≤|u|} q_i(u_i|c,b)`` is the factorized prefix mass
and ``α_T`` is the expected accepted acceptance length under the drafter's
factorized distribution (Proposition 1 of the paper).

Lemma 1 reduces the search space to the top-``K = min(B, |V|)`` tokens at
each depth. Algorithm 1 enumerates those prefixes in descending
log-probability order with a max-heap, popping one prefix per iteration and
pushing its first child and next sibling.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import List, Tuple

import mlx.core as mx


@dataclass
class DDTreeNode:
    """A single tree node, identified by its rank-tuple path from the root.

    ``ranks[i]`` is the 1-indexed rank of this node's token at depth ``i+1``
    (1 = highest-probability token at that depth). ``token_ids`` is the
    resolved vocab id at each depth. ``parent`` is the index (in the flat
    tree list returned by ``build_ddtree``) of this node's parent, or -1 for
    depth-1 nodes whose parent is the root bonus token.
    """

    ranks: Tuple[int, ...]
    token_ids: Tuple[int, ...]
    log_prob: float  # Σ log q_i (rank_i)
    depth: int
    parent: int = -1


def _top_k_tokens_per_depth(log_probs: mx.array, K: int) -> Tuple[mx.array, mx.array]:
    """Return (top_log_probs[L, K], top_ids[L, K]) along the vocabulary
    axis. ``log_probs`` has shape ``[L, V]``.
    """
    L, V = log_probs.shape
    K = min(K, V)
    # Fall back to argpartition for stability with large V.
    top_ids = mx.argsort(-log_probs, axis=-1)[:, :K]
    gather_idx = top_ids[None, :, :]  # [1, L, K] for take_along_axis
    top_lp = mx.take_along_axis(log_probs, top_ids, axis=-1)
    return top_lp, top_ids


def build_ddtree(
    drafter_logits: mx.array,
    budget: int,
    *,
    slot_offset: int = 1,
) -> List[DDTreeNode]:
    """Algorithm 1 — return up to ``budget`` tree nodes in the order they
    are popped (= descending log-probability).

    Parameters
    ----------
    drafter_logits : mx.array
        Shape ``[1, noise_len, V]``. Logits from the DFlash drafter forward
        pass for one sequence in the batch.
    budget : int
        Maximum number of tree nodes to return (``B`` in the paper). The
        root bonus token is *not* counted toward the budget.
    slot_offset : int, default=1
        Starting slot index inside ``drafter_logits``. When the drafter's
        noise block is laid out as ``[b, m, m, …, m]`` (length ``L+1``),
        depth-1 predictions live at slot index 1, so the default of 1 is
        correct. Set to 0 if you fed a noise block of pure masks.

    Returns
    -------
    List[DDTreeNode]
        Flat list of nodes. ``list[i].parent`` indexes earlier entries in
        the same list (or ``-1`` for depth-1 children of the root).
    """
    assert drafter_logits.ndim == 3 and drafter_logits.shape[0] == 1
    V = drafter_logits.shape[-1]
    L_total = drafter_logits.shape[1]
    L = L_total - slot_offset  # usable draft depths
    if L <= 0 or budget <= 0:
        return []

    logits_2d = drafter_logits[0, slot_offset:, :]  # [L, V]
    # log-softmax along vocab — reuses the same op the paper uses to turn
    # products into sums for numerical stability.
    log_probs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)

    K = min(budget, V)
    top_lp, top_ids = _top_k_tokens_per_depth(log_probs, K)
    mx.eval(top_lp, top_ids)
    top_lp_np = top_lp.tolist()  # [L][K]
    top_ids_np = top_ids.tolist()  # [L][K]

    # Heap entries: (neg_log_prob, insertion_index, ranks_tuple, parent_idx)
    # Negate log-prob to simulate a max-heap with heapq's min-heap.
    heap: List[Tuple[float, int, Tuple[int, ...], int]] = []
    counter = 0

    def _lp_of_ranks(ranks: Tuple[int, ...]) -> float:
        s = 0.0
        for depth, r in enumerate(ranks):
            s += top_lp_np[depth][r - 1]
        return s

    # Seed with the length-1 tuple (1,) = the top-1 token at depth 1.
    start_ranks = (1,)
    heappush(heap, (-_lp_of_ranks(start_ranks), counter, start_ranks, -1))
    counter += 1

    tree: List[DDTreeNode] = []
    rank_to_index: dict = {}  # ranks tuple -> flat index (for parent lookup)

    while heap and len(tree) < budget:
        neg_lp, _, ranks, parent = heappop(heap)
        lp = -neg_lp
        depth = len(ranks)
        token_ids = tuple(top_ids_np[d][r - 1] for d, r in enumerate(ranks))
        node = DDTreeNode(
            ranks=ranks, token_ids=token_ids, log_prob=lp, depth=depth, parent=parent
        )
        idx = len(tree)
        tree.append(node)
        rank_to_index[ranks] = idx

        # Push next sibling: same parent, last rank + 1
        last_rank = ranks[-1]
        if last_rank + 1 <= K:
            new_ranks = ranks[:-1] + (last_rank + 1,)
            heappush(
                heap,
                (
                    -_lp_of_ranks(new_ranks),
                    counter,
                    new_ranks,
                    parent,
                ),
            )
            counter += 1

        # Push first child: extend by one more depth, rank 1
        if depth < L:
            new_ranks = ranks + (1,)
            heappush(
                heap,
                (
                    -_lp_of_ranks(new_ranks),
                    counter,
                    new_ranks,
                    idx,
                ),
            )
            counter += 1

    return tree

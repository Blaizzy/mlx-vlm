# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# FlexiCubes portions adapted under Apache-2.0; see mesh_tables.py.
"""Inference-only FlexiCubes extraction on a sparse MLX grid."""

import mlx.core as mx

from .mesh_tables import check_table, dmc_table, num_vd_table
from .sparse import nonzero

CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
EDGES = (0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4)


def _keys(x, resolution):
    return (x[..., 0] * resolution + x[..., 1]) * resolution + x[..., 2]


def _decode(keys, resolution):
    return mx.stack(
        [
            keys // (resolution * resolution),
            keys // resolution % resolution,
            keys % resolution,
        ],
        axis=-1,
    )


def unique(keys):
    """Sorted unique keys and inverse indices, without NumPy."""
    order = mx.argsort(keys)
    sorted_keys = keys[order]
    start = mx.concatenate([mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
    first = nonzero(start)
    groups = mx.cumsum(start.astype(mx.int32)) - 1
    inverse = mx.zeros(keys.size, mx.int32).at[order].add(groups)
    return sorted_keys[first], inverse


def _lookup(keys, query):
    idx = mx.minimum(mx.searchsorted(keys, query.reshape(-1)), keys.size - 1).reshape(
        query.shape
    )
    return mx.where(keys[idx] == query, idx, keys.size)


def _empty():
    return {
        "vertices": mx.zeros((0, 3)),
        "faces": mx.zeros((0, 3), mx.int32),
        "vertex_colors": mx.zeros((0, 6)),
    }


def extract_mesh(features, grid):
    """Decode all learned SDF, deformation, alpha/beta/gamma and color values.

    Candidate cubes include neighbors of predicted vertices. This matches
    the dense reference's default +1 SDF outside the sparse prediction.
    """
    if not features.shape[0]:
        return _empty()
    r = grid.resolution
    coords = grid.coords[:, 1:]
    f = features.astype(mx.float32)
    corner = mx.array(CORNERS, dtype=mx.int32)
    vkeys, inverse = unique(_keys(coords[:, None] + corner[None], r + 1).reshape(-1))
    values = mx.concatenate(
        [
            f[:, :8, None] - 1 / r,
            f[:, 8:32].reshape(-1, 8, 3),
            f[:, 53:].reshape(-1, 8, 6),
        ],
        axis=-1,
    ).reshape(-1, 10)
    sums = mx.zeros((vkeys.size, 10)).at[inverse].add(values)
    counts = mx.zeros((vkeys.size, 1)).at[inverse].add(mx.ones((inverse.size, 1)))
    attrs = sums / counts
    vcoords = _decode(vkeys, r + 1)
    candidates = (vcoords[:, None] - corner[None]).reshape(-1, 3)
    valid = nonzero(mx.all((candidates >= 0) & (candidates < r), axis=-1))
    ckeys, _ = unique(_keys(candidates[valid], r))
    ccoords = _decode(ckeys, r)
    corners_keys = _keys(ccoords[:, None] + corner[None], r + 1)
    lookup = _lookup(vkeys, corners_keys)
    sdf = mx.concatenate([attrs[:, 0], mx.ones(1)])
    occupancy = sdf[lookup] < 0
    sums_occ = occupancy.sum(axis=-1)
    surface = nonzero((sums_occ > 0) & (sums_occ < 8))
    if not surface.size:
        return _empty()
    ccoords, ckeys = ccoords[surface], ckeys[surface]
    corners_keys = corners_keys[surface]
    cases = mx.sum(
        occupancy[surface] * mx.array([1, 2, 4, 8, 16, 32, 64, 128]), axis=-1
    )
    # Resolve the ambiguous C16/C19 cases against adjacent surface cubes.
    checks = mx.array(check_table, mx.int32)[cases]
    adjacent = ccoords + checks[:, 1:4]
    adjacent_index = _lookup(ckeys, _keys(adjacent, r))
    problem_flags = mx.concatenate([checks[:, 0], mx.zeros(1, mx.int32)])
    invert = (
        (checks[:, 0] == 1)
        & mx.all((adjacent >= 0) & (adjacent < r), axis=-1)
        & (problem_flags[adjacent_index] == 1)
    )
    cases = mx.where(invert, checks[:, -1], cases)

    # Retain the dense reference's vertex IDs, including default vertices.
    all_vkeys, cube_idx = unique(corners_keys.reshape(-1))
    cube_idx = cube_idx.reshape(-1, 8)
    attr_indices = _lookup(vkeys, all_vkeys)
    default_attr = mx.array([[1.0] + [0.0] * 9])
    attrs = mx.concatenate([attrs, default_attr])[attr_indices]
    vertices = (
        _decode(all_vkeys, r + 1).astype(mx.float32) / r
        - 0.5
        + mx.tanh(attrs[:, 1:4]) * ((1 - 1e-8) / (r * 2))
    )
    sdf, colors = attrs[:, 0], mx.sigmoid(attrs[:, 4:])
    source_keys = _keys(coords, r)
    source_order = mx.argsort(source_keys)
    weight_indices = _lookup(source_keys[source_order], ckeys)
    weights = mx.concatenate([f[source_order, 32:53], mx.zeros((1, 21))])[
        weight_indices
    ]
    beta, alpha = 1 + 0.99 * mx.tanh(weights[:, :12]), 1 + 0.99 * mx.tanh(
        weights[:, 12:20]
    )
    gamma = 0.99 * mx.sigmoid(weights[:, 20]) + 0.005
    edges = mx.array(EDGES, mx.int32)
    all_edges = cube_idx[:, edges].reshape(-1, 2)
    edge_keys = all_edges[:, 0].astype(mx.int64) * all_vkeys.size + all_edges[:, 1]
    unique_edge_keys, edge_inverse = unique(edge_keys)
    unique_edges = mx.stack(
        [unique_edge_keys // all_vkeys.size, unique_edge_keys % all_vkeys.size], axis=-1
    ).astype(mx.int32)
    crossing = mx.sum(sdf[unique_edges] < 0, axis=-1) == 1
    cross_ids = nonzero(crossing)
    surface_edges = unique_edges[cross_ids]
    mapping = mx.full((unique_edges.shape[0],), -1, mx.int32)
    mapping[cross_ids] = mx.arange(cross_ids.size)
    idx_map = mapping[edge_inverse].reshape(-1, 12)
    edge_counts = (
        mx.zeros((unique_edges.shape[0],), mx.int32)
        .at[edge_inverse]
        .add(mx.ones(edge_inverse.size, mx.int32))[edge_inverse]
    )

    # Each topology emits one to four dual vertices; seven edges per group.
    table = mx.array(dmc_table, mx.int32)
    numbers = mx.array(num_vd_table, mx.int32)[cases]
    edge_group, group_vertex, group_cube, gamma_parts = [], [], [], []
    total = 0
    for number in range(1, 5):
        cubes = nonzero(numbers == number)
        if not cubes.size:
            continue
        group = table[cases[cubes], :number].reshape(-1, 7)
        valid = nonzero(group >= 0)
        edge_group.append(group.reshape(-1)[valid])
        group_vertex.append(mx.repeat(mx.arange(group.shape[0]) + total, 7)[valid])
        group_cube.append(mx.repeat(cubes, number * 7)[valid])
        gamma_parts.append(mx.repeat(gamma[cubes], number))
        total += group.shape[0]
    edge_group, group_vertex, group_cube = (
        mx.concatenate(x) for x in (edge_group, group_vertex, group_cube)
    )
    vd_gamma = mx.concatenate(gamma_parts)
    selected = idx_map[group_cube, edge_group]
    pair = surface_edges[selected]
    alpha_pair = alpha[:, edges].reshape(-1, 12, 2)[group_cube, edge_group]
    signed = sdf[pair] * alpha_pair
    interp = mx.stack([signed[:, 1], -signed[:, 0]], axis=-1)
    interp = interp / interp.sum(axis=-1, keepdims=True)
    crossing_vertices = mx.sum(vertices[pair] * interp[..., None], axis=1)
    crossing_colors = mx.sum(colors[pair] * interp[..., None], axis=1)
    weight = beta[group_cube, edge_group, None]
    denominator = mx.zeros((total, 1)).at[group_vertex].add(weight)
    vd = (
        mx.zeros((total, 3)).at[group_vertex].add(crossing_vertices * weight)
        / denominator
    )
    vd_colors = (
        mx.zeros((total, colors.shape[-1]))
        .at[group_vertex]
        .add(crossing_colors * weight)
        / denominator
    )
    vd_idx = mx.zeros((cases.size * 12,), mx.int32)
    vd_idx[group_cube * 12 + edge_group] = group_vertex

    group_mask = nonzero((edge_counts == 4) & crossing[edge_inverse])
    group = idx_map.reshape(-1)[group_mask]
    # Explicit tie breaker gives the stable edge order used by the reference.
    order = mx.argsort(group.astype(mx.int64) * group.size + mx.arange(group.size))
    quads = vd_idx[group_mask][order].reshape(-1, 4)
    edge_ids = group[order].reshape(-1, 4)[:, 0]
    flip = sdf[surface_edges[edge_ids, 0]] > 0
    quads = mx.concatenate(
        [
            quads[nonzero(flip)][:, mx.array([0, 1, 3, 2])],
            quads[nonzero(~flip)][:, mx.array([2, 3, 1, 0])],
        ]
    )
    g = vd_gamma[quads]
    split = g[:, 0] * g[:, 2] > g[:, 1] * g[:, 3]
    faces = mx.where(
        split[:, None],
        quads[:, mx.array([0, 1, 2, 0, 2, 3])],
        quads[:, mx.array([0, 1, 3, 3, 1, 2])],
    ).reshape(-1, 3)
    return {"vertices": vd, "faces": faces.astype(mx.int32), "vertex_colors": vd_colors}

"""Mesh extraction and cleanup for Hi3D (NumPy).

Ports Sparcubes vertex averaging, sparse FlexiCubes dual marching cubes,
boundary-loop hole filling and the released post-processing. Mesh
simplification uses the optional ``pyfqmr`` package.
"""

import logging

import numpy as np

from ..sam3d_objects.mesh_tables import check_table, dmc_table, num_vd_table

CUBE_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)
CUBE_EDGES = np.array(
    [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4]
)
QUAD_SPLIT = np.array([0, 1, 3, 3, 1, 2])
COMPONENT_FACE_RATIO = 0.005
HOLE_LENGTH_LIMIT = 10000

_DMC = np.array(dmc_table, dtype=np.int64)
_NUM_VD = np.array(num_vd_table, dtype=np.int64)
_CHECK = np.array(check_table, dtype=np.int64)


def _keys(coords, extent):
    c = coords.astype(np.int64)
    return (c[..., 0] * extent + c[..., 1]) * extent + c[..., 2]


def _scatter_mean(index, values, size):
    sums = np.stack(
        [np.bincount(index, values[:, i], size) for i in range(values.shape[1])], -1
    )
    return (sums / np.bincount(index, minlength=size)[:, None]).astype(np.float32)


def grid_matrix(coords):
    """Homogeneous transform placing grid ``coords`` in the unit ball."""
    coords = np.asarray(coords, dtype=np.float32)
    center = coords.mean(axis=0)
    scale = np.linalg.norm(coords - center, axis=1).max() + np.float32(1e-8)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] /= scale
    matrix[:3, 3] = -center / scale
    return matrix


def decoded_mesh(sdf_sign, feats, coords, resolution, matrix):
    """Extract the isosurface of decoded voxels.

    Args:
        sdf_sign: ``(N, 8)`` corner sign logits.
        feats: ``(N, 32)`` tanh-activated corner SDFs (8) and deformations (24).
        coords: ``(N, 3)`` voxel coordinates on a ``resolution`` grid.
        matrix: Grid-to-object transform of the decoder's input grid.

    Returns:
        ``(vertices, faces)``; both empty when no cube straddles the surface.
    """
    coords = np.asarray(coords, dtype=np.int64)
    # FlexiCubes' ambiguity resolution assumes lexicographically ordered cubes.
    order = np.argsort(_keys(coords, resolution), kind="stable")
    coords = coords[order]
    feats = np.asarray(feats, dtype=np.float32)[order]
    sign = 1.0 / (1.0 + np.exp(-np.asarray(sdf_sign, dtype=np.float32)[order]))
    corner_feats = np.concatenate(
        [
            sign[..., None],
            feats[:, :8, None],
            feats[:, 8:32].reshape(-1, 8, 3) * np.float32(0.5),
        ],
        axis=-1,
    ).reshape(-1, 5)
    corners = (coords[:, None] + CUBE_CORNERS[None]).reshape(-1, 3)
    vertex_keys, inverse = np.unique(
        _keys(corners, resolution + 1), return_inverse=True
    )
    vertex_feats = _scatter_mean(inverse.reshape(-1), corner_feats, len(vertex_keys))
    extent = resolution + 1
    grid_vertices = np.stack(
        [
            vertex_keys // (extent * extent),
            vertex_keys // extent % extent,
            vertex_keys % extent,
        ],
        -1,
    ).astype(np.float32)
    positions = grid_vertices + vertex_feats[:, 2:5]
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
    sdf = np.where(vertex_feats[:, 0] >= 0.5, 1.0, -1.0) * np.abs(vertex_feats[:, 1])
    cubes = inverse.reshape(-1, 8)
    return flexicubes(
        positions.astype(np.float32), sdf.astype(np.float32), cubes, resolution, coords
    )


def flexicubes(vertices, sdf, cubes, resolution, cube_coords):
    """Dual marching cubes over sparse cubes (FlexiCubes without learned weights)."""
    occupied = sdf < 0
    corners = occupied[cubes]
    total = corners.sum(-1)
    surface = (total > 0) & (total < 8)
    if not surface.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    case_ids = _case_ids(corners, surface, resolution, cube_coords)
    edges, idx_map, edge_counts, edge_mask = _surface_edges(occupied, cubes, surface)
    vd, vd_idx_map = _dual_vertices(vertices, edges, sdf, case_ids, idx_map)
    return vd, _triangulate(sdf, edges, edge_counts, idx_map, vd_idx_map, edge_mask)


def _case_ids(corners, surface, resolution, cube_coords):
    case_ids = (corners[surface].astype(np.int64) << np.arange(8)).sum(-1)
    problem = _CHECK[case_ids]
    to_check = problem[:, 0] == 1
    problem = problem[to_check]
    index = cube_coords[surface][to_check]
    adjacent = index + problem[:, 1:4]
    within = np.all((adjacent >= 0) & (adjacent < resolution), axis=-1)
    index, adjacent, problem = index[within], adjacent[within], problem[within]
    if not len(problem):
        return case_ids
    problem_keys = _keys(index, resolution)
    order = np.argsort(problem_keys, kind="stable")
    sorted_keys = problem_keys[order]
    query = _keys(adjacent, resolution)
    position = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
    found = sorted_keys[position] == query
    neighbor = np.where(found, order[position], -1)
    padded = np.concatenate([problem, np.zeros((1, 5), problem.dtype)])
    invert = padded[neighbor][:, 0] == 1
    rows = np.arange(len(case_ids))[to_check][within][invert]
    case_ids[rows] = problem[invert][:, -1]
    return case_ids


def _surface_edges(occupied, cubes, surface):
    all_edges = cubes[surface][:, CUBE_EDGES].reshape(-1, 2).astype(np.int64)
    width = int(cubes.max()) + 1
    keys, inverse, counts = np.unique(
        all_edges[:, 0] * width + all_edges[:, 1],
        return_inverse=True,
        return_counts=True,
    )
    inverse = inverse.reshape(-1)
    unique_edges = np.stack([keys // width, keys % width], -1)
    crossing = occupied[unique_edges].sum(-1) == 1
    mapping = np.full(len(keys), -1, np.int64)
    mapping[crossing] = np.arange(crossing.sum())
    return unique_edges[crossing], mapping[inverse], counts[inverse], crossing[inverse]


def _dual_vertices(vertices, edges, sdf, case_ids, idx_map):
    edge_x = vertices[edges.reshape(-1)].reshape(-1, 2, 3)
    edge_s = sdf[edges.reshape(-1)].reshape(-1, 2, 1)
    idx_map = idx_map.reshape(-1, 12)
    num_vd = _NUM_VD[case_ids]
    groups, group_vd, group_cube = [], [], []
    total = 0
    for num in np.unique(num_vd):
        cubes = np.nonzero(num_vd == num)[0]
        count = len(cubes) * num
        group = _DMC[case_ids[cubes], :num].reshape(-1, num * 7)
        to_vd = (np.arange(count)[:, None].repeat(7, 1) + total).reshape(group.shape)
        to_cube = cubes[:, None].repeat(num * 7, 1).reshape(group.shape)
        total += count
        mask = group != -1
        groups.append(group[mask])
        group_vd.append(to_vd[mask])
        group_cube.append(to_cube[mask])
    groups = np.concatenate(groups)
    group_vd = np.concatenate(group_vd)
    group_cube = np.concatenate(group_cube)
    index = group_cube * 12 + groups
    edge_ids = idx_map.reshape(-1)[index]
    x, s = edge_x[edge_ids], edge_s[edge_ids].astype(np.float32)
    weights = np.concatenate([s[:, 1:2], -s[:, 0:1]], axis=1)
    crossing = (x * weights).sum(1) / weights.sum(1)
    vd = _scatter_mean(group_vd, crossing, total)
    vd_idx_map = np.zeros(len(case_ids) * 12, np.int64)
    vd_idx_map[index] = group_vd
    return vd, vd_idx_map


def _triangulate(sdf, edges, edge_counts, idx_map, vd_idx_map, edge_mask):
    mask = (edge_counts == 4) & edge_mask
    group = idx_map.reshape(-1)[mask]
    vd_idx = vd_idx_map[mask]
    order = np.argsort(group, kind="stable")
    quads = vd_idx[order].reshape(-1, 4)
    first = edges[group[order].reshape(-1, 4)[:, 0]]
    flip = sdf[first[:, 0]] > 0
    quads = np.concatenate(
        [quads[flip][:, [0, 1, 3, 2]], quads[~flip][:, [2, 3, 1, 0]]]
    )
    return quads[:, QUAD_SPLIT].reshape(-1, 3).astype(np.int32)


def fill_holes(vertices, faces, length_limit=HOLE_LENGTH_LIMIT):
    """Close boundary loops of up to ``length_limit`` vertices by ear clipping."""
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lo, hi = directed.min(1), directed.max(1)
    width = int(faces.max()) + 1
    _, inverse, counts = np.unique(
        lo * width + hi, return_inverse=True, return_counts=True
    )
    boundary = directed[counts[inverse.reshape(-1)] == 1]
    new = [
        _ear_clip(vertices, loop)
        for loop in _boundary_loops(boundary)
        if 3 <= len(loop) <= length_limit
    ]
    return np.concatenate([faces] + new) if new else faces


def _boundary_loops(boundary):
    outgoing = {}
    for idx, (start, _) in enumerate(boundary.tolist()):
        outgoing.setdefault(start, []).append(idx)
    edges = boundary.tolist()
    visited = [False] * len(edges)
    loops = []
    for i in range(len(edges)):
        if visited[i]:
            continue
        visited[i] = True
        start = edges[i][0]
        loop = [start, edges[i][1]]
        while loop[-1] != start:
            for nxt in outgoing.get(loop[-1], []):
                if not visited[nxt]:
                    visited[nxt] = True
                    loop.append(edges[nxt][1])
                    break
            else:
                break
        if loop[0] == loop[-1] and len(loop) > 3:
            loop.pop()
            loops.append(loop)
    return loops


def _ear_clip(vertices, boundary):
    """Triangulate one loop, repeatedly clipping its sharpest corner."""
    line = np.asarray(vertices, dtype=np.float32)[np.array(boundary)]
    line_idx = np.array(boundary, dtype=np.int64)
    faces = []
    while True:
        if len(line_idx) == 3:
            faces.append(line_idx[[2, 1, 0]])
            break
        nxt = np.roll(line, -1, axis=0)
        t = nxt - line
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + np.float32(1e-6))
        t_prev = np.roll(t, 1, axis=0)
        n = np.cross(-t_prev, t)
        center = line.mean(axis=0, keepdims=True)
        sin_sign = np.sign(
            (np.cross(nxt - center, line - center) * n).sum(1, keepdims=True)
        )
        sin_val = np.linalg.norm(n, axis=1, keepdims=True) * sin_sign
        cos_val = -(t_prev * t).sum(axis=1, keepdims=True)
        angle = np.arctan2(sin_val, cos_val).reshape(-1)
        collapsed = angle == 0.0
        score = np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, angle))
        ear = int(np.argmax(score))
        faces.append(
            line_idx[[(ear + 1) % len(line_idx), ear, (ear - 1) % len(line_idx)]]
        )
        line = np.delete(line, ear, axis=0)
        line_idx = np.delete(line_idx, ear)
    return np.vstack(faces)


def _face_components(faces):
    """Connected components of faces joined by edges shared by exactly two faces
    (the adjacency ``trimesh.Trimesh.face_adjacency`` uses)."""
    count = len(faces)
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    owner = np.tile(np.arange(count), 3)
    width = int(faces.max()) + 1
    keys = edges[:, 0] * width + edges[:, 1]
    order = np.argsort(keys, kind="stable")
    keys, owner = keys[order], owner[order]
    _, start, counts = np.unique(keys, return_index=True, return_counts=True)
    pairs = start[counts == 2]
    a, b = owner[pairs], owner[pairs + 1]
    parent = np.arange(count)
    while True:
        pa, pb = parent[a], parent[b]
        np.minimum.at(parent, np.maximum(pa, pb), np.minimum(pa, pb))
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
        if np.array_equal(parent[a], parent[b]):
            return parent


def postprocess(vertices, faces, target_faces, component_ratio=COMPONENT_FACE_RATIO):
    """Drop debris components, simplify, then center and scale to unit radius."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    labels = _face_components(faces)
    sizes = np.bincount(labels, minlength=len(faces))
    largest = int(np.argmax(sizes))
    keep = (labels == largest) | (sizes[labels] / len(faces) >= component_ratio)
    faces = faces[keep]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices, faces = vertices[used], inverse.reshape(-1, 3)
    try:
        import pyfqmr
    except ImportError:
        logging.warning(
            "pyfqmr is not installed; the Hi3D mesh is not simplified to %d faces",
            target_faces,
        )
    else:
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(vertices, faces)
        simplifier.simplify_mesh(
            target_count=target_faces,
            aggressiveness=5,
            preserve_border=True,
            verbose=False,
        )
        vertices, faces, _ = simplifier.getMesh()
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
    vertices = vertices - vertices.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(vertices, axis=1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("mesh cannot be normalized to a positive finite radius")
    return (vertices / radius).astype(np.float32), faces.astype(np.int32)


def coarse_from_sam3d(result):
    """Coarse mesh for Hi3D from a SAM 3D Objects result.

    The object-local mesh is posed into SAM 3D's camera frame (+X left, +Y
    up, +Z forward; ``camera = (scale * local) @ R + t``) and turned 180
    degrees about Y, so the side the photo shows faces +Z with +X to the
    right and +Y up.
    """
    pose, mesh = result["pose"], result["mesh"]
    v = np.array(mesh["vertices"], np.float64) * np.array(pose["scale"])
    v = v @ np.array(pose["rotation_matrix"]) + np.array(pose["translation"])
    v = (v - v.mean(axis=0)) * np.array([-1.0, 1.0, -1.0])
    return v.astype(np.float32), np.array(mesh["faces"], np.int64)


def read_mesh(path):
    """Vertices and triangle faces from an OBJ file, or any format trimesh reads."""
    path = str(path)
    if path.lower().endswith(".obj"):
        vertices, faces = [], []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append([float(p) for p in parts[1:4]])
                elif parts[0] == "f":
                    corners = [int(p.split("/")[0]) for p in parts[1:]]
                    corners = [c - 1 if c > 0 else len(vertices) + c for c in corners]
                    for i in range(1, len(corners) - 1):
                        faces.append([corners[0], corners[i], corners[i + 1]])
        return np.array(vertices, np.float32), np.array(faces, np.int64)
    try:
        import trimesh
    except ImportError as error:
        raise ImportError("Reading non-OBJ meshes requires trimesh") from error
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64)


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        np.savetxt(f, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, np.asarray(faces) + 1, fmt="f %d %d %d")

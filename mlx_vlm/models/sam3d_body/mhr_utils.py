"""Rotation math and index arrays for MHR body model."""

import mlx.core as mx


def cross(a: mx.array, b: mx.array) -> mx.array:
    """Cross product along last axis. MLX lacks mx.cross."""
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def rot6d_to_rotmat(x: mx.array) -> mx.array:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        x: (..., 6) first two columns of rotation matrix
    Returns:
        (..., 3, 3) rotation matrix
    """
    x1 = x[..., :3]
    x2 = x[..., 3:]
    x1 = x1 / (mx.linalg.norm(x1, axis=-1, keepdims=True) + 1e-8)
    z = cross(x1, x2)
    z = z / (mx.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = cross(z, x1)
    return mx.stack([x1, y, z], axis=-1)


def rotmat_to_euler_ZYX(R: mx.array) -> mx.array:
    """Convert 3x3 rotation matrix to ZYX euler angles.

    Args:
        R: (..., 3, 3) rotation matrix
    Returns:
        (..., 3) euler angles in ZYX order
    """
    sy = mx.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = (sy < 1e-6).astype(mx.float32)

    x = (
        mx.arctan2(R[..., 2, 1], R[..., 2, 2]) * (1 - singular)
        + mx.arctan2(-R[..., 1, 2], R[..., 1, 1]) * singular
    )
    y = mx.arctan2(-R[..., 2, 0], sy)
    z = mx.arctan2(R[..., 1, 0], R[..., 0, 0]) * (1 - singular)

    return mx.stack([z, y, x], axis=-1)


def batch_xyz_from_6d(poses: mx.array) -> mx.array:
    """Convert 6D rotation to XYZ euler angles.

    Args:
        poses: (..., 6)
    Returns:
        (..., 3) XYZ euler angles
    """
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = x_raw / (mx.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = cross(x, y_raw)
    z = z / (mx.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = cross(z, x)

    matrix = mx.stack([x, y, z], axis=-1)  # (..., 3, 3)

    sy = mx.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    singular = (sy < 1e-6).astype(mx.float32)

    ex = mx.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    ey = mx.arctan2(-matrix[..., 2, 0], sy)
    ez = mx.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])

    exs = mx.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1])

    return mx.stack(
        [
            ex * (1 - singular) + exs * singular,
            ey,
            ez * (1 - singular),
        ],
        axis=-1,
    )


def sincos_to_angle(sc: mx.array) -> mx.array:
    """Convert (sin, cos) pair to angle via atan2.

    Args:
        sc: (..., 2) sin and cos values
    Returns:
        (...,) angles
    """
    return mx.arctan2(sc[..., 0], sc[..., 1])


# --- Index arrays ---
# 3-DOF joints: 23 groups of 3 indices into the 133D output
ALL_PARAM_3DOF_ROT_IDXS = [
    (0, 2, 4),
    (6, 8, 10),
    (12, 13, 14),
    (15, 16, 17),
    (18, 19, 20),
    (21, 22, 23),
    (24, 25, 26),
    (27, 28, 29),
    (34, 35, 36),
    (37, 38, 39),
    (44, 45, 46),
    (53, 54, 55),
    (64, 65, 66),
    (85, 69, 73),
    (86, 70, 79),
    (87, 71, 82),
    (88, 72, 76),
    (91, 92, 93),
    (112, 96, 100),
    (113, 97, 106),
    (114, 98, 109),
    (115, 99, 103),
    (130, 131, 132),
]

# 1-DOF joints: 58 indices into the 133D output
ALL_PARAM_1DOF_ROT_IDXS = [
    1,
    3,
    5,
    7,
    9,
    11,
    30,
    31,
    32,
    33,
    40,
    41,
    42,
    43,
    47,
    48,
    49,
    50,
    51,
    52,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    67,
    68,
    74,
    75,
    77,
    78,
    80,
    81,
    83,
    84,
    89,
    90,
    94,
    95,
    101,
    102,
    104,
    105,
    107,
    108,
    110,
    111,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
]

# Translation indices: 6 into the 133D output
ALL_PARAM_1DOF_TRANS_IDXS = [124, 125, 126, 127, 128, 129]

# Hand parameter mask (indices 62-115 in the 133D output)
MHR_PARAM_HAND_IDXS = list(range(62, 116))

# Hand DOF pattern for compact_cont_to_model_params_hand
HAND_DOFS_IN_ORDER = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]


def compact_cont_to_model_params_body(body_pose_cont: mx.array) -> mx.array:
    """Convert 260D continuous body pose to 133D euler angles.

    Input layout (260D):
      - 23 x 6D (3-DOF joints) = 138D
      - 58 x 2D (1-DOF joints, sin/cos) = 116D
      - 6 x 1D (translation params) = 6D
    Total: 138 + 116 + 6 = 260D

    Output: 133D = 69 (3-DOF euler) + 58 (1-DOF angles) + 6 (translations)
    """
    B = body_pose_cont.shape[0]
    result = mx.zeros((B, 133))

    # Process 3-DOF joints: 23 groups, each 6D -> 3D euler
    offset = 0
    parts_3dof = []
    for i, (idx_x, idx_y, idx_z) in enumerate(ALL_PARAM_3DOF_ROT_IDXS):
        chunk = body_pose_cont[:, offset : offset + 6]  # (B, 6)
        euler = batch_xyz_from_6d(chunk)  # (B, 3)
        parts_3dof.append((idx_x, euler[:, 0:1]))
        parts_3dof.append((idx_y, euler[:, 1:2]))
        parts_3dof.append((idx_z, euler[:, 2:3]))
        offset += 6
    # offset should be 138

    # Process 1-DOF joints: 58 pairs, each 2D (sin,cos) -> 1D angle
    parts_1dof = []
    for i, idx in enumerate(ALL_PARAM_1DOF_ROT_IDXS):
        sc = body_pose_cont[:, offset : offset + 2]  # (B, 2)
        angle = sincos_to_angle(sc)  # (B,)
        parts_1dof.append((idx, angle[:, None]))
        offset += 2
    # offset should be 254

    # Process translations: 6 pass-through values
    parts_trans = []
    for i, idx in enumerate(ALL_PARAM_1DOF_TRANS_IDXS):
        val = body_pose_cont[:, offset : offset + 1]  # (B, 1)
        parts_trans.append((idx, val))
        offset += 1
    # offset should be 260

    # Build output by scattering values into correct indices
    # Collect all (index, value) pairs and build via advanced indexing
    all_parts = parts_3dof + parts_1dof + parts_trans
    all_indices = []
    all_values = []
    for idx, val in all_parts:
        all_indices.append(idx)
        all_values.append(val)

    indices = mx.array(all_indices)  # (133,)
    values = mx.concatenate(all_values, axis=1)  # (B, 133)

    # Sort by index to build output in order
    sort_order = mx.argsort(indices)
    # Build result: values[:, sort_order] placed at indices[sort_order]
    # Since indices covers all 133 positions exactly once, values in sorted order IS the result
    result = values[:, sort_order]

    return result


def compact_cont_to_model_params_hand(hand_cont: mx.array) -> mx.array:
    """Convert 54D continuous hand pose to 27D euler angles.

    Input layout follows HAND_DOFS_IN_ORDER pattern:
      3-DOF: 6D -> 3D euler
      1-DOF: 2D (sin,cos) -> 1D angle
      2-DOF: 4D (2x sin,cos) -> 2D angles
    """
    B = hand_cont.shape[0]
    parts = []
    offset = 0

    for dof in HAND_DOFS_IN_ORDER:
        if dof == 3:
            chunk = hand_cont[:, offset : offset + 6]
            euler = batch_xyz_from_6d(chunk)  # (B, 3)
            parts.append(euler)
            offset += 6
        elif dof == 1:
            sc = hand_cont[:, offset : offset + 2]
            angle = sincos_to_angle(sc)[:, None]  # (B, 1)
            parts.append(angle)
            offset += 2
        elif dof == 2:
            # Two 1-DOF joints packed together
            sc1 = hand_cont[:, offset : offset + 2]
            sc2 = hand_cont[:, offset + 2 : offset + 4]
            a1 = sincos_to_angle(sc1)[:, None]  # (B, 1)
            a2 = sincos_to_angle(sc2)[:, None]  # (B, 1)
            parts.append(mx.concatenate([a1, a2], axis=1))
            offset += 4

    return mx.concatenate(parts, axis=1)  # (B, 27)


def quat_to_rotmat(q: mx.array) -> mx.array:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix.

    Args:
        q: (..., 4) quaternion [x, y, z, w]
    Returns:
        (..., 3, 3) rotation matrix
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1 - 2 * (y2 + z2)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (x2 + z2)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (x2 + y2)

    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def euler_xyz_to_rotmat(angles: mx.array) -> mx.array:
    """Convert XYZ euler angles to 3x3 rotation matrix.

    Args:
        angles: (..., 3) euler angles [x, y, z]
    Returns:
        (..., 3, 3) rotation matrix R = Rz @ Ry @ Rx
    """
    cx = mx.cos(angles[..., 0])
    sx = mx.sin(angles[..., 0])
    cy = mx.cos(angles[..., 1])
    sy = mx.sin(angles[..., 1])
    cz = mx.cos(angles[..., 2])
    sz = mx.sin(angles[..., 2])

    # R = Rz @ Ry @ Rx
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx
    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx

    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def rotmat_to_quat(R: mx.array) -> mx.array:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w).

    Uses Shepperd's method with all 4 branches for numerical stability.
    Selects the branch with the largest diagonal element to avoid
    division by near-zero values (critical for ~180° rotations).

    Args:
        R: (..., 3, 3)
    Returns:
        (..., 4) quaternion [x, y, z, w]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    trace = R00 + R11 + R22

    # Shepperd's 4 candidates: pick largest of {trace, R00, R11, R22}
    # to maximise the denominator and avoid sqrt-of-negative / div-by-zero.
    # Branch 0: trace is largest  -> w is largest quat component
    s0 = mx.sqrt(mx.maximum(trace + 1.0, 1e-10)) * 2.0
    w0 = 0.25 * s0
    x0 = (R21 - R12) / (s0 + 1e-10)
    y0 = (R02 - R20) / (s0 + 1e-10)
    z0 = (R10 - R01) / (s0 + 1e-10)
    q0 = mx.stack([x0, y0, z0, w0], axis=-1)

    # Branch 1: R00 is largest diagonal -> x is largest
    s1 = mx.sqrt(mx.maximum(1.0 + R00 - R11 - R22, 1e-10)) * 2.0
    x1 = 0.25 * s1
    w1 = (R21 - R12) / (s1 + 1e-10)
    y1 = (R01 + R10) / (s1 + 1e-10)
    z1 = (R02 + R20) / (s1 + 1e-10)
    q1 = mx.stack([x1, y1, z1, w1], axis=-1)

    # Branch 2: R11 is largest diagonal -> y is largest
    s2 = mx.sqrt(mx.maximum(1.0 - R00 + R11 - R22, 1e-10)) * 2.0
    y2 = 0.25 * s2
    w2 = (R02 - R20) / (s2 + 1e-10)
    x2 = (R01 + R10) / (s2 + 1e-10)
    z2 = (R12 + R21) / (s2 + 1e-10)
    q2 = mx.stack([x2, y2, z2, w2], axis=-1)

    # Branch 3: R22 is largest diagonal -> z is largest
    s3 = mx.sqrt(mx.maximum(1.0 - R00 - R11 + R22, 1e-10)) * 2.0
    z3 = 0.25 * s3
    w3 = (R10 - R01) / (s3 + 1e-10)
    x3 = (R02 + R20) / (s3 + 1e-10)
    y3 = (R12 + R21) / (s3 + 1e-10)
    q3 = mx.stack([x3, y3, z3, w3], axis=-1)

    # Select best branch per element
    candidates = mx.stack([trace, R00, R11, R22], axis=-1)  # (N, 4)
    best = mx.argmax(candidates, axis=-1)  # (N,)

    # Build result via where-chain (MLX has no advanced scatter)
    result = q0
    result = mx.where((best == 1)[..., None], q1, result)
    result = mx.where((best == 2)[..., None], q2, result)
    result = mx.where((best == 3)[..., None], q3, result)

    return result.reshape(*batch_shape, 4)

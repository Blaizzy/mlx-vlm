"""MHR body model: FK, skinning, blend shapes — pure MLX replacement for JIT model."""

import mlx.core as mx
import mlx.nn as nn

from .mhr_utils import euler_xyz_to_rotmat, quat_to_rotmat


class MHRBodyModel(nn.Module):
    """Pure MLX replacement for the TorchScript JIT body model.

    Forward: (shape_params, model_params, expr_params) -> (skinned_verts, skel_state)

    Weight keys (under 'character.' prefix):
        skeleton.joint_translation_offsets: (127, 3)
        skeleton.joint_prerotations: (127, 4)  quaternions
        skeleton.joint_parents: (127,) int32
        skeleton.pmi: (2, 266) int32
        mesh.rest_vertices: (18439, 3)
        mesh.faces: (36874, 3) int32
        blend_shape.base_shape: (18439, 3)
        blend_shape.shape_vectors: (45, 18439, 3)
        parameter_transform.parameter_transform: (889, 249)
        parameter_transform.pose_parameters: (249,) uint8
        parameter_transform.rigid_parameters: (249,) uint8
        parameter_transform.scaling_parameters: (249,) uint8
        parameter_limits.minmax_min: (198,)
        parameter_limits.minmax_max: (198,)
        parameter_limits.minmax_weight: (198,)
        parameter_limits.minmax_parameter_index: (198,) int32
        linear_blend_skinning.inverse_bind_pose: (127, 8)
        linear_blend_skinning.skin_indices_flattened: (51337,) int32
        linear_blend_skinning.skin_weights_flattened: (51337,)
        linear_blend_skinning.vert_indices_flattened: (51337,) int32/int64

    Additional buffers loaded separately:
        face_expressions.shape_vectors: (72, 18439, 3)
        pose_correctives.pose_dirs_predictor.0.sparse_indices: (2, 53136) int32
        pose_correctives.pose_dirs_predictor.0.sparse_weight: (53136,)
        pose_correctives.pose_dirs_predictor.2.weight: (55317, 3000)
    """

    def __init__(self, num_joints: int = 127, num_verts: int = 18439):
        super().__init__()
        self.num_joints = num_joints
        self.num_verts = num_verts

        # These will be loaded from weights. Declare as frozen arrays.
        # Skeleton
        self.joint_translation_offsets = mx.zeros((num_joints, 3))
        self.joint_prerotations = mx.zeros((num_joints, 4))
        self.joint_parents = mx.zeros((num_joints,), dtype=mx.int32)

        # Parameter transform
        self.parameter_transform = mx.zeros((889, 249))
        self.pose_parameters = mx.zeros((249,), dtype=mx.uint8)
        self.rigid_parameters = mx.zeros((249,), dtype=mx.uint8)
        self.scaling_parameters = mx.zeros((249,), dtype=mx.uint8)

        # Parameter limits
        self.minmax_min = mx.zeros((198,))
        self.minmax_max = mx.zeros((198,))
        self.minmax_weight = mx.zeros((198,))
        self.minmax_parameter_index = mx.zeros((198,), dtype=mx.int32)

        # Blend shapes
        self.base_shape = mx.zeros((num_verts, 3))
        self.shape_vectors = mx.zeros((45, num_verts, 3))

        # Face expressions
        self.face_shape_vectors = mx.zeros((72, num_verts, 3))

        # Skinning
        self.inverse_bind_pose = mx.zeros((num_joints, 8))
        self.skin_indices = mx.zeros((51337,), dtype=mx.int32)
        self.skin_weights = mx.zeros((51337,))
        self.vert_indices = mx.zeros((51337,), dtype=mx.int32)

        # Pose correctives (sparse layer + linear)
        self.pc_sparse_indices = mx.zeros((2, 53136), dtype=mx.int32)
        self.pc_sparse_weight = mx.zeros((53136,))
        self.pc_linear_weight = mx.zeros((55317, 3000))

    def _apply_parameter_limits(self, model_params: mx.array) -> mx.array:
        """Apply min/max clamping to model parameters.

        Uses soft clamping with the minmax_weight for smooth gradients.
        """
        indices = self.minmax_parameter_index  # (198,) indices into 204-dim
        mins = self.minmax_min  # (198,)
        maxs = self.minmax_max  # (198,)

        param_vals = model_params[:, indices]  # (B, 198)
        clamped = mx.clip(param_vals, mins[None, :], maxs[None, :])

        # Scatter clamped values back into the full 204-dim parameter vector
        replacement = mx.zeros_like(model_params)  # (B, 204)
        mask = mx.zeros((204,))
        for i in range(198):
            idx = int(indices[i].item())
            replacement = _set_column(replacement, idx, clamped[:, i])
            mask = mask.at[idx].add(1.0)

        mask = mask[None, :]  # (1, 204)
        return model_params * (1 - mask) + replacement * mask

    def _parameter_transform(self, model_params: mx.array) -> mx.array:
        """Apply parameter transform matrix.

        model_params: (B, 204) -> joint_dofs: (B, 889)
        PT matrix is (889, 249), we pad model_params from 204 to 249.
        """
        B = model_params.shape[0]
        # Pad to 249 (extra 45 columns are unused, all zeros in PT)
        padded = mx.concatenate(
            [model_params, mx.zeros((B, 249 - 204))], axis=1
        )  # (B, 249)

        joint_dofs = padded @ self.parameter_transform.T  # (B, 889)
        return joint_dofs

    def _forward_kinematics(self, joint_dofs: mx.array) -> mx.array:
        """Compute global transforms from joint DOFs.

        joint_dofs: (B, 889) where each joint j has 7 DOFs at j*7..j*7+6
            [tx, ty, tz, rx, ry, rz, scale]

        Returns:
            skel_state: (B, 127, 8) = [x, y, z, qx, qy, qz, qw, scale]
        """
        B = joint_dofs.shape[0]
        # Reshape to (B, 127, 7)
        jd = joint_dofs.reshape(B, self.num_joints, 7)

        # Extract per-joint local transforms
        local_trans = jd[..., :3]  # (B, 127, 3)
        local_rot_euler = jd[..., 3:6]  # (B, 127, 3)
        local_scale = jd[..., 6:7]  # (B, 127, 1)

        # Convert euler angles to rotation matrices
        local_rot = euler_xyz_to_rotmat(local_rot_euler)  # (B, 127, 3, 3)

        # Apply prerotation (quaternion -> rotmat, then compose)
        prerot_mat = quat_to_rotmat(self.joint_prerotations)  # (127, 3, 3)
        # local_rot_composed = prerot @ local_rot for each joint
        # prerot_mat: (127, 3, 3), local_rot: (B, 127, 3, 3)
        local_rot_composed = mx.einsum("jpq,bjqr->bjpr", prerot_mat, local_rot)

        # Build 4x4 local transforms
        # translation = joint_translation_offsets + local_trans
        trans = self.joint_translation_offsets[None, :, :] + local_trans  # (B, 127, 3)

        # Scale factor: exp(dof * ln(2)) = 2^dof (matches PyTorch JIT convention)
        scale = mx.exp(local_scale * 0.6931471824645996)  # (B, 127, 1)

        # FK chain: iterate joints in parent order
        parents = self.joint_parents  # (127,)

        # Store global position, rotation (as 3x3), and scale for each joint
        # We'll accumulate lists and stack at the end
        global_pos_list = []
        global_rot_list = []
        global_scale_list = []

        for j in range(self.num_joints):
            parent = int(parents[j].item())
            lr = local_rot_composed[:, j]  # (B, 3, 3)
            lt = trans[:, j]  # (B, 3)
            ls = scale[:, j]  # (B, 1)

            if parent == -1:
                # Root joint
                gp = lt
                gr = lr
                gs = ls
            else:
                # Global = parent_global @ local
                pr = global_rot_list[parent]  # (B, 3, 3)
                pp = global_pos_list[parent]  # (B, 3)
                ps = global_scale_list[parent]  # (B, 1)

                # Position: parent_pos + parent_scale * (parent_rot @ local_trans)
                gp = pp + ps * mx.einsum("bij,bj->bi", pr, lt)
                # Rotation: parent_rot @ local_rot
                gr = mx.einsum("bij,bjk->bik", pr, lr)
                # Scale
                gs = ps * ls

            global_pos_list.append(gp)
            global_rot_list.append(gr)
            global_scale_list.append(gs)

        # Stack results
        global_pos = mx.stack(global_pos_list, axis=1)  # (B, 127, 3)
        global_rot = mx.stack(global_rot_list, axis=1)  # (B, 127, 3, 3)
        global_scale = mx.stack(global_scale_list, axis=1)  # (B, 127, 1)

        # Convert rotation matrices to quaternions for skel_state
        from .mhr_utils import rotmat_to_quat

        global_quat = rotmat_to_quat(global_rot)  # (B, 127, 4) [x,y,z,w]

        # skel_state: (B, 127, 8) = [x, y, z, qx, qy, qz, qw, scale]
        skel_state = mx.concatenate(
            [
                global_pos,
                global_quat,
                global_scale,
            ],
            axis=-1,
        )

        return skel_state, global_pos, global_rot, global_scale

    def _linear_blend_skinning(
        self,
        rest_verts: mx.array,
        global_pos: mx.array,
        global_rot: mx.array,
        global_scale: mx.array,
    ) -> mx.array:
        """Apply linear blend skinning.

        Args:
            rest_verts: (B, num_verts, 3) vertices after blend shapes
            global_pos: (B, 127, 3) joint world positions
            global_rot: (B, 127, 3, 3) joint world rotations
            global_scale: (B, 127, 1) joint world scales

        Returns:
            posed_verts: (B, num_verts, 3)
        """
        B = rest_verts.shape[0]

        # inverse_bind_pose: (127, 8) = [tx, ty, tz, qx, qy, qz, qw, scale]
        ibp_trans = self.inverse_bind_pose[:, :3]  # (127, 3)
        ibp_quat = self.inverse_bind_pose[:, 3:7]  # (127, 4)
        ibp_scale = self.inverse_bind_pose[:, 7:8]  # (127, 1)
        ibp_rot = quat_to_rotmat(ibp_quat)  # (127, 3, 3)

        # For each skinning entry (vert_idx, joint_idx, weight):
        # The transform for joint j applied to a vertex v:
        # v_local = ibp_scale[j] * (ibp_rot[j] @ v) + ibp_trans[j]  (bind pose inverse)
        # v_global = global_scale[j] * (global_rot[j] @ v_local) + global_pos[j]

        # Precompute combined transform per joint:
        # combined_rot[j] = global_rot[j] @ ibp_rot[j]  (broadcast over batch)
        # combined_trans[j] = global_pos[j] + global_rot[j] @ ibp_trans[j]
        # combined_scale[j] = global_scale[j] * ibp_scale[j]

        # ibp_rot: (127, 3, 3), global_rot: (B, 127, 3, 3)
        combined_rot = mx.einsum(
            "bjik,jkl->bjil", global_rot, ibp_rot
        )  # (B, 127, 3, 3)
        # global_pos: (B, 127, 3), ibp_trans: (127, 3)
        combined_trans = (
            global_pos + mx.einsum("bjik,jk->bji", global_rot, ibp_trans) * global_scale
        )  # (B, 127, 3)
        combined_scale = global_scale * ibp_scale[None, :, :]  # (B, 127, 1)

        # Now for each skinning triple, compute:
        # v_posed = sum_over_joints(weight * (combined_scale * combined_rot @ v + combined_trans))

        # Gather the relevant data per skinning entry
        si = self.skin_indices  # (N,) joint indices
        sw = self.skin_weights  # (N,) weights
        vi = self.vert_indices  # (N,) vertex indices
        N = si.shape[0]

        # Gather vertex positions for each entry
        # rest_verts: (B, V, 3), vi: (N,)
        v = rest_verts[:, vi, :]  # (B, N, 3)

        # Gather combined transforms for each entry's joint
        cr = combined_rot[:, si, :, :]  # (B, N, 3, 3)
        ct = combined_trans[:, si, :]  # (B, N, 3)
        cs = combined_scale[:, si, :]  # (B, N, 1)

        # Transform each vertex
        v_transformed = cs * mx.einsum("bnij,bnj->bni", cr, v) + ct  # (B, N, 3)

        # Weight
        v_weighted = v_transformed * sw[None, :, None]  # (B, N, 3)

        # Scatter-add into output
        # Use scatter add: for each entry i, add v_weighted[:, i] to output[:, vi[i]]
        posed_verts = mx.zeros((B, self.num_verts, 3))

        # MLX doesn't have scatter_add directly. Use a loop-free approach:
        # Reshape vi for indexing: we need to accumulate across the N dimension
        # grouped by vertex index.
        #
        # Since we can't do true scatter-add in MLX, we use the fact that
        # vert_indices is sorted (or can be sorted) and accumulate manually.
        # Actually, we can use mx.scatter_add-like behavior with:
        vi_expanded = vi[None, :, None].astype(mx.int32)  # (1, N, 1)
        vi_expanded = mx.broadcast_to(vi_expanded, (B, N, 3))

        # For scatter add, iterate over batch (B is typically 1-2)
        results = []
        for b in range(B):
            out = mx.zeros((self.num_verts, 3))
            # Group by vertex index using numpy-style accumulation
            # This is equivalent to scatter_add
            w = v_weighted[b]  # (N, 3)
            # Use put_along_axis or manual accumulation
            # MLX approach: build sparse accumulation
            for d in range(3):
                vals = w[:, d]  # (N,)
                idx = vi  # (N,)
                # Accumulate: out[:, d] = scatter_add(vals, idx, num_verts)
                col = _scatter_add_1d(vals, idx, self.num_verts)
                out = _set_column_2d(out, d, col)
            results.append(out)

        posed_verts = mx.stack(results, axis=0)  # (B, V, 3)
        return posed_verts

    def _blend_shapes(
        self,
        shape_params: mx.array,
        expr_params: mx.array = None,
    ) -> mx.array:
        """Apply blend shapes and face expressions.

        Args:
            shape_params: (B, 45)
            expr_params: (B, 72) or None
        Returns:
            verts: (B, num_verts, 3)
        """
        verts = self.base_shape[None] + mx.einsum(
            "bs,svd->bvd", shape_params, self.shape_vectors
        )
        if expr_params is not None:
            verts = verts + mx.einsum(
                "bf,fvd->bvd", expr_params, self.face_shape_vectors
            )
        return verts

    def _pose_features_from_joint_dofs(self, joint_dofs: mx.array) -> mx.array:
        """Convert 889D joint DOFs to 750D pose features for correctives.

        Matches PyTorch's _pose_features_from_joint_params:
        1. Reshape to (B, 127, 7)
        2. Skip joints 0,1 → take euler angles (B, 125, 3)
        3. Convert to 6D rotation (first two columns of rotation matrix)
        4. Subtract identity (elements 0 and 4)
        5. Flatten to (B, 750)
        """
        B = joint_dofs.shape[0]
        jd = joint_dofs.reshape(B, self.num_joints, 7)  # (B, 127, 7)
        euler = jd[:, 2:, 3:6]  # (B, 125, 3) — skip root and joint 1

        # Build rotation matrix columns from euler XYZ (R = Rz @ Ry @ Rx)
        cx = mx.cos(euler[..., 0])
        sx = mx.sin(euler[..., 0])
        cy = mx.cos(euler[..., 1])
        sy = mx.sin(euler[..., 1])
        cz = mx.cos(euler[..., 2])
        sz = mx.sin(euler[..., 2])

        # 6D = [R00, R10, R20, R01, R11, R21]
        r00 = cy * cz
        r10 = cy * sz
        r20 = -sy
        r01 = -cx * sz + sx * sy * cz
        r11 = cx * cz + sx * sy * sz
        r21 = sx * cy

        feat = mx.stack([r00, r10, r20, r01, r11, r21], axis=-1)  # (B, 125, 6)

        # Subtract identity: R00 - 1 and R11 - 1
        identity_sub = mx.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        feat = feat - identity_sub

        return feat.reshape(B, -1)  # (B, 750)

    def _pose_correctives(
        self,
        joint_dofs: mx.array,
        num_verts: int,
    ) -> mx.array:
        """Compute pose-dependent vertex corrections.

        Pipeline: joint_dofs → 6D pose features (750D) → sparse → ReLU → dense
        """
        B = joint_dofs.shape[0]

        # Convert joint DOFs to 6D pose features (matches PyTorch preprocessing)
        pose_feats = self._pose_features_from_joint_dofs(joint_dofs)  # (B, 750)

        # Sparse layer: indices (2, K), weights (K,)
        out_indices = self.pc_sparse_indices[0]  # (K,)
        in_indices = self.pc_sparse_indices[1]  # (K,)
        weights = self.pc_sparse_weight  # (K,)

        # Gather input values from pose features
        input_vals = pose_feats[:, in_indices]  # (B, K)
        weighted = input_vals * weights[None, :]  # (B, K)

        # Scatter-add into sparse output
        out_size = 3000
        sparse_out_list = []
        for b in range(B):
            col = _scatter_add_1d(weighted[b], out_indices, out_size)
            sparse_out_list.append(col)
        sparse_out = mx.stack(sparse_out_list, axis=0)  # (B, 3000)

        # ReLU
        sparse_out = nn.relu(sparse_out)

        # Dense layer: (55317, 3000) -> (B, 55317)
        dense_out = sparse_out @ self.pc_linear_weight.T  # (B, 55317)

        # Reshape to (B, V, 3) where V = 55317 / 3 = 18439
        corrections = dense_out.reshape(B, -1, 3)  # (B, 18439, 3)

        return corrections

    def __call__(
        self,
        shape_params: mx.array,
        model_params: mx.array,
        expr_params: mx.array = None,
    ):
        """Forward pass of MHR body model.

        Args:
            shape_params: (B, 45)
            model_params: (B, 204) [trans(3), rot(3), body(130), scales(68)]
            expr_params: (B, 72) or None

        Returns:
            skinned_verts: (B, 18439, 3)
            skel_state: (B, 127, 8) [x, y, z, qx, qy, qz, qw, scale]
        """
        # 1. Parameter transform: 204 -> 889 joint DOFs
        # NOTE: parameter_limits is NOT applied during inference.
        # The JIT model skips it entirely in its forward pass.
        joint_dofs = self._parameter_transform(model_params)

        # 3. Blend shapes
        verts = self._blend_shapes(shape_params, expr_params)

        # 4. Pose correctives
        corrections = self._pose_correctives(joint_dofs, self.num_verts)
        verts = verts + corrections

        # 5. Forward kinematics
        skel_state, global_pos, global_rot, global_scale = self._forward_kinematics(
            joint_dofs
        )

        # 6. Linear blend skinning
        skinned_verts = self._linear_blend_skinning(
            verts, global_pos, global_rot, global_scale
        )

        return skinned_verts, skel_state


def _scatter_add_1d(values: mx.array, indices: mx.array, size: int) -> mx.array:
    """Scatter-add values into a 1D array of given size.

    values: (N,)
    indices: (N,) int32
    size: output size
    Returns: (size,)

    Note: uses numpy for the scatter-add because MLX lacks a native scatter_add op.
    This causes a GPU->CPU->GPU round-trip per skinning call. Functionally correct
    but a performance bottleneck for video pipelines.
    """
    import numpy as np

    idx_np = np.array(indices, copy=False)
    val_np = np.array(values, copy=False)
    out_np = np.zeros(size, dtype=np.float32)
    np.add.at(out_np, idx_np, val_np)
    return mx.array(out_np)


def _set_column(arr: mx.array, col: int, values: mx.array) -> mx.array:
    """Set a column in a 2D array (functional, returns new array)."""
    cols = []
    for c in range(arr.shape[1]):
        if c == col:
            cols.append(values[:, None] if values.ndim == 1 else values)
        else:
            cols.append(arr[:, c : c + 1])
    return mx.concatenate(cols, axis=1)


def _set_column_2d(arr: mx.array, col: int, values: mx.array) -> mx.array:
    """Set a column in a 2D array where values is 1D."""
    cols = []
    for c in range(arr.shape[1]):
        if c == col:
            cols.append(values[:, None])
        else:
            cols.append(arr[:, c : c + 1])
    return mx.concatenate(cols, axis=1)

"""MHR pose head: proj FFN -> parameter extraction -> body model."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import SAM3DConfig
from .mhr_body import MHRBodyModel
from .mhr_utils import (
    MHR_PARAM_HAND_IDXS,
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    rot6d_to_rotmat,
    rotmat_to_euler_ZYX,
)
from .transformer import DecoderFFN


class MHRHead(nn.Module):
    """MHR pose prediction head.

    Takes a (B, 1024) pose token from the decoder and predicts body parameters,
    then runs the MHR body model to get vertices and keypoints.

    Weight key mapping:
        proj.layers.0.0.{weight,bias}: (1024, 1024) - first linear + ReLU
        proj.layers.1.{weight,bias}: (519, 1024) - output linear

    Buffers (loaded as frozen params):
        joint_rotation: (127, 3, 3)
        scale_mean: (68,)
        scale_comps: (28, 68)
        faces: (36874, 3) int32
        hand_pose_mean: (54,)
        hand_pose_comps: (54, 54)
        hand_joint_idxs_left: (27,) int32
        hand_joint_idxs_right: (27,) int32
        keypoint_mapping: (308, 18566)
        right_wrist_coords: (3,)
        root_coords: (3,)
        local_to_world_wrist: (3, 3)
        nonhand_param_idxs: (145,) int32
    """

    def __init__(self, input_dim: int = 1024, config: SAM3DConfig = None):
        super().__init__()
        if config is None:
            config = SAM3DConfig()

        self.config = config
        output_dim = config.pose_output_dim  # 519

        # Proj FFN: same nested list pattern as decoder FFN
        self.proj = DecoderFFN(input_dim, input_dim)
        # Override output layer to have 519 output dim instead of input_dim
        self.proj.layers[1] = nn.Linear(input_dim, output_dim)

        # Body model
        self.body_model = MHRBodyModel(
            num_joints=config.num_joints,
            num_verts=config.num_vertices,
        )

        # Buffers (frozen, loaded from weights)
        self.joint_rotation = mx.zeros((config.num_joints, 3, 3))
        self.scale_mean = mx.zeros((68,))
        self.scale_comps = mx.zeros((28, 68))
        self.faces = mx.zeros((config.num_faces, 3), dtype=mx.int32)
        self.hand_pose_mean = mx.zeros((54,))
        self.hand_pose_comps = mx.zeros((54, 54))
        self.hand_joint_idxs_left = mx.zeros((27,), dtype=mx.int32)
        self.hand_joint_idxs_right = mx.zeros((27,), dtype=mx.int32)
        self.keypoint_mapping = mx.zeros((308, 18566))
        self.right_wrist_coords = mx.zeros((3,))
        self.root_coords = mx.zeros((3,))
        self.local_to_world_wrist = mx.zeros((3, 3))
        self.nonhand_param_idxs = mx.zeros((145,), dtype=mx.int32)

        # Precompute hand param mask
        self._hand_mask = mx.array(MHR_PARAM_HAND_IDXS, dtype=mx.int32)

    def _replace_hands_in_pose(
        self,
        full_pose_params: mx.array,
        hand_pose_params: mx.array,
    ) -> mx.array:
        """Decode hand PCA params and insert into full pose vector.

        full_pose_params: (B, 136) = [trans(3), rot(3), body(130)]
        hand_pose_params: (B, 108) = [left(54), right(54)]

        The hand_joint_idxs_left/right index into the 136-dim pose vector.
        """
        B = full_pose_params.shape[0]
        left_cont = hand_pose_params[:, :54]
        right_cont = hand_pose_params[:, 54:]

        # PCA decode: mean + coeffs @ components
        left_decoded = compact_cont_to_model_params_hand(
            self.hand_pose_mean[None, :] + left_cont @ self.hand_pose_comps
        )  # (B, 27)
        right_decoded = compact_cont_to_model_params_hand(
            self.hand_pose_mean[None, :] + right_cont @ self.hand_pose_comps
        )  # (B, 27)

        # Insert decoded hand params at the correct indices
        # hand_joint_idxs_left/right: (27,) indices into 136-dim pose vector
        result = _scatter_set(full_pose_params, self.hand_joint_idxs_left, left_decoded)
        result = _scatter_set(result, self.hand_joint_idxs_right, right_decoded)
        return result

    def __call__(
        self,
        x: mx.array,
        init_estimate: mx.array = None,
    ) -> dict:
        """Forward pass.

        Args:
            x: (B, 1024) pose token from decoder
            init_estimate: (B, 519) optional initial parameter estimate

        Returns:
            dict with pred_vertices, pred_keypoints_3d, pred_joint_coords,
            pred_model_params, pred_shape
        """
        # Project to parameter space
        pred = self.proj(x)  # (B, 519)
        if init_estimate is not None:
            pred = pred + init_estimate

        # Split predictions
        global_rot_6d = pred[:, :6]
        pred_pose_cont = pred[:, 6:266]  # 260D continuous body pose
        pred_shape = pred[:, 266:311]  # 45D shape params
        pred_scale = pred[:, 311:339]  # 28D scale PCA coefficients
        pred_hand = pred[:, 339:447]  # 108D hand params
        pred_face = pred[:, 447:519] * 0  # 72D face, zeroed

        # Global rotation: 6D -> rotmat -> euler
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)  # (B, 3, 3)
        global_rot_euler = rotmat_to_euler_ZYX(global_rot_rotmat)  # (B, 3)
        B = x.shape[0]
        global_trans = mx.zeros((B, 3))

        # Body pose: 260D continuous -> 133D euler
        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)  # (B, 133)

        # Zero out hand params in body pose (they come from the hand head)
        pred_pose_euler = _zero_at_indices(pred_pose_euler, self._hand_mask)

        # Zero out jaw (last 3 params)
        pred_pose_euler = _zero_last_n(pred_pose_euler, 3)

        # Take first 130 body params (exclude translations which are in last 6)
        body_pose_params = pred_pose_euler[:, :130]

        # Scale decomposition: mean + PCA
        scales = self.scale_mean[None, :] + pred_scale @ self.scale_comps  # (B, 68)

        # Assemble full pose: [trans*10, rot, body_pose]
        full_pose_params = mx.concatenate(
            [
                global_trans * 10,
                global_rot_euler,
                body_pose_params,
            ],
            axis=1,
        )  # (B, 136)

        # Replace hand joints with decoded PCA hand params
        full_pose_params = self._replace_hands_in_pose(full_pose_params, pred_hand)

        # Combine with scales for full model params
        model_params = mx.concatenate([full_pose_params, scales], axis=1)  # (B, 204)

        # Run body model
        skinned_verts, skel_state = self.body_model(pred_shape, model_params, pred_face)

        # Parse skeleton state
        joint_coords = skel_state[:, :, :3]  # (B, 127, 3)

        # Scale from centimeters to meters
        verts = skinned_verts / 100.0
        joint_coords = joint_coords / 100.0

        # Compute keypoints via mapping
        # keypoint_mapping: (308, 18566) where 18566 = 18439 verts + 127 joints
        model_vert_joints = mx.concatenate(
            [verts, joint_coords], axis=1
        )  # (B, 18566, 3)

        keypoints = mx.einsum(
            "kv,bvd->bkd", self.keypoint_mapping, model_vert_joints
        )  # (B, 308, 3)
        keypoints = keypoints[:, :70]  # Take first 70

        # Flip Y, Z for camera coordinate system
        verts = _flip_yz(verts)
        keypoints = _flip_yz(keypoints)
        joint_coords = _flip_yz(joint_coords)

        return {
            "pred_vertices": verts,
            "pred_keypoints_3d": keypoints,
            "pred_joint_coords": joint_coords,
            "pred_model_params": model_params,
            "pred_shape": pred_shape,
        }

    def load_all_weights(self, safetensors_path: str):
        """Load head_pose and mhr body model weights from safetensors.

        Handles the key remapping from safetensors structure to model parameters.
        """
        from safetensors import safe_open

        # Key mapping: safetensors prefix -> (model prefix, attr remap)
        MHR_KEY_MAP = {
            "character.skeleton.joint_translation_offsets": "joint_translation_offsets",
            "character.skeleton.joint_prerotations": "joint_prerotations",
            "character.skeleton.joint_parents": "joint_parents",
            "character.skeleton.pmi": None,  # skip — not used at inference
            "character.mesh.rest_vertices": None,  # skip — skinning uses blend_shape result
            "character.mesh.faces": None,  # skip
            "character.mesh.texcoords": None,  # skip
            "character.mesh.texcoord_faces": None,  # skip
            "character.parameter_transform.parameter_transform": "parameter_transform",
            "character.parameter_transform.pose_parameters": "pose_parameters",
            "character.parameter_transform.rigid_parameters": "rigid_parameters",
            "character.parameter_transform.scaling_parameters": "scaling_parameters",
            "character.parameter_limits.minmax_min": "minmax_min",
            "character.parameter_limits.minmax_max": "minmax_max",
            "character.parameter_limits.minmax_weight": "minmax_weight",
            "character.parameter_limits.minmax_parameter_index": "minmax_parameter_index",
            "character.parameter_limits.ellipsoid_ellipsoid": None,
            "character.parameter_limits.ellipsoid_ellipsoid_inv": None,
            "character.parameter_limits.ellipsoid_offset": None,
            "character.blend_shape.base_shape": "base_shape",
            "character.blend_shape.shape_vectors": "shape_vectors",
            "character.linear_blend_skinning.inverse_bind_pose": "inverse_bind_pose",
            "character.linear_blend_skinning.skin_indices_flattened": "skin_indices",
            "character.linear_blend_skinning.skin_weights_flattened": "skin_weights",
            "character.linear_blend_skinning.vert_indices_flattened": "vert_indices",
            "face_expressions.shape_vectors": "face_shape_vectors",
            "pose_correctives.pose_dirs_predictor.0.sparse_indices": "pc_sparse_indices",
            "pose_correctives.pose_dirs_predictor.0.sparse_weight": "pc_sparse_weight",
            "pose_correctives.pose_dirs_predictor.2.weight": "pc_linear_weight",
        }

        weights = []
        with safe_open(safetensors_path, framework="numpy") as f:
            for key in f.keys():
                tensor = mx.array(f.get_tensor(key))

                if key.startswith("head_pose.") and not key.startswith(
                    "head_pose_hand."
                ):
                    model_key = key[len("head_pose.") :]
                    weights.append((model_key, tensor))

                elif key.startswith("mhr."):
                    mhr_key = key[len("mhr.") :]
                    mapped = MHR_KEY_MAP.get(mhr_key)
                    if mapped is None:
                        continue  # skip unused buffers
                    weights.append(("body_model." + mapped, tensor))

        self.load_weights(weights)


def _flip_yz(x: mx.array) -> mx.array:
    """Negate Y and Z axes: x[..., 1] *= -1, x[..., 2] *= -1."""
    sign = mx.array([1.0, -1.0, -1.0])
    return x * sign


def _zero_at_indices(arr: mx.array, indices: mx.array) -> mx.array:
    """Zero out specific column indices in a (B, D) array.

    Since MLX arrays are immutable, we build a mask.
    """
    D = arr.shape[1]
    mask = mx.ones((D,))
    # Build mask with zeros at specified indices
    mask_np = np.ones(D, dtype=np.float32)
    idx_np = np.array(indices, copy=False)
    mask_np[idx_np] = 0.0
    mask = mx.array(mask_np)
    return arr * mask[None, :]


def _zero_last_n(arr: mx.array, n: int) -> mx.array:
    """Zero out the last n columns of a (B, D) array."""
    D = arr.shape[1]
    mask = mx.concatenate(
        [
            mx.ones((D - n,)),
            mx.zeros((n,)),
        ]
    )
    return arr * mask[None, :]


def _scatter_set(
    arr: mx.array,
    indices: mx.array,
    values: mx.array,
) -> mx.array:
    """Set arr[:, indices[i]] = values[:, i] for each i.

    Functional replacement for arr.at[:, indices].set(values).
    Works by building a full replacement with masking.
    """
    B, D = arr.shape
    _, K = values.shape

    idx_np = np.array(indices, copy=False).astype(np.int64)

    # Build mask: 1 at index positions, 0 elsewhere
    mask_np = np.zeros(D, dtype=np.float32)
    mask_np[idx_np] = 1.0
    mask = mx.array(mask_np)  # (D,)

    # Build replacement array: scatter values into correct positions
    replacement_np = np.zeros((1, D), dtype=np.float32)
    # We need per-batch replacement, but since indices are fixed, we can
    # build a gather-scatter mapping

    # Create a mapping: for each output column d, which input column k provides it?
    # Only valid for columns in idx_np
    col_map_np = np.zeros(D, dtype=np.int64)
    for k, d in enumerate(idx_np):
        col_map_np[d] = k

    col_map = mx.array(col_map_np)  # (D,)

    # Gather from values using col_map, then mask
    # values: (B, K), we need to gather along axis 1
    # For columns not in indices, the gathered value doesn't matter (masked out)
    full_values = values[:, col_map]  # (B, D)

    # Blend: keep original where mask=0, use full_values where mask=1
    result = arr * (1 - mask[None, :]) + full_values * mask[None, :]
    return result

# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from Transformers' MiniCPM-V 4.7 canvas M-RoPE implementation.

import itertools

import numpy as np


class CanvasRoPE:
    """Lay image thumbnails, slices, and video frames on a shared 3-D canvas."""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _frame_end_idx(last_crop_end_idx, input_ids, markers, limit):
        end_idx = last_crop_end_idx
        while end_idx < limit and input_ids[end_idx] in markers:
            end_idx += 1
        return end_idx

    def _group_visual_frames(self, input_ids, mm_token_type_ids):
        markers = (
            self.config.slice_start_id,
            self.config.slice_end_id,
            self.config.image_start_id,
            self.config.image_end_id,
            self.config.newline_id,
        )
        slice_start_id = markers[0]
        seq_len = len(input_ids)
        groups = []
        open_slices = None
        previous_end_idx = 0
        for modality, run in itertools.groupby(
            enumerate(mm_token_type_ids), lambda item: item[1]
        ):
            if modality == 0:
                continue
            run = list(run)
            crop = (run[0][0], run[-1][0] + 1)
            if open_slices is not None and input_ids[crop[0] - 1] == slice_start_id:
                open_slices.append(crop)
            else:
                open_slices = []
                frame = (modality, crop[0], crop[1], open_slices)
                gap = input_ids[previous_end_idx : crop[0] - 1]
                adjacent = bool(groups) and all(
                    (token_id in markers for token_id in gap)
                )
                if adjacent:
                    groups[-1][1].append(frame)
                else:
                    groups.append((crop[0] - 1, [frame]))
            previous_end_idx = crop[1]
        visual_groups = []
        for group_start_idx, frames in groups:
            frames = [
                (
                    modality,
                    thumb_start_idx,
                    thumb_end_idx,
                    self._frame_end_idx(
                        slices[-1][1] if slices else thumb_end_idx,
                        input_ids,
                        markers,
                        seq_len,
                    ),
                    slices,
                )
                for modality, thumb_start_idx, thumb_end_idx, slices in frames
            ]
            group_end_idx = frames[-1][3]
            visual_groups.append((group_start_idx, group_end_idx, frames))
        return visual_groups

    def get_vision_position_ids(
        self,
        start_position,
        grid_thw,
        canvas_height=0,
        canvas_width=0,
        h_offset=0,
        w_offset=0,
        spatial_merge_size=1,
        device=None,
    ):
        llm_grid_h = int(grid_thw[0]) // spatial_merge_size
        llm_grid_w = int(grid_thw[1]) // spatial_merge_size
        canvas_height = canvas_height or llm_grid_h
        canvas_width = canvas_width or llm_grid_w
        h_coords = (
            np.rint(np.linspace(0, canvas_height - 1, llm_grid_h)).astype(np.int32)
            + h_offset
        )
        w_coords = (
            np.rint(np.linspace(0, canvas_width - 1, llm_grid_w)).astype(np.int32)
            + w_offset
        )
        H_grid, W_grid = np.meshgrid(h_coords, w_coords, indexing="ij")
        T_grid = np.zeros_like(H_grid)
        position_ids = np.stack([T_grid, H_grid, W_grid], axis=0).reshape(3, -1)
        position_ids += start_position
        return position_ids

    def get_rope_index(
        self,
        input_ids,
        mm_token_type_ids=None,
        target_sizes=None,
        target_sizes_videos=None,
        downsample_mode=None,
        attention_mask=None,
        **kwargs,
    ):
        downsample_mode = downsample_mode or self.config.downsample_mode
        merge_factor = 2 if downsample_mode == "4x" else 4
        input_ids = np.asarray(input_ids)
        mm_token_type_ids = np.asarray(mm_token_type_ids)
        position_ids = np.zeros((3, *input_ids.shape), dtype=np.int32)
        grid_iters = {
            1: iter(target_sizes.tolist()) if target_sizes is not None else None,
            2: (
                iter(target_sizes_videos.tolist())
                if target_sizes_videos is not None
                else None
            ),
        }
        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx]
            current_mm_token_type_ids = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                valid_mask = np.asarray(attention_mask[batch_idx], dtype=bool)
                current_input_ids = current_input_ids[valid_mask]
                current_mm_token_type_ids = current_mm_token_type_ids[valid_mask]
            else:
                valid_mask = slice(None)
            current_input_ids = current_input_ids.tolist()
            current_mm_token_type_ids = current_mm_token_type_ids.tolist()
            seq_len = len(current_input_ids)
            curr_position_ids = np.broadcast_to(np.arange(seq_len), (3, seq_len)).copy()
            current_pos = 0
            current_idx = 0
            for group_start_idx, group_end_idx, frames in self._group_visual_frames(
                current_input_ids, current_mm_token_type_ids
            ):
                if group_start_idx > current_idx:
                    text_len = group_start_idx - current_idx
                    curr_position_ids[:, current_idx:group_start_idx] = (
                        np.arange(text_len) + current_pos
                    )
                    current_pos += text_len
                frame_cursor_idx = group_start_idx
                for (
                    modality,
                    thumb_start_idx,
                    thumb_end_idx,
                    frame_end_idx,
                    slices,
                ) in frames:
                    target_sizes_thumb = next(grid_iters[modality])
                    frame_start_idx = thumb_start_idx - 1
                    if frame_start_idx > frame_cursor_idx:
                        gap_len = frame_start_idx - frame_cursor_idx
                        curr_position_ids[:, frame_cursor_idx:frame_start_idx] = (
                            np.arange(gap_len) + current_pos
                        )
                        current_pos += gap_len + 1
                    frame_end_idx = min(frame_end_idx, group_end_idx)
                    canvas_origin = current_pos
                    halo_before_canvas = max(canvas_origin - 1, 0)
                    llm_slice_h, llm_slice_w = (0, 0)
                    num_rows, num_cols = (0, 0)
                    canvas_height = target_sizes_thumb[0] // merge_factor
                    canvas_width = target_sizes_thumb[1] // merge_factor
                    if slices:
                        num_cols = len(slices)
                        for k in range(len(slices) - 1):
                            if slices[k + 1][0] - slices[k][1] > 2:
                                num_cols = k + 1
                                break
                        num_rows = len(slices) // num_cols if num_cols > 0 else 1
                        if num_rows * num_cols != len(slices):
                            num_rows, num_cols = (1, len(slices))
                        target_sizes_first_slice = next(grid_iters[modality])
                        llm_slice_h = target_sizes_first_slice[0] // merge_factor
                        llm_slice_w = target_sizes_first_slice[1] // merge_factor
                        canvas_height = num_rows * llm_slice_h
                        canvas_width = num_cols * llm_slice_w
                    curr_position_ids[:, frame_start_idx:frame_end_idx] = canvas_origin
                    curr_position_ids[1:, frame_start_idx] = halo_before_canvas
                    if thumb_end_idx < frame_end_idx:
                        curr_position_ids[1, thumb_end_idx] = (
                            canvas_origin + canvas_height
                        )
                        curr_position_ids[2, thumb_end_idx] = (
                            canvas_origin + canvas_width
                        )
                    curr_position_ids[:, thumb_start_idx:thumb_end_idx] = (
                        self.get_vision_position_ids(
                            start_position=canvas_origin,
                            grid_thw=target_sizes_thumb,
                            canvas_height=canvas_height,
                            canvas_width=canvas_width,
                            spatial_merge_size=merge_factor,
                        )
                    )
                    for k, (slice_start_idx, slice_end_idx) in enumerate(slices):
                        h_off = k // num_cols * llm_slice_h
                        w_off = k % num_cols * llm_slice_w
                        target_sizes_slice = (
                            target_sizes_first_slice
                            if k == 0
                            else next(grid_iters[modality])
                        )
                        slice_h = target_sizes_slice[0] // merge_factor
                        slice_w = target_sizes_slice[1] // merge_factor
                        slice_start_marker_idx = slice_start_idx - 1
                        if slice_start_marker_idx >= frame_start_idx:
                            curr_position_ids[1, slice_start_marker_idx] = (
                                canvas_origin + h_off
                            )
                            curr_position_ids[2, slice_start_marker_idx] = (
                                canvas_origin + w_off
                            )
                        if slice_end_idx < frame_end_idx:
                            curr_position_ids[1, slice_end_idx] = (
                                canvas_origin + h_off + slice_h - 1
                            )
                            curr_position_ids[2, slice_end_idx] = (
                                canvas_origin + w_off + slice_w - 1
                            )
                        curr_position_ids[:, slice_start_idx:slice_end_idx] = (
                            self.get_vision_position_ids(
                                start_position=canvas_origin,
                                grid_thw=target_sizes_slice,
                                h_offset=h_off,
                                w_offset=w_off,
                                spatial_merge_size=merge_factor,
                            )
                        )
                    for k in range(len(slices) - 1):
                        gap_start_idx, gap_end_idx = (slices[k][1], slices[k + 1][0])
                        if gap_end_idx - gap_start_idx <= 2:
                            continue
                        boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                        right_edge_w = num_cols * llm_slice_w
                        for newline_idx in range(gap_start_idx + 1, gap_end_idx - 1):
                            curr_position_ids[1, newline_idx] = (
                                canvas_origin + boundary_h
                            )
                            curr_position_ids[2, newline_idx] = (
                                canvas_origin + right_edge_w
                            )
                    current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
                    frame_cursor_idx = frame_end_idx
                if frame_cursor_idx < group_end_idx:
                    trail_len = group_end_idx - frame_cursor_idx
                    curr_position_ids[:, frame_cursor_idx:group_end_idx] = (
                        np.arange(trail_len) + current_pos
                    )
                    current_pos += trail_len
                current_idx = group_end_idx
            if current_idx < seq_len:
                curr_position_ids[:, current_idx:seq_len] = (
                    np.arange(seq_len - current_idx) + current_pos
                )
            position_ids[:, batch_idx][:, valid_mask] = curr_position_ids
        if attention_mask is not None:
            seq_lens = np.asarray(attention_mask).sum(-1)
        else:
            seq_lens = np.full((input_ids.shape[0],), input_ids.shape[1])
        rope_deltas = position_ids.max(axis=(0, 2))[:, None] + 1 - seq_lens[:, None]
        return position_ids, rope_deltas.astype(np.int32)

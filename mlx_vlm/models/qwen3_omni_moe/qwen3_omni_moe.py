from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_omni_moe.code2wav import Code2WavModel
from mlx_vlm.models.qwen3_omni_moe.talker import Talker
from mlx_vlm.models.qwen3_omni_moe.thinker import Thinker

from .config import ModelConfig


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)

    return final_embedding


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.thinker = Thinker(config.thinker_config)
        self.has_talker = config.enable_audio_output
        if self.has_talker:
            self.talker = Talker(config.talker_config)
            self.code2wav = Code2WavModel(config.code2wav_config)
        else:
            self.talker = None
            self.code2wav = None

    def enable_talker(self):
        if not self.has_talker:
            self.talker = Talker(self.config.talker_config)
            self.code2wav = Code2WavModel(self.config.code2wav_config)
            self.has_talker = True

    def disable_talker(self):
        if self.has_talker:
            self.talker = None
            self.code2wav = None
            self.has_talker = False

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        audio_feature_lengths: Optional[mx.array] = None,
        **kwargs,
    ):
        return self.thinker.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=input_features_mask,
            audio_feature_lengths=audio_feature_lengths,
        )

    def get_audio_features(
        self,
        input_features: mx.array,
        input_features_mask: Optional[mx.array] = None,
        audio_feature_lengths: Optional[mx.array] = None,
    ):
        return self.thinker.get_audio_features(
            input_features=input_features,
            feature_attention_mask=input_features_mask,
            audio_feature_lengths=audio_feature_lengths,
        )

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
    ):
        dtype = self.thinker.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        vision_output = self.thinker.vision_tower(pixel_values, image_grid_thw)
        if isinstance(vision_output, tuple):
            return vision_output[0]
        return vision_output

    @property
    def layers(self):
        return self.thinker.language_model.layers

    def extract_thinker_hidden_states(self, input_ids, target_layer_idx, **kwargs):
        embed_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
            ]
        }
        input_embedding_features = self.thinker.get_input_embeddings(
            input_ids, **embed_kwargs
        )
        inputs_embeds = input_embedding_features.inputs_embeds

        lm_kwargs = {
            k: v for k, v in kwargs.items() if k in ["image_grid_thw", "video_grid_thw"]
        }

        outputs = self.thinker.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            **lm_kwargs,
        )

        hidden_states = outputs.hidden_states[target_layer_idx + 1]

        return hidden_states, inputs_embeds

    def _get_talker_user_parts(
        self,
        im_start_index: int,
        segment_end_index: int,
        multimodal_mask: mx.array,
        thinker_hidden: mx.array,
        thinker_embed: mx.array,
    ):
        seq_len = segment_end_index - im_start_index
        user_talker_part = mx.zeros(
            (1, seq_len, self.config.talker_config.text_config.hidden_size),
            dtype=thinker_embed.dtype,
        )
        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]
        user_thinker_hidden_mm = thinker_hidden[:, im_start_index:segment_end_index]
        user_thinker_embed_seg = thinker_embed[:, im_start_index:segment_end_index]

        if mx.any(user_mm_mask):
            mm_indices = mx.array(
                np.where(np.array(mx.reshape(user_mm_mask, (-1,))))[0]
            )
            user_thinker_hidden_mm_flat = mx.reshape(
                user_thinker_hidden_mm, (-1, user_thinker_hidden_mm.shape[-1])
            )
            mm_hidden_flat = mx.take(user_thinker_hidden_mm_flat, mm_indices, axis=0)
            mm_hidden = self.talker.hidden_projection(mm_hidden_flat)
            user_talker_part_flat = mx.reshape(
                user_talker_part, (-1, user_talker_part.shape[-1])
            )
            user_talker_part_flat[mm_indices] = mm_hidden
            user_talker_part = mx.reshape(user_talker_part_flat, user_talker_part.shape)

        text_mask = ~user_mm_mask
        if mx.any(text_mask):
            text_indices = mx.array(np.where(np.array(mx.reshape(text_mask, (-1,))))[0])
            user_thinker_embed_flat = mx.reshape(
                user_thinker_embed_seg, (-1, user_thinker_embed_seg.shape[-1])
            )
            text_embed_flat = mx.take(user_thinker_embed_flat, text_indices, axis=0)
            user_text_hidden = self.talker.text_projection(text_embed_flat)
            user_talker_part_flat = mx.reshape(
                user_talker_part, (-1, user_talker_part.shape[-1])
            )
            user_talker_part_flat[text_indices] = user_text_hidden
            user_talker_part = mx.reshape(user_talker_part_flat, user_talker_part.shape)

        return user_talker_part

    def _get_talker_assistant_parts(
        self,
        im_start_index: int,
        segment_end_index: int,
        speaker_id: int,
        thinker_embed: mx.array,
        tts_pad_embed: mx.array,
        tts_bos_embed: mx.array,
        tts_eos_embed: mx.array,
    ):
        assistant_hidden = self.talker.text_projection(
            thinker_embed[:, im_start_index:segment_end_index]
        )
        assistant_text_hidden = mx.concatenate(
            (
                assistant_hidden[:, :3],
                mx.broadcast_to(tts_pad_embed, (1, 4, tts_pad_embed.shape[-1])),
                tts_bos_embed,
                assistant_hidden[:, 3:4],
            ),
            axis=1,
        )
        codec_special_tokens = mx.array(
            [
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    speaker_id,
                    self.config.talker_config.codec_pad_id,
                    self.config.talker_config.codec_bos_id,
                ]
            ],
            dtype=mx.int32,
        )
        assistant_codec_hidden = mx.concatenate(
            (
                mx.zeros(
                    (1, 3, self.config.talker_config.text_config.hidden_size),
                    dtype=thinker_embed.dtype,
                ),
                self.talker.model.codec_embedding(codec_special_tokens),
            ),
            axis=1,
        )
        trailing_text_hidden = mx.concatenate(
            (
                assistant_hidden[:, 4:],
                tts_eos_embed,
            ),
            axis=1,
        )
        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = mx.full(
            (1, assistant_text_hidden.shape[1]),
            self.config.tts_pad_token_id,
            dtype=mx.int32,
        )
        return input_embeds, input_ids, trailing_text_hidden

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        return self.thinker(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def generate(
        self,
        input_ids: mx.array,
        speaker: str = "Ethan",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 50,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_audio_output in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker

        if not return_audio:
            from mlx_vlm.generate import generate_step

            thinker_kwargs = {
                "max_tokens": thinker_max_new_tokens,
                "eos_tokens": [thinker_eos_token_id],
            }
            for key, value in kwargs.items():
                if key.startswith("thinker_"):
                    thinker_kwargs[key[len("thinker_") :]] = value
                elif key in (
                    "input_features",
                    "feature_attention_mask",
                    "audio_feature_lengths",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                ):
                    thinker_kwargs[key] = value

            generator = generate_step(
                input_ids,
                self.thinker,
                thinker_kwargs.get("pixel_values"),
                kwargs.get("mask"),
                **{
                    k: v
                    for k, v in thinker_kwargs.items()
                    if k not in ("pixel_values", "mask")
                },
            )
            sequences = [input_ids]
            for token, _ in generator:
                sequences.append(mx.array([[token]]))
                if token == thinker_eos_token_id:
                    break
            thinker_result = type(
                "obj",
                (object,),
                {
                    "sequences": mx.concatenate(sequences, axis=1),
                    "hidden_states": None,
                },
            )()
            return thinker_result, None

        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "Qwen3-Omni currently does not support batched inference with audio output"
            )

        speaker_id = self.config.talker_config.speaker_id.get(speaker.lower())
        if speaker_id is None:
            raise NotImplementedError(f"Speaker {speaker} not implemented")

        from mlx_vlm.generate import generate_step

        thinker_kwargs = {
            "max_tokens": thinker_max_new_tokens,
            "eos_tokens": [thinker_eos_token_id],
            "output_hidden_states": True,
        }
        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key in (
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            ):
                thinker_kwargs[key] = value

        generator = generate_step(
            input_ids,
            self.thinker,
            thinker_kwargs.get("pixel_values"),
            kwargs.get("mask"),
            **{
                k: v
                for k, v in thinker_kwargs.items()
                if k not in ("pixel_values", "mask", "output_hidden_states")
            },
        )
        sequences = [input_ids]
        hidden_states_list = []
        for token, _ in generator:
            sequences.append(mx.array([[token]]))
            if token == thinker_eos_token_id:
                break

        thinker_result_sequences = mx.concatenate(sequences, axis=1)

        thinker_hidden_all, thinker_embed_all = self.extract_thinker_hidden_states(
            thinker_result_sequences,
            target_layer_idx=self.config.talker_config.accept_hidden_layer,
            **kwargs,
        )

        im_start_indexes = mx.concatenate(
            (
                mx.array(
                    np.where(np.array(input_ids[0] == self.config.im_start_token_id))[0]
                ),
                mx.array([thinker_result_sequences.shape[-1]], dtype=mx.int32),
            ),
            axis=0,
        )
        multimodal_mask = (
            (thinker_result_sequences == self.config.thinker_config.audio_token_id)
            | (thinker_result_sequences == self.config.thinker_config.image_token_id)
            | (thinker_result_sequences == self.config.thinker_config.video_token_id)
        )

        talker_special_tokens = mx.array(
            [
                [
                    self.config.tts_bos_token_id,
                    self.config.tts_eos_token_id,
                    self.config.tts_pad_token_id,
                ]
            ],
            dtype=input_ids.dtype,
        )
        talker_special_embeds = self.thinker.language_model.model.embed_tokens(
            talker_special_tokens
        )
        talker_special_embeds_proj = self.talker.text_projection(talker_special_embeds)
        tts_bos_embed = talker_special_embeds_proj[:, 0:1]
        tts_eos_embed = talker_special_embeds_proj[:, 1:2]
        tts_pad_embed = talker_special_embeds_proj[:, 2:3]

        talker_input_embeds = []
        talker_input_ids = []

        for i in range(len(im_start_indexes) - 1):
            im_start_index = int(im_start_indexes[i])
            segment_end_index = int(im_start_indexes[i + 1])
            role_token = int(input_ids[0, im_start_index + 1])

            if role_token == self.config.system_token_id:
                continue
            elif role_token == self.config.user_token_id:
                talker_user_part = self._get_talker_user_parts(
                    im_start_index,
                    segment_end_index,
                    multimodal_mask,
                    thinker_hidden_all,
                    thinker_embed_all,
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(
                    thinker_result_sequences[:, im_start_index:segment_end_index]
                )
            elif (
                role_token == self.config.assistant_token_id
                and i == len(im_start_indexes) - 2
            ):
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = (
                    self._get_talker_assistant_parts(
                        im_start_index,
                        segment_end_index,
                        speaker_id,
                        thinker_embed_all,
                        tts_pad_embed,
                        tts_bos_embed,
                        tts_eos_embed,
                    )
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
            elif (
                role_token == self.config.assistant_token_id
                and i != len(im_start_indexes) - 2
            ):
                continue
            else:
                raise AssertionError(
                    "Expect role id after <|im_start|> (assistant, user, system)"
                )

        if len(talker_input_embeds) == 0:
            return (
                type(
                    "obj",
                    (object,),
                    {
                        "sequences": thinker_result_sequences,
                        "hidden_states": None,
                    },
                )(),
                None,
            )

        talker_input_embed = mx.concatenate(talker_input_embeds, axis=1)
        talker_input_id = mx.concatenate(talker_input_ids, axis=1)

        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,
            max_new_tokens=talker_max_new_tokens,
            temperature=talker_temperature,
            top_p=talker_top_p,
        )

        valid_codes = [
            hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None
        ]
        if not valid_codes:
            talker_wavs = mx.zeros((1, 1, 1000))
        else:
            talker_codes = mx.stack(valid_codes, axis=1).transpose(0, 2, 1)
            talker_wavs = self.code2wav.chunked_decode(
                talker_codes, chunk_size=300, left_context_size=25
            )

        thinker_result = type(
            "obj",
            (object,),
            {
                "sequences": thinker_result_sequences,
                "hidden_states": None,
            },
        )()

        return thinker_result, talker_wavs.astype(mx.float32)

    def generate_stream(
        self,
        input_ids: mx.array,
        speaker: str = "Ethan",
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        chunk_size: int = 300,
        left_context_size: int = 25,
        **kwargs,
    ):
        if not self.has_talker:
            raise ValueError("Cannot stream audio without talker module")
        if input_ids.shape[0] != 1:
            raise NotImplementedError("Streaming does not support batched inference")

        speaker_id = self.config.talker_config.speaker_id.get(speaker.lower())
        if speaker_id is None:
            raise NotImplementedError(f"Speaker {speaker} not implemented")

        from mlx_vlm.generate import generate_step

        thinker_kwargs = {
            "max_tokens": thinker_max_new_tokens,
            "eos_tokens": [thinker_eos_token_id],
        }
        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key in (
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            ):
                thinker_kwargs[key] = value

        generator = generate_step(
            input_ids,
            self.thinker,
            thinker_kwargs.get("pixel_values"),
            kwargs.get("mask"),
            **{
                k: v
                for k, v in thinker_kwargs.items()
                if k not in ("pixel_values", "mask")
            },
        )
        sequences = [input_ids]
        for token, _ in generator:
            sequences.append(mx.array([[token]]))
            if token == thinker_eos_token_id:
                break

        thinker_result_sequences = mx.concatenate(sequences, axis=1)
        thinker_hidden_all, thinker_embed_all = self.extract_thinker_hidden_states(
            thinker_result_sequences,
            target_layer_idx=self.config.talker_config.accept_hidden_layer,
            **kwargs,
        )

        im_start_indexes = mx.concatenate(
            (
                mx.array(
                    np.where(np.array(input_ids[0] == self.config.im_start_token_id))[0]
                ),
                mx.array([thinker_result_sequences.shape[-1]], dtype=mx.int32),
            ),
            axis=0,
        )
        multimodal_mask = (
            (thinker_result_sequences == self.config.thinker_config.audio_token_id)
            | (thinker_result_sequences == self.config.thinker_config.image_token_id)
            | (thinker_result_sequences == self.config.thinker_config.video_token_id)
        )

        talker_special_tokens = mx.array(
            [
                [
                    self.config.tts_bos_token_id,
                    self.config.tts_eos_token_id,
                    self.config.tts_pad_token_id,
                ]
            ],
            dtype=input_ids.dtype,
        )
        talker_special_embeds = self.thinker.language_model.model.embed_tokens(
            talker_special_tokens
        )
        talker_special_embeds_proj = self.talker.text_projection(talker_special_embeds)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            talker_special_embeds_proj[:, 0:1],
            talker_special_embeds_proj[:, 1:2],
            talker_special_embeds_proj[:, 2:3],
        )

        talker_input_embeds, talker_input_ids = [], []
        trailing_text_hidden = None

        for i in range(len(im_start_indexes) - 1):
            im_start_index, segment_end_index = int(im_start_indexes[i]), int(
                im_start_indexes[i + 1]
            )
            role_token = int(input_ids[0, im_start_index + 1])

            if role_token == self.config.system_token_id:
                continue
            elif role_token == self.config.user_token_id:
                talker_input_embeds.append(
                    self._get_talker_user_parts(
                        im_start_index,
                        segment_end_index,
                        multimodal_mask,
                        thinker_hidden_all,
                        thinker_embed_all,
                    )
                )
                talker_input_ids.append(
                    thinker_result_sequences[:, im_start_index:segment_end_index]
                )
            elif (
                role_token == self.config.assistant_token_id
                and i == len(im_start_indexes) - 2
            ):
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = (
                    self._get_talker_assistant_parts(
                        im_start_index,
                        segment_end_index,
                        speaker_id,
                        thinker_embed_all,
                        tts_pad_embed,
                        tts_bos_embed,
                        tts_eos_embed,
                    )
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)

        if not talker_input_embeds:
            return

        talker_input_embed = mx.concatenate(talker_input_embeds, axis=1)
        talker_input_id = mx.concatenate(talker_input_ids, axis=1)

        generated_tokens = thinker_result_sequences[0, input_ids.shape[1] :].tolist()
        yield ("text", generated_tokens)

        codes_list = []
        decoded_len = 0

        for residual_codes in self.talker.generate_stream(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,
            max_new_tokens=talker_max_new_tokens,
            temperature=talker_temperature,
            top_p=talker_top_p,
        ):
            codes_list.append(residual_codes)
            if len(codes_list) >= chunk_size:
                codes_buffer = mx.stack(codes_list, axis=1).transpose(0, 2, 1)
                wav_chunk, decoded_len = self.code2wav.stream_decode(
                    codes_buffer, chunk_size, left_context_size, decoded_len
                )
                if wav_chunk is not None:
                    mx.eval(wav_chunk)
                    yield ("audio", wav_chunk.astype(mx.float32))

        if codes_list:
            codes_buffer = mx.stack(codes_list, axis=1).transpose(0, 2, 1)
            wav_chunk = self.code2wav.flush_decode(
                codes_buffer, left_context_size, decoded_len
            )
            if wav_chunk is not None:
                mx.eval(wav_chunk)
                yield ("audio", wav_chunk.astype(mx.float32))

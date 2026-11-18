# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import logging
from typing import List, Optional

import torch
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    get_pos_emb_on_this_cp_rank,
)
from torch import Tensor, nn

from mbridge.models.qwen3_vl.transformer_config import Qwen3VLTransformerConfig

# Prefer fused RoPE from Apex as we need the `transpose_output_memory` argument for the bshd trick.
# See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/2469.
try:
    # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
except ImportError:
    fused_apply_rotary_pos_emb = None
from mbridge.models.qwen3_vl.rope_utils import Qwen3VLMultimodalRotaryEmbedding

logger = logging.getLogger(__name__)


# Slightly modified from Qwen3_5VLModel.get_rope_index
def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    image_position_ids: Optional[List[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    image_grid_thw_list = (
        image_grid_thw.tolist() if image_grid_thw is not None else None
    )
    video_grid_thw_list = (
        video_grid_thw.tolist() if video_grid_thw is not None else None
    )

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                explicit_image_positions = None
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw_list[image_index]
                    explicit_image_positions = (
                        image_position_ids[image_index]
                        if image_position_ids is not None
                        else None
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw_list[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

                if ed_image < ed_video and explicit_image_positions is not None:
                    visual_positions = explicit_image_positions.to(
                        device=input_ids.device, dtype=input_ids.dtype
                    )
                    if visual_positions.ndim != 2 or visual_positions.shape[-1] != 3:
                        raise ValueError(
                            "image_position_ids entries must have shape [visual_tokens, 3], "
                            f"got {tuple(visual_positions.shape)}"
                        )
                    visual_positions = visual_positions.transpose(0, 1).contiguous()
                else:
                    # Video timestamp spans and native image spans retain the
                    # regular Qwen merged-grid positions.
                    t_index = (
                        torch.arange(llm_grid_t, device=input_ids.device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=input_ids.device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    visual_positions = torch.stack([t_index, h_index, w_index])
                visual_length = visual_positions.shape[1]
                expected_token = image_token_id if ed_image < ed_video else video_token_id
                actual_visual_length = 0
                while (
                    ed + actual_visual_length < len(input_tokens)
                    and input_tokens[ed + actual_visual_length] == expected_token
                ):
                    actual_visual_length += 1
                if actual_visual_length != visual_length:
                    raise ValueError(
                        f"visual placeholder length={actual_visual_length} does not match "
                        f"position length={visual_length} for token_id={expected_token}"
                    )
                llm_pos_ids_list.append(
                    visual_positions + text_len + st_idx
                )
                st = ed + visual_length

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if llm_positions.shape[1] != input_ids.shape[0]:
                raise ValueError(
                    f"M-RoPE produced {llm_positions.shape[1]} positions for "
                    f"{input_ids.shape[0]} valid input tokens"
                )
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                device=position_ids.device, dtype=position_ids.dtype
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        if image_position_ids is not None and image_index != len(image_position_ids):
            raise ValueError(
                f"consumed {image_index} explicit image position tensors, "
                f"but received {len(image_position_ids)}"
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def apply_rotary_pos_emb_thd_absolute(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    return _apply_rotary_pos_emb_bshd(
        t[:, None],
        freqs,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        mscale=mscale,
    ).squeeze(1)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config: Qwen3VLTransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    bshd (conventional) / thd (packed seq) format

    In Qwen3-VL, the shape of freqs is (seq_length, bs, 1, 2 * dim) instead of [max_seqlen, 1, 1, 2 * dim]
    """
    assert not config.apply_rope_fusion
    orig_t_dtype = t.dtype
    if config.apply_rotary_pos_emb_in_fp32:
        t = t.float()

    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale,
        )
    else:
        result = apply_rotary_pos_emb_thd_absolute(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale,
            cp_group=cp_group,
        )

    if config.apply_rotary_pos_emb_in_fp32:
        result = result.to(orig_t_dtype)

    return result

"""Patch vLLM 0.6.3 prefix-cache and MRoPE chunk alignment bugs."""

from __future__ import annotations

import pathlib

from patch_utils import package_root, replace_once


HELPER_ANCHOR = """\
logger = init_logger(__name__)

LORA_WARMUP_RANK = 8"""

HELPER_REPLACEMENT = """\
logger = init_logger(__name__)


def _slice_mrope_positions(positions, start, stop, expected_len):
    if positions is None or len(positions) != 3:
        raise RuntimeError("MRoPE positions must contain three axes")
    sliced = [axis[start:stop] for axis in positions]
    lengths = [len(axis) for axis in sliced]
    if lengths != [expected_len] * 3:
        raise RuntimeError(
            "MRoPE/input token length mismatch after chunk alignment: "
            f"positions={lengths}, input_tokens={expected_len}, "
            f"slice=({start}, {stop})")
    return sliced


LORA_WARMUP_RANK = 8"""

PREFIX_PAST_ANCHOR = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass"""

PREFIX_PAST_REPLACEMENT = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            # Must clear prefix_cache_hit so _add_seq_group uses the full
            # block_tables (prefix + previous-chunk blocks) instead of only
            # computed_block_nums (prefix only).  Without this, block_tables
            # passed to _forward_prefix_pytorch is too narrow for context_len,
            # causing an empty blk_ids slice and a zero-dim amax() crash.
            inter_data.prefix_cache_hit = False"""

PARTIAL_HIT_ANCHOR = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len"""

PARTIAL_HIT_REPLACEMENT = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len
            if inter_data.mrope_input_positions is not None:
                positions = inter_data.mrope_input_positions[seq_idx]
                if positions is not None:
                    inter_data.mrope_input_positions[seq_idx] = \\
                        _slice_mrope_positions(
                            positions, uncomputed_start, None,
                            inter_data.query_lens[seq_idx])"""

FULL_HIT_ANCHOR = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1"""

FULL_HIT_REPLACEMENT = """\
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1
            if inter_data.mrope_input_positions is not None:
                positions = inter_data.mrope_input_positions[seq_idx]
                if positions is not None:
                    inter_data.mrope_input_positions[seq_idx] = \\
                        _slice_mrope_positions(positions, -1, None, 1)"""

MULTIMODAL_MROPE_ANCHOR = """\
                mrope_input_positions, mrope_position_delta = \\
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_token_id=hf_config.image_token_id,
                        video_token_id=hf_config.video_token_id,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        vision_end_token_id=hf_config.vision_end_token_id,
                        spatial_merge_size=hf_config.vision_config.
                        spatial_merge_size,
                        context_len=inter_data.context_lens[seq_idx],
                    )

                seq_data.mrope_position_delta = mrope_position_delta
                inter_data.mrope_input_positions[
                    seq_idx] = mrope_input_positions"""

MULTIMODAL_MROPE_REPLACEMENT = """\
                # vLLM 0.6.3 returns positions through the end of token_ids,
                # while chunked prefill sends only [context_len:seq_len].
                # Compute the full MRoPE map once so the delta remains tied to
                # the complete request, then select exactly the physical query.
                mrope_input_positions, mrope_position_delta = \\
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_token_id=hf_config.image_token_id,
                        video_token_id=hf_config.video_token_id,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        vision_end_token_id=hf_config.vision_end_token_id,
                        spatial_merge_size=hf_config.vision_config.
                        spatial_merge_size,
                        context_len=0,
                    )
                mrope_input_positions = _slice_mrope_positions(
                    mrope_input_positions,
                    inter_data.context_lens[seq_idx],
                    inter_data.seq_lens[seq_idx],
                    len(inter_data.input_tokens[seq_idx]))

                seq_data.mrope_position_delta = mrope_position_delta
                inter_data.mrope_input_positions[
                    seq_idx] = mrope_input_positions"""

MODEL_INPUT_FIELDS_ANCHOR = """\
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None"""

MODEL_INPUT_FIELDS_REPLACEMENT = """\
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    # BI100 scheduler-owned GDN prefix-cache actions. These plain Python
    # objects are included in the multiprocess model-input broadcast.
    gdn_restore_key: Optional[Tuple[int, bytes]] = None
    gdn_capture_points: Optional[List[Tuple[int, Tuple[int, bytes]]]] = None
    gdn_evict_keys: Optional[List[Tuple[int, bytes]]] = None
    gdn_segment_offsets: Optional[List[int]] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None"""

BASE_BROADCAST_ANCHOR = """\
            \"multi_modal_kwargs\": self.multi_modal_kwargs,
            \"prompt_adapter_mapping\": self.prompt_adapter_mapping,
            \"prompt_adapter_requests\": self.prompt_adapter_requests,
            \"virtual_engine\": self.virtual_engine,
            \"request_ids_to_seq_ids\": self.request_ids_to_seq_ids,
            \"finished_requests_ids\": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod"""

BASE_BROADCAST_REPLACEMENT = """\
            \"multi_modal_kwargs\": self.multi_modal_kwargs,
            \"gdn_restore_key\": self.gdn_restore_key,
            \"gdn_capture_points\": self.gdn_capture_points,
            \"gdn_evict_keys\": self.gdn_evict_keys,
            \"gdn_segment_offsets\": self.gdn_segment_offsets,
            \"prompt_adapter_mapping\": self.prompt_adapter_mapping,
            \"prompt_adapter_requests\": self.prompt_adapter_requests,
            \"virtual_engine\": self.virtual_engine,
            \"request_ids_to_seq_ids\": self.request_ids_to_seq_ids,
            \"finished_requests_ids\": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod"""

SAMPLING_BROADCAST_ANCHOR = """\
            \"multi_modal_kwargs\": self.multi_modal_kwargs,
            \"prompt_adapter_mapping\": self.prompt_adapter_mapping,
            \"prompt_adapter_requests\": self.prompt_adapter_requests,
            \"virtual_engine\": self.virtual_engine,
            \"request_ids_to_seq_ids\": self.request_ids_to_seq_ids,
            \"finished_requests_ids\": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)"""

SAMPLING_BROADCAST_REPLACEMENT = """\
            \"multi_modal_kwargs\": self.multi_modal_kwargs,
            \"gdn_restore_key\": self.gdn_restore_key,
            \"gdn_capture_points\": self.gdn_capture_points,
            \"gdn_evict_keys\": self.gdn_evict_keys,
            \"gdn_segment_offsets\": self.gdn_segment_offsets,
            \"prompt_adapter_mapping\": self.prompt_adapter_mapping,
            \"prompt_adapter_requests\": self.prompt_adapter_requests,
            \"virtual_engine\": self.virtual_engine,
            \"request_ids_to_seq_ids\": self.request_ids_to_seq_ids,
            \"finished_requests_ids\": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)"""

BUILDER_INIT_ANCHOR = """\
        self.finished_requests_ids = finished_requests_ids
        self.decode_only = True

        # Intermediate data"""

BUILDER_INIT_REPLACEMENT = """\
        self.finished_requests_ids = finished_requests_ids
        self.decode_only = True
        self.gdn_restore_key = None
        self.gdn_capture_points = None
        self.gdn_evict_keys = None
        self.gdn_segment_offsets = None

        # Intermediate data"""

ADD_SEQ_GROUP_ANCHOR = """\
    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        \"\"\"Add a sequence group to the builder.\"\"\"
        seq_ids = seq_group_metadata.seq_data.keys()"""

ADD_SEQ_GROUP_REPLACEMENT = """\
    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        \"\"\"Add a sequence group to the builder.\"\"\"
        gdn_actions = (
            seq_group_metadata.gdn_restore_key,
            seq_group_metadata.gdn_capture_points,
            seq_group_metadata.gdn_evict_keys,
            seq_group_metadata.gdn_segment_offsets,
        )
        if any(value is not None for value in gdn_actions):
            if not seq_group_metadata.is_prompt:
                raise RuntimeError(\"GDN prefix-cache actions require prefill\")
            if any(value is not None for value in (
                    self.gdn_restore_key, self.gdn_capture_points,
                    self.gdn_evict_keys, self.gdn_segment_offsets)):
                raise RuntimeError(
                    \"only one GDN prefix-cache action group is supported\")
            (self.gdn_restore_key, self.gdn_capture_points,
             self.gdn_evict_keys, self.gdn_segment_offsets) = gdn_actions
        seq_ids = seq_group_metadata.seq_data.keys()"""

BUILD_RESULT_ANCHOR = """\
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,"""

BUILD_RESULT_REPLACEMENT = """\
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            gdn_restore_key=self.gdn_restore_key,
            gdn_capture_points=self.gdn_capture_points,
            gdn_evict_keys=self.gdn_evict_keys,
            gdn_segment_offsets=self.gdn_segment_offsets,
            request_ids_to_seq_ids=request_ids_to_seq_ids,"""

EXECUTE_KWARGS_ANCHOR = """\
        seqlen_agnostic_kwargs = {
            \"finished_requests_ids\": model_input.finished_requests_ids,
            \"request_ids_to_seq_ids\": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        if (self.observability_config is not None"""

EXECUTE_KWARGS_REPLACEMENT = """\
        seqlen_agnostic_kwargs = {
            \"finished_requests_ids\": model_input.finished_requests_ids,
            \"request_ids_to_seq_ids\": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        gdn_prefix_kwargs = {}
        if model_input.gdn_restore_key is not None:
            gdn_prefix_kwargs[\"gdn_restore_key\"] = model_input.gdn_restore_key
        if model_input.gdn_capture_points is not None:
            gdn_prefix_kwargs[\"gdn_capture_points\"] = (
                model_input.gdn_capture_points)
        if model_input.gdn_evict_keys is not None:
            gdn_prefix_kwargs[\"gdn_evict_keys\"] = model_input.gdn_evict_keys
        if model_input.gdn_segment_offsets is not None:
            gdn_prefix_kwargs[\"gdn_segment_offsets\"] = (
                model_input.gdn_segment_offsets)
        if (self.observability_config is not None"""

MODEL_CALL_ANCHOR = """\
                **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                             device=self.device),
                **seqlen_agnostic_kwargs)"""

MODEL_CALL_REPLACEMENT = """\
                **MultiModalInputs.as_kwargs(multi_modal_kwargs,
                                             device=self.device),
                **seqlen_agnostic_kwargs,
                **gdn_prefix_kwargs)"""

PROFILE_KV_LAYERS_ANCHOR = """\
        num_layers = self.model_config.get_num_layers(self.parallel_config)"""

PROFILE_KV_LAYERS_REPLACEMENT = """\
        num_layers = self.model_config.get_num_attention_layers(
            self.parallel_config)"""


def patch_model_runner(model_runner: pathlib.Path) -> None:
    replace_once(
        model_runner,
        HELPER_ANCHOR,
        HELPER_REPLACEMENT,
        required=True,
        already_contains="def _slice_mrope_positions(",
    )
    replace_once(
        model_runner,
        PREFIX_PAST_ANCHOR,
        PREFIX_PAST_REPLACEMENT,
        required=True,
        already_contains="Must clear prefix_cache_hit so _add_seq_group",
    )
    replace_once(
        model_runner,
        PARTIAL_HIT_ANCHOR,
        PARTIAL_HIT_REPLACEMENT,
        required=True,
        already_contains="positions, uncomputed_start, None,",
    )
    replace_once(
        model_runner,
        FULL_HIT_ANCHOR,
        FULL_HIT_REPLACEMENT,
        required=True,
        already_contains="_slice_mrope_positions(positions, -1, None, 1)",
    )
    replace_once(
        model_runner,
        MULTIMODAL_MROPE_ANCHOR,
        MULTIMODAL_MROPE_REPLACEMENT,
        required=True,
        already_contains="Compute the full MRoPE map once",
    )
    replace_once(
        model_runner,
        MODEL_INPUT_FIELDS_ANCHOR,
        MODEL_INPUT_FIELDS_REPLACEMENT,
        already_contains="gdn_restore_key: Optional[Tuple[int, bytes]]",
    )
    replace_once(
        model_runner,
        BASE_BROADCAST_ANCHOR,
        BASE_BROADCAST_REPLACEMENT,
        already_contains=BASE_BROADCAST_REPLACEMENT,
    )
    replace_once(
        model_runner,
        SAMPLING_BROADCAST_ANCHOR,
        SAMPLING_BROADCAST_REPLACEMENT,
        already_contains=SAMPLING_BROADCAST_REPLACEMENT,
    )
    replace_once(
        model_runner,
        BUILDER_INIT_ANCHOR,
        BUILDER_INIT_REPLACEMENT,
        already_contains="self.gdn_restore_key = None",
    )
    replace_once(
        model_runner,
        ADD_SEQ_GROUP_ANCHOR,
        ADD_SEQ_GROUP_REPLACEMENT,
        already_contains="gdn_actions = (",
    )
    replace_once(
        model_runner,
        BUILD_RESULT_ANCHOR,
        BUILD_RESULT_REPLACEMENT,
        already_contains="gdn_restore_key=self.gdn_restore_key",
    )
    replace_once(
        model_runner,
        EXECUTE_KWARGS_ANCHOR,
        EXECUTE_KWARGS_REPLACEMENT,
        already_contains="gdn_prefix_kwargs = {}",
    )
    replace_once(
        model_runner,
        MODEL_CALL_ANCHOR,
        MODEL_CALL_REPLACEMENT,
        already_contains="**gdn_prefix_kwargs)",
    )
    replace_once(
        model_runner,
        PROFILE_KV_LAYERS_ANCHOR,
        PROFILE_KV_LAYERS_REPLACEMENT,
        required=True,
        already_contains=PROFILE_KV_LAYERS_REPLACEMENT,
    )


if __name__ == "__main__":
    patch_model_runner(package_root("vllm") / "worker" / "model_runner.py")

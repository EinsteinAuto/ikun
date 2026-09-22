"""Install the optional BI100 prefix-cache diagnostic trace."""
from patch_utils import package_root, replace_once, replace_one_of

VLLM_ROOT = package_root("vllm")
TARGET = VLLM_ROOT / "core" / "block_manager_v2.py"
OUTPUTS_TARGET = VLLM_ROOT / "outputs.py"

HELPER = '''
    def _bi100_capture_cache_trace(self, seq_group, seq, block_table) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return

        session = getattr(self, "_bi100_trace_session", None)
        if session is None:
            session = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
            self._bi100_trace_session = session

        self._bi100_trace_ordinal = getattr(self, "_bi100_trace_ordinal", 0) + 1
        request_id_sha256 = hashlib.sha256(
            str(seq_group.request_id).encode("utf-8")).hexdigest()[:16]

        prompt_tokens = len(seq.get_token_ids())
        requests = getattr(self, "_bi100_trace_requests", None)
        if requests is None:
            requests = {}
            self._bi100_trace_requests = requests

        requests[seq.seq_id] = {
            "version": 4,
            "trace_session_sha256": session,
            "ordinal": self._bi100_trace_ordinal,
            "request_id_sha256": request_id_sha256,
            "prompt_tokens": prompt_tokens,
            "prompt_allocated_blocks": (
                (prompt_tokens + self.block_size - 1) // self.block_size
            ),
            "block_size": self.block_size,
            "capacity_blocks": self.num_total_gpu_blocks,
        }
        setattr(seq_group, "_bi100_cache_trace_seq_id", seq.seq_id)
        setattr(seq_group, "_bi100_cache_trace_emit",
                self._bi100_emit_cache_trace)

    def _bi100_update_cache_trace(
            self, seq, raw_kv_hit_blocks, restore_key, capture_actions,
            evict_keys, policy) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return
        requests = getattr(self, "_bi100_trace_requests", None)
        if not requests or seq.seq_id not in requests:
            return
        record = requests[seq.seq_id]
        record["gdn_policy"] = policy
        if "initial_raw_kv_contiguous_hit_blocks" not in record:
            record["initial_raw_kv_contiguous_hit_blocks"] = max(
                0, int(raw_kv_hit_blocks))
            record["gdn_restore_digest_base64"] = (
                base64.b64encode(restore_key[1]).decode("ascii")
                if restore_key is not None else None)
        record["raw_kv_contiguous_hit_blocks"] = max(
            int(raw_kv_hit_blocks),
            int(record.get("raw_kv_contiguous_hit_blocks", 0)))
        effective_blocks = int(restore_key[0]) if restore_key is not None else 0
        record["effective_gdn_hit_blocks"] = max(
            effective_blocks, int(record.get("effective_gdn_hit_blocks", 0)))

        admissions = record.setdefault("gdn_admissions", [])
        for key, reason in capture_actions:
            admissions.append({
                "block_count": int(key[0]),
                "digest_base64": base64.b64encode(key[1]).decode("ascii"),
                "reason": str(reason),
            })
        evictions = record.setdefault("gdn_evictions", [])
        for key in evict_keys:
            evictions.append({
                "block_count": int(key[0]),
                "digest_base64": base64.b64encode(key[1]).decode("ascii"),
                "reason": "capacity_lru",
            })

    def _bi100_finalize_cache_trace(self, seq, block_table) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return

        requests = getattr(self, "_bi100_trace_requests", None)
        if not requests:
            return

        record = requests.get(seq.seq_id)
        if record is None:
            return

        total_tokens = len(seq.get_token_ids())
        block_hashes = block_table.get_content_hashes()
        for block_hash in block_hashes:
            if not isinstance(block_hash, bytes) or len(block_hash) != 32:
                raise RuntimeError(
                    "BI100 cache trace requires 32-byte content hashes")
        full_blocks = len(block_hashes)
        record.update({
            "total_tokens": total_tokens,
            "allocated_blocks": (
                (total_tokens + self.block_size - 1) // self.block_size
            ),
            "full_blocks": full_blocks,
            "hash_encoding": "sha256_base64",
            "block_hashes": base64.b64encode(b"".join(block_hashes)).decode("ascii"),
            "_finalized": True,
        })
        generated_tokens = max(0, total_tokens - record["prompt_tokens"])
        record["generated_tokens"] = generated_tokens

    def _bi100_emit_cache_trace(self, seq_group) -> None:
        if os.getenv("BI100_CACHE_TRACE", "0") != "1":
            return
        seq_id = getattr(seq_group, "_bi100_cache_trace_seq_id", None)
        requests = getattr(self, "_bi100_trace_requests", None)
        if seq_id is None or not requests:
            return
        record = requests.pop(seq_id, None)
        if record is None:
            return
        if record.pop("_finalized", False) is not True:
            raise RuntimeError(
                "BI100 cache trace emitted before block finalization")

        metrics = getattr(seq_group, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        first_token = getattr(metrics, "first_token_time", None)
        finished = getattr(metrics, "finished_time", None)
        queue = getattr(metrics, "time_in_queue", None)
        cached = getattr(metrics, "num_cached_tokens", None)
        if any(value is None for value in (
                arrival, first_token, finished, queue)):
            raise RuntimeError(
                "BI100 cache trace requires finalized request metrics")
        record["ttft_s"] = max(0.0, float(first_token - arrival))
        record["request_latency_s"] = max(
            0.0, float(finished - arrival))
        record["time_in_queue_s"] = max(0.0, float(queue))
        record["observed_effective_cached_tokens"] = max(
            0, int(cached or 0))
        ttft_s = record["ttft_s"]
        if ttft_s > 0:
            record["observed_input_tps"] = record["prompt_tokens"] / ttft_s
        generated_tokens = record["generated_tokens"]
        if generated_tokens > 1:
            decode_s = finished - first_token
            if decode_s > 0:
                record["observed_output_tps"] = (
                    (generated_tokens - 1) / decode_s)
        print("[BI100_CACHE_TRACE] " + json.dumps(record, separators=(",", ":"),
                                             sort_keys=True), flush=True)
'''


def main():
    replace_once(TARGET, "from collections.abc import Mapping\n",
                 "from collections.abc import Mapping\nimport base64\nimport json\nimport os\n",
                 required=True, already_contains="import base64\n")
    replace_once(TARGET, "class BlockSpaceManagerV2(BlockSpaceManager):\n",
                 "class BlockSpaceManagerV2(BlockSpaceManager):\n" + HELPER,
                 required=True, already_contains="def _bi100_capture_cache_trace(")
    replace_once(TARGET,
                 "        self.block_tables[seq.seq_id] = block_table\n\n        # Track seq",
                 "        self.block_tables[seq.seq_id] = block_table\n        self._bi100_capture_cache_trace(\n            seq_group, seq, block_table)\n\n        # Track seq",
                 required=True,
                 already_contains="self.block_tables[seq.seq_id] = block_table\n"
                                "        self._bi100_capture_cache_trace(")
    replacements = []
    for table_key in ("seq_id", "seq.seq_id"):
        prefix = (
            "        self._last_access_blocks_tracker."
            "update_seq_blocks_last_access(\n"
            f"            seq_id, self.block_tables[{table_key}]."
            "physical_block_ids)\n")
        replacements.append((
            prefix + "\n        # Untrack seq",
            prefix + "        self._bi100_finalize_cache_trace(\n"
            f"            seq, self.block_tables[{table_key}])\n\n"
            "        # Untrack seq",
        ))
    replace_one_of(
        TARGET,
        replacements,
        required=True,
        already_contains="        self._bi100_finalize_cache_trace(\n"
                         "            seq, self.block_tables[")
    replace_once(
        OUTPUTS_TARGET,
        "        seq_group.set_finished_time(finished_time)\n\n"
        "        init_args = (seq_group.request_id, prompt, prompt_token_ids,\n",
        "        seq_group.set_finished_time(finished_time)\n"
        "        if finished_time is not None:\n"
        "            cache_trace_emit = getattr(\n"
        "                seq_group, \"_bi100_cache_trace_emit\", None)\n"
        "            if callable(cache_trace_emit):\n"
        "                cache_trace_emit(seq_group)\n"
        "                delattr(seq_group, \"_bi100_cache_trace_emit\")\n"
        "                delattr(seq_group, \"_bi100_cache_trace_seq_id\")\n\n"
        "        init_args = (seq_group.request_id, prompt, prompt_token_ids,\n",
        required=True,
        already_contains="if finished_time is not None:\n"
                         "            cache_trace_emit = getattr(\n",
    )


if __name__ == "__main__":
    main()

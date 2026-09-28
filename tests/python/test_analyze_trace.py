"""
test_analyze_trace.py — Unit tests for the trace analyzer tool.

Tests kernel categorization, NCCL analysis, item sync detection,
and roadmap generation against known trace data.
"""
import json
import os
import sys
import tempfile
import unittest

# Make sure we can import the analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from analyze_trace import (
    categorize_kernel,
    analyze_kernels,
    analyze_nccl,
    analyze_item_sync,
    analyze_dtype_casts,
    analyze_gpu_utilization,
    compute_roadmap,
)


class TestKernelCategorization(unittest.TestCase):
    """Verify every kernel name maps to the expected category."""

    def test_nccl_fence_wait(self):
        self.assertEqual(
            categorize_kernel("legacy::_fenceFlagWaitE(unsigned long*, unsigned long, int)"),
            "NCCL:fenceWait",
        )

    def test_nccl_fence_ops(self):
        self.assertEqual(
            categorize_kernel("legacy::__global__fenceOps(legacy::syncDevicesHelper)"),
            "NCCL:fenceOps",
        )

    def test_gemm(self):
        self.assertEqual(
            categorize_kernel("void cuinfer::impl::kernel::general_gemm_f_f_h_h_h_tcuh<128u, 32u>"),
            "GEMM",
        )

    def test_moe_w13(self):
        self.assertEqual(
            categorize_kernel("(anonymous namespace)::direct_w13_kernel(__half const*)"),
            "MoE:w13",
        )

    def test_moe_w2_reduce(self):
        self.assertEqual(
            categorize_kernel("(anonymous namespace)::direct_w2_reduce_kernel(__half const*)"),
            "MoE:w2_reduce",
        )

    def test_moe_topk(self):
        self.assertEqual(
            categorize_kernel("void (anonymous namespace)::topk_gating_softmax<float, 8, 256>"),
            "MoE:topk_softmax",
        )

    def test_gdn_packed_decode(self):
        self.assertEqual(
            categorize_kernel("(anonymous namespace)::gdn_packed_decode_kernel(float*)"),
            "GDN:packed_decode",
        )

    def test_elementwise_sum(self):
        self.assertEqual(
            categorize_kernel("void legacy::elementWiseKernel<false, false, __half, legacy::sumGpu>"),
            "Elementwise:sum(NCCL)",
        )

    def test_elementwise_other(self):
        self.assertEqual(
            categorize_kernel("void at::native::modern::elementwise_kernel<at::native::sigmoid>"),
            "Activation:sigmoid",
        )

    def test_paged_attention(self):
        self.assertEqual(
            categorize_kernel("void cuinfer::impl::kernel::single_query_cached_kv_attention_kernel_256"),
            "PagedAttention",
        )

    def test_unknown_kernel(self):
        cat = categorize_kernel("some_unknown_kernel_name_here")
        self.assertTrue(cat.startswith("Other:"))


class TestSyntheticTrace(unittest.TestCase):
    """Test analysis passes against a minimal synthetic trace."""

    def setUp(self):
        """Build a minimal synthetic trace with known kernel counts."""
        self.events = []
        ts = 1000000

        # 6 fenceWait (= 1 logical AR for TP=4 ring: 6 steps × 1 wait each... 
        # actually we model 2 fenceWait per step × 3 steps = 6, 
        # plus 3 fenceOps per reduce-scatter + 3 per all-gather = 6 total)
        for _ in range(12):
            self.events.append({"cat": "kernel", "name": "legacy::_fenceFlagWaitE",
                                "ts": ts, "dur": 15})
            ts += 20
        for _ in range(6):
            self.events.append({"cat": "kernel", "name": "legacy::__global__fenceOps",
                                "ts": ts, "dur": 10})
            ts += 15

        # 4 GEMM
        for _ in range(4):
            self.events.append({"cat": "kernel", "name": "void cuinfer::impl::kernel::general_gemm_f_f_h_h_h_tcuh",
                                "ts": ts, "dur": 50})
            ts += 60

        # 2 MoE w13
        for _ in range(2):
            self.events.append({"cat": "kernel", "name": "(anon)::direct_w13_kernel",
                                "ts": ts, "dur": 35})
            ts += 40

        # 1 aten::item
        self.events.append({"cat": "cpu_op", "name": "aten::item", "ts": ts, "dur": 370})
        ts += 400

        # 1 aten::to
        self.events.append({"cat": "cpu_op", "name": "aten::to", "ts": ts, "dur": 10})
        ts += 15

    def test_kernel_analysis_totals(self):
        result = analyze_kernels(self.events)
        self.assertGreater(result["total_kernel_us"], 0)
        cats = result["categories"]
        self.assertIn("NCCL", cats)
        self.assertIn("GEMM", cats)
        self.assertEqual(cats["NCCL"]["count"], 18)  # 12 waits + 6 ops
        self.assertEqual(cats["GEMM"]["count"], 4)

    def test_nccl_logical_ar_count(self):
        result = analyze_nccl(self.events)
        # 6 fenceOps / 6 = 1 logical AR
        self.assertEqual(result["logical_allreduce"], 1)
        self.assertGreater(result["per_ar_cost_us"], 0)

    def test_item_sync_detection(self):
        result = analyze_item_sync(self.events)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["total_us"], 370)

    def test_dtype_cast_detection(self):
        result = analyze_dtype_casts(self.events)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["total_us"], 10)

    def test_gpu_utilization(self):
        result = analyze_gpu_utilization(self.events)
        self.assertGreater(result["utilization_pct"], 0)
        self.assertLessEqual(result["utilization_pct"], 100)

    def test_roadmap_has_three_priorities(self):
        kernels = analyze_kernels(self.events)
        nccl = analyze_nccl(self.events)
        item = analyze_item_sync(self.events)
        dtype = analyze_dtype_casts(self.events)
        roadmap = compute_roadmap(nccl, item, dtype, kernels)
        self.assertEqual(len(roadmap), 3)
        priorities = [r["priority"] for r in roadmap]
        self.assertEqual(priorities, ["P0", "P1", "P2"])

    def test_roadmap_savings_positive(self):
        kernels = analyze_kernels(self.events)
        nccl = analyze_nccl(self.events)
        item = analyze_item_sync(self.events)
        dtype = analyze_dtype_casts(self.events)
        roadmap = compute_roadmap(nccl, item, dtype, kernels)
        for r in roadmap:
            self.assertGreaterEqual(r["estimated_saving_us"], 0)


class TestRealTraceIfAvailable(unittest.TestCase):
    """Integration test: run analysis against real trace if present."""

    TRACE_PATH = "/tmp/trace_test/ikun_server_trace.json"

    def setUp(self):
        if not os.path.isfile(self.TRACE_PATH):
            self.skipTest("Real trace not found at " + self.TRACE_PATH)
        with open(self.TRACE_PATH) as f:
            self.data = json.load(f)
        self.events = self.data.get("traceEvents", [])

    def test_real_trace_has_expected_event_count(self):
        self.assertGreater(len(self.events), 10000)

    def test_real_nccl_81_logical_ar(self):
        result = analyze_nccl(self.events)
        self.assertEqual(result["logical_allreduce"], 81,
                         "Expected 81 logical all-reduces for 64-layer Qwen3.5-35B")

    def test_real_item_count_10(self):
        result = analyze_item_sync(self.events)
        self.assertEqual(result["count"], 10,
                         "Expected 10 aten::item calls (one per full attention layer)")

    def test_real_gpu_util_under_50pct(self):
        result = analyze_gpu_utilization(self.events)
        self.assertLess(result["utilization_pct"], 50,
                        "GPU utilization should be under 50% (known bottleneck)")


if __name__ == "__main__":
    unittest.main()

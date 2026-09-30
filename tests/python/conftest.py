"""
conftest.py — pytest configuration for ikun tests

isolate test discovery from the root __init__.py which imports vllm
(vllm requires pydantic, triton etc which are not available in CI)
"""
import sys
import os

# ensure tools/ is importable for test_analyze_trace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

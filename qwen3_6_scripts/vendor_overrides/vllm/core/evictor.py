# SPDX-License-Identifier: Apache-2.0
"""Alias: vllm.core.evictor -> vllm.core.evictor_v2"""
from vllm.core.evictor_v2 import *  # noqa: F401,F403
from vllm.core.evictor_v2 import (EvictionPolicy, Evictor, make_evictor,
                                   BlockMetaData, LRUEvictor)

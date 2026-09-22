# SPDX-License-Identifier: Apache-2.0
"""Alias: SelfAttnBlockSpaceManager -> BlockSpaceManagerV2.

New vllm names the v2 block manager SelfAttnBlockSpaceManager in
vllm/core/block_manager.py. The vendor_overrides still ship the v2
implementation under its old name (block_manager_v2.py /
BlockSpaceManagerV2). This module bridges the two.
"""
from vllm.core.block_manager_v2 import BlockSpaceManagerV2

SelfAttnBlockSpaceManager = BlockSpaceManagerV2

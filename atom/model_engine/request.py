# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RequestOutput:
    """Output structure passed to stream callback."""

    request_id: int
    output_tokens: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    kv_transfer_params_output: Optional[Dict[str, Any]] = None
    num_cached_tokens: int = 0

    # Readings from whichever clock the engine core is running -- wall time
    # normally, virtual time under Compass. They travel with the output because
    # the process that streams to the client is not the process that owns the
    # sequence, and under a simulated run the streaming side's own clock
    # measures the simulator rather than the system being simulated.
    # Zero means "not stamped yet"; the first token has no finish time.
    arrive_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0

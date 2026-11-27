# SPDX-License-Identifier: MIT
"""``COMPASS_MOE_KEYLOG=1`` (opt-in; a real run recording its steps): each worker logs the first aiter MoE tuning
lookup of each key (`aiter.fused_moe.get_2stage_cfgs`, whose first argument is the padded token count, aiter's
`get_padded_M`) as a line of ``<record>/moe_keys/<pid>.jsonl``: the monotonic time, the padded M, the token count
of the call that asked (the first ``token_num``, ``M`` or ``hidden_states`` rows up the stack), the model and layer
classes up the stack (the target's, or DSpark's drafter's), the step the recording runner is on, and the key. Which
call first asks an untuned tier (MiniMax-M3.1's M=256, whose first use is not its forward's token count) is then
read off the lines. It wraps the lookup only: its arguments and what it returns, and so every kernel, are
unchanged, and any failure to log is swallowed."""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path

ENV = "COMPASS_MOE_KEYLOG"
#: The frame variables that hold the token count of the MoE call that asked.
TOKEN_NAMES = ("token_num", "M", "m")


def _tokens(frame) -> int | None:
    """The token count the nearest caller up from `frame` holds, None without one."""
    while frame is not None:
        f = frame.f_locals
        for name in TOKEN_NAMES:
            v = f.get(name)
            if isinstance(v, int):
                return v
        h = f.get("hidden_states")
        shape = getattr(h, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        frame = frame.f_back
    return None


def _modules(frame, limit: int = 8) -> list[str]:
    """The classes of the `torch.nn.Module` instances up from `frame` (`self` of each frame), innermost first."""
    out = []
    while frame is not None and len(out) < limit:
        owner = frame.f_locals.get("self")
        if owner is not None and any(c.__name__ == "Module" for c in type(owner).__mro__):
            name = type(owner).__name__
            if not out or out[-1] != name:
                out.append(name)
        frame = frame.f_back
    return out


def install(runner, record) -> bool:
    """Wrap aiter's MoE tuning lookup under `ENV`, logging into `record` (the parity record's root); False when off
    or with no aiter."""
    if os.environ.get(ENV) != "1":
        return False
    try:
        import aiter.fused_moe as fused_moe
    except ImportError:
        return False
    inner = fused_moe.get_2stage_cfgs
    if getattr(inner, "_compass_keylog", False):
        return True
    path = Path(record) / "moe_keys" / f"{os.getpid()}.jsonl"
    seen: set = set()

    @functools.wraps(inner)
    def logged(*args, **kwargs):
        try:
            key = (args, tuple(sorted(kwargs.items())))
            if key not in seen:
                seen.add(key)
                frame = sys._getframe(1)
                parity = getattr(runner, "_parity_record", None)
                line = {
                    "t_ns": time.monotonic_ns(),
                    "padded_m": args[0] if args else kwargs.get("token"),
                    "tokens": _tokens(frame),
                    "modules": _modules(frame),
                    "step": getattr(parity, "step", None),
                    "key": [str(a) for a in args],
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as out:
                    out.write(json.dumps(line) + "\n")
        except Exception:  # noqa: BLE001  the lookup never pays for the log
            pass
        return inner(*args, **kwargs)

    logged._compass_keylog = True
    fused_moe.get_2stage_cfgs = logged
    return True

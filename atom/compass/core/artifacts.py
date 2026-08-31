"""Where a rank's artifacts live, and how to find them again.

Every rank records its own graph and its own step timings, because under any
parallelism the ranks genuinely differ and one shared path makes them race for
it. The write side has always known that; the read side did not, and the gap is
invisible at a single rank because no suffix is applied there. A calibration run
at TP=2 wrote ``steps.tp0.jsonl`` and ``steps.tp1.jsonl`` and the predicting run
then asked for ``steps.jsonl``, which no longer existed.

So the convention lives here once, and both sides go through it.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

__all__ = ["rank_path", "resolve_rank_path"]


def rank_path(path: str, coords: Mapping[str, int]) -> str:
    """The path ``coords`` writes to, given the shared ``path`` it was asked for.

    ``g.json`` at ``{"tp": 1}`` becomes ``g.tp1.json``. Coordinates are sorted so
    the name is stable regardless of how the mapping was built, and an empty
    mapping leaves the path alone.
    """
    suffix = "-".join(f"{name}{index}" for name, index in sorted(coords.items()))
    stem, ext = os.path.splitext(path)
    return f"{stem}.{suffix}{ext}" if suffix else path


def resolve_rank_path(
    path: str, coords: Optional[Mapping[str, int]] = None
) -> tuple[str, bool]:
    """Find the artifact this rank should read.

    Returns the path to use and whether it is this rank's own. Prefers the
    rank's file, falls back to the unsuffixed one, and returns the rank's path
    when neither exists so the caller's error names what it actually wanted.

    The fallback is deliberate rather than an oversight: reusing one rank's
    calibration across a wider run is a legitimate thing to want — the ranks of a
    symmetric TP group time within a fraction of a percent of each other — and it
    is the caller's business to say so, not this function's to forbid.
    """
    if coords:
        mine = rank_path(path, coords)
        if os.path.exists(mine):
            return mine, True
        if os.path.exists(path):
            return path, False
        return mine, True
    return path, False

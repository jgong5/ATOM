"""Serving replay with no device in the loop.

A simulated run under `--compass-mode predict` already replaces the forward
pass with a cost estimate, but it still starts a real ATOM deployment: worker
processes, a device context, real weights, a real KV allocation. That proves the
cost model without proving the thing the PoC actually claims -- that a
configuration can be evaluated *without* the hardware it describes.

This package supplies the two pieces that removes:

- :class:`~atom.compass.replay.local_proc.LocalProcManager`, which stands where
  ``AsyncIOProcManager`` stands and runs the model runner in this process.
- :class:`~atom.compass.replay.runner.ReplayModelRunner`, which answers the
  runner's RPCs from captured target metadata instead of from a device.

Everything between them is ATOM's: the scheduler, the block manager, the
sequence lifecycle, admission, the output path. None of it is reimplemented
here, and that is the point -- a replay that reimplemented the scheduler would
be predicting its own policy rather than ATOM's.

What is bypassed is named in :mod:`~atom.compass.replay.local_proc`.

Both names are resolved on first attribute access, not at import time. The
one module that must run *before* anything else --
:mod:`~atom.compass.replay.bootstrap`, which answers AITER's import-time
architecture query from the captured target -- lives in this package, and
importing the package would otherwise drag the runner (and everything the
runner imports) in ahead of it. Same reason as `atom/__init__.py`.
"""

__all__ = ["LocalProcManager", "ReplayModelRunner"]


def __getattr__(name: str):
    if name == "LocalProcManager":
        from atom.compass.replay.local_proc import LocalProcManager

        return LocalProcManager
    if name == "ReplayModelRunner":
        from atom.compass.replay.runner import ReplayModelRunner

        return ReplayModelRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

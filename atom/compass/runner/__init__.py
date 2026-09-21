# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The model runner a simulated run installs in place of ATOM's own.

Two modules, split by what each is allowed to import:

- `overrides` holds the behaviour -- the methods that keep construction off the
  device -- and imports nothing from the engine. That is not a stylistic
  preference. Importing `atom.model_engine.model_runner` runs aiter's
  architecture probe, which shells out to `rocminfo` and raises in a container
  with no driver, so anything reachable from that import cannot be exercised by
  a test that runs without one.
- `model_runner` binds those overrides onto `ModelRunner`, and is therefore the
  only module here that needs a driver to import. It is also where the composed
  class is checked against `overrides.RPC_SURFACE`, the table of names a worker
  dispatches: the worker skips a name the runner lacks without raising, so a
  hole in that surface is a caller that waits forever rather than an error.

This package stays empty at import time for the same reason, so naming it costs
nothing.

`COMPASS_RUNNER_QUALNAME` is the string that selects the runner: a `Config`
field, read where the worker process instantiates its runner. Installing this
class needs no change to ATOM.
"""

COMPASS_RUNNER_QUALNAME = "atom.compass.runner.model_runner.CompassModelRunner"

__all__ = ["COMPASS_RUNNER_QUALNAME"]

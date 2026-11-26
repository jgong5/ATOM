# SPDX-License-Identifier: MIT
"""Compass adapter for agentx-harness (aiperf).

aiperf's plugin discovery imports this package while ``aiperf.plugin.plugins``
is still initialising, so nothing here imports aiperf at module level: an
import error escaping from here makes discovery drop the whole plugin with one
WARNING line.

Importing the package installs an import hook that rebinds module globals of
each aiperf module in ``REBIND`` once that module has executed. The runner's
``LoopScheduler`` becomes ``ClockPacedLoopScheduler``, so every phase runner's
scheduler, and the branch orchestrator and replay barrier it is handed to, pace
on the Compass clock; the hook then refuses the run unless this package's
strategy is the registered ``agentic_replay``. The runner's
``ReplayBarrierCoordinator`` becomes ``ClockReplayBarrierCoordinator``, whose
per-trace idle cap times out on that clock. The timing manager's
``StickyCreditRouter`` becomes ``CompassCreditRouter``, which binds that clock
to the traffic LP. The orchestrator's ``PhaseRunner`` becomes
``ClockPhaseRunner``, whose phase deadlines wait on that clock. ``time`` in the
phase lifecycle, the credit issuer and the strategy reads the clock, so phase
stamps and windows, ``issued_at_ns`` and the system idle cap are simulated; the
CLI's ``uuid4`` is fixed, so prompt token counts repeat from run to run. The
runner's ``asyncio`` becomes ``ClockAsyncio``, so its cancel drain times out on
that clock too. Where aiperf mints a session's id, ``uuid`` becomes
``session_ids``, so the id is drawn from the session's trace and branch, as a
real run's client draws it.
"""

import gc
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from compass_harness import fingerprint, session_ids

# Under its run file's ``rare_full_gc``, every aiperf process, which plugin
# discovery imports this package into, raises its generation-2 collection
# threshold as `atom.compass.run.rare_full_gc` has the run's other processes
# do, each name here that module's. Not by importing that module: it is heavy,
# and discovery runs in every aiperf process.
_run_file = os.environ.get("ATOM_COMPASS_RUN")  # `atom.compass.run.ENV`
if _run_file and json.loads(Path(_run_file).read_text()).get("rare_full_gc") is True:
    _t0, _t1, _ = gc.get_threshold()
    gc.set_threshold(_t0, _t1, 1_000_000)  # `atom.compass.run.RARE_FULL_GC_GEN2`

RUNNER = "aiperf.timing.phase.runner"
STRATEGY = "compass_harness.strategy:CompassAgenticReplay"
#: The path prefix of the adapter's two sockets, shared by every aiperf process.
ADDRESS_ENV = "COMPASS_HARNESS_IPC"
SIM_TIME = (("time", "compass_harness.timesource:SimTime"),)
#: aiperf module -> ((a global to rebind, the replacement as module:qualname, or
#: module: for the module itself), ...).
REBIND = {
    RUNNER: (
        ("LoopScheduler", "compass_harness.scheduler:ClockPacedLoopScheduler"),
        ("asyncio", "compass_harness.timesource:ClockAsyncio"),
        (
            "ReplayBarrierCoordinator",
            "compass_harness.scheduler:ClockReplayBarrierCoordinator",
        ),
    ),
    "aiperf.timing.manager": (
        ("StickyCreditRouter", "compass_harness.router:CompassCreditRouter"),
    ),
    "aiperf.timing.phase_orchestrator": (
        ("PhaseRunner", "compass_harness.strategy:ClockPhaseRunner"),
    ),
    "aiperf.timing.phase.lifecycle": SIM_TIME,
    "aiperf.credit.issuer": SIM_TIME,
    "aiperf.timing.strategies.agentic_replay": SIM_TIME,
    "aiperf.cli_runner": (("uuid4", "compass_harness.timesource:fixed_uuid4"),),
}
for _module in session_ids.MODULES:  # as a real run's aiperf binds it
    REBIND[_module] = REBIND.get(_module, ()) + (("uuid", session_ids.__name__ + ":"),)


def check_registered() -> None:
    """Raise unless ``agentic_replay`` resolves to this package's strategy."""
    from aiperf.plugin import plugins

    found = plugins.get_entry("timing_strategy", "agentic_replay").class_path
    if found != STRATEGY:
        raise RuntimeError(
            f"Compass plugin not registered: timing_strategy agentic_replay is "
            f"{found}, not {STRATEGY}. Look for a 'Plugin discovery' warning "
            "naming compass."
        )


class _RebindHook:
    """Meta-path finder that wraps each ``REBIND`` module's loader, then removes itself."""

    def __init__(self, pending: dict) -> None:
        self.pending = pending

    def find_spec(self, name, path=None, target=None):
        if name not in self.pending:
            return None
        rebinds = self.pending.pop(name)
        if not self.pending:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            for attr, replacement in rebinds:
                path, _, qualname = replacement.partition(":")
                value = importlib.import_module(path)
                setattr(module, attr, getattr(value, qualname) if qualname else value)
            if name == RUNNER:
                check_registered()

        spec.loader.exec_module = exec_module
        return spec


# Every aiperf process is spawned after the CLI process ran this, so all of
# them share one adapter socket address.
os.environ.setdefault(ADDRESS_ENV, f"/tmp/compass-harness-{os.getpid()}")
if pending := {m: r for m, r in REBIND.items() if m not in sys.modules}:
    sys.meta_path.insert(0, _RebindHook(pending))

# After the hook: when discovery drops a refused plugin, the hook re-imports this
# package as the runner loads, and the run fails on the same message.
fingerprint.check()

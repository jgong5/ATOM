# SPDX-License-Identifier: MIT
"""Compass adapter for agentx-harness (aiperf).

aiperf's plugin discovery imports this package while ``aiperf.plugin.plugins``
is still initialising, so nothing here imports aiperf at module level: an
import error escaping from here makes discovery drop the whole plugin with one
WARNING line.

Importing the package installs a one-shot import hook. When
``aiperf.timing.phase.runner`` has executed, the hook rebinds its
``LoopScheduler`` to ``ClockPacedLoopScheduler``, so every phase runner's
scheduler, and the branch orchestrator and replay barrier it is handed to, pace
on the Compass clock. It then refuses the run unless this package's strategy is
the registered ``agentic_replay``.
"""

import importlib.util
import sys

from compass_harness import fingerprint

RUNNER = "aiperf.timing.phase.runner"
STRATEGY = "compass_harness.strategy:CompassAgenticReplay"


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


class _RunnerHook:
    """Meta-path finder that wraps the runner module's loader, then removes itself."""

    def find_spec(self, name, path=None, target=None):
        if name != RUNNER:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            from compass_harness.scheduler import ClockPacedLoopScheduler

            module.LoopScheduler = ClockPacedLoopScheduler
            check_registered()

        spec.loader.exec_module = exec_module
        return spec


if RUNNER not in sys.modules:
    sys.meta_path.insert(0, _RunnerHook())

# After the hook: when discovery drops a refused plugin, the hook re-imports this
# package as the runner loads, and the run fails on the same message.
fingerprint.check()

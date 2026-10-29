# SPDX-License-Identifier: MIT
"""`time` for the modules whose clock reads all take the simulated clock.

Every name is the standard library's own, looked up on each read, except
`time`, `monotonic` and `perf_counter`: while an LP runtime is installed they
read its clock (`LPRuntime.read_clock`), and while none is, each is the
standard library's function itself. The other real clock reads (the `_ns`
forms, `clock_gettime`, `process_time`, `thread_time`) are not exposed: a
module that needs a real read keeps `import time` and `atom.utils.clock.now`
at each read that takes the simulated clock.

Import the module (`from atom.utils import time_shim as time`): a name
imported from it is whichever function was current at import. A module that
had `from time import monotonic` imports `read_monotonic as monotonic`, which
picks on each call. Setting or deleting a name sets or deletes it on `time`,
as it did when the importing module's `time` was `time` itself.
"""

import sys
import time as _time
from types import ModuleType

from atom.utils import clock

_REAL_ONLY = frozenset(
    (
        "time_ns",
        "monotonic_ns",
        "perf_counter_ns",
        "clock_gettime",
        "clock_gettime_ns",
        "process_time",
        "process_time_ns",
        "thread_time",
        "thread_time_ns",
    )
)


class _TimeShim(ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__") or name in _REAL_ONLY:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return getattr(_time, name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            setattr(_time, name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("__"):
            super().__delattr__(name)
        else:
            delattr(_time, name)


def _simulated(name: str) -> property:
    # A property, not `__getattr__`: a read on a real run costs no failed lookup.
    def read(self):
        rt = clock.installed()
        return getattr(_time, name) if rt is None else rt.read_clock

    return property(read)


for _name in ("time", "monotonic", "perf_counter"):
    setattr(_TimeShim, _name, _simulated(_name))


def read_monotonic() -> float:
    """`time.monotonic()`, or the LP clock while a runtime is installed."""
    return clock.now(_time.monotonic)


sys.modules[__name__].__class__ = _TimeShim

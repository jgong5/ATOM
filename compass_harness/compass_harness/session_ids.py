# SPDX-License-Identifier: MIT
"""Each agentic session's id from its trace and branch, the same in every run.

aiperf mints a session's id, which it sends as X-Correlation-ID and, as the
environment asks, X-Session-ID or X-Dynamo-Session-ID, with ``uuid.uuid4()``;
ATOM's router breaks an equal load between DP ranks by it
(``CoreManager._select_new_session_rank_locked``), so with random ids two runs
of one workload could place a new session apart. In each module ``MODULES``
names, every ``uuid4`` call mints a session id, and ``uuid`` is this module:
its ``uuid4`` digests the conversation and the parent session's id the
minting function holds (``SITES``) and how many ids that function drew for
them before, which counts a trace the sampler drew again, in its order.

aiperf mints them in its timing manager, a service process. A simulated run
binds this module in every aiperf process through ``compass_harness.REBIND``,
as plugin discovery imports the package. A real run's aiperf, where importing
``compass_harness`` would pace it on a simulated clock, loads this file by path
before its CLI (``client.sh``, ``cctraces_sim.sh``) and calls ``bind_by_fork``;
so it imports neither.
"""

import hashlib
import sys
import uuid as _uuid
from collections import Counter

MODULES = (
    "aiperf.timing.conversation_source",
    "aiperf.timing.trajectory_source",
    "aiperf.timing.strategies.agentic_replay",
)
#: Each minting function: the locals that may name its session's conversation,
#: then its parent's id; the first one bound of each.
SITES = {
    "next": (("conversation_id",), ()),
    "start_branch_child": (("child_conversation_id",), ("parent_correlation_id",)),
    "start_pre_session_child": (("child_conversation_id",), ()),
    "__init__": (("conversation_id",), ()),  # Trajectory's default id
    "_snapshot_for": (("child_cid", "root_id"), ("parent_corr",)),
    "_build_handoff_trajectories": (("trace_id",), ()),
    "_build_session_for_trace": (("trace_id",), ()),
}
_drawn: Counter = Counter()


def uuid4() -> _uuid.UUID:
    """The id the calling minting function's session gets."""
    frame = sys._getframe(1)
    if frame.f_code.co_name == "<lambda>":  # a dataclass field's default
        frame = frame.f_back
    site, local = frame.f_code.co_name, frame.f_locals
    if site not in SITES:
        raise RuntimeError(f"{__name__}: {site} mints no session id it knows")
    names, parents = SITES[site]
    conversation = next(local[n] for n in names if n in local)
    parent = next((local[n] for n in parents if local.get(n)), "")
    key = f"{site}/{parent}/{conversation}"
    _drawn[key] += 1
    digest = hashlib.blake2b(f"{key}/{_drawn[key]}".encode(), digest_size=16)
    return _uuid.UUID(bytes=digest.digest(), version=4)


def bind_by_fork() -> None:
    """Bind this module, loaded as ``session_ids``, as ``uuid`` in each module
    ``MODULES`` names, and have every process aiperf starts forked from this
    one, so its services inherit the binding whatever the interpreter's default
    start method (forkserver from Python 3.14). Exits once aiperf asks for
    another start method, as its CLI does where configured to."""
    import importlib
    import multiprocessing

    for name in MODULES:
        importlib.import_module(name).uuid = sys.modules[__name__]
    multiprocessing.set_start_method("fork", force=True)

    def only_fork(method, force=False):
        if method != "fork":
            sys.exit(
                f"{__name__}: aiperf asked for start method {method}, but its "
                "services take each session's id from this process by fork"
            )

    multiprocessing.set_start_method = only_fork


def __getattr__(name):
    return getattr(_uuid, name)

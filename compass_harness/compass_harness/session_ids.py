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

A simulated run binds it through ``compass_harness.REBIND``. A real run's aiperf
loads this file by path (``client.sh``, ``cctraces_sim.sh``) and binds it the
same, since importing ``compass_harness`` would pace aiperf on a simulated
clock; so it imports neither.
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


def __getattr__(name):
    return getattr(_uuid, name)

# SPDX-License-Identifier: MIT
"""Identity, order and lookahead for the simulated clock.

Three things live here, and nothing that uses them: who the participants are
(`LpId`), the order they are served in (`LpRegistry`), and how far each may run
ahead of each other one (`LookaheadMatrix`). The rule that reads them, the state
each participant carries, and the transport that carries a request are separate
and sit elsewhere.

Nothing here imports a device runtime, reads a clock, or opens a socket, which
is what makes it testable on any machine.

That claim is enforced, not asserted: every `.py` file under this package, at
any depth, is held to a three-module standard-library allowlist and to building
no `set`. Tooling that has to read the tree, parse source, or talk to anything
therefore does not belong here even when it is about time -- put it beside the
package, not inside it. The build-time audit of ATOM's blocking calls sits in
`atom/compass/audit/` for exactly this reason.
"""

from .identity import LpId
from .lookahead import (
    TRAFFIC_TO_ENGINE_FLOOR_SECONDS,
    InterLpLink,
    LinkClass,
    LookaheadMatrix,
)
from .registry import LpRegistry

__all__ = [
    "TRAFFIC_TO_ENGINE_FLOOR_SECONDS",
    "InterLpLink",
    "LinkClass",
    "LookaheadMatrix",
    "LpId",
    "LpRegistry",
]

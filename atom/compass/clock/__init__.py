# SPDX-License-Identifier: MIT
"""Identity, order, channels and the grant rule for the simulated clock.

What lives here: who the participants are (`LpId`), the order they are served
in (`LpRegistry`), the channels between them with the lookahead each declares
and the path distances that follow (`ChannelTable`), and the Clock Authority
that grants each one time from those distances and the messages it has
registered (`ClockAuthority`). The transport that carries a request to the
authority, and the runtime inside each participant, sit elsewhere.

Nothing here imports a device runtime, reads a clock, or opens a socket, which
is what makes it testable on any machine.

That claim is enforced, not asserted: every `.py` file under this package, at
any depth, is held to a standard-library allowlist and to building
no `set`. Tooling that has to read the tree, parse source, or talk to anything
therefore does not belong here even when it is about time -- put it beside the
package, not inside it. The build-time audit of ATOM's blocking calls sits in
`atom/compass/audit/` for exactly this reason.
"""

from .authority import (
    END,
    NER,
    TAR,
    BackdatedEvent,
    ClockAbort,
    ClockAuthority,
    LpRow,
)
from .channels import (
    Channel,
    ChannelTable,
    prefill_decode_table,
    single_engine_table,
)
from .identity import LpId
from .registry import LpRegistry

__all__ = [
    "END",
    "NER",
    "TAR",
    "BackdatedEvent",
    "Channel",
    "ChannelTable",
    "ClockAbort",
    "ClockAuthority",
    "LpId",
    "LpRegistry",
    "LpRow",
    "prefill_decode_table",
    "single_engine_table",
]

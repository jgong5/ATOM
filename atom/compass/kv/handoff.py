# SPDX-License-Identifier: MIT
"""What one deployment hands the next, and what a simulated peer may say.

A disaggregated prefill ends with a parameter blob on the producer's response.
The HTTP router does not interpret it -- it copies it onto the decode request
verbatim -- but it does refuse a prefill response that carries none, so the
blob is not optional decoration: without it there is no decode leg at all.

**The field set is not a choice.** It is re-derived from the pull backend's own
`request_finished`, which is the shape the router and the consumer were built
against, and the test beside this module parses that source rather than
trusting a copy of the list, so a field added there turns this red instead of
leaving a simulated deployment quietly one field short.

**Four of the fields name an endpoint, and a simulated deployment has none.**
The real backends fill them from the host they run on -- the machine's own
address and a port they actually bound -- which is the one answer a simulated
producer must not give: that blob names the machine running the simulation, so
anything that read it would open a connection to a real host that has no KV to
serve and no idea it was named. The answers here are the other way round --
well-formed, of the type the field carries, and impossible to mistake for a
listener:

| field | value | why it cannot be an endpoint |
|---|---|---|
| `remote_host` | `compass-simulated.invalid` | `.invalid` is reserved by RFC 2606 precisely so that it can never be delegated, so the name resolves nowhere, on any network, forever -- and it says in the name what it is |
| `remote_port` | `0` | the sockets API's own "no port": nothing listens on it, and a connection to it is refused at once rather than reaching whatever else is up on that host |
| `remote_handshake_port` | `0` | the same, for the same reason: there is no handshake because there are no two processes to hold one |
| `remote_engine_id` | `compass-simulated` | an identity, not an address. The real backends carry a constant here too -- the literal string `"None"` in both -- so nothing downstream derives a route from it |

The rest of the blob is the request's own truth and is taken from the sequence:
the block table it occupies, the parallel widths it was launched under, the
first sampled token the consumer resumes from, and the drafts and prefix-cache
hit that ride beside it.
"""

from __future__ import annotations

from typing import Any

#: A name reserved by RFC 2606 so that it can never resolve anywhere.
SIMULATED_HOST = "compass-simulated.invalid"

#: No socket was bound, so there is no port; this is what that is spelled as.
SIMULATED_PORT = 0

#: Who served the prefill. An identity, and deliberately not an address.
SIMULATED_ENGINE_ID = "compass-simulated"


def transfer_params(seq: Any, *, tp_size: int, dp_rank: int) -> dict[str, Any]:
    """The blob the router relays for *seq*, as a simulated producer sees it.

    `tp_size` and `dp_rank` describe the deployment rather than the request,
    so they are passed in from the config the connector was built with. Both
    are cast to `int` because the router takes `dp_rank` only if it is a
    number, and otherwise **substitutes its own registry value for the
    prefilling worker**, dropping it outright only when it has none either
    way. So a rank of the wrong type is not lost loudly; it is replaced by a
    plausible one, with no error anywhere.
    """
    drafts = getattr(seq, "spec_token_ids", None)
    draft_token_ids = (
        [int(token) for token in drafts] if drafts is not None and len(drafts) else []
    )
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": list(seq.block_table),
        "remote_engine_id": SIMULATED_ENGINE_ID,
        "remote_host": SIMULATED_HOST,
        "remote_port": SIMULATED_PORT,
        "remote_handshake_port": SIMULATED_PORT,
        "tp_size": int(tp_size),
        "dp_rank": int(dp_rank),
        "transfer_id": seq.id,
        "first_token_id": seq.output_tokens[0] if seq.output_tokens else None,
        "draft_token_ids": draft_token_ids,
        "prefix_cache_hit_tokens": getattr(seq, "prefix_cache_hit_tokens", 0),
    }

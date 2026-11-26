# SPDX-License-Identifier: MIT
"""A step's value of each named law column, for a design's identifiability.

The column set is an input (`deploy`'s `columns`). A column this tree's step
linear backend prices is counted by the backend itself: the step is priced by
a law whose every coefficient is one second, so each term's seconds are the
column's count, the same function a record's fit and a simulated run price
by. A column the backend does not price yet (an opt-in column the backend
does not carry yet) is counted by its provisional definition in
`PROVISIONAL` until the backend prices it (`provisional` names them); a
column neither knows refuses.
"""

from __future__ import annotations

from functools import cache

from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import DUTY_TAU, BatchView, sum_context, sum_tokens
from atom.compass.backends.step_linear import (
    DECODE,
    EAGER,
    PREFILL,
    StepLinearBackend,
    StepLinearLaw,
)


def _row_padding(view: BatchView, dp: int, buckets, min_spread: float = 0.0) -> float:
    """Decode: how uneven the rank's rows are, rows x max N_KV - Sum N_KV."""
    rows = view.decode
    return len(rows) * max(r.context_tokens for r in rows) - sum_context(rows)


def _peer_tokens(view: BatchView, dp: int, buckets, min_spread: float = 0.0) -> float:
    """Decode: the group's MoE tokens less the rank's own, the tokens its
    peers decode at the step; 0 on one rank."""
    if view.moe_tokens is None:
        return 0.0
    return float(max(view.moe_tokens - sum_tokens(view.decode), 0))


def _rows(view: BatchView, dp: int, buckets, min_spread: float = 0.0) -> float:
    """Decode: the rows the rank decodes, a per-sequence cost apart from their
    context (Kimi-K3.1's DCP exchange at each full-attention layer)."""
    return float(len(view.decode))


def _bucket(view: BatchView, buckets) -> int | None:
    """The token-graph bucket the step replays; None when it runs eagerly."""
    if view.group_needs_eager or view.tbo_tokens:
        return None
    peak = view.group_max_tokens or sum_tokens(view.requests)
    return next((b for b in sorted(buckets or ()) if b >= peak), None)


def _moe_imbalance(view: BatchView, dp: int, buckets, min_spread: float = 0.0) -> float:
    """Prefill: on a step running the variable-length gather (no token graph),
    the rows its busiest rank holds past an even share, dp x its tokens less
    the group's rows, where that is at least `min_spread` of the group's rows;
    else 0, and 0 on one rank."""
    if view.moe_rows is None or view.group_max_tokens is None:
        return 0.0
    if _bucket(view, buckets) is not None:
        return 0.0
    spread = max(dp * view.group_max_tokens - view.moe_rows, 0)
    return float(spread) if spread >= min_spread * view.moe_rows else 0.0


def _gather_rows(view: BatchView, dp: int, buckets, min_spread: float = 0.0) -> float:
    """Prefill: the rows the gather moves, every rank at the whole token-graph
    bucket the busiest rank's tokens take, or the group's rows past the
    largest bucket (eager)."""
    bucket = _bucket(view, buckets)
    if bucket is None:
        return float(view.moe_rows or sum_tokens(view.requests))
    return float(dp * bucket)


#: Provisional definitions of LLM-Booster#739's opt-in columns, by kind, used
#: only while this tree's backend does not price the column: row_padding,
#: peer_tokens and rows on decode, gather_rows and moe_imbalance over
#: `min_spread` on prefill.
PROVISIONAL = {
    "decode": {"row_padding": _row_padding, "peer_tokens": _peer_tokens, "rows": _rows},
    "prefill": {"moe_imbalance": _moe_imbalance, "gather_rows": _gather_rows},
}


def _vector(features, extra=None) -> dict:
    axes = [f for f in features if f != "step"]
    return {
        "coefficients": dict.fromkeys(features, 1.0),
        "hull": {"equal": dict.fromkeys(axes, 0.0), "facets": []},
        **(extra or {}),
    }


@cache
def _ones(
    kind: str,
    names: tuple,
    rung,
    buckets: tuple,
    routing,
    dp: int = 2,
    min_spread: float = 0.0,
) -> StepLinearBackend:
    """A backend whose law prices `kind` at `names`, every coefficient 1 s."""
    extra = {}
    if kind == "prefill":
        if "cold_peer_rows" in names:
            extra["duty_tau"] = DUTY_TAU
        if "padding" in names:
            extra["buckets"] = list(buckets)
        if {"gather_rows", "moe_imbalance"} & set(names):
            # a gather column states its group's width and its floor
            extra |= {"ranks": max(dp, 2), "min_spread": min_spread}
        law = {"provenance": "ones", "prefill": _vector(names, extra), "decode": {}}
    else:
        key = EAGER if rung is None else str(rung)
        # rowmod counts at its vector's modulus; one probes that it is priced
        modulus = {"rowmod": 1} if "rowmod" in names else None
        law = {"provenance": "ones", "decode": {key: _vector(names, modulus)}}
    if routing:
        law["moe_routing"] = {"experts": routing[0], "top_k": routing[1]}
    return StepLinearBackend(StepLinearLaw(law), dp=dp)


@cache
def backend_prices(kind: str, name: str) -> bool:
    """Whether this tree's step linear backend prices column `name` of `kind`."""
    base = PREFILL if kind == "prefill" else DECODE
    if name in base:
        return True
    probe = (*base, name) + (("padding",) if name == "gather_rows" else ())
    try:
        _ones(kind, probe, 1, (256,), (8, 2))
    except ValueError:
        return False
    return True


def provisional(kind: str, names) -> list[str]:
    """The columns of `names` counted by a provisional definition."""
    return [n for n in names if not backend_prices(kind, n)]


def values(
    view: BatchView, kind: str, names, deployment, min_spread: float = 0.0
) -> tuple[float, ...]:
    """`view`'s value of each of `names`, a `kind` vector's columns, in order;
    `min_spread` the share of the gather's rows below which `moe_imbalance`
    reads 0, the floor the law's fit took. `rowmod` refuses: its count is
    over the modulus a fitted law's vector states, which a config does not."""
    names = tuple(names)
    if kind == "decode" and "rowmod" in names:
        raise ValueError(
            "decode column rowmod is counted at its law's modulus, which a "
            "config does not state"
        )
    known = tuple(n for n in names if backend_prices(kind, n))
    other = [n for n in names if n not in known]
    unknown = [n for n in other if n not in PROVISIONAL[kind]]
    if unknown:
        raise ValueError(
            f"{kind} columns {unknown}: neither this tree's step linear backend "
            "nor a provisional definition counts them"
        )
    got = {}
    if known:
        # a law's vector holds its kind's base columns, whichever are asked;
        # gather_rows reads the vector's buckets, which come with padding
        base = PREFILL if kind == "prefill" else DECODE
        pad = ("padding",) if "gather_rows" in known else ()
        need = tuple(dict.fromkeys((*base, *known, *pad)))
        backend = _ones(
            kind,
            need,
            view.capture_rung,
            tuple(deployment.prefill_buckets),
            deployment.moe_routing,
            deployment.dp,
            min_spread,
        )
        try:
            cost = backend.price(view)
        except CostRefused as e:
            raise ValueError(f"the backend refuses the {kind} view: {e}") from e
        for term in cost.terms:
            what, _, feature = term.name.partition(".")
            if what == kind:
                got[feature] = term.seconds
    for n in other:
        got[n] = PROVISIONAL[kind][n](
            view, deployment.dp, deployment.prefill_buckets, min_spread
        )
    return tuple(float(got[n]) for n in names)

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law: one rank's step priced by a fitted linear law.

A step is prefill or decode, never both. Prefill has one coefficient vector,
decode one per capture rung, each over the features `StepStubBackend` reads:

    prefill:        [1, tokens, Sum N_Q^2, Sum N_Q.(N_KV - N_Q)]
    decode, rung r: [1, Sum N_KV, r.max N_KV - Sum N_KV]

A prefill vector may add `history`, Sum (N_KV - N_Q): the KV a chunk reads
on its cached history, which a short chunk on a deep history pays apart from
its attention arithmetic; then `cached_seqs`, its rows with cached context
(N_KV above N_Q), each of which adds a cost whatever its history's length;
and then `moe_rows`, the rows the data-parallel group's MoE gather carries
(`BatchView.moe_rows`), which every rank's MoE runs on.
After `moe_rows` it may add `serial_moe_rows`: `moe_rows` on a step the group did
not split for two-batch overlap (`BatchView.tbo_tokens` None), else 0, the MoE
rows no second micro-batch overlaps.
Then it may add `cold_peer_rows` (`cold_peer_rows`): the rows the group's MoE
gather carries from the rank's peers, `moe_rows` less its own tokens, times one
less its `BatchView.prefill_duty`, the share of the last seconds the rank spent
in its group's prefill steps (`shape.PrefillDuty`, at `DUTY_TAU`). On cell A
(DeepSeek-V4.1, DPA4-EP4) a prefill step that several ranks run right after
decode steps costs more than the same step deep in a stretch of prefill steps,
by about the peers' share of the gather: the prefill vector fitted without the
column priced the stretches (the warmup's first turns, c=256's prefill queue)
about 5% high and the isolated prefills of c=8 to c=128 2-4% low (#3747). A
vector with the column states its `duty_tau`, which must be `DUTY_TAU`, and a
step stating no `prefill_duty` is refused.
Last, a prefill vector may add `padding`, the rows its token graph runs past
the step's tokens (`token_padding`): ATOM replays a prefill's first layers at
the smallest of the vector's `buckets`, the prefill token-graph buckets the
runner recorded, that holds the most tokens any rank of the data-parallel group
scheduled. A data-parallel step stating no `group_max_tokens` is refused unless
its group split it for two-batch overlap or runs it eagerly for an attention
backend (`token_padding`), and a vector with a `padding` coefficient states its
`buckets`.
After it a prefill vector may add `eager_dummy` and `eager_dummy_tail`
(`eager_dummy`): 1 on a data-parallel rank's dummy forward in a prefill step
its group runs eagerly, past the largest of `buckets` (or with none), and the
second only where the group also runs a finishing prompt's tail, else 0. Such a
dummy (`dummy_view`, the row a record's dummy line states) is priced by the
prefill vector like any rank, and the group's step is the max over its ranks:
on real DPA records it holds its peers for about as long as a full-budget
prefill, whatever their tokens, which a one-token row's other features do not
carry.

A law prices one rank's step; the runner takes the group's step as the max
over its ranks. Its parallel layout is its `engine_config`'s, which a run
checks; a law stating `key` is refused. A backend built for a data-parallel
width above 1 names `UNIFORM_ROUTING` among every step's treatments: the MoE
segment is priced as if routing spread tokens evenly over the experts, which
no feature carries. A run refuses a law whose `platform` differs from the one
its spec states in any field, naming each and both values.

A decode vector may add one MoE column over `n`, the real tokens the step's
MoE runs on: a data-parallel group's gathered tokens (`BatchView.moe_tokens`),
or on one rank, which has no `moe_rows`, the step's own; a data-parallel step
stating no `moe_tokens` is refused. The column is `moe_tokens`, `n` itself, or
`moe_experts`, the routed experts those tokens are expected to reach,
`E (1 - (1 - K/E)^m)` for the law's `moe_routing` `{"experts": E, "top_k": K}`,
over `m`, `n` with the group's dummy forwards (`BatchView.group_dummies`)
counted as one token: each dummy runs one row, token 0 at position 0, so
together they reach one token's experts, where `n` counts each dummy's row.
Under a drafter a dummy runs `mtp_k + 1` rows, and `m` merges one row per dummy.
The gather pads every rank to the rung, and a pad row reaches no expert
(before #2340, the one expert set of token 0), so at one rung the MoE's work
follows the group's real tokens, which an uneven spread over the ranks leaves
well below the rung's rows. A law whose decode vector carries `moe_experts`
and that states no `moe_routing` is refused.
After it a decode vector may add `rowmod`, `rowmod(k)` over its rows in scheduled
order: the largest summed N_KV over the rows `i = r (mod k)`, with `k` the
vector's `"rowmod"` entry. It prices an order the same multiset of rows can
take at a different cost; `rowmod(1)` is Sum N_KV.

A decode step that replays no graph (an eager runner, or a batch wider than
the widest captured size) is the rung named `eager`, whose padding is zero.
At that rung, under a law with no `eager` vector, a DP dummy forward and a decode
while a peer prefills (`BatchView.dp_dummy`, `group_prefills`) cost 0 s, each
under its treatment (`eager_beside_peer`), not refused. So does, at any rung
the law holds no vector for, a decode-phase dummy every rank of its group runs
(`idle_group_dummy`): a P/D producer's, whose decodes are all such steps.

Every term is `estimated`, but a vector's stating `declared`, the origin of one
fitted on another subject: those are `declared`, with that note, never validated.
Each vector carries the convex hull of the features
it was fitted on, the `EAGER_DUMMY` indicators left out (`hull_axes`). A step outside it is priced by the same vector, its terms'
source is `EXTRAPOLATED` and their detail names the hull; it is not refused.
A vector may also carry, under `held_out`, the hull of the steps
`step_linear_fit` priced with a law fitted on the other records and their
errors per band: a term is validated iff its step lies inside that hull, so a
law without one prices every term unvalidated. A step the law has no vector for
is refused: prefill when the law has no prefill vector, a decode rung with
none, a step mixing prefill and decode rows, or one with image tokens. `compass
fit steps` leaves no captured rung without one: a rung no record reached gets
a declared vector filled from the fitted rungs beside it, under a hull holding
no point (`step_linear_fit.fill_rungs`), named under the law's `filled` and, by
law, its `filled_rungs`.

A law fitted on an MTP record holds a draft law of its own form under `draft`.
Its terms, named `draft.<kind>.<feature>`, are the step's `draft_terms`, priced from the
same features, on a step that produces no output too: there the draft is the
pass that fills the drafter's KV over the chunk.
A draft law may also hold a `propose` vector over `[1, seqs]`, seqs the step's
requests: on a prefill step that produces output it adds the drafter's
`propose` pass, which a middle chunk does not run. Its prefill vector may add
`seqs` and `out`, the step's requests and 1 on a prefill step that produces
output and 0 on one that does not: the draft-KV pass projects and writes a tail
only for requests that finish their prompts. Under either, a prefill step that
does not say whether it produces output is refused.
Beside `propose` it may hold a `short_propose` vector over `[1, seqs]` with a
`below` entry, a token count: an output prefill of fewer tokens prices its
pass by it instead.
A draft law may also hold a `draft_kv` vector over `[1, seqs]`: on a decode
step it adds the drafter's draft-KV pass, which runs before
`forward_done_event`, so its terms are the target's. A draft law with
`output_prefill_kv` true prices that pass on a prefill step that produces
output by its prefill vector, whose terms there are the target's too.
A data-parallel rank's dummy forward (`BatchView.dp_dummy`) runs no draft-KV
pass, so neither prices one there: its draft is the rest of the draft law's.
A serial drafter's draft law may hold a `forward` vector over the prefill
vector's base features, with `passes`, the draft forwards its `propose` runs.
On a decode and on a prefill step that produces output it prices the draft
as each of those forwards at its own rows (`serial_passes`), named
`draft.pass<i>.<feature>`, plus its `propose` vector, the rest of the draft,
over `[1, seqs]`; there neither its decode vectors nor its prefill vector
price the draft, which on a middle chunk is the draft-KV pass its prefill
vector prices.

The law is a mapping, as a run file holds it, `prefill` and `draft` optional:

    {"provenance": "<what it was fitted on>",
     "platform": {"device": ..., "software_stack": ..., "ptl": ..., "torch": ...},
     "prefill": VECTOR,   # with a padding column, its "buckets"
     "decode": {"eager": VECTOR, "1": VECTOR, "2": VECTOR, ...},
     "moe_routing": {"experts": E, "top_k": K},   # with a moe_experts column
     "draft": {"provenance": ..., "prefill": VECTOR, "decode": {...},
               "propose": VECTOR, "draft_kv": VECTOR,
               "short_propose": VECTOR,     # with "below": tokens
               "output_prefill_kv": true,
               "forward": VECTOR}}          # with "passes": K; each optional

    VECTOR = {"coefficients": {<feature>: seconds, ...},
              "hull": HULL,
              "rowmod": k,    # a decode vector with a `rowmod` coefficient only
              "duty_tau": s,  # a prefill vector with a `cold_peer_rows` coefficient only
              "declared": "<where it was fitted>",  # optional
              "held_out": {"hull": HULL, "bands": {<band>: ERRORS, ...}}}  # optional

    HULL = {"equal": {<feature>: value, ...}, "facets": [[a_1, ..., a_k, c], ...]}
    ERRORS = {"steps": n, "p10": e, "p50": e, "p90": e}, e = priced / real - 1

`coefficients` names every feature of its kind, `step` for the intercept.
`equal` holds each feature that was constant over the fit and the value a step
must have; `facets` bound the remaining non-intercept features, in the order
above, as `a . x + c <= 0` inside: the rows of qhull's `ConvexHull.equations`,
and two opposed rows holding each feature that was an affine function of
another over the fit to it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import (
    CostTerm,
    StepCost,
    eager_beside_peer,
    idle_group_dummy,
)
from atom.compass.backends.hull import Hull
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import (
    DUTY_TAU,
    IMAGES,
    BatchView,
    RequestShape,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.work import activated_experts

PREFILL = ("step", "tokens", "query_square", "query_cached")
HISTORY = PREFILL + ("history",)
#: The prefill vector's optional features, in the order they follow `PREFILL`.
PREFILL_EXTRA = (
    "history",
    "cached_seqs",
    "moe_rows",
    "serial_moe_rows",
    "cold_peer_rows",
    "seqs",
    "out",
    "padding",
    "eager_dummy",
    "eager_dummy_tail",
)
#: The prefill vector's columns over a DP dummy forward run eagerly (`eager_dummy`).
EAGER_DUMMY = ("eager_dummy", "eager_dummy_tail")
DECODE = ("step", "context", "graph_padding")
#: The MoE columns a decode vector may add, over its step's real MoE tokens.
MOE_COLUMNS = ("moe_tokens", "moe_experts")
#: The decode vector's optional features, in the order they follow `DECODE`.
DECODE_EXTRA = MOE_COLUMNS + ("rowmod",)
PROPOSE = ("step", "seqs")
EAGER = "eager"
SOURCE = "step-linear"
EXTRAPOLATED = "step-linear-extrapolated"
UNIFORM_ROUTING = "uniform routing"
#: The entries a vector keeps of what earlier steps priced (`_Vector.kept`).
KEPT = 1024


class _Kept(dict):
    """What a vector's earlier steps priced, by the inputs it was priced from;
    emptied once it holds `KEPT` entries, so a run's growing contexts do not
    grow it without bound."""

    def keep(self, key, value):
        if len(self) >= KEPT:
            self.clear()
        self[key] = value
        return value


class _Vector:
    """One kind's (and rung's) coefficients and the hull they were fitted on.

    `verdicts` keeps the hulls' verdicts on each point a step priced, and
    `kept` each term a step priced by its feature's count and those verdicts:
    a later step whose inputs repeat them takes them back rather than
    recomputing them (`StepLinearBackend._priced`)."""

    def __init__(self, name: str, features: tuple[str, ...], spec: Mapping) -> None:
        self.name = name
        coefficients = spec["coefficients"]
        if set(coefficients) != set(features):
            raise ValueError(
                f"the {name} vector has coefficients for {sorted(coefficients)}, "
                f"and its features are {list(features)}"
            )
        for feature, value in coefficients.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"the {name} vector's {feature} coefficient is {value}; a term "
                    "is a non-negative duration"
                )
        self.coefficients = tuple(float(coefficients[f]) for f in features)
        # The intercept, first in `features`, is not a hull coordinate, nor is
        # an `EAGER_DUMMY` column, an indicator (`hull_axes`).
        self.axes = hull_axes(features)
        axes = tuple(features[i] for i in self.axes)
        self.hull = Hull(f"the {name} hull", axes, spec["hull"])
        held = spec.get("held_out")
        self.held_out = held and Hull(f"the {name} held-out hull", axes, held["hull"])
        self.features = features
        self.declared = str(spec.get("declared", "")).strip()
        self.rowmod = int(spec["rowmod"]) if "rowmod" in features else None
        if self.rowmod is not None and self.rowmod < 1:
            raise ValueError(f"the {name} vector's rowmod modulus is {self.rowmod}")
        self.duty_tau = None
        if "cold_peer_rows" in features:
            self.duty_tau = spec.get("duty_tau")
            if self.duty_tau != DUTY_TAU:
                raise ValueError(
                    f"the {name} vector carries cold_peer_rows over a duty of time "
                    f"constant {self.duty_tau!r}; a view states its duty at "
                    f"{DUTY_TAU} s"
                )
        self.buckets = None
        if "padding" in features:
            self.buckets = tuple(sorted(int(b) for b in spec.get("buckets") or ()))
            if not self.buckets:
                raise ValueError(
                    f"the {name} vector carries padding and states no buckets, "
                    "the prefill token-graph buckets it pads to"
                )
        self.verdicts, self.kept = _Kept(), _Kept()


def moe_tokens(view: BatchView) -> int | None:
    """The real tokens `view`'s MoE runs on: its data-parallel group's, or on
    one rank, which has no `moe_rows`, its own decode rows'. None for a
    data-parallel view that states no group tokens."""
    if view.moe_rows is None:
        return sum_tokens(view.decode)
    return view.moe_tokens


def moe_column(
    feature: str, tokens: int, routing: tuple[int, int] | None, dummies: int = 0
) -> float:
    """MoE column `feature` of `MOE_COLUMNS` for a step of `tokens` real MoE
    tokens, `dummies` of them its data-parallel group's dummy forwards' rows:
    `moe_tokens` counts each row, `moe_experts`, over `routing`, the (experts,
    top-k) of `moe_routing`, counts the dummies' rows as one token, since
    every dummy runs the same row."""
    if feature == "moe_experts":
        reached = tokens - max(dummies - 1, 0)
        # An idle rank's dummy is an empty batch that adds no token to `tokens` (#3682), so with three idle ranks
        # beside a one-token decode the subtraction passes below zero and the column, then its term, went negative
        # (CostTerm refuses it). The step reaches at least the experts of one token; where `reached` was already
        # one or more, nothing changes.
        return activated_experts(*routing, max(reached, 1 if tokens or dummies else 0))
    return float(tokens)


def token_padding(view: BatchView, buckets: tuple[int, ...]) -> int | None:
    """The rows `view`'s prefill token graph runs past its tokens: the smallest
    of `buckets`, ascending, that holds its group's `group_max_tokens` (on one
    rank its own tokens), less its own; 0 past the largest, on a step the
    group split for two-batch overlap and on one an attention backend refuses
    a graph for (`group_needs_eager`), all of which ATOM runs eagerly
    (`ForwardMode._pad_to_token_graph`). None for a data-parallel view
    stating no `group_max_tokens`."""
    if view.tbo_tokens or view.group_needs_eager:
        return 0
    tokens = sum_tokens(view.requests)
    peak = tokens if view.moe_rows is None else view.group_max_tokens
    if peak is None:
        return None
    return next((b - tokens for b in buckets if b >= peak), 0)


def hull_axes(features) -> tuple[int, ...]:
    """The positions in `features` of a vector's hull coordinates: all but the
    intercept, first, and the `EAGER_DUMMY` indicators, which a hull in their
    extra dimensions would only split into its 0 and 1 faces, at a cost
    qhull's facet count grows by."""
    return tuple(i for i, f in enumerate(features) if i and f not in EAGER_DUMMY)


def dummy_view(view: BatchView) -> BatchView:
    """The view of a data-parallel rank's dummy forward at the prefill step a
    peer's `view` states: the one-token row at position 0 a record's dummy
    line holds (`ModelRunner.dummy_execution`), producing output as those
    lines say, with the group's fields its peer states."""
    import dataclasses

    return dataclasses.replace(
        view,
        requests=(RequestShape(1, 1, False),),
        capture_rung=None,
        produces_output=True,
        dp_dummy=True,
        group_prefills=True,
        draft_bs=None,
        target_bs=None,
        image_tokens=0,
    )


def eager_dummy(view: BatchView, buckets: tuple[int, ...]) -> tuple[int, int] | None:
    """`EAGER_DUMMY`'s counts for `view`: (1, 1 where its group runs a
    finishing prompt's tail, `group_tail`, else 0) on a data-parallel rank's
    dummy forward in a prefill step whose body runs eagerly, past the largest
    of `buckets` or with none (`token_padding`), else (0, 0); None on such a
    dummy whose view states no `group_max_tokens`."""
    if not (view.dp_dummy and view.prefill):
        return 0, 0
    if buckets:
        padding = token_padding(view, buckets)
        if padding is None:
            return None
        if padding:
            return 0, 0
    return 1, int(view.group_tail)


def cold_peer_rows(view: BatchView) -> float | None:
    """`view`'s `cold_peer_rows`: the rows its data-parallel group's MoE gather
    carries from its peers, `moe_rows` less its own tokens (a dummy forward's
    one row), times one less its `prefill_duty`; 0 on one rank, which has no
    `moe_rows`. None for a data-parallel view stating no `prefill_duty`."""
    if view.moe_rows is None:
        return 0.0
    if view.prefill_duty is None:
        return None
    peers = max(view.moe_rows - sum_tokens(view.requests), 0)
    return peers * (1.0 - min(max(view.prefill_duty, 0.0), 1.0))


def cached_seqs(rows) -> int:
    """How many of `rows` have cached context, N_KV above N_Q."""
    return sum(1 for r in rows if r.cached_tokens)


def rowmod(context, k: int) -> int:
    """The largest summed context over rows `i = r (mod k)`, in `context`'s order."""
    return max(sum(context[r::k]) for r in range(min(k, len(context))))


def serial_passes(view: BatchView, passes: int) -> list[tuple]:
    """The `PREFILL` features of each of a serial drafter's `passes` draft
    forwards over `view`'s step: the first runs the step's own rows; each
    later one a row a request, at its context, run at `view.draft_bs` rows
    where the drafter pads them to a draft graph's batch."""
    first = view.requests
    later = tuple(RequestShape(1, r.context_tokens, True) for r in first)
    rows = len(later) if view.draft_bs is None else view.draft_bs

    def features(shapes, tokens):
        return (1, tokens, sum_query_square(shapes), sum_query_cached(shapes))

    return [features(first, sum_tokens(first))] + [features(later, rows)] * (passes - 1)


class StepLinearLaw:
    """A prefill vector, a decode vector per rung, each with its hull."""

    def __init__(self, law: Mapping) -> None:
        self.fitted_on = str(law.get("provenance", "")).strip()
        if not self.fitted_on:
            raise ValueError("a law states what it was fitted on, under provenance")
        if "key" in law:
            raise ValueError(
                f"the law states key {law['key']}; its engine_config states the "
                "parallel layout it was fitted at"
            )
        self.prefill = None
        if "prefill" in law:
            prefill = law["prefill"]
            extra = [f for f in PREFILL_EXTRA if f in prefill["coefficients"]]
            self.prefill = _Vector("prefill", PREFILL + tuple(extra), prefill)
        self.decode = {
            None if rung == EAGER else int(rung): _Vector(
                f"decode rung {rung}",
                DECODE + tuple(f for f in DECODE_EXTRA if f in spec["coefficients"]),
                spec,
            )
            for rung, spec in law["decode"].items()
        }
        routing = law.get("moe_routing")
        self.moe_routing = routing and (int(routing["experts"]), int(routing["top_k"]))
        if not self.moe_routing and any(
            "moe_experts" in v.features for v in self.decode.values()
        ):
            raise ValueError(
                "a decode vector carries moe_experts and the law states no "
                "moe_routing, the experts and top-k it is counted over"
            )
        self.draft = StepLinearLaw(law["draft"]) if "draft" in law else None
        self.propose = law.get("propose") and _Vector(
            "propose", PROPOSE, law["propose"]
        )
        short = law.get("short_propose")
        self.short_propose = short and _Vector("short_propose", PROPOSE, short)
        if short and ("below" not in short or not self.propose):
            lacks = "no below" if "below" not in short else "no propose vector"
            raise ValueError(
                "the short_propose vector prices a pass under its below token "
                f"count, beside a propose vector; the law states {lacks}"
            )
        self.below = short and int(short["below"])
        self.draft_kv = law.get("draft_kv") and _Vector(
            "draft_kv", PROPOSE, law["draft_kv"]
        )
        self.output_prefill_kv = bool(law.get("output_prefill_kv"))
        self.forward = law.get("forward") and _Vector(
            "forward", PREFILL, law["forward"]
        )
        self.passes = self.forward and int(law["forward"]["passes"])
        if self.forward and self.passes < 1:
            raise ValueError(f"the forward vector states {self.passes} passes")


class StepLinearBackend(CostBackend):
    """Prices a step as its features times the law's coefficients, one term each."""

    def __init__(self, law: StepLinearLaw, dp: int = 1) -> None:
        self.law = law
        self.dp = dp
        self.treatments = (UNIFORM_ROUTING,) if dp > 1 else ()

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def _refuse(self, view: BatchView, reason: str) -> CostRefused:
        return CostRefused(view, [Refusal(SOURCE, reason)])

    def price(self, batch_view: BatchView) -> StepCost:
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}"
            )
        prefill, decode = batch_view.prefill, batch_view.decode
        if batch_view.image_tokens:
            raise self._refuse(batch_view, IMAGES)
        if prefill and decode:
            raise self._refuse(
                batch_view,
                "a step mixing prefill and decode rows; "
                "the law prices a step of one kind",
            )
        beside = eager_beside_peer(batch_view, SOURCE)
        if beside is not None and None not in self.law.decode:
            return beside
        idle = idle_group_dummy(batch_view, self.dp, SOURCE)
        if idle is not None and batch_view.capture_rung not in self.law.decode:
            return idle
        terms, kv = self._terms(self.law, batch_view, "")
        drafts = []
        if self.law.draft is not None:
            drafts, draft_kv = self._terms(self.law.draft, batch_view, "draft ")
            kv += draft_kv
        return StepCost(terms + kv, self.treatments, drafts)

    def _terms(self, law: StepLinearLaw, batch_view: BatchView, what: str) -> tuple:
        """`law`'s terms for the step, named for `what`, and apart its draft-KV
        pass's, which are the target's."""
        prefill, decode = batch_view.prefill, batch_view.decode
        if law.forward and (decode or batch_view.produces_output):
            return self._serial(law, batch_view, what)
        if prefill:
            vector, kind = law.prefill, "prefill"
            if vector is None:
                raise self._refuse(batch_view, f"no {what}prefill vector")
            counts = (
                1,
                sum_tokens(prefill),
                sum_query_square(prefill),
                sum_query_cached(prefill),
            )
            if "history" in vector.features:
                counts += (sum_context(prefill) - sum_tokens(prefill),)
            if "cached_seqs" in vector.features:
                counts += (cached_seqs(prefill),)
            if "moe_rows" in vector.features:
                if batch_view.moe_rows is None:
                    raise self._refuse(batch_view, "no MoE gather rows on this step")
                counts += (batch_view.moe_rows,)
            if "serial_moe_rows" in vector.features:
                counts += (0 if batch_view.tbo_tokens else batch_view.moe_rows,)
            if "cold_peer_rows" in vector.features:
                cold = cold_peer_rows(batch_view)
                if cold is None:
                    raise self._refuse(
                        batch_view,
                        "no prefill duty on this data-parallel step, which its "
                        "peers' MoE rows are priced by",
                    )
                counts += (cold,)
            output = batch_view.produces_output
            by_output = {
                "seqs": len(prefill) if output else 0,
                "out": int(bool(output)),
            }
            mine = [f for f in by_output if f in vector.features]
            if output is None and (law.propose or mine):
                raise self._refuse(
                    batch_view,
                    "a prefill step that does not say whether it produces output, "
                    f"which its {what}prefill terms depend on",
                )
            counts += tuple(by_output[f] for f in mine)
            if vector.buckets:
                padding = token_padding(batch_view, vector.buckets)
                if padding is None:
                    raise self._refuse(
                        batch_view,
                        "no group_max_tokens on this data-parallel step, which "
                        "its prefill token graph's bucket holds",
                    )
                counts += (padding,)
            if eager := [f for f in EAGER_DUMMY if f in vector.features]:
                got = eager_dummy(batch_view, vector.buckets or ())
                if got is None:
                    raise self._refuse(
                        batch_view,
                        "no group_max_tokens on this data-parallel dummy forward, "
                        "which says whether its group runs the step eagerly",
                    )
                counts += tuple(dict(zip(EAGER_DUMMY, got))[f] for f in eager)
        else:
            rung, kind = batch_view.capture_rung, "decode"
            vector = law.decode.get(rung)
            if vector is None:
                raise self._refuse(
                    batch_view,
                    f"no {what}decode vector for rung {EAGER if rung is None else rung}",
                )
            counts = (1, sum_context(decode), batch_view.graph_padding)
            if moe := [f for f in MOE_COLUMNS if f in vector.features]:
                tokens = moe_tokens(batch_view)
                if tokens is None:
                    raise self._refuse(batch_view, "no MoE group tokens on this step")
                dummies = batch_view.group_dummies
                counts += tuple(
                    moe_column(f, tokens, law.moe_routing, dummies) for f in moe
                )
            if vector.rowmod:
                counts += (rowmod([r.context_tokens for r in decode], vector.rowmod),)
        terms, kv = self._priced(law, vector, kind, counts, what), []
        dummy = batch_view.dp_dummy
        if prefill and law.output_prefill_kv and batch_view.produces_output:
            terms, kv = [], ([] if dummy else terms)
        seqs = (1, len(batch_view.requests))
        if prefill and law.propose is not None and batch_view.produces_output:
            propose = law.propose
            if law.short_propose and sum_tokens(prefill) < law.below:
                propose = law.short_propose
            terms += self._priced(law, propose, "propose", seqs, what)
        if decode and law.draft_kv is not None and not dummy:
            kv = self._priced(law, law.draft_kv, "draft_kv", seqs, what)
        return terms, kv

    def _serial(self, law: StepLinearLaw, batch_view: BatchView, what: str):
        """A serial draft's terms: each draft forward at its rows, then the
        rest of its draft; on a decode, apart, its draft-KV pass's."""
        terms, kv, seqs = [], [], (1, len(batch_view.requests))
        for i, counts in enumerate(serial_passes(batch_view, law.passes)):
            terms += self._priced(law, law.forward, f"pass{i}", counts, what)
        if law.propose is not None:
            terms += self._priced(law, law.propose, "propose", seqs, what)
        if batch_view.decode and law.draft_kv is not None:
            kv = self._priced(law, law.draft_kv, "draft_kv", seqs, what)
        return terms, kv

    def _priced(self, law, vector: _Vector, kind: str, counts: tuple, what: str):
        """`vector`'s terms for a step of features `counts`. The hulls'
        verdicts on a point, and a term from its count and those verdicts,
        are what an earlier step got from the same inputs (`_Vector.kept`)."""
        point = tuple(counts[i] for i in vector.axes)
        verdicts = vector.verdicts.get(point)
        if verdicts is None:
            inside = vector.hull.contains(point)
            validated = bool(vector.held_out and vector.held_out.contains(point))
            verdicts = vector.verdicts.keep(point, (inside, validated))
        terms = []
        for i, count in enumerate(counts):
            # The count's type too: 1 and 1.0 price alike and read apart.
            key = (what, kind, i, type(count), count, verdicts)
            term = vector.kept.get(key)
            if term is None:
                term = vector.kept.keep(
                    key, _term(law, vector, kind, what, i, count, verdicts)
                )
            terms.append(term)
        return terms

    def describe(self) -> str:
        return f"step linear law; {self.law.fitted_on}"


def _term(
    law, vector: _Vector, kind: str, what: str, i: int, count, verdicts
) -> CostTerm:
    """`vector`'s term for feature `i` at `count`, under the hulls' verdicts
    (inside the hull, inside the held-out hull) on the step's point."""
    (inside, validated), detail, source = verdicts, law.fitted_on, SOURCE
    if not inside:
        source = EXTRAPOLATED
        detail = f"{detail}; outside the {what}{vector.name} hull"
    coefficient = vector.coefficients[i]
    return CostTerm(
        f"{what.replace(' ', '.')}{kind}.{vector.features[i]}",
        count * coefficient,
        Provenance(
            Method.DECLARED if vector.declared else Method.ESTIMATED,
            f"{coefficient:g} s x {count}; {detail}",
            source,
            validated=validated and not vector.declared,
            note=vector.declared,
        ),
    )

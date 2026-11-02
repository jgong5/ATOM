# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law: one rank's step priced by a fitted linear law.

A step is prefill or decode, never both. Prefill has one coefficient vector,
decode one per capture rung, each over the features `StepStubBackend` reads:

    prefill:        [1, tokens, Sum N_Q^2, Sum N_Q.(N_KV - N_Q)]
    decode, rung r: [1, Sum N_KV, r.max N_KV - Sum N_KV]

A prefill vector may add `history`, Sum (N_KV - N_Q): the KV a chunk reads
on its cached history, which a short chunk on a deep history pays apart from
its attention arithmetic, and then `moe_rows`, the rows the data-parallel
group's MoE gather carries (`BatchView.moe_rows`), which every rank's MoE runs on.
After `moe_rows` it may add `serial_moe_rows`: `moe_rows` on a step the group did
not split for two-batch overlap (`BatchView.tbo_tokens` None), else 0, the MoE
rows no second micro-batch overlaps.

A law prices one rank's step; the runner takes the group's step as the max
over its ranks. Its parallel layout is its `engine_config`'s, which a run
checks; a law stating `key` is refused. A backend built for a data-parallel
width above 1 names `UNIFORM_ROUTING` among every step's treatments: the MoE
segment is priced as if routing spread tokens evenly over the experts, which
no feature carries. A run refuses a law whose
`machine_spec` is not its spec's `machine_spec_digest`, naming both.

A decode vector may add one MoE column over `n`, the real tokens the step's
MoE runs on: a data-parallel group's gathered tokens (`BatchView.moe_tokens`),
or on one rank, which has no `moe_rows`, the step's own; a data-parallel step
stating no `moe_tokens` is refused. The column is `moe_tokens`, `n` itself, or
`moe_experts`, the routed experts those tokens are expected to reach,
`E (1 - (1 - K/E)^n)` for the law's `moe_routing` `{"experts": E, "top_k": K}`.
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
A DP dummy forward (`BatchView.dp_dummy`) at that rung, under a law with no
`eager` vector, is `eager_dp_dummy`: 0 s, not refused.

Every term is `estimated`. Each vector carries the convex hull of the features
it was fitted on. A step outside it is priced by the same vector, its terms'
source is `EXTRAPOLATED` and their detail names the hull; it is not refused.
A vector may also carry, under `held_out`, the hull of the steps
`step_linear_fit` priced with a law fitted on the other records and their
errors per band: a term is validated iff its step lies inside that hull, so a
law without one prices every term unvalidated. A step the law has no vector for
is refused: prefill when the law has no prefill vector, a decode rung with
none, a step mixing prefill and decode rows, or one with image tokens.

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
Beside `propose` it may hold a `near_propose` vector over `[1, seqs]` with a
`lag` entry, in seconds: an output prefill whose device reaches the pass less
than `lag` behind its host, the view's `start_lag_s` plus the step's target
and draft-KV prices, prices the pass by it instead; a view stating no
`start_lag_s` is refused. The pass replays one draft graph at any prefill
length; what moves its window is how far the device runs behind its host.
A draft law may also hold a `draft_kv` vector over `[1, seqs]`: on a decode
step it adds the drafter's draft-KV pass, which runs before
`forward_done_event`, so its terms are the target's. A draft law with
`output_prefill_kv` true prices that pass on a prefill step that produces
output by its prefill vector, whose terms there are the target's too.

The law is a mapping, as a run file holds it, `prefill` and `draft` optional:

    {"provenance": "<what it was fitted on>",
     "machine_spec": "sha256:<hex>",
     "prefill": VECTOR,
     "decode": {"eager": VECTOR, "1": VECTOR, "2": VECTOR, ...},
     "moe_routing": {"experts": E, "top_k": K},   # with a moe_experts column
     "draft": {"provenance": ..., "prefill": VECTOR, "decode": {...},
               "propose": VECTOR, "draft_kv": VECTOR,
               "near_propose": VECTOR,      # with "lag": seconds
               "output_prefill_kv": true}}   # all four optional

    VECTOR = {"coefficients": {<feature>: seconds, ...},
              "hull": HULL,
              "rowmod": k,    # a decode vector with a `rowmod` coefficient only
              "held_out": {"hull": HULL, "bands": {<band>: ERRORS, ...}}}  # optional

    HULL = {"equal": {<feature>: value, ...}, "facets": [[a_1, ..., a_k, c], ...]}
    ERRORS = {"steps": n, "p10": e, "p50": e, "p90": e}, e = priced / real - 1

`coefficients` names every feature of its kind, `step` for the intercept.
`equal` holds each feature that was constant over the fit and the value a step
must have; `facets` bound the remaining non-intercept features, in the order
above, as `a . x + c <= 0` inside: the rows of qhull's `ConvexHull.equations`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, eager_dp_dummy
from atom.compass.backends.hull import Hull
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import (
    IMAGES,
    BatchView,
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.work import activated_experts

PREFILL = ("step", "tokens", "query_square", "query_cached")
HISTORY = PREFILL + ("history",)
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


class _Vector:
    """One kind's (and rung's) coefficients and the hull they were fitted on."""

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
        # The intercept, first in `features`, is not a hull coordinate.
        self.hull = Hull(f"the {name} hull", features[1:], spec["hull"])
        held = spec.get("held_out")
        self.held_out = held and Hull(
            f"the {name} held-out hull", features[1:], held["hull"]
        )
        self.features = features
        self.rowmod = int(spec["rowmod"]) if "rowmod" in features else None
        if self.rowmod is not None and self.rowmod < 1:
            raise ValueError(f"the {name} vector's rowmod modulus is {self.rowmod}")


def moe_tokens(view: BatchView) -> int | None:
    """The real tokens `view`'s MoE runs on: its data-parallel group's, or on
    one rank, which has no `moe_rows`, its own decode rows'. None for a
    data-parallel view that states no group tokens."""
    if view.moe_rows is None:
        return sum_tokens(view.decode)
    return view.moe_tokens


def moe_column(feature: str, tokens: int, routing: tuple[int, int] | None) -> float:
    """MoE column `feature` of `MOE_COLUMNS` for a step of `tokens` real MoE
    tokens, `moe_experts` over `routing`, the (experts, top-k) of `moe_routing`."""
    if feature == "moe_experts":
        return activated_experts(*routing, tokens)
    return float(tokens)


def rowmod(context, k: int) -> int:
    """The largest summed context over rows `i = r (mod k)`, in `context`'s order."""
    return max(sum(context[r::k]) for r in range(min(k, len(context))))


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
            extra = [
                f
                for f in ("history", "moe_rows", "serial_moe_rows", "seqs", "out")
                if f in prefill["coefficients"]
            ]
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
        if "short_propose" in law:
            raise ValueError(
                "the law splits the propose pass at a token count (short_propose), "
                "which a lag has replaced (near_propose); refit it"
            )
        near = law.get("near_propose")
        self.near_propose = near and _Vector("near_propose", PROPOSE, near)
        if near and ("lag" not in near or not self.propose):
            lacks = "no lag" if "lag" not in near else "no propose vector"
            raise ValueError(
                "the near_propose vector prices a pass the device reaches under "
                f"its lag, beside a propose vector; the law states {lacks}"
            )
        self.lag = near and float(near["lag"])
        self.draft_kv = law.get("draft_kv") and _Vector(
            "draft_kv", PROPOSE, law["draft_kv"]
        )
        self.output_prefill_kv = bool(law.get("output_prefill_kv"))


class StepLinearBackend(CostBackend):
    """Prices a step as its features times the law's coefficients, one term each."""

    def __init__(self, law: StepLinearLaw, dp: int = 1) -> None:
        self.law = law
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
        eager = not prefill and batch_view.capture_rung is None
        if eager and batch_view.dp_dummy and None not in self.law.decode:
            return eager_dp_dummy(SOURCE)
        terms, kv = self._terms(self.law, batch_view, "")
        drafts = []
        if self.law.draft is not None:
            drafts, draft_kv = self._terms(
                self.law.draft, batch_view, "draft ", terms + kv
            )
            kv += draft_kv
        return StepCost(terms + kv, self.treatments, drafts)

    def _terms(
        self, law: StepLinearLaw, batch_view: BatchView, what: str, target=()
    ) -> tuple:
        """`law`'s terms for the step, named for `what`, and apart its draft-KV
        pass's, which are the target's; `target` holds the target's terms,
        which run before a draft's."""
        prefill, decode = batch_view.prefill, batch_view.decode
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
            if "moe_rows" in vector.features:
                if batch_view.moe_rows is None:
                    raise self._refuse(batch_view, "no MoE gather rows on this step")
                counts += (batch_view.moe_rows,)
            if "serial_moe_rows" in vector.features:
                counts += (0 if batch_view.tbo_tokens else batch_view.moe_rows,)
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
                counts += tuple(moe_column(f, tokens, law.moe_routing) for f in moe)
            if vector.rowmod:
                counts += (rowmod([r.context_tokens for r in decode], vector.rowmod),)
        terms, kv = self._priced(law, vector, kind, counts, what), []
        if prefill and law.output_prefill_kv and batch_view.produces_output:
            terms, kv = [], terms
        seqs = (1, len(batch_view.requests))
        if prefill and law.propose is not None and batch_view.produces_output:
            propose = law.propose
            if law.near_propose:
                if batch_view.start_lag_s is None:
                    raise self._refuse(
                        batch_view,
                        "an output prefill that does not say how far behind its "
                        "host the device starts it, which its propose pass "
                        "depends on",
                    )
                ahead = sum(t.seconds for t in (*target, *kv))
                if batch_view.start_lag_s + ahead < law.lag:
                    propose = law.near_propose
            terms += self._priced(law, propose, "propose", seqs, what)
        if decode and law.draft_kv is not None:
            kv = self._priced(law, law.draft_kv, "draft_kv", seqs, what)
        return terms, kv

    def _priced(self, law, vector: _Vector, kind: str, counts: tuple, what: str):
        """`vector`'s terms for a step of features `counts`."""
        detail, source = law.fitted_on, SOURCE
        if not vector.hull.contains(counts[1:]):
            source = EXTRAPOLATED
            detail = f"{detail}; outside the {what}{vector.name} hull"
        validated = bool(vector.held_out and vector.held_out.contains(counts[1:]))
        return [
            CostTerm(
                f"{what.replace(' ', '.')}{kind}.{feature}",
                count * coefficient,
                Provenance(
                    Method.ESTIMATED,
                    f"{coefficient:g} s x {count}; {detail}",
                    source,
                    validated=validated,
                ),
            )
            for feature, count, coefficient in zip(
                vector.features, counts, vector.coefficients
            )
        ]

    def describe(self) -> str:
        return f"step linear law; {self.law.fitted_on}"

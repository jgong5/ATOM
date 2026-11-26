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
After it a prefill vector may add `gather_rows` (or `gather_pad_rows`) and
`moe_imbalance` (`GATHER_COLUMNS`, `gather_column`), each over its
data-parallel group's MoE gather and 0 on one rank. `gather_rows`
(`gather_rows`) is the rows the gather carries: on a step whose group replays a
prefill token graph every rank runs the bucket (`token_bucket`,
`ForwardMode._pad_to_token_graph`), so the gather carries `ranks` buckets,
however few tokens each rank scheduled, its pad rows reaching no expert; on one
it runs eagerly the variable-length gather carries each rank's own count,
`moe_rows`. `gather_pad_rows` (`gather_pad_rows`) is the pad rows alone,
`gather_rows` less `moe_rows`: the gather's rows past those its ranks scheduled
under a token graph, 0 on a step run eagerly. It spans what `gather_rows` adds
to `moe_rows` without repeating it (#3720: on cell A's e3v2 record
`gather_rows` held a VIF of 150 against `moe_rows`, its pad rows 2.5), and a
vector takes one of the two. `moe_imbalance` (`moe_imbalance`) is the rows by which a group
running that variable-length gather is uneven: `ranks` times its
`group_max_tokens` less its `moe_rows`, the rows its busiest rank holds past an
even share, where that is at least `min_spread` of `moe_rows`, else 0, and 0
on a step that replays a token graph, whose ranks all run the bucket.
`uneven_rows` (`uneven_rows`), after it, is the `moe_rows` of a step whose
`moe_imbalance` at that `min_spread` is not 0, else 0: the rows of a gather its
group runs uneven, each of which waits on the busiest rank at every layer's
collective. It spans what `moe_imbalance` adds to the base columns without
repeating `tokens` and `moe_rows` on the busiest rank (#3720: on cell A's e3v2
record `moe_imbalance` held a VIF of 17.4 and pushed `moe_rows` to 18.8,
`uneven_rows` 1.7 and 5.7). A vector
with any of them states `ranks`, its group's width, and one with
`moe_imbalance` or `uneven_rows` its `min_spread`; `gather_rows` and
`gather_pad_rows` take the vector's `buckets`, so they come with `padding`. A
data-parallel step stating no `group_max_tokens` is refused.
After the gather columns it may add `tail_peers` (`tail_peers`), stating its
group's width `ranks` as they do: on a data-parallel rank's own prefill forward
at a step whose group runs a finishing prompt's tail (`BatchView.group_tail`),
the group's other ranks running real rows, `ranks` less one less its
`group_dummies`; else 0, on one rank and on a dummy forward, whose tail
`eager_dummy_tail` prices. The tail runs the decoder on every rank of the
group, its MoE collectives being the group's (DeepSeek-V4.1), and each peer
running a real prefill beside it holds the group past the max of its ranks'
prices: on #752's cell A v2 (DPA4, MoE over TP) the prefill vector fitted
without the column on its calibration record and three addenda priced the
output prefill steps of 2-4 real ranks 3-6% low in-sample (about 8 ms a real
peer) and on the c=64-512 targets alike, their middle chunks within 2%.
After `seqs` and `out` it may add `peer_small` and `peer_large` (`peer_share`):
on a data-parallel rank's own prefill forward, output or middle chunk, whose
peers run real rows (its group's MoE gather, `BatchView.moe_rows`, carrying
more rows than its own tokens), 1 in the first where those peer rows are under
`PEER_SMALL` of its own and 1 in the second where they are not, else 0; 0 on
one rank and on a dummy forward. Each rank's forward waits on its peers' at
every layer's MoE collective, and a peer's small real prefill costs the group
device time a max over its ranks' prices does not hold: on cell B's P (#3720,
DeepSeek-V4.1 DP2 EP2, store_bF5) steps whose largest rank ran 8k-24k tokens
beside a peer under a quarter of them ran 21-28 ms over the law on every
record, beside a larger peer 6-10 ms and beside a dummy 1-3 ms, both ranks'
device windows alike and their host launches long done.
Then, for a model with KDA layers (`kda_layers`: Kimi-K3), it may add `KDA`, the
recurrent state work its step does beside the tokens (`kda_columns`), four of
them the step's `BatchView.kda` counts: `kda_seqs`, the rows holding a KDA
state (each read and written once); `kda_saves`, the midstep checkpoints the
forward writes; `kda_forks`, the rows that read their state from another slot
(a prefix-cache hit resuming at a checkpoint slot; Kimi-K3.1's PAGE-backed
checkpoints restore instead, a move); `kda_moves`, the state slot moves and
paged checkpoint copies run before the forward; and, off the rows,
`kda_pad_blocks`, the padding of the prefill rows' last chunks in chunks,
`Sum ceil(N_Q / c) - N_Q / c` at the recurrence's chunk `c` (`KDA_CHUNK`). The chunk kernel's work is
`Sum ceil(N_Q / c)` chunks, which `tokens` holds but for that padding: a column
of the chunks themselves is `tokens / c` but for it: on synthetic Kimi records
(`tests/compass/test_step_linear_kda.py`) its VIF against the other columns was
1.2e5, the padding's 2.0. A vector with these columns states its
`kda_chunk`, which must be `KDA_CHUNK`, and a step stating no `kda` counts is
refused.
Last, a prefill vector may add `padding`, the rows its token graph runs past
the step's tokens (`token_padding`): ATOM replays a prefill's first layers at
the smallest of the vector's `buckets`, the prefill token-graph buckets the
runner recorded, that holds the most tokens any rank of the data-parallel group
scheduled. A data-parallel step stating no `group_max_tokens` is refused unless
its group split it for two-batch overlap or runs it eagerly for an attention
backend (`token_padding`), and a vector with a `padding` coefficient states its
`buckets`.
After it a prefill vector may add `graph_tokens` and `graph_query_cached`
(`GRAPH_COLUMNS`, `graph_columns`): its `tokens` and its `query_cached`,
Sum N_Q.(N_KV - N_Q), on a step that replays a prefill token graph
(`token_bucket` above 0), else 0. On #3720's cell A (DeepSeek-V4.1, DPA4-EP4)
a chunk the token graph replays pays about 3.5-4 us more per token than an
eager one, about what `padding` prices each pad row at, whatever its history:
the vector fitted without the columns priced eager prefills within 1% (p50)
and the prefilling rank of a token-graph step of 3k-7k tokens 10-20% low, on
the e3v2 calibration record and on the agentic targets alike. Its dummy peers,
priced at the bucket's pad rows, then set the group's price, which so holds
none of the chunk's attention over its history: 10-18% low at 150k-400k
cached tokens. Those chunks are the follow-up turns of agentic sessions, whose
device time sits on their TTFT at low concurrency. A vector with either column
states its `buckets`.
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
Before the MoE column a decode vector may add `row_padding` (`row_padding`), its
rows' own rectangle less their context, `len.max N_KV - Sum N_KV`: how uneven
the requests the step decodes are, apart from the slots its rung adds past them,
which `graph_padding` also holds. At one data-parallel rank the two follow each
other within a rung; a step whose rung a wider peer set separates them.
After it, still before the MoE column, a decode vector may add `ctx_hinge`
(`ctx_hinge`), its rows' context past the vector's knot `ctx_knot`, in tokens,
`Sum max(0, N_KV - ctx_knot)`: a context slope that bends at the knot, which
`context`'s one slope over every row does not carry. On #752's cell A
(DeepSeek-V4.1, DPA4) the calibration record's decode law, fitted without it,
priced its rung 1-2 steps, by their rows' mean context, 0.2% high below 32k
tokens, 0.4% low at 32k-131k, 0.8% low at 131k-524k and level from 524k. A
vector with the column states its `ctx_knot`, a positive token count.
After the MoE column it may add `peer_tokens` (`peer_tokens`), the real MoE
tokens its data-parallel peers bring to the gather, `moe_tokens` less its own
decode rows' tokens, 0 on one rank: at a rung whose `moe_experts` reaches
nearly every expert, the column no longer grows with the group's tokens while
the gather and the experts' rows still do.
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
              "ctx_knot": K,  # a decode vector with a `ctx_hinge` coefficient only
              "duty_tau": s,  # a prefill vector with a `cold_peer_rows` coefficient only
              "kda_chunk": c, # a prefill vector with `KDA` coefficients only
              "ranks": n,     # a prefill vector with a `GATHER_COLUMNS` column only
              "min_spread": f,  # a prefill vector with `moe_imbalance` or `uneven_rows` only
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
#: A KDA model's prefill columns: `BatchView.kda`'s counts in its order, then the
#: prefill rows' chunk padding (`kda_columns`).
KDA = ("kda_seqs", "kda_saves", "kda_forks", "kda_moves", "kda_pad_blocks")
#: The chunk, in tokens, Kimi-K3's KDA prefill recurrence runs its rows in: aiter's
#: FlashKDA path (`kimi_delta_attn.flash_kda.FLASH_KDA_CHUNK`), which its
#: `chunk_kimi_delta_attn` takes at Kimi-K3's K = V = 128 with a safe gate; its
#: default pipeline runs 64.
KDA_CHUNK = 32
#: The prefill vector's optional features, in the order they follow `PREFILL`.
PREFILL_EXTRA = (
    "history",
    "cached_seqs",
    "moe_rows",
    "serial_moe_rows",
    "cold_peer_rows",
    "gather_rows",
    "gather_pad_rows",
    "moe_imbalance",
    "uneven_rows",
    "tail_peers",
    "seqs",
    "out",
    "peer_small",
    "peer_large",
    *KDA,
    "padding",
    "graph_tokens",
    "graph_query_cached",
    "eager_dummy",
    "eager_dummy_tail",
)
#: The prefill vector's columns over a DP dummy forward run eagerly (`eager_dummy`).
EAGER_DUMMY = ("eager_dummy", "eager_dummy_tail")
#: The prefill vector's columns over a step that replays a prefill token graph
#: (`graph_columns`): its tokens and its Sum N_Q.(N_KV - N_Q) there, else 0.
GRAPH_COLUMNS = ("graph_tokens", "graph_query_cached")
#: The prefill vector's columns over its data-parallel peers' share of its rows
#: (`peer_share`): below `PEER_SMALL` of them, and at or above it.
PEER_SHARE = ("peer_small", "peer_large")
PEER_SMALL = 0.25
DECODE = ("step", "context", "graph_padding")
#: The MoE columns a decode vector may add, over its step's real MoE tokens.
MOE_COLUMNS = ("moe_tokens", "moe_experts")
#: The decode vector's optional features, in the order they follow `DECODE`:
#: first those `step_linear_fit` counts with the base columns on each rank's row,
#: `row_padding` then `ctx_hinge`; then those it adds over a step's group and
#: rows, the MoE column, `peer_tokens`, and `rowmod` last.
DECODE_EXTRA = ("row_padding", "ctx_hinge") + MOE_COLUMNS + ("peer_tokens", "rowmod")
#: The prefill vector's columns over its data-parallel group's MoE gather that
#: state the group's width, `ranks`.
GATHER_COLUMNS = (
    "gather_rows",
    "gather_pad_rows",
    "moe_imbalance",
    "uneven_rows",
    "tail_peers",
)
#: The `GATHER_COLUMNS` that state the vector's `min_spread`.
SPREAD_COLUMNS = ("moe_imbalance", "uneven_rows")
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
        self.ctx_knot = None
        if "ctx_hinge" in features:
            self.ctx_knot = spec.get("ctx_knot")
            knot = self.ctx_knot
            if isinstance(knot, bool) or not isinstance(knot, int) or knot < 1:
                raise ValueError(
                    f"the {name} vector carries ctx_hinge past a knot of {knot!r} "
                    "tokens; it states its knot, a positive token count, under "
                    "ctx_knot"
                )
        self.duty_tau = None
        if "cold_peer_rows" in features:
            self.duty_tau = spec.get("duty_tau")
            if self.duty_tau != DUTY_TAU:
                raise ValueError(
                    f"the {name} vector carries cold_peer_rows over a duty of time "
                    f"constant {self.duty_tau!r}; a view states its duty at "
                    f"{DUTY_TAU} s"
                )
        if any(f in KDA for f in features) and spec.get("kda_chunk") != KDA_CHUNK:
            raise ValueError(
                f"the {name} vector carries KDA columns over a chunk of "
                f"{spec.get('kda_chunk')!r} tokens; a view states its padding at "
                f"{KDA_CHUNK}"
            )
        self.buckets = None
        if "padding" in features:
            self.buckets = tuple(sorted(int(b) for b in spec.get("buckets") or ()))
            if not self.buckets:
                raise ValueError(
                    f"the {name} vector carries padding and states no buckets, "
                    "the prefill token-graph buckets it pads to"
                )
        self.ranks = self.min_spread = None
        if gathered := [f for f in GATHER_COLUMNS if f in features]:
            self.ranks = spec.get("ranks")
            if not isinstance(self.ranks, int) or self.ranks < 2:
                raise ValueError(
                    f"the {name} vector carries {' and '.join(gathered)} over a "
                    f"data-parallel group of {self.ranks!r} ranks; it states its "
                    "group's width, 2 or more, under ranks"
                )
        graphed = [f for f in GRAPH_COLUMNS if f in features]
        if graphed and not self.buckets:
            raise ValueError(
                f"the {name} vector carries {' and '.join(graphed)} and no "
                "padding, whose buckets say which steps replay a token graph"
            )
        if "gather_rows" in features and "gather_pad_rows" in features:
            raise ValueError(
                f"the {name} vector carries gather_rows and gather_pad_rows, "
                "the second the first less moe_rows"
            )
        for padded in ("gather_rows", "gather_pad_rows"):
            if padded in features and not self.buckets:
                raise ValueError(
                    f"the {name} vector carries {padded} and no padding, whose "
                    "buckets say which steps replay a token graph"
                )
        if spread := [f for f in SPREAD_COLUMNS if f in features]:
            self.min_spread = spec.get("min_spread")
            if not isinstance(self.min_spread, (int, float)) or self.min_spread < 0:
                raise ValueError(
                    f"the {name} vector carries {' and '.join(spread)} and states "
                    f"its min_spread as {self.min_spread!r}, not a non-negative share"
                )


def kda_layers(hf_config) -> int:
    """How many KDA layers the model config `hf_config` states, in its text
    config's `linear_attn_config` (Kimi-K3's, Kimi-Linear's); 0 for none."""
    text = getattr(hf_config, "text_config", None) or hf_config
    linear = getattr(text, "linear_attn_config", None) or {}
    return len(linear.get("kda_layers") or ()) if isinstance(linear, Mapping) else 0


def kda_columns(view: BatchView) -> tuple | None:
    """`KDA`'s columns of `view`'s step: its `kda` counts, then the padding of
    its prefill rows' last chunks of `KDA_CHUNK` tokens, in chunks, the
    chunked recurrence running each row from its own first token; no padding
    on a step whose rows hold no state (a data-parallel dummy's). None for a
    view stating no counts."""
    if view.kda is None:
        return None
    padded = (
        sum(-r.query_tokens % KDA_CHUNK for r in view.prefill) if view.kda[0] else 0
    )
    return (*view.kda, padded / KDA_CHUNK)


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


def token_bucket(view: BatchView, buckets: tuple[int, ...]) -> int | None:
    """The prefill token-graph bucket `view`'s step replays: the smallest of
    `buckets`, ascending, that holds its group's `group_max_tokens` (on one
    rank its own tokens); 0 past the largest, on a step the group split for
    two-batch overlap and on one an attention backend refuses a graph for
    (`group_needs_eager`), all of which ATOM runs eagerly
    (`ForwardMode._pad_to_token_graph`). None for a data-parallel view
    stating no `group_max_tokens`."""
    if view.tbo_tokens or view.group_needs_eager:
        return 0
    peak = sum_tokens(view.requests) if view.moe_rows is None else view.group_max_tokens
    if peak is None:
        return None
    return next((b for b in buckets if b >= peak), 0)


def token_padding(view: BatchView, buckets: tuple[int, ...]) -> int | None:
    """The rows `view`'s prefill token graph runs past its tokens: its
    `token_bucket` less its own tokens; 0 where it runs eagerly. None for a
    data-parallel view stating no `group_max_tokens`."""
    bucket = token_bucket(view, buckets)
    return bucket and bucket - sum_tokens(view.requests)


def graph_columns(view: BatchView, buckets: tuple[int, ...]) -> tuple | None:
    """`GRAPH_COLUMNS`' counts for `view`: its prefill rows' tokens and their
    Sum N_Q.(N_KV - N_Q) where its step replays a prefill token graph
    (`token_bucket` above 0), else (0, 0), as on a step run eagerly. None for
    a data-parallel view stating no `group_max_tokens`."""
    bucket = token_bucket(view, buckets)
    if bucket is None:
        return None
    if not bucket:
        return 0, 0
    return sum_tokens(view.prefill), sum_query_cached(view.prefill)


def gather_rows(view: BatchView, buckets: tuple[int, ...], ranks: int) -> int | None:
    """The rows `view`'s data-parallel group's MoE gather carries on a prefill
    step of `ranks` ranks: `ranks` times the `token_bucket` it replays, to
    which every rank pads its rows, or, run eagerly, the variable-length
    gather's `moe_rows`; 0 on one rank. None for a data-parallel view stating
    no `group_max_tokens`."""
    if view.moe_rows is None:
        return 0
    bucket = token_bucket(view, buckets)
    if bucket is None:
        return None
    return ranks * bucket if bucket else view.moe_rows


def gather_pad_rows(
    view: BatchView, buckets: tuple[int, ...], ranks: int
) -> int | None:
    """The pad rows `view`'s data-parallel group's MoE gather carries on a
    prefill step of `ranks` ranks, `gather_rows` less `moe_rows`: `ranks`
    times the `token_bucket` it replays less the rows its ranks scheduled,
    rows that reach no expert; 0 on a step run eagerly, whose
    variable-length gather carries no pad, and on one rank. None for a
    data-parallel view stating no `group_max_tokens`."""
    rows = gather_rows(view, buckets, ranks)
    if rows is None:
        return None
    return max(rows - (view.moe_rows or 0), 0)


def gather_column(
    feature: str,
    view: BatchView,
    buckets: tuple[int, ...],
    ranks: int,
    min_spread: float | None = None,
) -> int | None:
    """`view`'s count of `feature`, one of `GATHER_COLUMNS`, over a group of
    `ranks` ranks and the prefill token-graph `buckets`, `moe_imbalance` and
    `uneven_rows` at `min_spread`: the one function the fit and the backend
    count each by."""
    if feature == "gather_rows":
        return gather_rows(view, buckets, ranks)
    if feature == "gather_pad_rows":
        return gather_pad_rows(view, buckets, ranks)
    if feature == "moe_imbalance":
        return moe_imbalance(view, buckets, ranks, min_spread)
    if feature == "uneven_rows":
        return uneven_rows(view, buckets, ranks, min_spread)
    if feature == "tail_peers":
        return tail_peers(view, ranks)
    raise ValueError(f"{feature!r} is not one of {GATHER_COLUMNS}")


def tail_peers(view: BatchView, ranks: int) -> int:
    """`view`'s `tail_peers` in a data-parallel group of `ranks` ranks: on a
    rank's own prefill forward at a step its group runs a finishing prompt's
    tail at (`BatchView.group_tail`), the group's other ranks that run real
    rows, `ranks` less one less `group_dummies`; else 0, on one rank (no
    `moe_rows`), on a dummy forward, on a decode and at a step with no tail."""
    if view.moe_rows is None or view.dp_dummy or not view.prefill:
        return 0
    if not view.group_tail:
        return 0
    return max(ranks - 1 - view.group_dummies, 0)


def uneven_rows(
    view: BatchView, buckets: tuple[int, ...], ranks: int, min_spread: float
) -> int | None:
    """The rows of `view`'s group's variable-length MoE gather where its
    group of `ranks` ranks runs it uneven: its `moe_rows` where its
    `moe_imbalance` at `min_spread` is not 0, else 0 (on a step that replays
    a token graph and on one rank too). None for a data-parallel view stating
    no `group_max_tokens`."""
    idle = moe_imbalance(view, buckets, ranks, min_spread)
    if idle is None:
        return None
    return view.moe_rows if idle else 0


def moe_imbalance(
    view: BatchView, buckets: tuple[int, ...], ranks: int, min_spread: float
) -> int | None:
    """The rows by which `view`'s group of `ranks` ranks runs its
    variable-length MoE gather uneven: `ranks` times its `group_max_tokens`
    less its `moe_rows`, where that is at least `min_spread` times
    `moe_rows`, else 0; 0 on a step that replays a token graph
    (`token_bucket`), whose ranks all run the bucket, and on one rank. None
    for a data-parallel view stating no `group_max_tokens`."""
    if view.moe_rows is None:
        return 0
    bucket = token_bucket(view, buckets)
    if bucket is None or view.group_max_tokens is None:
        return None
    if bucket:
        return 0
    idle = ranks * view.group_max_tokens - view.moe_rows
    return idle if idle > 0 and idle >= min_spread * view.moe_rows else 0


def row_padding(rows) -> int:
    """`rows`' own rectangle less their context, `len.max N_KV - Sum N_KV`: how
    uneven the decode rows are, apart from the slots a rung adds past them."""
    if not rows:
        return 0
    return len(rows) * max(r.context_tokens for r in rows) - sum_context(rows)


def ctx_hinge(rows, knot: int) -> int:
    """`rows`' context past `knot` tokens, `Sum max(0, N_KV - knot)`: the KV
    the decode rows read beyond the knot, which a vector's `ctx_hinge` prices
    at a slope of its own."""
    return sum(max(0, r.context_tokens - knot) for r in rows)


def peer_tokens(view: BatchView) -> int | None:
    """The real MoE tokens `view`'s data-parallel peers bring to its group's
    gather: `moe_tokens` less its own decode rows' tokens, 0 on one rank. None
    for a data-parallel view stating no `moe_tokens`."""
    if view.moe_rows is None:
        return 0
    if view.moe_tokens is None:
        return None
    return max(view.moe_tokens - sum_tokens(view.decode), 0)


def hull_axes(features) -> tuple[int, ...]:
    """The positions in `features` of a vector's hull coordinates: all but the
    intercept, first, the `EAGER_DUMMY` and `PEER_SHARE` indicators and the
    `tail_peers` count of 0 to `ranks` less 1, which
    a hull in their extra dimensions would only split into its 0 and 1 faces,
    at a cost qhull's facet count grows by."""
    return tuple(
        i
        for i, f in enumerate(features)
        if i and f not in EAGER_DUMMY and f not in PEER_SHARE and f != "tail_peers"
    )


def dummy_view(view: BatchView) -> BatchView:
    """The view of a data-parallel rank's dummy forward at the prefill step a
    peer's `view` states: the one-token row at position 0 a record's dummy
    line holds (`ModelRunner.dummy_execution`), producing output as those
    lines say, with the group's fields its peer states; under a KDA model no
    state work, as its batch schedules no request holding a state."""
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
        kda=None if view.kda is None else (0,) * len(view.kda),
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


def peer_share(view: BatchView) -> tuple[int, int]:
    """`PEER_SHARE`'s counts for `view`: on a data-parallel rank's own prefill
    forward whose peers run real rows, `moe_rows` less its own tokens, (1, 0)
    where those rows are under `PEER_SMALL` of its own and (0, 1) where they
    are not; else (0, 0), on one rank, which has no `moe_rows`, and on a dummy
    forward too."""
    if view.moe_rows is None or view.dp_dummy:
        return 0, 0
    own = sum_tokens(view.requests)
    peers = view.moe_rows - own
    if own <= 0 or peers <= 0:
        return 0, 0
    small = peers < PEER_SMALL * own
    return int(small), int(not small)


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
            for feature in GATHER_COLUMNS:
                if feature not in vector.features:
                    continue
                got = gather_column(
                    feature,
                    batch_view,
                    vector.buckets or (),
                    vector.ranks,
                    vector.min_spread,
                )
                if got is None:
                    raise self._refuse(
                        batch_view,
                        "no group_max_tokens on this data-parallel step, which "
                        f"its group's MoE gather's {feature} is counted from",
                    )
                counts += (got,)
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
            if share := [f for f in PEER_SHARE if f in vector.features]:
                got = dict(zip(PEER_SHARE, peer_share(batch_view)))
                counts += tuple(got[f] for f in share)
            if kda := [f for f in KDA if f in vector.features]:
                if (columns := kda_columns(batch_view)) is None:
                    raise self._refuse(
                        batch_view,
                        "no KDA state counts on this step, which its prefill "
                        "vector's KDA columns price",
                    )
                counts += tuple(dict(zip(KDA, columns))[f] for f in kda)
            if vector.buckets:
                padding = token_padding(batch_view, vector.buckets)
                if padding is None:
                    raise self._refuse(
                        batch_view,
                        "no group_max_tokens on this data-parallel step, which "
                        "its prefill token graph's bucket holds",
                    )
                counts += (padding,)
            if graph := [f for f in GRAPH_COLUMNS if f in vector.features]:
                got = dict(
                    zip(GRAPH_COLUMNS, graph_columns(batch_view, vector.buckets))
                )
                counts += tuple(got[f] for f in graph)
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
            if "row_padding" in vector.features:
                counts += (row_padding(decode),)
            if "ctx_hinge" in vector.features:
                counts += (ctx_hinge(decode, vector.ctx_knot),)
            if moe := [f for f in MOE_COLUMNS if f in vector.features]:
                tokens = moe_tokens(batch_view)
                if tokens is None:
                    raise self._refuse(batch_view, "no MoE group tokens on this step")
                dummies = batch_view.group_dummies
                counts += tuple(
                    moe_column(f, tokens, law.moe_routing, dummies) for f in moe
                )
            if "peer_tokens" in vector.features:
                if (peers := peer_tokens(batch_view)) is None:
                    raise self._refuse(batch_view, "no MoE group tokens on this step")
                counts += (peers,)
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
        """`vector`'s terms for a step of features `counts`."""
        detail, source = law.fitted_on, SOURCE
        point = tuple(counts[i] for i in vector.axes)
        if not vector.hull.contains(point):
            source = EXTRAPOLATED
            detail = f"{detail}; outside the {what}{vector.name} hull"
        validated = bool(vector.held_out and vector.held_out.contains(point))
        method = Method.DECLARED if vector.declared else Method.ESTIMATED
        return [
            CostTerm(
                f"{what.replace(' ', '.')}{kind}.{feature}",
                count * coefficient,
                Provenance(
                    method,
                    f"{coefficient:g} s x {count}; {detail}",
                    source,
                    validated=validated and not vector.declared,
                    note=vector.declared,
                ),
            )
            for feature, count, coefficient in zip(
                vector.features, counts, vector.coefficients
            )
        ]

    def describe(self) -> str:
        return f"step linear law; {self.law.fitted_on}"

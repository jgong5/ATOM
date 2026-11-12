# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step linear law fitted from real step records.

    python -m atom.compass.backends.step_linear_fit RECORD... --out LAW --platform JSON [--form FORM] [--moe-routing E,K] [--engine FILE]

Each RECORD is a real run's step record (`steps/dp0.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`), labelled by `step_labels`, whose docstring
states the labels; the command writes each record's per-step table beside it.
A record refuses, before any table is written, when the run.json beside the
`steps` directory its path resolves into names JIT compiles its server ran
(`step_labels.refusal`).
A step whose device label is cut is counted under its reason and not fitted.
Prefill labels come from trace runs alone (`step_labels`); given none, the
prefill vector is left out and named under `unfitted`. A trace run gives no
decode or host label, so its record fits only prefill.
`fit` and `fit_host` take the labelled samples, grouped into runs by their
provenance's `run`, and each returns its laws and a report. `fit`'s report
holds the runs it was fitted on, each vector's fit figures and hull, and, once
`mark_held_out` has run, the held-out errors per run; its `seed` fixes the
`rowmod` shuffles, so the same samples and arguments give the same bytes.

The law states the records' `engine_config`, read from the `.engine.json`
beside each (`engine_config`) or from `--engine FILE` in their place, and a run
under another engine config refuses it. At its `data_parallel_size` N above 1
(EP off) the records come in runs of
N, each run's `dp0.jsonl` to `dp<N-1>.jsonl`. A step is its ranks' lines
joined on step index, labelled with the smallest rank label (the others wait
for the last rank to enter), and a prefill row adds the line's `moe_rows` and,
when some prefill step of the samples the group split for two-batch overlap,
`serial_moe_rows` (`StepLinearLaw`). A
rank that ran the step as a dummy forward adds no row. A step that some rank
lacks otherwise or cut, or whose ranks' rows fall to different vectors, is
counted under `ranks` and not fitted. The law reports each step's rank label
spread, `max / min - 1`, as its p50 and p90 under `spread`. The group's step
is the max over its ranks, so each vector is fitted by max-affine regression
(`_fit_form`): from the fit on every rank's row and from each feature's axis,
the lowest after a first round going on, each pass refits on the row
each step's price picks, a step line-searched until it lowers the squared
relative error, until no step changes rank or the step's ranks tie; MAD drops
rerun it. The vector's solve, warm steps and hull read each step's row at the
rank it prices the step as (`priced_rank`); a decode rung that adopts a
column (below) is fitted again under it. Each
vector reports, under `fit.all_rows`, the centre and scaled MAD of the fit on
every rank's row, each step priced as its max over ranks, and under
`fit.passes` the start, passes, how they stopped, `settled` or `tie`, and the
squared error `F`; more than `FIT_PASSES` passes refuse the fit. A held-out
step is priced and banded as its costliest rank.

Records with a host key (`dp0.key.json`, from the real runner) also give the
host laws as a run file's `host` entry: `key`, and `laws[kind][segment]`,
seconds on `(1, tokens, seqs)`, fitted on the samples' host labels. They take
the step linear law's solver, MAD drops and hulls; `seqs` in a fixed
proportion to `tokens` on every sample (equal, or a decode under MTP, where
each row computes its drafts too) gets zero. Two keys refuse. The law, as
`platform`, and the host key state the platform fields `--platform` gives for
their kinds, `step_linear_law` and `host_laws`, which a run's spec must match,
and the host key's `engine` is the law's `engine_config`. A segment outside `HOST_SEGMENTS` is
fitted for a kind whose samples carry it: `W_inputs` and `L_mtp`, on an MTP
record, and `W_recv`, on a later pipeline stage's. A forward whose `W_launch` its kind's
solve drops above the law paid a first-use JIT compile (`_jit`): it and the
forward after it, whose period takes up the stall, are left out of every host
law; the report lists the JIT forwards per run under `jit_steps`. A `W_launch`
label leaves out the time its launch spent in HIP's blocking calls, which the
report states per run and kind under `blocked` (`_blocked`); it names under
`unstamped` the runs whose records state none, so their `W_launch` labels are
wall times.
A forward's `E` that admits requests (`step_labels.admissions`) is left out of
its kind's law: the law then prices the engine loop alone, and `admit`, seconds
on `step_labels.admit_x`, prices what the forwards of every kind that admit spend
beyond it, fitted on their `E` with their kind's law as a fixed part. Records
on too few such forwards give `admit` null, so a run refuses a forward that
admits a request rather than charge it nothing, and the report says why under
`unfitted`.

Under a launch provider that declares features (`launch_features`), a step it
predicts otherwise than its line observed refuses, naming it; each host law
adopts features as columns (`_launch_columns`), their coefficients under the
entry's `columns[kind][segment]`, and `_jit` solves `W_launch` with every one.

The same solver fits the host entry's ``tokenizer`` term on a recorded
server's tokenizer calls (`fit_tokenizer`), and its ``http_intake`` term on its
HTTP intake stamps (`fit_intake`): ``per_byte_s`` and ``idle_s`` on the lone
requests, ``fixed_s`` from a burst, and ``bytes`` over every request's body.

Each equation is divided by its own label and the coefficients are bound to be
non-negative. Steps whose relative residual is more than `MAD_K` scaled MADs
from the median are dropped and the rest refitted, until the kept set settles.
The dropped steps that lead a record's steps of one vector, each slower than
the law, are its warm steps (shape compiles and first touches), and so is a
dropped step anywhere that ran more than `WARM_FACTOR` times its price, such as
the first short chunk on a deep history: counted with their seconds apart from
the dropped ones, never charged, and listed per run under `warm_steps`, in the
order the law's `runs` names the runs. A feature with no variance gets coefficient
zero and is named. Each vector reports its samples,
warm and dropped counts, the centre and scaled MAD of its relative residuals,
the condition number of its column-scaled design and
the convex hull of its fitted steps, which `StepLinearLaw` reads: each step's
row at the rank the vector, with any column it adopted, prices highest
(`priced_rank`), the rank a run prices the step as. A vector with
no more fitted steps than coefficients, or whose steps span no hull, is left
out and named under `unfitted`; `StepLinearBackend` refuses its steps.

On an MTP record a step also carries a draft label (`step_labels`). `fit` fits
the steps that carry one on their draft labels the same way, and writes that law
under the law's `draft`; it holds no `held_out` unless its records time a
serial drafter's passes (below). A step with a propose label
(`step_labels`) is fitted there on its draft label less that label, the
draft-KV pass a middle chunk runs too, and left out when nothing remains; the
propose labels fit the draft law's `propose` vector, seconds on `[1, seqs]`,
whatever lag the device ran at.
A decode's draft-KV label fits the draft law's `draft_kv` vector alike. An
output prefill's fits the draft law's prefill vector in place of its draft
label, which, all after `forward_done_event`, fits the `propose` vector, and
sets the draft law's `output_prefill_kv`. The target's vectors fit the device
label less the draft-KV label.
On an MTP record both prefill vectors, the target's and the draft's, add
`seqs` and `out` (`StepLinearLaw`).
Records whose decodes time a serial drafter's passes (`passes_s`) fit the
draft law's `forward` vector instead, each pass a sample at its own features
(`serial_passes`), with `passes` the count every such step ran, and its
`propose` vector on what each decode's draft label holds past its passes; the
draft law's prefill vector then fits the middle chunks alone, whose draft is
the draft-KV pass, in the `base` form, and it has no decode or `draft_kv`
vector. `mark_held_out`
gives each of that draft law's vectors `held_out` as it gives the law's, on
the steps each is fitted on.

Records whose engine files state the prefill token-graph buckets their runner
recorded (`PREFILL_BUCKETS`) fit a prefill vector that adds `padding` and
states them (`StepLinearLaw`); every record states the same buckets, or none
does. A record whose engine file predates the field states them in its host
key only where its runner wrote graph sets there (`stated_buckets`); `--engine`
states them for one that states none. Records stating them nowhere fit no
`padding`, named under `unfitted`.
The draft law adds none: #2976 measured the target's token graph alone.

Forms (`FORMS`): `base`, the features `StepLinearLaw` always prices; `history`,
prefill plus `Sum (N_KV - N_Q)`, the KV a short chunk reads on a deep history;
`shared`, decode with the context and padding slopes shared by every rung and
an intercept per rung; `cached_seqs`, prefill plus its rows with cached
context, each a cost whatever its history's length.

With `moe_routing`, the model's routed experts and top-k (`--moe-routing E,K`),
the law states `moe_routing` and, outside `shared`, a decode rung may adopt one
MoE column (`StepLinearLaw`), `moe_tokens` or `moe_experts`, by the rule
`_moe_column` states; the vector records under `fit.moe_column` the figures the
rule read.
An MTP record's draft law adopts none.

Outside `shared`, a decode rung may adopt a `rowmod(k)` column (`StepLinearLaw`),
read from its rows in scheduled order, by the rule `_rowmod` states; the
vector then names `k` and records, under `fit.rowmod`, the MAD without the
column, the mean MAD with each step's rows shuffled, and per held-out record
the p50 error without and with it.

With two or more runs it also prints, per form, the error on each run of a
law fitted on the others, warm steps left out: per kind and rung, or prefill
token band, the p10, p50 and p90 of `predicted / measured - 1`; and the same
per host law. The law's own form's errors are written into it: under each
vector's `held_out`, the convex hull of the rows they were priced by, each
step's costliest rank's, and their percentiles per band, which `StepLinearLaw`
reads to mark a term validated. A cheaper rank's row is not hulled: no error was
measured at it. A vector whose held-out steps span no hull gets none.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from atom.compass.backends import json_file, launch_features, step_labels
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.shape import (
    sum_context,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)
from atom.compass.backends.step_grid_plan import PREFILL_BUCKETS, keyed_of
from atom.compass.backends.step_labels import CUTS, HOST_SEGMENTS, host_kind
from atom.compass.backends.step_linear import (
    DECODE,
    EAGER,
    HISTORY,
    MOE_COLUMNS,
    PREFILL,
    PROPOSE,
    cached_seqs,
    moe_column,
    moe_tokens,
    rowmod,
    serial_passes,
    token_padding,
)

FORMS = ("base", "history", "shared", "cached_seqs")
CONTEXT_PARALLEL = ("prefill_context_parallel_size", "decode_context_parallel_size")
MAD_K = 3.5
#: A relative residual this small is never an outlier, however tight the rest.
MAD_FLOOR = 0.01
MAD_PASSES = 20
#: Passes after which a max-affine fit whose error still falls refuses. Near a
#: tie between ranks each line-searched pass gains little, so a fit that ends
#: at a tie takes far more passes than one that settles.
FIT_PASSES = 10_000
#: A step that ran this many times its price is a one-off first touch, not scatter.
WARM_FACTOR = 5.0
TOKEN_BANDS = (512, 2048, 8192)
#: Steps at its rung a record needs before a `rowmod` choice holds it out.
ROWMOD_CELL_STEPS = 100
#: The fraction of a rung's scaled MAD a `rowmod` column must cut: a smaller cut is
#: within the sampling error of a MAD over a few hundred steps.
ROWMOD_GAIN = 0.1
#: Seeded shuffles of each step's rows whose mean MAD the `rowmod` column must not beat.
ROWMOD_SHUFFLES = 8
HOST_KINDS = ("decode", "prefill", "middle")
HOST_FEATURES = ("step", "tokens", "seqs")
#: What a lone HTTP request's intake costs: per request, and per body byte.
INTAKE_FEATURES = ("idle", "bytes")
#: What a lone tokenizer call costs: per call, per token, and for a segment
#: tokenizer call per character its stored-prefix search compared.
TOKENIZER_FEATURES = ("fixed", "tokens", "chars")
#: What a forward that admits requests spends once, per prompt token past the
#: prefix-cache hit and per token within it (`step_labels.admit_x`).
ADMIT_FEATURES = ("admits", "uncached", "cached")
#: Why a law fitted on records stating their prefill buckets nowhere has no `padding`.
NO_BUCKETS = "no record states prefill buckets, so the prefill vector adds no padding"


def _rung_key(rung: int | None) -> str:
    return EAGER if rung is None else str(rung)


def _features(
    samples,
    form: str,
    dp: int,
    draft: bool = False,
    serial: bool = False,
    routing: tuple[int, int] | None = None,
    buckets: tuple[int, ...] = (),
) -> tuple[list[tuple], dict]:
    """One rank's fittable samples as (vector, features, seconds, step, context,
    tokens, dummies), and how many were cut, per reason.

    `vector` is `prefill` or a decode rung's key; features are in `StepLinearLaw`'s
    order, plus `history` or `cached_seqs` in its form, above `dp` 1 `moe_rows` on prefill and
    under `serial` `serial_moe_rows`, and under `draft` or on a step with a draft
    label `seqs` and `out` on prefill, with `buckets` its `padding`, as
    `StepLinearBackend` counts them;
    `context` is the decode rows' N_KV in scheduled order, empty for prefill;
    `tokens` a decode step's real MoE tokens under `routing`, the model's
    (experts, top-k), which its MoE column reads, else None; `dummies` its
    group's dummy forwards (`BatchView.group_dummies`), which the column's
    `moe_experts` counts as one token.
    """
    cut, rows = dict.fromkeys(CUTS, 0), []
    for s in samples:
        view = s.view
        if s.cut:
            cut[s.cut] += 1
        else:
            tokens = None
            if view.prefill:
                r = view.prefill
                key = "prefill"
                counts = (1, sum_tokens(r), sum_query_square(r), sum_query_cached(r))
                if form == "history":
                    counts += (sum_context(r) - sum_tokens(r),)
                if form == "cached_seqs":
                    counts += (cached_seqs(r),)
                if dp > 1:
                    if view.moe_rows is None:
                        raise ValueError(
                            f"{s.provenance['run']} step {s.step} has no moe_rows, which "
                            "a data-parallel law prices prefill from"
                        )
                    counts += (view.moe_rows,)
                    if serial:
                        counts += (0 if view.tbo_tokens else view.moe_rows,)
                if draft or s.draft_s is not None:
                    output = bool(view.produces_output)
                    counts += (len(r) if output else 0, int(output))
                if buckets:
                    if (padding := token_padding(view, buckets)) is None:
                        raise ValueError(
                            f"{s.provenance['run']} step {s.step} has no "
                            "group_max_tokens, which its prefill token graph's "
                            "bucket holds"
                        )
                    counts += (padding,)
            else:
                key = _rung_key(view.capture_rung)
                counts = (1, sum_context(view.decode), view.graph_padding)
                if routing and (tokens := moe_tokens(view)) is None:
                    raise ValueError(
                        f"{s.provenance['run']} step {s.step} has no moe_tokens, "
                        "which a data-parallel law's MoE column reads"
                    )
            context = tuple(r.context_tokens for r in view.decode)
            dummies = view.group_dummies
            rows.append((key, counts, s.device_s, s.step, context, tokens, dummies))
    return rows, cut


def _host_x(s) -> tuple[int, int, int]:
    return 1, sum_tokens(s.view.requests), len(s.view.requests)


def _host_rows(samples) -> list[tuple]:
    """(kind, segment, (1, tokens, seqs), seconds, run, launch) per host label
    of each sample, but the `E` of one that admits requests."""
    rows = []
    for s in samples:
        rows += [
            (host_kind(s.view), k, _host_x(s), v, s.provenance["run"], s.launch)
            for k, v in s.host.items()
            if k != "E" or not s.admitted[0]
        ]
    return rows


def _host_price(host, kind, segment, x, launch) -> float | None:
    """What the host entry `host` prices a `kind` forward's `segment` at `x`
    that launched `launch`, its columns included; None with no law."""
    c = host["laws"].get(kind, {}).get(segment)
    columns = host.get("columns", {}).get(kind, {}).get(segment, {})
    priced = sum(v * launch[f] for f, v in columns.items())
    return None if c is None else float(np.dot(c, x)) + priced


def _admit_rows(samples, host) -> list[tuple]:
    """(`admit_x` of admitted, `E` its kind's law prices, `E`) per sample
    that admits requests, has a positive `E` and a kind with an `E` law."""
    return [
        (step_labels.admit_x(s.admitted), base, s.host["E"])
        for s in samples
        if s.admitted[0]
        and s.host.get("E", 0) > 0
        and (base := _host_price(host, host_kind(s.view), "E", _host_x(s), s.launch))
        is not None
    ]


def label_records(records, dp: int = 1) -> tuple[list, dict]:
    """`records`, runs of `dp` ranks in rank order, as `step_labels` samples, each
    run named by its records' digests; and each record's table rows."""
    samples, tables = [], {}
    for i in range(0, len(records), dp):
        run = records[i : i + dp]
        name = ", ".join(map(step_labels.digest, run))
        for record, (got, table) in zip(run, step_labels.label_run(run, name)):
            samples += got
            tables[record] = table
    return samples, tables


def _runs(samples, dp: int) -> dict[str, list[list]]:
    """`samples` per run, in the order runs first appear, as each rank's samples."""
    runs: dict[str, dict[int, list]] = {}
    for s in samples:
        runs.setdefault(s.provenance["run"], {}).setdefault(s.rank, []).append(s)
    if any(len(ranks) != dp for ranks in runs.values()):
        n = sum(map(len, runs.values()))
        raise ValueError(f"{n} records are not runs of {dp} ranks")
    return {run: [ranks[r] for r in sorted(ranks)] for run, ranks in runs.items()}


def fit_host(samples) -> tuple[dict, dict]:
    """The host laws over `samples`, as a run file's `host` entry, and their report."""
    from scipy.spatial import QhullError

    keys = sorted({json.dumps(s.provenance["config"], sort_keys=True) for s in samples})
    if len(keys) != 1:
        raise ValueError(
            f"the records carry {len(keys)} host keys ({' / '.join(keys)}); "
            "host laws are fitted under one"
        )
    launch, graphs = launch_features.stated(json.loads(keys[0]))
    for s in samples:
        if (predicted := launch.predict(s.view, graphs)) != s.launch:
            raise ValueError(
                f"{s.provenance['run']} step {s.step} launched {s.launch}, and "
                f"its launch provider predicts {predicted}"
            )
    jit = _jit(samples, launch.features)
    after = {(run, rank, n + 1) for run, rank, n in sorted(jit)}
    kept = [
        s for s in samples if (s.provenance["run"], s.rank, s.step) not in jit | after
    ]
    rows = _host_rows(kept)
    host = {"key": json.loads(keys[0]), "laws": {}}
    report = {"cut": {"idle": sum(s.idle for s in samples)}, "unfitted": {}}
    report["blocked"], report["unstamped"] = _blocked(samples)
    report["jit_steps"] = {}
    for run, _, n in sorted(jit):
        report["jit_steps"].setdefault(run, []).append(n)
    for kind in HOST_KINDS:
        extra = sorted({r[1] for r in rows if r[0] == kind} - set(HOST_SEGMENTS))
        for segment in HOST_SEGMENTS[: 3 if kind == "middle" else 4] + tuple(extra):
            members = [r for r in rows if r[:2] == (kind, segment) and r[3] > 0]
            try:
                adopted, figures = _launch_columns(members, launch.features)
                columns = [(f, [r[5][f] for r in members]) for f in adopted]
                coef, fitted = _host_law(
                    [r[2] for r in members],
                    [r[3] for r in members],
                    HOST_FEATURES,
                    columns=columns,
                )
            except (ValueError, QhullError) as e:
                report["unfitted"][f"{kind}.{segment}"] = str(e).splitlines()[0]
                continue
            host["laws"].setdefault(kind, {})[segment] = coef[: len(HOST_FEATURES)]
            if adopted:
                c = dict(zip(adopted, coef[len(HOST_FEATURES) :]))
                host.setdefault("columns", {}).setdefault(kind, {})[segment] = c
            report[f"{kind}.{segment}"] = fitted | figures
    a, base, labels = list(zip(*_admit_rows(kept, host))) or [(), (), ()]
    try:
        host["admit"], report["admit"] = _host_law(a, labels, ADMIT_FEATURES, base)
    except (ValueError, QhullError) as e:
        host["admit"] = None  # a run refuses a forward that admits a request
        report["unfitted"]["admit"] = str(e).splitlines()[0]
    return host, report


def _blocked(samples) -> tuple[dict, list[str]]:
    """Per run whose lines state `blocked_ns` and kind, its launches, those
    that blocked in a HIP call, their blocked ms at p50 and p90, and each
    call's blocked ms in all; and the runs whose lines state none, so their
    `W_launch` labels are wall times."""
    runs: dict = {}
    for s in samples:
        if "W_launch" in s.host:
            runs.setdefault(s.provenance["run"], []).append(s)
    blocked = {}
    for run, launches in runs.items():
        if all(s.blocked is None for s in launches):
            continue
        for kind in HOST_KINDS:
            mine = [s.blocked or {} for s in launches if host_kind(s.view) == kind]
            ms = [sum(b.values()) * 1e3 for b in mine if b]
            calls: dict = {}
            for b in mine:
                for call, seconds in b.items():
                    calls[call] = calls.get(call, 0.0) + seconds * 1e3
            blocked.setdefault(run, {})[kind] = {
                "launches": len(mine),
                "blocked": len(ms),
                "ms": np.percentile(ms, (50, 90)).tolist() if ms else None,
                "calls": calls,
            }
    return blocked, sorted(runs.keys() - blocked.keys())


def fit_tokenizer(paths, intakes) -> tuple[dict, dict]:
    """The ``tokenizer`` term over the lone calls (`_lone_calls`) each
    tokenizer record in `paths` holds, beside the intake stamps in `intakes`,
    and its report: the records' backend and fingerprint, and per kind of
    call, ``fixed_s + tokens / tokens_per_s`` by the host-law solver on the
    calls' wall seconds, a segment tokenizer call's less its calls', over
    their ids, plus ``chars / chars_per_s`` over the characters its
    stored-prefix search compared where every call's record holds them
    (`_own_work`).
    One-item ``encode_batch`` calls are ``encode``'s and several-item ones
    their own kind. A ``decode`` with no positive slope is ``fixed_s`` alone
    over ``tokens``, the ``[lo, hi]`` token counts its calls span. A kind the
    calls cannot give is left out and its reason listed under the report's
    ``unfitted``, a ``segment`` without ``chars_per_s`` as
    ``segment.chars_per_s``; the report gives each record's p10, p50 and p90
    relative residual per kind. Raises on records of two tokenizers."""
    from scipy.spatial import QhullError

    heads, rows = {}, []
    for i, (path, intake) in enumerate(zip(paths, intakes)):
        head, *lines = Path(path).read_text().splitlines()
        heads[json.dumps(json.loads(head), sort_keys=True)] = None
        got = [json.loads(line) for line in lines]
        for c in _lone_calls([c for c in got if c["call"] != "segment"], intake):
            kind = c["call"] if c["items"] > 1 or c["call"] == "decode" else "encode"
            s = (c["t_end_ns"] - c["t_start_ns"]) * 1e-9
            rows.append((i, kind, c["tokens"], s))
        rows += [(i, "segment", n, s, chars) for n, chars, s in _own_work(got, intake)]
    if len(heads) != 1:
        raise ValueError(f"the records name {len(heads)} tokenizers: {sorted(heads)}")
    term, report = json.loads(next(iter(heads))), {"unfitted": {}}
    for kind in ("encode", "encode_batch", "decode", "segment"):
        got = np.array([r[2:4] for r in rows if r[1] == kind and r[3] > 0])
        chars = [r[4] for r in rows if r[1] == kind and r[3] > 0 and len(r) > 4]
        record = np.array([r[0] for r in rows if r[1] == kind and r[3] > 0])
        try:
            if not len(got):
                raise ValueError(f"the records hold no lone {kind} call")
            x = np.column_stack([np.ones(len(got)), got[:, 0]])
            if len(chars) == len(got) and None not in chars:
                x = np.column_stack([x, chars])
            coef, fitted = _host_law(x, got[:, 1], TOKENIZER_FEATURES[: x.shape[1]])
        except (ValueError, QhullError) as e:
            report["unfitted"][kind] = str(e)
            continue
        if coef[1] > 0:
            term[kind] = {"fixed_s": coef[0], "tokens_per_s": 1 / coef[1]}
            if len(coef) > 2 and coef[2] > 0:
                term[kind]["chars_per_s"] = 1 / coef[2]
            elif kind == "segment":
                lack = chars.count(None)
                report["unfitted"]["segment.chars_per_s"] = (
                    f"{lack} of {len(chars)} lone segment calls' records hold no "
                    "compared_chars"
                    if lack
                    else "the records give no time per compared character"
                )
        elif kind == "decode":
            term[kind] = {"fixed_s": coef[0], "tokens": _span(fitted, "tokens")}
        else:
            report["unfitted"][kind] = f"{kind}'s calls take no time per token"
            continue
        error = x @ coef / got[:, 1] - 1
        fitted["records"] = {
            str(p): np.percentile(error[record == i], [10, 50, 90]).tolist()
            for i, p in enumerate(paths)
            if (record == i).any()
        }
        report[kind] = fitted
    return term, report


def _span(fitted: dict, name: str) -> list:
    """The ``[lo, hi]`` of the one feature `name` of `_host_law`'s report."""
    if name in fitted["hull"]["equal"]:
        return [fitted["hull"]["equal"][name]] * 2
    (_, lo), (_, hi) = fitted["hull"]["facets"]
    return [lo, -hi]


def _stamps(path) -> list[tuple]:
    """The HTTP intake stamps `path` holds (`atom.compass.run.INTAKE_RECORD`),
    each ``(start ns, end ns, kind, size, start cpu ns, end cpu ns)``, in order
    of start."""
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    return sorted(
        (int(r[3]), int(r[0]), r[2], int(r[4]), int(r[6]), int(r[5])) for r in rows
    )


def _requests(ev) -> tuple[list, np.ndarray]:
    """Each request's app call among the stamps `ev`, in order, and every
    engine send's end, sorted."""
    apps = [e[0] for e in ev if e[2] == "app"]
    # A request's app call is the last before its body's first read; any
    # other, such as a health probe's, has no body.
    calls = sorted(
        {apps[np.searchsorted(apps, e[0]) - 1] for e in ev if e[2] == "body"}
    )
    return calls, np.array(sorted(e[1] for e in ev if e[2] == "send"))


def _lone_calls(calls: list[dict], intake) -> list[dict]:
    """The tokenizer calls among `calls` that ran lone: no other call, and no
    request's intake (`_requests` over the stamps in `intake`) in flight from
    the call's start to its end but, for an encode, its own."""
    apps, sends = _requests(_stamps(intake))
    calls = sorted(calls, key=lambda c: c["t_start_ns"])
    start = np.array([c["t_start_ns"] for c in calls])
    end = np.array([c["t_end_ns"] for c in calls])
    before = np.maximum.accumulate(np.r_[-math.inf, end[:-1]])
    after = np.r_[start[1:], math.inf]
    arrived = np.searchsorted(apps, end, "right") - np.searchsorted(apps, start)
    flight = np.searchsorted(apps, start) - np.searchsorted(sends, start)
    own = np.array([c["call"] != "decode" for c in calls])
    lone = (before < start) & (after > end) & (arrived == 0) & (flight <= own)
    return [c for c, k in zip(calls, lone) if k]


def _own_work(calls: list[dict], intake) -> list[tuple[int, int | None, float]]:
    """Each ``segment`` call among `calls` lone (`_lone_calls`) among the calls it
    did not make, its thread's that start in its span, as the ids those returned,
    the characters its stored-prefix search compared if its record holds them,
    and its wall seconds less theirs."""
    own, ids, made, last = {}, {}, set(), None
    for c in sorted(calls, key=lambda c: (c["thread"], c["t_start_ns"])):
        start, end = c["t_start_ns"], c["t_end_ns"]
        if last and last["thread"] == c["thread"] and start < last["t_end_ns"]:
            own[id(last)] -= end - start
            ids[id(last)] += c["tokens"]
            made.add(id(c))
        elif c["call"] == "segment":
            own[id(c)], ids[id(c)], last = end - start, 0, c
    lone = _lone_calls([c for c in calls if id(c) not in made], intake)
    return [
        (ids[id(c)], c.get("compared_chars"), own[id(c)] * 1e-9)
        for c in lone
        if c["call"] == "segment"
    ]


def _intakes(path) -> tuple[list, list, list]:
    """The HTTP intake stamps `path` holds as each lone request's ``(body
    bytes, CPU seconds)``, each burst's ``(body bytes, template start CPU
    ns)`` per request, and every request's body bytes.

    A request is its app call and the body, template, pool job and engine send
    that follow it. A lone one found every earlier request sent and its own
    sent before the next app call, with nothing else stamped between: its
    seconds are the CPU time the loop's thread spent from the app call to the
    send's end, less what it spent from its pool job's submit to its resume
    (``wait``), the job's and other requests' work. A burst is a run of requests each of whose app call came
    before the template of the one ahead of it began, the loop taking them in
    one after another."""
    ev = _stamps(path)
    at = [e[0] for e in ev]
    cpu = {e[0]: e[4] for e in ev if e[2] == "app"}
    calls, sends = _requests(ev)
    lone = []
    for i, (call, after) in enumerate(zip(calls, calls[1:] + [math.inf])):
        span = ev[np.searchsorted(at, call, "right") : np.searchsorted(at, after)]
        # An output taken in or a head sent is the output's or the response's.
        skip = ("app", "recv", "wait", "out", "head")
        between = [e for e in span if e[2] not in skip]
        waits = sum(e[5] - e[4] for e in span if e[2] == "wait")
        if np.searchsorted(sends, call) != i or [e[2] for e in between] != [
            "body",
            "tmpl",
            "prep",
            "send",
        ]:
            continue
        body, *_, send = between
        lone.append((body[3], (send[5] - cpu[call] - waits) * 1e-9))
    # The loop renders each request's template as soon as its body is read.
    tmpl = [e for e in ev if e[2] == "tmpl"]
    sizes = [e[3] for e in sorted(ev, key=lambda e: e[1]) if e[2] == "body"]
    # A template whose start finds a later request arrived and not yet rendered.
    queued = [np.searchsorted(calls, t[0]) > k + 1 for k, t in enumerate(tmpl)]
    bursts, run = [], []
    for k, t in enumerate(tmpl):
        run.append((sizes[k], t[0], t[4]))
        if not queued[k]:
            if len(run) > 1:
                bursts.append(run)
            run = []
    return lone, bursts, sizes


def fit_intake(paths) -> tuple[dict, dict]:
    """The ``http_intake`` term over the intake stamps each record in `paths`
    holds (`_intakes`), and its report: ``idle_s`` and ``per_byte_s`` by the
    host-law solver on the lone requests' CPU seconds over their body bytes;
    ``bytes`` spans every request's body; ``fixed_s`` the loop's CPU time per
    request between template starts in the longest burst, less ``per_byte_s``
    times its mean body, each gap the next request's. Raises on too few lone
    requests or no burst."""
    lone, bursts, bodies = [], [], []
    for path in paths:
        got = _intakes(path)
        lone += got[0]
        bursts += got[1]
        bodies += got[2]
    nbytes, seconds = np.array(lone, dtype=np.float64).reshape(-1, 2).T
    x = np.column_stack([np.ones(len(nbytes)), nbytes])
    (idle, per_byte), report = _host_law(x, seconds, INTAKE_FEATURES)
    if not bursts:
        raise ValueError(f"no burst among {len(lone)} lone requests")
    sizes, starts, cpu = np.array(max(bursts, key=len), dtype=np.float64).T
    fixed = np.diff(cpu).mean() * 1e-9 - per_byte * sizes[1:].mean()
    report["burst"] = {
        "requests": len(sizes),
        "span_s": (starts[-1] - starts[0]) * 1e-9,
    }
    term = {
        "fixed_s": fixed,
        "per_byte_s": per_byte,
        "idle_s": idle,
        "bytes": [min(bodies), max(bodies)],
    }
    return term, report


#: The host entry's ``hops`` `fit_hops` fits (`atom.compass.run.HOPS`).
HOPS = ("admission_s", "ipc_s", "ipc_out_s", "stream_s")


def fit_hops(intakes, clients) -> tuple[dict, dict]:
    """The ``hops`` term over the records whose intake stamps `intakes` names,
    each with the engine's queue handoffs and step lines beside them
    (``dp<r>.handoff.jsonl``, ``dp<r>.jsonl``) and its client requests'
    ``sent`` and ``acked``, CLOCK_MONOTONIC ns, in `clients`, or the reason
    there are none as ``missing``; and its report.

    Each hop is the p10 of its messages alone on it (`_alone`) in every
    record: ``admission_s``, a client send to its request's app call;
    ``ipc_s``, an engine send call's return to the engine input thread's receipt,
    inside a forward of that DP rank, since an idle engine's loop holds the GIL
    the thread waits on; ``ipc_out_s``, an output's put on the engine's output
    queue to the server's output thread taking its first row; ``stream_s``,
    an SSE head's send call's return to the client's ack. A hop the records cannot give is
    left out and its reason listed under the report's ``unfitted``."""
    took, report = {h: [] for h in HOPS}, {"unfitted": {}}
    for intake, client in zip(intakes, clients):
        for hop, messages in _hop_messages(Path(intake), client).items():
            try:
                took[hop] += messages()
            except ValueError as e:
                report["unfitted"][hop] = f"{intake}: {e}"
    term = {}
    for hop, s in took.items():
        if hop in report["unfitted"]:
            continue
        if not s:
            report["unfitted"][hop] = "no message ran alone on it"
            continue
        p10, p50, p90 = np.percentile(np.array(s) * 1e-9, [10, 50, 90]).tolist()
        term[hop] = p10
        report[hop] = {"p10_s": p10, "p50_s": p50, "p90_s": p90, "messages": len(s)}
    return term, report


def _alone(sent, got, returned=None) -> tuple[np.ndarray, np.ndarray]:
    """The arrival and ``got - sent`` of each message alone on its hop: sent
    once every message ahead of it arrived, and arrived before the next was
    sent. One side may miss messages the other saw, as a client misses those
    in flight at its stop (`_paired`). With `returned`, the return of each send
    call, made one at a time, a message is timed from it, and is not alone when
    it arrived first. Raises on a side with no message."""
    sent, got = np.sort(np.array(sent, np.int64)), np.sort(np.array(got, np.int64))
    if not len(sent) or not len(got):
        raise ValueError(f"{len(sent)} sends and {len(got)} arrivals")
    if len(got) >= len(sent):
        k, i = _paired(sent, got)
    else:  # in reverse time each arrival is a send, and each send an arrival
        i, k = _paired(-got[::-1], -sent[::-1])
        k, i = len(sent) - 1 - k, len(got) - 1 - i
    if returned is not None:
        sent = np.sort(np.array(returned, np.int64))
    took = got[i] - sent[k]
    return got[i][took > 0], took[took > 0]


def _paired(sent, got) -> tuple[np.ndarray, np.ndarray]:
    """The indices into `sent` and `got`, both sorted, of each message alone:
    exactly one arrival between its send and the next, and as many before its
    send as at any send before it. `got` may hold arrivals whose sends `sent`
    misses, each raising that count for good."""
    before = np.searchsorted(got, sent, "right")
    upto = np.searchsorted(got, np.r_[sent[1:], np.iinfo(np.int64).max], "right")
    ahead = before - np.arange(len(sent))
    alone = (upto - before == 1) & (ahead == np.maximum.accumulate(ahead))
    return np.flatnonzero(alone), before[alone]


def _hop_messages(intake: Path, client) -> dict:
    """Each hop's alone messages' ns in the record whose intake stamps are at
    `intake`, as a call per hop that raises on what the record cannot give;
    `client` is as `fit_hops` takes it."""
    ev = _stamps(intake)
    calls, sends = _requests(ev)
    recvs, puts, forwards = {}, [], {}
    for path in intake.parent.glob("dp*.handoff.jsonl"):
        rank = path.name.split(".")[0]
        for h in json_file.read_jsonl(path):
            if "t_recv_ns" in h:
                recvs[h["t_recv_ns"]] = rank
            elif h["rows"]:
                puts.append((h["t_out_put_ns"], h["rows"]))
        steps = json_file.read_jsonl(path.with_name(f"{rank}.jsonl"))
        forwards[rank] = np.array(
            sorted((s["t_enter_ns"], s["t_exit_ns"]) for s in steps)
        )

    def stepping(t, rank) -> bool:
        enter, exit_ = forwards[rank].reshape(-1, 2).T
        i = np.searchsorted(enter, t, "right") - 1
        return i >= 0 and t <= exit_[i]

    def of_client(field: str) -> list:
        if "missing" in client:
            raise ValueError(client["missing"])
        return client[field]

    def handed() -> None:
        if not forwards:
            raise ValueError("no engine's dp<r>.handoff.jsonl is beside it")

    def ipc():
        handed()
        called = [e[0] for e in ev if e[2] == "send"]
        got, took = _alone(called, list(recvs), returned=sends)
        return [d for g, d in zip(got, took) if stepping(g, recvs[g])]

    def ipc_out():
        # From the first request on, each output timed to its first row: the
        # server stamps each row it takes in, and the message came with its first.
        handed()
        if not len(sends):
            raise ValueError("no request was sent")
        put = sorted(p for p in puts if p[0] > sends[0])
        rows = sorted(e[0] for e in ev if e[2] == "out" and e[0] > sends[0])
        firsts = np.cumsum([0] + [n for _, n in put])
        if firsts[-1] != len(rows):
            raise ValueError(f"{len(rows)} rows taken in, not the {firsts[-1]} put")
        return _alone([t for t, _ in put], [rows[i] for i in firsts[:-1]])[1].tolist()

    heads = [e for e in ev if e[2] == "head"]
    return {
        "admission_s": lambda: _alone(of_client("sent"), calls)[1].tolist(),
        "ipc_s": ipc,
        "ipc_out_s": ipc_out,
        "stream_s": lambda: _alone(
            [e[0] for e in heads], of_client("acked"), [e[1] for e in heads]
        )[1].tolist(),
    }


def _host_law(x, y, names, offset=0.0, columns=()) -> tuple[list, dict]:
    """The host law on `names` over the forwards `x` labelled `y`, the part
    `offset` of each label fixed, and its report. Raises on too few forwards or
    no hull; with the last of `names` in a fixed proportion to the one before on
    every forward, it gets zero. Each of `columns`, (feature, values), follows."""
    x = np.array(x, dtype=np.float64).reshape(-1, len(names))
    if len(x) <= (k := len(names) + len(columns)):
        raise ValueError(f"{len(x)} forwards for {k} coefficients")
    width = len(names)
    if np.linalg.matrix_rank(x[:, 1:]) < width - 1:
        width -= 1
        x[:, width] = 0.0
    wide = np.column_stack([x, *(v for _, v in columns)])
    names = [*names, *(f for f, _ in columns)]
    s = _solve(wide, np.array(y), names, np.array(offset), names[0])
    hull = _hull(x[s["keep"]][:, 1:width], names[1:width])
    return s["coef"].tolist(), {
        "hull": hull,
        "samples": int(s["keep"].sum()),
        "dropped": int((~s["keep"]).sum()),
        **{k: s[k] for k in ("condition", "centre", "mad", "zero")},
    }


def _jit(samples, features=()) -> set[tuple]:
    """(run, rank, step) of each forward whose `W_launch` the fit of its kind
    drops above its law, a first-use JIT compile; on a decode, whose launch
    of a millisecond or so drops on host noise, only past `WARM_FACTOR` times
    its price. Each launch feature of `features` is a column of that fit."""
    jit = set()
    for kind in HOST_KINDS:
        members = [
            s
            for s in samples
            if host_kind(s.view) == kind and s.host.get("W_launch", 0) > 0
        ]
        if len(members) <= len(HOST_FEATURES) + len(features):
            continue
        x = np.array(
            [(*_host_x(m), *(m.launch[f] for f in features)) for m in members],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(x[:, 1:3]) < 2:
            x[:, 2] = 0.0
        y = np.array([m.host["W_launch"] for m in members])
        solved = _solve(x, y, [*HOST_FEATURES, *features])
        floor = 1 / WARM_FACTOR - 1 if kind == "decode" else 0.0
        jit |= {
            (m.provenance["run"], m.rank, m.step)
            for m, keep, r in zip(members, solved["keep"], solved["residual"])
            if not keep and r < floor
        }
    return jit


def _launch_columns(members, features) -> tuple[list[str], dict]:
    """The `features` host-law `members` (`_host_rows`) adopt as columns, in
    order, and the report's `launch_columns`: each round's held-out error per
    design tried (`_held_out_error`, each run held out). A round adds the
    feature of least error while it is below the last design's. Members of one
    run adopt none."""
    if not features:
        return [], {}
    x = np.array([r[2] for r in members], dtype=np.float64).reshape(-1, 3)
    if np.linalg.matrix_rank(x[:, 1:]) < 2:
        x[:, 2] = 0.0
    y, cell = np.array([r[3] for r in members]), np.array([r[4] for r in members])

    def error(names):
        values = [[r[5][f] for r in members] for f in names]
        design = np.column_stack([x, *values])
        return _held_out_error(_folds(design, y, cell, [*HOST_FEATURES, *names]))

    adopted, rounds = [], [{"none": error([])}]
    while rounds[0]["none"] is not None and len(adopted) < len(features):
        last = min(rounds[-1].values())
        rounds.append({f: error([*adopted, f]) for f in features if f not in adopted})
        if not min(rounds[-1].values()) < last:
            break
        adopted.append(min(rounds[-1], key=rounds[-1].get))
    return adopted, {"launch_columns": {"adopted": adopted, "held_out": rounds}}


def _solve(
    x: np.ndarray, y: np.ndarray, names: list[str], offset=0.0, kept: str = "step"
) -> dict:
    """Non-negative least squares on relative error, MAD-dropped until settled;
    `offset` is a fixed part of each label's price, outside the coefficients:
    `_fit_form` on one row per step."""
    return _fit_form(
        x[:, None], np.ones((len(y), 1), dtype=bool), y, names, offset, kept
    )


def _fit_form(
    x: np.ndarray,
    real: np.ndarray,
    y: np.ndarray,
    names: list[str],
    offset=0.0,
    kept: str = "step",
) -> dict:
    """One form fitted by max-affine regression: step `i`'s label `y[i]` is the
    max over its ranks' rows `x[i, r]`, those `real[i, r]` holds, of their
    price, plus `offset`, a fixed part of it outside the coefficients.

    `F(c) = sum_i ((max_r x_ir.c + offset_i) / y_i - 1)^2` over the kept steps.
    The start is the non-negative least-squares solve of every rank's row, each
    equation divided by its label. Each pass solves on the row each step's
    current price picks, a Gauss-Newton step on `F`, and halves it until `F`
    falls. The passes end settled, the solve picking the rows it was solved
    on, or at a tie between ranks, when no step that moves some price by more
    than the square root of the float epsilon makes `F` fall; more than
    `FIT_PASSES` passes in all raise. Each feature that differs between some
    step's ranks is a start too, its axis scaled to the labels: the first
    round runs from each start, every step kept, and the lowest `F` goes on.
    Steps whose residual is more than
    `MAD_K` scaled MADs from the median are then dropped and the passes rerun
    from there, until the kept set settles or for `MAD_PASSES` rounds.

    Returns the solve's figures: `pick`, each step's rank, at a tie the rank
    of larger features (`priced_rank`, prices folded as a run folds them);
    `passes`, its start, the passes over every round, the last stop and `F`;
    and `all_rows`, the centre and scaled MAD of the first start's residuals.
    """
    from scipy.optimize import nnls

    n, width, k = x.shape
    a, b = x / y[:, None, None], 1.0 - offset / y
    flat = a.reshape(-1, k)
    # An intercept, a column whose name starts with `kept`, is kept whatever
    # its variance; any other constant column is not.
    vary = [
        j for j in range(k) if names[j].startswith(kept) or np.ptp(x[real][:, j]) > 0
    ]
    tol = np.finfo(np.float64).eps ** 0.5
    steps = np.arange(n)

    def solve(rows, labels):
        norms = np.linalg.norm(rows[:, vary], axis=0)
        # A column with no non-zero kept row (a rung this pass dropped every
        # step of) sits out of this pass's solve, at coefficient 0.
        solved = norms > 0
        cols = rows[:, vary][:, solved] / norms[solved]
        coef = np.zeros(k)
        coef[np.array(vary)[solved]] = nnls(cols, labels)[0] / norms[solved]
        return coef, cols

    def price(c):
        return np.where(real, (flat @ c).reshape(n, width), -np.inf)

    def cost(c):
        return float(((price(c).max(axis=1) - b)[keep] ** 2).sum())

    keep = np.ones(n, dtype=bool)
    c, cols = solve(a[real], np.broadcast_to(b[:, None], real.shape)[real])
    start = price(c).max(axis=1) - b

    def alternate(c, cols, on, passes):
        """Passes from `c` over the kept steps until settled or at a tie."""
        while True:
            pick = price(c).argmax(axis=1)
            picked = np.zeros_like(real)
            picked[steps, pick] = True
            if on is not None and (on[keep] == picked[keep]).all():
                return c, cols, on, passes, "settled"
            if passes == FIT_PASSES:
                raise ValueError(
                    f"the alternation on {names} still moved after {FIT_PASSES} passes"
                )
            passes += 1
            full, solved = solve(a[steps, pick][keep], b[keep])
            if (price(full).argmax(axis=1)[keep] == pick[keep]).all():
                c, cols, on = full, solved, picked
            else:
                d, f, t = full - c, cost(c), 1.0
                moves = np.abs(price(full - c)[real & keep[:, None]]).max()
                while cost(c + t * d) >= f and t * moves > tol:
                    t /= 2
                if t * moves <= tol:
                    return c, cols, on, passes, "tie"
                c, cols, on = c + t * d, solved, picked if t == 1 else None

    # Each feature that differs between some step's ranks starts from its axis,
    # scaled by the median of each step's label over its largest value of it.
    starts = {"all_rows": (c, real)}
    top = np.where(real[..., None], x, -np.inf).max(axis=1)
    low = np.where(real[..., None], x, np.inf).min(axis=1)
    for j in np.flatnonzero((top > low).any(axis=0)):
        lit = top[:, j] > 0
        axis = np.zeros(k)
        axis[j] = np.median((y - offset)[lit] / top[lit, j])
        starts[names[j]] = (axis, None)
    # Each start's first round keeps every step; the lowest `F` wins.
    ends = {s: alternate(c0, cols, on, 0) for s, (c0, on) in starts.items()}
    first = min(ends, key=lambda s: cost(ends[s][0]))
    c, cols, on, passes, stop = ends[first]
    # Refitted until the kept set stops moving, so a step an outlier's pull
    # dropped on the first pass comes back once the outlier is gone.
    for rounds in range(1, MAD_PASSES + 1):
        if rounds > 1:
            c, cols, on, passes, stop = alternate(c, cols, on, passes)
        residual = price(c).max(axis=1) - b
        centre = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - centre))
        moved = np.abs(residual - centre) <= max(MAD_K * mad, MAD_FLOOR)
        if (moved == keep).all() or rounds == MAD_PASSES:
            break
        keep, on = moved, None
    pick = np.zeros(n, dtype=int)
    if width > 1:
        pick = [
            priced_rank(
                {
                    r: (fold_seconds(v * w for v, w in zip(x[i, r], c)), tuple(x[i, r]))
                    for r in np.flatnonzero(real[i])
                }
            )
            for i in steps
        ]
    return {
        "coef": c,
        "keep": keep,
        "residual": residual,
        "centre": float(centre),
        "mad": float(mad),
        "condition": float(np.linalg.cond(cols)),
        "zero": [names[j] for j in range(k) if j not in vary],
        "pick": np.asarray(pick),
        "passes": {"start": first, "count": passes, "stop": stop, "F": cost(c)},
        "all_rows": {
            "centre": float(np.median(start)),
            "mad": float(1.4826 * np.median(np.abs(start - np.median(start)))),
        },
    }


def _folds(design, y, cell, names) -> dict:
    """Per record of `cell` beside which another holds rows, (its rows, the p50
    error on them of the solve of `design` on the other records' rows)."""
    out = {}
    for c in sorted(set(cell.tolist())):
        other = cell != c
        if other.any():
            coef = _solve(design[other], y[other], names)["coef"]
            error = design[~other] @ coef / y[~other] - 1.0
            out[c] = (int((~other).sum()), float(np.median(error)))
    return out


def _held_out_error(folds: dict) -> float | None:
    """`_folds`' held-out error: the mean of each record's held-out p50 error's
    magnitude, weighted by its rows; None with no record held out."""
    rows = sum(n for n, _ in folds.values())
    return sum(n * abs(p) for n, p in folds.values()) / rows if rows else None


def _moe_column(members, x, names, routing) -> tuple:
    """The MoE column decode rung `members` adopts, or None, and the figures the
    rule read, as the vector records them under `fit.moe_column`.

    `members` are the rung's `fit` rows, `x` their design over `names`. The
    forms are no column and each of `MOE_COLUMNS` over the steps' real MoE
    tokens, `moe_experts` under `routing` and with each step's group dummies
    as one token (`moe_column`). Each form is solved with each record
    held out of the rung's rows, its held-out error `_held_out_error`'s. The
    form of smallest held-out error is picked, no column on a tie, and a
    picked column is adopted, returned as (feature, column, solve). A rung of
    one record has no held-out error and adopts none.
    """
    cell = np.array([r[0] for r in members])
    y = np.array([r[3] for r in members])
    forms = {"none": (x, names)}
    for feature in MOE_COLUMNS:
        column = [moe_column(feature, r[6], routing, r[7]) for r in members]
        forms[feature] = (np.column_stack([x, column]), [*names, feature])
    held, error = {}, {}
    for form, (design, wide) in forms.items():
        held[form] = _folds(design, y, cell, wide)
        error[form] = _held_out_error(held[form])
    picked = min(error, key=error.get) if held["none"] else None
    adopted = None if picked in (None, "none") else picked
    figures = {
        "picked": picked,
        "adopted": adopted,
        "held_out": {
            form: {
                "error": error[form],
                "folds": {str(c): list(f) for c, f in held[form].items()},
            }
            for form in forms
        },
    }
    if adopted is None:
        return None, figures
    design, wide = forms[adopted]
    return (adopted, design[:, -1], _solve(design, y, wide)), figures


def _rowmod(members, x, rung: int, base: dict, seed: int, names) -> tuple | None:
    """The `rowmod(k)` column decode rung `rung` adopts, or None.

    `members` are the rung's `fit` rows, `x` their design over `names`, `base`
    its solve without the column.
    `k` runs over the powers of two up to the rung. A `k` qualifies when its
    solve cuts `base`'s scaled MAD by `ROWMOD_GAIN`, the mean MAD of the same
    solve over `ROWMOD_SHUFFLES` shuffles of each step's rows, by `seed`, does not,
    and no record (a cell) priced by a solve on the others has a p50 error
    further from zero than without the column. Every cell
    needs `ROWMOD_CELL_STEPS` steps at the rung, and a rung needs two cells.
    The qualifying `k` of smallest MAD is adopted: returned with its solve and
    the evidence the law records.
    """
    cell = np.array([r[0] for r in members])
    cells = sorted(set(cell.tolist()))
    if len(cells) < 2 or min((cell == c).sum() for c in cells) < ROWMOD_CELL_STEPS:
        return None
    y = np.array([r[3] for r in members])
    rng = np.random.default_rng(seed)
    shuffles = [
        [rng.permutation(r[5]) for r in members] for _ in range(ROWMOD_SHUFFLES)
    ]
    wide = [*names, "rowmod"]

    def mad(rows, k):
        column = [rowmod(r, k) for r in rows]
        return _solve(np.column_stack([x, column]), y, wide)["mad"]

    def p50(design):
        folds = _folds(design, y, cell, wide[: design.shape[1]])
        return [p for _, p in folds.values()]

    without, best, k = p50(x), None, 2
    target = (1.0 - ROWMOD_GAIN) * base["mad"]
    while k <= rung:
        column = np.array([rowmod(r[5], k) for r in members], dtype=np.float64)
        design = np.column_stack([x, column])
        solved = _solve(design, y, wide)
        if solved["mad"] < min(target, best[1]["mad"] if best else target):
            shuffled_mad = float(np.mean([mad(s, k) for s in shuffles]))
            held = p50(design)
            if shuffled_mad >= target and all(
                abs(w) <= abs(o) for w, o in zip(held, without)
            ):
                evidence = {
                    "mad_without": base["mad"],
                    "shuffled_mad": shuffled_mad,
                    "held_out_p50": {
                        str(c): [o, w] for c, o, w in zip(cells, without, held)
                    },
                }
                best = (k, solved, evidence)
        k *= 2
    return best


def _hull(points: np.ndarray, names: tuple[str, ...]) -> dict:
    """`StepLinearLaw`'s hull over the non-intercept features of the fitted steps.
    qhull's precision checks depend on the order the points come in: where it
    refuses them, it hulls their distinct points in sorted order, the same hull.
    A set that spans none still raises."""
    from scipy.spatial import ConvexHull, QhullError

    equal = {
        names[j]: int(points[0, j])
        for j in range(points.shape[1])
        if np.ptp(points[:, j]) == 0
    }
    vary = [j for j in range(points.shape[1]) if names[j] not in equal]
    x = points[:, vary]
    lo, hi = x.min(axis=0), x.max(axis=0)
    if len(vary) == 1:
        facets = [[-1.0, float(lo[0])], [1.0, -float(hi[0])]]
    elif vary:
        # Hulled in the unit box, where qhull's tolerances suit every feature.
        unit = (x - lo) / (hi - lo)
        try:
            eq = ConvexHull(unit).equations
        except QhullError:
            eq = ConvexHull(np.unique(unit, axis=0)).equations
        normals = eq[:, :-1] / (hi - lo)
        facets = np.column_stack([normals, eq[:, -1] - normals @ lo]).tolist()
    else:
        facets = []
    return {"equal": equal, "facets": facets}


def _groups(
    run,
    form: str,
    draft: bool = False,
    steps=None,
    serial: bool = False,
    routing: tuple[int, int] | None = None,
    dummies: list[frozenset] | None = None,
    buckets: tuple[int, ...] = (),
) -> tuple[list[dict[int, tuple]], dict, list[float]]:
    """One run's steps, each {rank: its `_features` row} labelled with the
    smallest rank label, a rank that ran the step as a dummy forward holding
    none; what was cut; and each step's rank label spread. With `steps`, only
    the steps numbered in it; `serial`, `routing` and `buckets` as
    `_features` reads them;
    `dummies`, each rank's steps run as dummy forwards, read from its samples
    when None: a rank that gave no sample needs them given."""
    if dummies is None:
        dummies = [rank[0].dummies for rank in run]
    run = [[s for s in rank if steps is None or s.step in steps] for rank in run]
    got = [
        _features(rank, form, len(run), draft, serial, routing, buckets) for rank in run
    ]
    cut = {k: sum(c[k] for _, c in got) for k in got[0][1]}
    if len(run) == 1:
        return [{0: r} for r in got[0][0]], cut, []
    ranks = [{r[3]: r for r in rows} for rows, _ in got]
    groups, spread, cut["ranks"] = [], [], 0
    for step in sorted(set().union(*ranks)):
        group = {i: rank[step] for i, rank in enumerate(ranks) if step in rank}
        ran = len(group) + sum(step in d for d in dummies)
        if ran < len(ranks) or len({r[0] for r in group.values()}) > 1:
            cut["ranks"] += 1
            continue
        labels = [r[2] for r in group.values()]
        spread.append(max(labels) / min(labels) - 1.0)
        groups.append({i: (*r[:2], min(labels), *r[3:]) for i, r in group.items()})
    return groups, cut, spread


def _fit_steps(steps: list[list[tuple]], names: list[str], design) -> dict:
    """`_fit_form` of `steps`, each its ranks' `fit` rows, on `design(row)`, the
    form's features of a row; with `rows`, each step's row at the rank picked.
    A rank that ran the step as a dummy forward holds no row."""
    width = max(map(len, steps))
    x = np.zeros((len(steps), width, len(names)))
    real = np.zeros((len(steps), width), dtype=bool)
    for i, step in enumerate(steps):
        for r, row in enumerate(step):
            x[i, r], real[i, r] = design(row), True
    solved = _fit_form(x, real, np.array([s[0][3] for s in steps]), names)
    solved["rows"] = [s[p] for s, p in zip(steps, solved["pick"])]
    return solved


def _serial(samples, options) -> bool:
    """Whether the prefill vector adds `serial_moe_rows`: as `options` states,
    else when the group split some data-parallel prefill step of `samples`;
    with none split the column would repeat `moe_rows`."""
    split = any(s.view.prefill and s.view.tbo_tokens for s in samples)
    return (options or {}).get("serial", _options(options)[1] > 1 and split)


def _options(options) -> tuple[str, int]:
    options = {"form": "base", "dp": 1, **(options or {})}
    if options["form"] not in FORMS:
        raise ValueError(f"form {options['form']!r} is not one of {FORMS}")
    return options["form"], options["dp"]


def _routing(options) -> tuple[int, int] | None:
    """`options`' `moe_routing`, (routed experts, top-k), or None; refused under
    `shared`, whose decode solve has no MoE column."""
    routing = (options or {}).get("moe_routing")
    if routing and _options(options)[0] == "shared":
        raise ValueError("the shared form fits no MoE column")
    return tuple(routing) if routing else None


def _buckets(options) -> tuple[int, ...]:
    """`options`' `buckets`, the records' prefill token-graph buckets, ascending."""
    return tuple(sorted((options or {}).get("buckets") or ()))


def _law_routing(law: dict) -> tuple[int, int] | None:
    """`law`'s `moe_routing` as (experts, top-k), or None."""
    routing = law.get("moe_routing")
    return routing and (routing["experts"], routing["top_k"])


def fit(samples, options=None, seed: int = 0, draft: bool = False) -> tuple[dict, dict]:
    """The law over `samples`, as the mapping `StepLinearLaw` reads, and its report.

    `options` takes `form` (`base` by default), `dp`, the ranks each run
    holds (1 by default), `moe_routing`, the model's (routed experts,
    top-k), under which each decode rung but the draft law's may adopt a MoE
    column (`_moe_column`), and `buckets`, the records' prefill token-graph
    buckets, under which the prefill vector adds `padding`; `seed` fixes the
    `rowmod` shuffles; `draft`, or a sample with a draft label, adds `seqs` and
    `out` to the prefill vector.
    """
    from scipy.spatial import QhullError

    form, dp = _options(options)
    routing = _routing(options)
    serial = _serial(samples, options)
    buckets = _buckets(options)
    kvs = [s for s in samples if s.draft_kv_s is not None and not s.view.prefill]
    samples = [
        (
            s
            if s.draft_kv_s is None
            else dataclasses.replace(s, device_s=s.device_s - s.draft_kv_s)
        )
        for s in samples
    ]
    runs = _runs(samples, dp)
    groups, cut, spread = [], {"warm": 0, "warm_s": 0.0}, []
    for i, run in enumerate(runs.values()):
        got, c, s = _groups(
            run, form, draft, serial=serial, routing=routing, buckets=buckets
        )
        groups += [[(i, *r) for r in g.values()] for g in got]
        spread += s
        for k, v in c.items():
            cut[k] = cut.get(k, 0) + v
    prefill = list(HISTORY if form == "history" else PREFILL)
    prefill += ["cached_seqs"] * (form == "cached_seqs") + ["moe_rows"] * (dp > 1)
    prefill += ["serial_moe_rows"] * serial
    prefill += ["seqs", "out"] * (draft or any(s.draft_s is not None for s in samples))
    prefill += ["padding"] * bool(buckets)
    law = {
        "provenance": f"step_linear_fit {form} on {len(runs) * dp} real records "
        f"({', '.join(runs)})",
        "form": form,
        "cut": cut,
        "runs": list(runs),
        "warm_steps": [[] for _ in runs],
        "unfitted": {},
        "decode": {},
    }
    if routing:
        law["moe_routing"] = {"experts": routing[0], "top_k": routing[1]}
    if spread:
        law["spread"] = np.percentile(spread, (50, 90)).tolist()
    keys = sorted(
        {g[0][1] for g in groups},
        key=lambda k: (k != "prefill", k != EAGER, k.isdigit() and int(k)),
    )
    rungs = [k for k in keys if k != "prefill"]
    # Each solve: its column names, its rows, and their design matrix.
    solves = [("prefill", prefill)]
    if form == "shared" and rungs:
        solves.append(
            (None, ["context", "graph_padding"] + [f"step@{k}" for k in rungs])
        )
    else:
        solves += [(k, list(DECODE)) for k in rungs]
    for key, names in solves:
        steps = [
            g
            for g in groups
            if g[0][1] == key or (key is None and g[0][1] != "prefill")
        ]
        if not steps:
            law["unfitted"][key] = (
                "no prefill step carries a trace run's busy time"
                if key == "prefill"
                else f"0 steps for {len(names)} coefficients"
            )
            continue
        if key is None:
            solved = _fit_steps(
                steps, names, lambda r: (*r[2][1:], *(r[1] == k for k in rungs))
            )
        else:
            solved = _fit_steps(steps, names, lambda r: r[2])
        rowmod_k = figures = None
        if key and key.isdigit():
            # The column and `rowmod` are chosen on the base form's rows.
            members, base = solved["rows"], solved
            counts = np.array([r[2] for r in members], dtype=np.float64)
            if routing:
                chosen, figures = _moe_column(members, counts, names, routing)
                if chosen:
                    feature, column, base = chosen
                    counts = np.column_stack([counts, column])
                    names = [*names, feature]
            chosen = _rowmod(members, counts, int(key), base, seed, names)
            if chosen:
                rowmod_k, _, evidence = chosen
                names = [*names, "rowmod"]
            if len(names) > len(DECODE):
                adopted = {"coefficients": names}
                if rowmod_k:
                    adopted["rowmod"] = rowmod_k
                solved = _fit_steps(
                    steps,
                    names,
                    lambda r, v=adopted: _counts(v, routing, r[2], *r[5:]),
                )
        members = solved["rows"]
        slow = ~solved["keep"] & (solved["residual"] < 0)
        once = solved["residual"] < 1 / WARM_FACTOR - 1
        warm = np.zeros(len(members), dtype=bool)
        for rec in range(len(runs)):
            leading = True
            for j in (j for j, r in enumerate(members) if r[0] == rec):
                leading = leading and slow[j]
                if not (leading or slow[j] and once[j]):
                    continue
                warm[j] = True
                law["warm_steps"][rec].append(members[j][4])
                cut["warm"] += 1
                cut["warm_s"] += members[j][3]
        for k in rungs if key is None else [key]:
            mine = np.array([r[1] == k for r in members])
            coef = solved["coef"]
            if key is None:
                coef = (coef[2 + rungs.index(k)], *coef[:2])
            features = names if key else DECODE
            coefficients = dict(zip(features, map(float, coef)))
            priced = {"coefficients": coefficients}
            if rowmod_k:
                priced["rowmod"] = rowmod_k
            kept = mine & solved["keep"]
            try:
                if kept.sum() <= len(features):
                    raise ValueError(
                        f"{int(kept.sum())} steps for {len(features)} coefficients"
                    )
                points = [
                    _counts(priced, routing, r[2], *r[5:])
                    for r, keep in zip(members, kept)
                    if keep
                ]
                hull = _hull(
                    np.array(points, dtype=np.float64)[:, 1:], tuple(features[1:])
                )
            except (ValueError, QhullError) as e:
                law["unfitted"][k] = str(e).splitlines()[0]
                continue
            vector = {
                "coefficients": coefficients,
                "hull": hull,
                "fit": {
                    "samples": int(kept.sum()),
                    "warm": int((mine & warm).sum()),
                    "dropped": int((mine & ~solved["keep"] & ~warm).sum()),
                    "condition": solved["condition"],
                    "centre": solved["centre"],
                    "mad": solved["mad"],
                    "zero": solved["zero"],
                },
            }
            if max(map(len, steps)) > 1:
                vector["fit"]["all_rows"] = solved["all_rows"]
                vector["fit"]["passes"] = solved["passes"]
            if figures:
                vector["fit"]["moe_column"] = figures
            if rowmod_k:
                vector["rowmod"] = rowmod_k
                vector["fit"]["rowmod"] = evidence
            if k == "prefill":
                if buckets:
                    vector["buckets"] = list(buckets)
                law["prefill"] = vector
            else:
                law["decode"][k] = vector
    vectors = [("prefill", law.get("prefill")), *law["decode"].items()]
    report = {
        "provenance": [
            {k: v for k, v in run[0][0].provenance.items() if k != "requests"}
            for run in runs.values()
        ],
        "vectors": {k: {**v["fit"], "hull": v["hull"]} for k, v in vectors if v},
        "held_out": {},
    }
    drafts, serial = _draft_samples(samples), [s for s in samples if s.passes_s]
    if serial and not drafts:
        law["draft"] = {
            "provenance": f"step_linear_fit draft passes on {len(runs) * dp} "
            f"real records ({', '.join(runs)})",
            "decode": {},
        }
    elif drafts:
        drafted = _draft_options(options, bool(serial))
        law["draft"] = fit(drafts, drafted, seed, draft=True)[0]
    if serial:
        law["draft"]["forward"] = _forward(serial)
        law["draft"]["propose"] = _propose(_rest(serial))
    elif drafts:
        if proposes := [
            s if s.draft_kv_s is None else dataclasses.replace(s, propose_s=s.draft_s)
            for s in samples
            if s.propose_s is not None
        ]:
            law["draft"]["propose"] = _propose(proposes)
        if kvs:
            law["draft"]["draft_kv"] = _propose(kvs, "draft_kv_s")
        if any(s.view.prefill and s.draft_kv_s is not None for s in samples):
            law["draft"]["output_prefill_kv"] = True
    return law, report


def _draft_options(options, serial: bool) -> dict:
    """`options` for the draft law, which adopts no MoE column and no padding;
    under a serial drafter whose records time its passes, the `base` form: its
    prefill vector fits the middle chunks alone, each one request's chunk at
    the step's token budget, whose `history` is its `query_cached` over it."""
    drafted = {**(options or {}), "moe_routing": None, "buckets": None}
    return {**drafted, "form": "base"} if serial else drafted


def _draft_samples(samples) -> list:
    """The samples the draft law's prefill and decode vectors are fitted on,
    each labelled with its draft label (`_draft_kv`). Under a serial drafter
    whose records time its passes, only the middle chunks: its decodes' and
    output prefills' drafts are its passes and the rest of their draft."""
    serial = any(s.passes_s for s in samples)
    return [
        dataclasses.replace(
            s,
            device_s=_draft_kv(s),
            draft_s=None,
            propose_s=None,
            draft_kv_s=None,
            passes_s=None,
        )
        for s in samples
        if s.draft_s is not None
        and _draft_kv(s) > 0
        and not (serial and (s.passes_s or s.view.produces_output))
    ]


def _rest(samples) -> list:
    """`samples` labelled, as propose labels, with what their draft label
    holds past their passes; those whose passes hold it all left out."""
    rest = [
        dataclasses.replace(s, propose_s=s.draft_s - sum(s.passes_s)) for s in samples
    ]
    return [s for s in rest if s.propose_s > 0]


def _forward(samples) -> dict:
    """The draft law's `forward` vector over `samples`' passes, each a sample
    at its features (`serial_passes`), solved and hulled as a step vector is,
    with `passes`, how many each holds; samples holding other counts refuse."""
    passes = sorted({len(s.passes_s) for s in samples})
    if len(passes) != 1:
        raise ValueError(f"the records' steps ran {passes} draft passes")
    (k,) = passes
    x = [c for s in samples for c in serial_passes(s.view, k)]
    y = [p for s in samples for p in s.passes_s]
    return {**_vector(np.array(x, dtype=np.float64), np.array(y), PREFILL), "passes": k}


def _draft_kv(s) -> float:
    """The draft prefill vector's label of `s`: an output prefill's draft-KV
    label, else its draft label less any propose label."""
    if s.view.prefill and s.draft_kv_s is not None:
        return s.draft_kv_s
    return s.draft_s - (s.propose_s or 0.0)


def _propose(samples, label: str = "propose_s") -> dict:
    """The draft law's `propose` vector over `samples`' propose labels, or
    another `label`, on `PROPOSE`, solved and hulled as a step vector is."""
    x = np.array([(1, len(s.view.requests)) for s in samples], dtype=np.float64)
    return _vector(x, np.array([getattr(s, label) for s in samples]), PROPOSE)


def _vector(x, y, names) -> dict:
    """A vector on `names` over the rows `x` labelled `y`, solved and hulled."""
    s = _solve(x, y, list(names))
    fitted = {k: s[k] for k in ("condition", "centre", "mad", "zero")}
    return {
        "coefficients": dict(zip(names, map(float, s["coef"]))),
        "hull": _hull(x[s["keep"]][:, 1:], names[1:]),
        "fit": {
            "samples": int(s["keep"].sum()),
            "dropped": int((~s["keep"]).sum()),
            **fitted,
        },
    }


def token_band(tokens: int) -> str:
    """The band of `TOKEN_BANDS` `tokens` falls in: `lo-hi`, or `lo+` past the last."""
    lo = max([0] + [b for b in TOKEN_BANDS if b <= tokens])
    hi = min([b for b in TOKEN_BANDS if b > tokens], default=None)
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def _band(key: str, counts) -> str:
    if key != "prefill":
        return f"decode rung {key}"
    return f"prefill tokens {token_band(counts[1])}"


def priced_rank(ranks: dict):
    """The rank a DP step is priced and banded as, of `ranks`, {rank: (price,
    features)}: its costliest, a tie going to the rank of larger features, which a
    law blind to them prices the same and the step still waits for."""
    return max(ranks, key=ranks.get)


def _held_out(samples, options, seed, warm=None, draft: bool = False):
    """Each step of each run, warm steps left out, priced as its costliest rank
    by the law fitted on the other runs, under `draft` as `fit` fits a draft
    law: (run, vector, band, that rank's (counts, context, tokens, dummies),
    relative error), the error nan where that law has no vector."""
    form, dp = _options(options)
    options = {**(options or {}), "serial": _serial(samples, options)}
    runs = _runs(samples, dp)
    if warm is None:
        warm = fit(samples, options, seed, draft)[0]["warm_steps"]
    for i, (run, ranks) in enumerate(runs.items()):
        others = [s for s in samples if s.provenance["run"] != run]
        law = fit(others, options, seed, draft)[0]
        routing = _law_routing(law)
        for group in _groups(
            ranks,
            form,
            draft,
            serial=options["serial"],
            routing=_routing(options),
            buckets=_buckets(options),
        )[0]:
            key, _, seconds, step, *_ = next(iter(group.values()))
            if step in warm[i]:
                continue
            vector = law["decode"].get(key) if key != "prefill" else law.get(key)
            mine = [(r[1], *r[4:7]) for r in group.values()]
            price, error = [c[1] for c, *_ in mine], float("nan")
            if vector is not None:
                c = list(vector["coefficients"].values())
                price = [float(np.dot(c, _counts(vector, routing, *m))) for m in mine]
                error = max(price) / seconds - 1.0
            top = mine[priced_rank({j: (p, mine[j][0]) for j, p in enumerate(price)})]
            yield run, key, _band(key, top[0]), top, error


def _counts(vector: dict, routing, counts, context, tokens, dummies) -> tuple:
    """A step's `counts` with the MoE column and `rowmod` count `vector` adds,
    over its rows' `context` and its real MoE `tokens`, `dummies` of them its
    group's dummy forwards', under `routing`."""
    moe = [f for f in MOE_COLUMNS if f in vector["coefficients"]]
    counts = (*counts, *(moe_column(f, tokens, routing, dummies) for f in moe))
    return counts + ((rowmod(context, vector["rowmod"]),) if "rowmod" in vector else ())


def held_out(samples, options=None, seed: int = 0) -> dict:
    """Per band, the relative errors on each run of the law fitted on the others."""
    errors: dict[str, list[float]] = {}
    for _, _, band, _, error in _held_out(samples, options, seed):
        band = band if not math.isnan(error) else f"{band} (no vector)"
        errors.setdefault(band, []).append(error)
    return errors


def _bands(rows) -> dict:
    """Per band of `_held_out`'s rows, their steps and error percentiles."""
    bands = {}
    for band in sorted({r[2] for r in rows}):
        e = [r[4] for r in rows if r[2] == band]
        p10, p50, p90 = np.percentile(e, (10, 50, 90)).tolist()
        bands[band] = {"steps": len(e), "p10": p10, "p50": p50, "p90": p90}
    return bands


def mark_held_out(
    law: dict, report: dict, samples, options=None, seed: int = 0, draft=False
):
    """Write the held-out errors: under each of `law`'s vectors, as `held_out`,
    the hull of the rank rows `_held_out` priced its steps by, their errors per
    band; under `report`'s `held_out`, each run's errors per band. A draft
    law fitted on a serial drafter's passes, its vectors alike, on the steps
    each is fitted on, their bands named `draft` (`_mark_passes`); `draft`
    says `law` is a draft law."""
    from scipy.spatial import QhullError

    rows = [
        r
        for r in _held_out(samples, options, seed, law["warm_steps"], draft)
        if not math.isnan(r[4])
    ]
    for run in dict.fromkeys(r[0] for r in rows):
        report["held_out"][run] = _bands([r for r in rows if r[0] == run])
    routing = _law_routing(law)
    for key, vector in [("prefill", law.get("prefill")), *law["decode"].items()]:
        mine = [r for r in rows if r[1] == key]
        if vector is None or not mine:
            continue
        points = [_counts(vector, routing, *r[3]) for r in mine]
        try:
            hull = _hull(
                np.array(points, dtype=np.float64)[:, 1:],
                tuple(vector["coefficients"])[1:],
            )
        except (ValueError, QhullError):
            continue
        vector["held_out"] = {"hull": hull, "bands": _bands(mine)}
    if "forward" not in law.get("draft", {}):
        return
    serial = [s for s in samples if s.passes_s]
    drafts, mine = _draft_samples(samples), {"held_out": {}}
    if len({s.provenance["run"] for s in drafts}) > 1:
        drafted = _draft_options(options, True)
        mark_held_out(law["draft"], mine, drafts, drafted, seed, True)
    if len({s.provenance["run"] for s in serial}) > 1:
        _mark_passes(law["draft"], mine, serial)
    for run, bands in mine["held_out"].items():
        named = {f"draft {b}": v for b, v in bands.items()}
        report["held_out"].setdefault(run, {}).update(named)


def _mark_passes(law: dict, report: dict, samples) -> None:
    """`mark_held_out` for a draft law's `forward` and `propose` vectors:
    each run's passes and propose rests (`_rest`) priced by the vectors fitted
    on the other runs, banded by pass."""
    from scipy.spatial import QhullError

    rows = {"forward": [], "propose": []}
    for run in dict.fromkeys(s.provenance["run"] for s in samples):
        others = [s for s in samples if s.provenance["run"] != run]
        mine = [s for s in samples if s.provenance["run"] == run]
        forward = _forward(others)
        for s in mine:
            passes = serial_passes(s.view, forward["passes"])
            for i, (x, seconds) in enumerate(zip(passes, s.passes_s)):
                price = np.dot(list(forward["coefficients"].values()), x)
                rows["forward"].append((run, "", f"pass {i}", x, price / seconds - 1))
        propose = _propose(_rest(others))
        for s in _rest(mine):
            x = (1, len(s.view.requests))
            price = np.dot(list(propose["coefficients"].values()), x)
            rows["propose"].append((run, "", "propose", x, price / s.propose_s - 1))
    for key, got in rows.items():
        for run in dict.fromkeys(r[0] for r in got):
            mine = _bands([r for r in got if r[0] == run])
            report["held_out"].setdefault(run, {}).update(mine)
        names = tuple(law[key]["coefficients"])[1:]
        try:
            hull = _hull(np.array([r[3] for r in got], dtype=np.float64)[:, 1:], names)
        except (ValueError, QhullError):
            continue
        law[key]["held_out"] = {"hull": hull, "bands": _bands(got)}


def host_held_out(samples) -> dict:
    """Per host law, the relative errors on each run of the laws fitted on the
    others; under `admit E`, those of the `E` of the forwards that admit
    requests, priced by their kind's law and `admit`."""
    errors: dict[str, list[float]] = {}
    for run in dict.fromkeys(s.provenance["run"] for s in samples):
        others = [s for s in samples if s.provenance["run"] != run]
        host = fit_host(others)[0]
        admit = host.get("admit")
        mine = [s for s in samples if s.provenance["run"] == run]
        rows = [
            (f"{kind} {segment}", _host_price(host, kind, segment, x, launch), seconds)
            for kind, segment, x, seconds, _, launch in _host_rows(mine)
        ]
        rows += [
            ("admit E", admit and base + float(np.dot(admit, a)), e)
            for a, base, e in _admit_rows(mine, host)
        ]
        for band, price, seconds in rows:
            if seconds > 0:
                errors.setdefault(
                    f"host {band}" + ("" if price is not None else " (no law)"), []
                ).append(price / seconds - 1.0 if price is not None else float("nan"))
    return errors


def _table(errors: dict) -> str:
    lines = ["band | steps | p10 | p50 | p90"]
    for band in sorted(errors):
        e = np.array(errors[band])
        p = ["-"] * 3
        if not np.isnan(e).any():
            p = [f"{v:+.1%}" for v in np.percentile(e, (10, 50, 90))]
        lines.append(f"{band} | {len(e)} | " + " | ".join(p))
    return "\n".join(lines)


def engine_config(records) -> dict:
    """The engine config `records` ran under: `keyed_of` the `.engine.json`
    beside each. A record without one, or records under two, refuse, naming
    the fields that differ."""
    held = []
    for record in records:
        engine = Path(record).with_suffix(".engine.json")
        if not engine.exists():
            raise ValueError(
                f"{record} has no {engine.name}, the engine config it ran under"
            )
        held.append(keyed_of(json.loads(engine.read_text())))
    names = sorted({k for e in held for k in e})
    if differ := [k for k in names if any(e.get(k) != held[0].get(k) for e in held)]:
        raise ValueError(
            f"the records ran under two engine configs, differing in {differ}"
        )
    return held[0]


def stated_buckets(record) -> list | None:
    """The prefill token-graph buckets the record `record` (`dp<r>.jsonl`) states:
    its engine file's, else its host key's graph sets'; None where neither does."""
    engine = Path(record).with_suffix(".engine.json")
    raw = json.loads(engine.read_text()) if engine.exists() else {}
    if PREFILL_BUCKETS in raw:
        return raw[PREFILL_BUCKETS]
    key = Path(record).with_suffix(".key.json")
    graphs = launch_features.stated(key.exists() and json.loads(key.read_text()))[1]
    return None if graphs is None else graphs["token"]["prefill"]


def recorded_buckets(stated: dict) -> tuple[int, ...] | None:
    """The prefill token-graph buckets `stated`, {where: buckets or None}, states,
    None when none states any. Two sets, or some stating them and others not,
    refuse, naming each."""
    held: dict = {}
    for where, buckets in stated.items():
        stated_as = None if buckets is None else tuple(buckets)
        held.setdefault(stated_as, []).append(where)
    if len(held) > 1:
        raise ValueError(
            "the records state other prefill buckets: "
            + "; ".join(
                f"{'none' if b is None else list(b)} in {', '.join(f)}"
                for b, f in held.items()
            )
        )
    return next(iter(held), None)


def refuse_context_parallel(engine: dict, where: str) -> None:
    """Refuse an engine config that states a context-parallel width: a step
    linear law states none, so a run at width 1 would price with it."""
    if widths := [f"{k} {engine[k]}" for k in CONTEXT_PARALLEL if k in engine]:
        raise ValueError(
            f"{where} states {' and '.join(widths)}, and a step linear law states "
            "no context-parallel width, so a run at width 1 would price with it"
        )


def join_width(engine: dict) -> int:
    """The ranks a run's records join at: `engine`'s `data_parallel_size`.
    An engine config stating a context-parallel width refuses."""
    refuse_context_parallel(engine, "the records' engine config")
    if "data_parallel_size" not in engine:
        raise ValueError(
            "the records' engine config states no data_parallel_size, the ranks "
            "a step joins; it predates the field"
        )
    return engine["data_parallel_size"]


def _routing_arg(value: str) -> tuple[int, int]:
    """`--moe-routing`'s `E,K`: two positive integers, `K` at most `E`."""
    try:
        experts, top_k = map(int, value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not E,K") from None
    if not 0 < top_k <= experts:
        raise argparse.ArgumentTypeError(f"{value!r} routes to {top_k} of {experts}")
    return experts, top_k


#: The kinds `--platform` states platform fields for: the law's and its host laws'.
PLATFORM_KINDS = ("step_linear_law", "host_laws")


def _platform_arg(value: str) -> dict:
    """`--platform`'s JSON object, an object of platform fields per kind."""
    try:
        given = json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"{value!r} is not JSON") from None
    if not isinstance(given, dict) or not all(
        isinstance(given.get(k), dict) for k in PLATFORM_KINDS
    ):
        raise argparse.ArgumentTypeError(
            f"{value!r} states no object for each of {', '.join(PLATFORM_KINDS)}"
        )
    return given


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.backends.step_linear_fit",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("records", nargs="+", help="real step records, steps/dp0.jsonl")
    parser.add_argument(
        "--out", required=True, help="where the law is written, as JSON"
    )
    parser.add_argument("--form", choices=FORMS, default="base")
    parser.add_argument(
        "--moe-routing",
        metavar="E,K",
        type=_routing_arg,
        help="the model's routed experts and top-k: each decode rung may adopt its "
        "steps' real MoE tokens, or the experts they reach, as a column",
    )
    parser.add_argument(
        "--platform",
        required=True,
        type=_platform_arg,
        help="the platform the records ran on, as the law's and its host laws' "
        'keys state it: {"step_linear_law": {...}, "host_laws": {...}}',
    )
    parser.add_argument(
        "--engine",
        help="the engine file the records ran under, stated in place of the "
        ".engine.json beside each, its prefill buckets for a record that states "
        "none; refused where one contradicts it",
    )
    args = parser.parse_args(argv)
    for record in args.records:
        run = Path(record).resolve().parent.parent / "run.json"
        if run.is_file() and (
            built := step_labels.refusal(json.loads(run.read_text()))
        ):
            raise ValueError(f"{record} is not fitted: {built}")
    owns = [Path(r).with_suffix(".engine.json") for r in args.records]
    stated = {r: stated_buckets(r) for r in args.records}
    if args.engine:
        engine = keyed_of(json.loads(Path(args.engine).read_text()))
        for own in owns:
            raw = json.loads(own.read_text()) if own.exists() else {}
            held = keyed_of(raw)
            refuse_context_parallel(held, str(own))
            if differ := [k for k, v in held.items() if engine.get(k, v) != v]:
                raise ValueError(f"{own} contradicts {args.engine} in {differ}")
        given = json.loads(Path(args.engine).read_text()).get(PREFILL_BUCKETS)
        stated = {args.engine: given} | {
            r: b for r, b in stated.items() if b is not None
        }
    else:
        engine = engine_config(args.records)
    dp = join_width(engine)
    options = {"form": args.form, "dp": dp, "moe_routing": args.moe_routing}
    options["buckets"] = recorded_buckets(stated)
    samples, tables = label_records(args.records, dp)
    for record, table in tables.items():
        step_labels.write_table(record, table)
    law, report = fit(samples, options)
    if options["buckets"] is None:
        law["unfitted"]["padding"] = NO_BUCKETS
    if any(s.provenance["config"] for s in samples):
        law["host"], law["host_fit"] = fit_host(samples)
    law["platform"] = args.platform["step_linear_law"]
    law["engine_config"] = engine
    if "host" in law:
        stated = law["host"]["key"]["engine"]  # its graph sets stay
        graphs = {k: v for k, v in stated.items() if k == launch_features.GRAPHS}
        law["host"]["key"] |= {
            **args.platform["host_laws"],
            "engine": engine | graphs,
        }
    if len(args.records) > dp:
        mark_held_out(law, report, samples, options)
    Path(args.out).write_text(json.dumps(law) + "\n")
    print(f"cut: {json.dumps(law['cut'])}")
    print(f"unfitted: {json.dumps(law['unfitted'])}")
    for name, vector in [("prefill", law.get("prefill"))] + list(law["decode"].items()):
        if vector:
            print(
                f"{name}: {json.dumps(vector['coefficients'])} {json.dumps(vector['fit'])}"
            )
    for name, vector in law.get("draft", {}).items():
        if isinstance(vector, dict) and "coefficients" in vector:
            held = vector.get("held_out", {}).get("bands", {})
            print(f"draft {name}: {json.dumps(vector['coefficients'])} held out {held}")
    if "host" in law:
        print(f"host: {json.dumps(law['host']['laws'])} {json.dumps(law['host_fit'])}")
    if len(args.records) > dp:
        for form in FORMS:
            routing = None if form == "shared" else args.moe_routing
            errors = held_out(
                samples, {**options, "form": form, "moe_routing": routing}
            )
            print(f"\nheld out, form {form}\n{_table(errors)}")
        if "host" in law:
            print(f"\nheld out, host laws\n{_table(host_held_out(samples))}")


if __name__ == "__main__":
    main()

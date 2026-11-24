# SPDX-License-Identifier: MIT
"""`compass fit host`'s opt-in host fit of a theory-designed calibration (#3720).

    COMPASS_FIT_HOST_KIND_BLOCKS=1 compass fit host RECORD... --client-run RECORD EXPORT...
        [--decode-client-run RECORD EXPORT...] --store ROOT --out DIR

With `KIND_BLOCKS_ENV` set to ``1``, `refit` replaces the decode, prefill
and middle kinds' launch laws, their `PATH_SEGMENTS`, that
`step_linear_fit.fit_host` gave: a forward's launch is priced at one constant
per launch path (`launch_features.path`: a decode's CUDA-graph rung or eager, a
prefill's or middle chunk's replayed or eager body and tail), the slopes on
``tokens`` and ``seqs`` shared, in place of an intercept and the launch columns
a held-out choice adopts. Each path with `PATH_MIN` forwards or more takes
a constant: the kind's commonest the law's own, ``laws[kind][segment][0]``,
which a run also prices any path no constant names at
(`runner.overrides.HostDeviceRecurrence`), any other one under the entry's
``paths``. A path short of `PATH_MIN` forwards is left out of the fit
(`common`), too few to fit or hold out a constant on, and filled: a decode rung of the host key's ladder linearly between
the fitted rungs either side (`fill_rungs`), and any other combination of the
launch features at the price of the law `fit_host` gave, its intercept plus
the columns it adopted (`fill_flags`); the report names them under
``filled``. The path constants are adopted, by the rule #739 section 2.3
states, only where they lower the error held out in the blocks below against
the law `fit_host` gave (``choice``; a tie keeps that law, the path fit then
reported under ``paths_tried``). The engine loop's segments keep `fit_host`'s
law. Every law is
validated held out in `BLOCKS` blocks per path kind: each path's forwards, in
step order, are cut into `BLOCKS` contiguous blocks, and block `b` of every
path held out together, so every path is held out in every block. The
report's ``<kind>.<segment>`` gives each path's forwards and, under
``held_out``, the error per block and per path of the law fitted on the other
blocks; a launch law's also the base path, the paths left out (``short``)
and the same blocks' error of a law with no path term (``error_pooled``). Its
``kind_blocks`` states the checks a calibration record's host fit is held to
(#739 section 2.2), each with what it read and whether it passed:

- H1: every launch path of every segment has `PATH_MIN` forwards (a launch
  law leaves a path short of them out and fills it);
- H2: every path appears in every held-out block;
- H3: each per-token tokenizer term's lone calls span `H3_DECADES` decades of
  tokens, per record;
- H4: the host laws are fitted on steady windows only: the graded spans
  (`fit.samples.GRADED_ONLY_ENV`) and the decode drains below;
- H5: the lone requests' bodies span `H5_DECADES` decades of bytes, per record.

Under the switch the ``tokenizer`` and ``http_intake`` terms are fitted on the
steady client runs' windows only (`steady_windows`: each `--client-run`'s
graded span, from its first step's entry to its last step's exit), as the
host laws are; each call or request is still read among all of its record's
stamps (`step_linear_fit.within`).

`--decode-client-run RECORD EXPORT` names a client run its record served, as
`--client-run` does, whose decode drains alone give host samples
(`drain_cut`): the steps of its span in which no rank prefills, a batch
admitted at once and decoded down through the rungs, and none of the steps
of its admissions' bursts. The report's ``decode_client_runs`` gives each
one's span, drains and samples left out and kept.
"""

import dataclasses
import itertools
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from atom.compass.backends import launch_features, step_labels
from atom.compass.backends.json_file import read_json
from atom.compass.backends.launch_features import PATHS
from atom.compass.backends.step_labels import HOST_SEGMENTS, host_kind
from atom.compass.backends.step_linear_fit import (
    HOST_FEATURES,
    HOST_KINDS,
    _host_law,
    _host_x,
    _intakes,
    _jit,
    _lone_calls,
    _own_work,
    _solve,
    within,
)
from atom.compass.fit.samples import GRADED_ONLY_ENV, client_spans
from atom.compass.parity.served import served
from atom.compass.validate.runs import SIDES, named_by_prefill

#: The variable that, set to ``1``, fits the host laws per launch path and
#: holds them out in blocks per path kind (module docstring).
KIND_BLOCKS_ENV = "COMPASS_FIT_HOST_KIND_BLOCKS"
#: The held-out blocks each path's forwards are cut into.
BLOCKS = 5
#: The forwards a launch path needs for a constant of its own (H1): two per
#: held-out block.
PATH_MIN = 2 * BLOCKS
#: The relative difference in held-out error within which two laws tie: a tie
#: keeps the law `fit_host` gave (#739 section 2.3, the fewer changes).
TIE = 1e-6
#: The decades of tokens a per-token tokenizer term's lone calls span (H3).
H3_DECADES = 1.0
#: The decades of body bytes the lone requests span (H5).
H5_DECADES = 2.0
#: The segments a forward's launch path sets, which take one constant per path:
#: the launch. The engine loop's segments keep `fit_host`'s law: on #3720's
#: e3v2 record a constant per decode rung for `E`, `W_pre` and `E_out` took up
#: what its load put at each rung, and priced the c=64 target's decode `E` 23%
#: over where the law priced it 10% over.
PATH_SEGMENTS = ("W_launch",)
#: The tokenizer kinds charged per token (`step_linear_fit.fit_tokenizer`).
PER_TOKEN = ("encode", "encode_batch", "segment")


def enabled() -> bool:
    return os.environ.get(KIND_BLOCKS_ENV) == "1"


def price(
    host: dict, kind: str, segment: str, x, path: str | None, launch=None
) -> float | None:
    """What the host entry `host` prices a `kind` forward's `segment` at `x`
    on launch `path`, as a run does: its path constant in place of the law's
    own where its ``paths`` state one, and its launch columns on the features
    `launch`; None with no law."""
    law = host["laws"].get(kind, {}).get(segment)
    if law is None:
        return None
    own = host.get(PATHS, {}).get(kind, {}).get(segment, {}).get(path)
    shift = 0.0 if own is None else own - law[0]
    columns = host.get("columns", {}).get(kind, {}).get(segment, {})
    priced = sum(c * (launch or {}).get(f, 0.0) for f, c in columns.items())
    return float(np.dot(law, x)) + shift * x[0] + priced


def refit(samples, host: dict, report: dict) -> None:
    """Refit `host`'s decode, prefill and middle `PATH_SEGMENTS` laws on
    `samples` with one constant per launch path, in place of their intercept
    and launch columns, with their `report` entries; validate every other law
    `fit_host` gave in the same held-out blocks; and state the report's
    ``kind_blocks`` (module docstring). The forwards `fit_host` leaves out as
    paying a JIT compile, and each forward after one, are left out alike."""
    from scipy.spatial import QhullError

    launch, _ = launch_features.stated(host["key"])
    jit = _jit(samples, launch.features)
    gone = jit | {(run, rank, n + 1) for run, rank, n in jit}
    kept = [s for s in samples if (s.provenance["run"], s.rank, s.step) not in gone]
    cells: dict = {}
    for s in sorted(kept, key=lambda s: (s.provenance["run"], s.step, s.rank)):
        p = launch_features.path(s.view, s.launch)
        for segment, seconds in s.host.items():
            if seconds > 0 and (segment != "E" or not s.admitted[0]):
                row = (_host_x(s), seconds, p, s.launch)
                cells.setdefault((host_kind(s.view), segment), []).append(row)
    columns = host.get("columns", {})
    checks: dict = {"blocks": BLOCKS, "path_min": PATH_MIN, "H1": {}, "H2": {}}
    for kind in HOST_KINDS:
        adopted = columns.get(kind, {})
        laws = host["laws"].get(kind, {})
        extra = sorted({g for k, g in cells if k == kind} - set(HOST_SEGMENTS))
        for segment in HOST_SEGMENTS[: 3 if kind == "middle" else 4] + tuple(extra):
            name, members = f"{kind}.{segment}", cells.get((kind, segment), [])
            if segment not in PATH_SEGMENTS:
                if segment in laws:
                    cols = list(adopted.get(segment, {}))
                    report[name] = report.get(name, {}) | _validated(members, cols)
                    _check(checks, name, report[name])
                continue
            given = laws.get(segment)
            flags = adopted.get(segment, {})
            try:
                law, own, fitted = _path_law(members)
            except (ValueError, QhullError) as e:
                if given is None:
                    continue
                law, own, fitted = None, {}, {"why": str(e).splitlines()[0]}
            # Both laws held out on the same forwards: those of a path with a
            # constant of its own.
            mine = common(members)
            theirs = None if given is None else _validated(mine, list(flags))
            choice = _choice(fitted, theirs)
            if choice["kept"]:
                # The law `fit_host` gave holds out no worse: it stays.
                report[name] = report.get(name, {}) | theirs
                report[name] |= {"choice": choice, "paths_tried": fitted}
                report[name]["paths"] = dict(
                    sorted(Counter(m[2] for m in members).items())
                )
                _check(checks, name, report[name])
                continue
            fitted["choice"] = choice
            adopted.pop(segment, None)
            if kind == "decode":
                ladder = (host["key"].get("engine") or {}).get("ladder") or ()
                fitted["filled"] = fill_rungs(law[0], fitted["base"], own, ladder)
            elif given is not None:
                fitted["filled"] = fill_flags(
                    given[0], flags, launch.features, fitted["base"], own
                )
            report["unfitted"].pop(name, None)
            host["laws"].setdefault(kind, {})[segment] = law
            if own:
                host.setdefault(PATHS, {}).setdefault(kind, {})[segment] = own
            report[name] = fitted
            _check(checks, name, fitted)
        if kind in columns and not adopted:
            del columns[kind]
    for kind in [k for k, v in host["laws"].items() if not v]:
        del host["laws"][kind]
    if "columns" in host and not columns:
        del host["columns"]
    checks["H1"]["pass"] = all(v["pass"] for v in checks["H1"].values())
    checks["H2"]["pass"] = all(v["pass"] for v in checks["H2"].values())
    report["kind_blocks"] = checks


def _choice(fitted: dict, theirs: dict | None) -> dict:
    """Whether the law `fit_host` gave a launch segment stays over its law per
    launch path, whose report is `fitted` (no ``held_out`` where that law
    could not be fitted): by the rule #739 section 2.3 states, the path
    constants are adopted only where they lower the error held out in the same
    blocks, `theirs` that law's (`_validated`, None with no such law), a tie
    keeping the change out; and both errors."""
    own = fitted.get("held_out", {}).get("error")
    other = None if theirs is None else theirs["held_out"]["error"]
    lower = own is not None and other is not None and own < other * (1 - TIE)
    keep = theirs is not None and (own is None or (other is not None and not lower))
    return {"paths": own, "fit_host": other, "kept": keep}


def _check(checks: dict, name: str, fitted: dict) -> None:
    """H1 and H2 of the segment `name` from its report entry `fitted`: its
    paths short of `PATH_MIN` forwards, and those its held-out blocks miss."""
    short = sorted(p for p, n in fitted["paths"].items() if n < PATH_MIN)
    checks["H1"][name] = {"short": short, "pass": not short}
    held = fitted["held_out"]["paths"]
    missed = sorted(
        p for p in fitted["paths"] if held.get(p, {}).get("blocks", 0) < BLOCKS
    )
    checks["H2"][name] = {"short": missed, "pass": not missed}


def _validated(members: list, columns: list) -> dict:
    """A law `fit_host` gave, on ``(1, tokens, seqs)`` and the launch
    `columns` it adopted, held out in the same blocks per path (`_blocks`)
    over its `members` (`refit`'s rows): each path's forwards and its held-out
    figures."""
    paths = [m[2] for m in members]
    x = np.array([m[0] for m in members], dtype=np.float64).reshape(-1, 3)
    if np.linalg.matrix_rank(x[:, 1:]) < 2:
        x[:, 2] = 0.0
    design = np.column_stack([x, *([m[3][f] for m in members] for f in columns)])
    y = np.array([m[1] for m in members])
    held = _blocks(design, y, paths, [*HOST_FEATURES, *columns])
    return {"paths": dict(sorted(Counter(paths).items())), "held_out": held}


def fill_flags(intercept: float, columns: dict, features, base: str, own: dict):
    """Fill into `own`, a prefill or middle segment's path constants, each
    combination of the launch `features` whose path has no constant of its
    own, the base path `base` aside, at what the law `fit_host` gave prices
    it: its `intercept` plus each of its features' adopted `columns`. A path
    the records ran too few forwards on, or none, such as a middle chunk that
    replays its body, keeps that additive price rather than the base path's
    constant; return what it filled."""
    filled = {}
    for n in range(len(features) + 1):
        for chosen in itertools.combinations(sorted(features), n):
            p = "+".join(chosen) or "replay"
            if p != base and p not in own:
                c = intercept + sum(columns.get(f, 0.0) for f in chosen)
                filled[p] = own[p] = float(c)
    return filled


def fill_rungs(base_s: float, base: str, own: dict, ladder) -> dict:
    """Fill into `own`, a decode segment's path constants, each rung of
    `ladder` whose path (``rung<r>``) has no constant of its own, the base
    path's `base` (whose constant is `base_s`) aside: linear in the rung
    between the fitted rungs either side of it, the nearest one's past the
    last; and return what it filled. A rung the records ran too few decodes
    at, or none, is priced between its neighbours rather than at the base's
    constant."""
    fitted = {int(p[4:]): c for p, c in own.items() if p[4:].isdigit()}
    if base.startswith("rung") and base[4:].isdigit():
        fitted[int(base[4:])] = base_s
    if not fitted:
        return {}
    known = sorted(fitted)
    filled = {}
    for rung in ladder:
        if rung in fitted:
            continue
        lo = max((r for r in known if r < rung), default=None)
        hi = min((r for r in known if r > rung), default=None)
        if lo is None or hi is None:
            value = fitted[lo if hi is None else hi]
        else:
            share = (rung - lo) / (hi - lo)
            value = fitted[lo] + share * (fitted[hi] - fitted[lo])
        filled[f"rung{rung}"] = own[f"rung{rung}"] = float(value)
    return filled


def common(members: list) -> list:
    """The `members` (`refit`'s rows) on a path of `PATH_MIN` forwards or
    more, every one when none has as many: a path short of them has too few
    to fit or hold out a constant on, and is filled (`fill_rungs`,
    `fill_flags`)."""
    counts = Counter(m[2] for m in members)
    return [m for m in members if counts[m[2]] >= PATH_MIN] or members


def _path_law(members: list) -> tuple[list, dict, dict]:
    """The law of `members`, `refit`'s rows ``((1, tokens, seqs), seconds,
    path, launch)`` in step order, on a path of `PATH_MIN` forwards or more
    (`common`), with one constant per path but its commonest, ``[base, per
    token, per seq]``; those paths' constants; and its report: `_host_law`'s,
    each path's forwards, the base path, the paths `short` of `PATH_MIN`
    forwards left out, and the held-out figures (`_blocks`)."""
    counts = Counter(m[2] for m in members)
    members = common(members)
    paths = [m[2] for m in members]
    if not paths:
        raise ValueError("0 forwards for 3 coefficients")
    held_counts = Counter(paths)
    base = max(held_counts, key=lambda p: (held_counts[p], p))
    own = sorted(p for p in held_counts if p != base)
    short = sorted(p for p in counts if p not in held_counts)
    x = np.array([m[0] for m in members], dtype=np.float64).reshape(-1, 3)
    x[:, 0] = [p == base for p in paths]
    y = np.array([m[1] for m in members])
    values = [np.array([float(q == p) for q in paths]) for p in own]
    names = [f"path:{p}" for p in own]
    coef, fitted = _host_law(x, y, HOST_FEATURES, columns=list(zip(names, values)))
    design = x.copy()
    if np.linalg.matrix_rank(design[:, 1:]) < 2:
        design[:, 2] = 0.0
    pooled_x = design.copy()
    pooled_x[:, 0] = 1.0
    design = np.column_stack([design, *values])
    held = _blocks(design, y, paths, [*HOST_FEATURES, *names])
    held["error_pooled"] = _blocks(pooled_x, y, paths, list(HOST_FEATURES))["error"]
    report = fitted | {
        "paths": dict(sorted(counts.items())),
        "base": base,
        "short": short,
        "held_out": held,
    }
    return coef[:3], dict(zip(own, coef[3:])), report


def _blocks(design, y, paths, names) -> dict:
    """The held-out figures of the host-law solve of `design` on labels `y`,
    held out in `BLOCKS` blocks per path: each path's rows, in order, cut into
    `BLOCKS` contiguous blocks. Per block its rows, the p50 of ``price / label
    - 1`` and the p90 of its magnitude; per path its held-out rows, their p50
    and the blocks it appears in; and ``error``, the mean of each block's and
    path's p50 magnitude weighted by its rows (`_held_out_error`)."""
    block = np.zeros(len(y), dtype=int)
    for p in sorted(set(paths)):
        mine = np.array([i for i, q in enumerate(paths) if q == p])
        block[mine] = np.arange(len(mine)) * BLOCKS // len(mine)
    error = np.full(len(y), math.nan)
    for b in range(BLOCKS):
        held = block == b
        if held.any() and (~held).any():
            coef = _solve(design[~held], y[~held], names, drops=False)["coef"]
            error[held] = design[held] @ coef / y[held] - 1.0
    blocks, per, cells = [], {}, []
    labels = np.array(paths)
    for b in range(BLOCKS):
        e = error[(block == b) & ~np.isnan(error)]
        blocks.append(
            {
                "forwards": len(e),
                "p50": float(np.median(e)) if len(e) else None,
                "p90": float(np.percentile(np.abs(e), 90)) if len(e) else None,
            }
        )
        for p in sorted(set(paths)):
            if len(mine := e[labels[(block == b) & ~np.isnan(error)] == p]):
                cells.append((len(mine), abs(float(np.median(mine)))))
    for p in sorted(set(paths)):
        e = error[(labels == p) & ~np.isnan(error)]
        per[p] = {
            "forwards": len(e),
            "p50": float(np.median(e)) if len(e) else None,
            "blocks": len(set(block[(labels == p) & ~np.isnan(error)].tolist())),
        }
    rows = sum(n for n, _ in cells)
    return {
        "blocks": blocks,
        "paths": per,
        "error": sum(n * e for n, e in cells) / rows if rows else None,
    }


def record_name(record) -> str:
    """The run name `fit.samples.load` gives `record`'s samples."""
    steps = Path(record) / "steps"
    files = sorted(steps.glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:]))
    return ", ".join(map(step_labels.digest, files))


def client_runs(record, exports) -> list[dict]:
    """Each of the client runs `exports` that `record` served, in order: its
    graded span (`fit.samples.client_spans`), its host-clock window, from its
    first step's entry to its last step's exit over the DP ranks, and the
    steps of its span in which some rank, not a DP dummy, prefills. A prefill
    side's exports are read as `fit.publish.inputs` reads them."""
    decode = Path(record).parent / SIDES[1]
    if Path(record).name == SIDES[0] and decode.is_dir():
        exports = [
            named_by_prefill(read_json(e), record, decode)["requests"] for e in exports
        ]
    steps = served(Path(record) / "steps", exports)
    enter, exit_, prefill = {}, {}, set()
    for lines in steps.values():
        for s in lines:
            n = s["step"]
            enter[n] = min(enter.get(n, math.inf), s["t_enter_ns"])
            exit_[n] = max(exit_.get(n, -math.inf), s["t_exit_ns"])
            if not s["is_dummy_run"] and any(not row[2] for row in s["rows"]):
                prefill.add(n)
    out = []
    for k, (a, b) in sorted(client_spans(steps).items()):
        out.append(
            {
                "span": [a, b],
                "window": [enter[a], exit_[b]],
                "prefills": [n for n in sorted(prefill) if a <= n <= b],
            }
        )
    return out


def steady_windows(runs: dict, decode: set) -> dict:
    """Per record, the host-clock windows of its steady client runs: `runs`
    holds each record's `client_runs` with their exports, `decode` the
    ``(record, export)`` pairs named as decode client runs, which are not."""
    return {
        record: [tuple(r["window"]) for e, r in got if (record, e) not in decode]
        for record, got in runs.items()
    }


def drain_cut(samples, runs: dict, decode: set) -> tuple[list, list]:
    """`samples` less each decode client run's steps in which some rank
    prefills, their admissions' bursts, and the dummy forwards there: its
    host samples are its drains', the stretches of its span its ranks only
    decode in. And the report's ``decode_client_runs``: each run's record,
    export, span, its drains (count and steps), and its samples left out and
    kept."""
    cut: dict = {}
    report = []
    for record, got in runs.items():
        name = record_name(record)
        for export, r in got:
            if (record, export) not in decode:
                continue
            (a, b), bursts = r["span"], set(r["prefills"])
            cut.setdefault(name, set()).update(bursts)
            drains = [n for n in range(a, b + 1) if n not in bursts]
            stretches = sum(
                1 for i, n in enumerate(drains) if i == 0 or drains[i - 1] != n - 1
            )
            report.append(
                {
                    "record": record,
                    "export": export,
                    "span": [a, b],
                    "drains": stretches,
                    "drain_steps": len(drains),
                }
            )
    kept, dropped = [], Counter()
    for s in samples:
        bursts = cut.get(s.provenance["run"], ())
        if s.step in bursts:
            dropped[s.provenance["run"], s.step] += 1
            continue
        dummies = tuple(d for d in s.dummy_host if d[0] not in bursts)
        if dummies != s.dummy_host:
            s = dataclasses.replace(s, dummy_host=dummies)
        kept.append(s)
    for entry in report:
        name = record_name(entry["record"])
        a, b = entry["span"]
        entry["dropped"] = sum(
            v for (r, n), v in dropped.items() if r == name and a <= n <= b
        )
        entry["kept"] = sum(
            s.provenance["run"] == name and a <= s.step <= b for s in kept
        )
    return kept, report


def spans(records, windows: dict) -> dict:
    """H3 and H5 per record of `records` holding intake stamps, on its steady
    `windows`: per per-token tokenizer kind, its lone calls' token span in
    decades (a segment call's, the ids its own calls returned), and its lone
    requests' body span in decades, each with whether it reaches `H3_DECADES`
    or `H5_DECADES`; a kind with no lone call is not held to H3."""
    import json

    from atom.compass.run import INTAKE_RECORD, TOKENIZER_RECORD

    h3, h5 = {}, {}
    for record in records:
        intake = Path(record) / "steps" / INTAKE_RECORD
        if not intake.is_file():
            continue
        calls = intake.with_name(TOKENIZER_RECORD)
        w = windows.get(record)
        lone = _intakes(intake, calls if calls.is_file() else None, w)[0]
        h5[record] = _decades([b for b, _ in lone], H5_DECADES, "requests", "bytes")
        if not calls.is_file():
            continue
        _, *lines = calls.read_text().splitlines()
        got = [json.loads(line) for line in lines]
        got = [c for c in got if within(c["t_start_ns"], w)]
        tokens: dict = {k: [] for k in PER_TOKEN}
        for c in _lone_calls([c for c in got if c["call"] != "segment"], intake):
            if c["call"] == "decode":
                continue
            kind = c["call"] if c["items"] > 1 else "encode"
            tokens[kind].append(c["tokens"])
        tokens["segment"] = [n for n, _, _ in _own_work(got, intake)]
        h3[record] = {
            k: _decades(v, H3_DECADES, "calls", "tokens") for k, v in tokens.items()
        }
    held = [v for r in h3.values() for v in r.values() if v["calls"]]
    return {
        "H3": h3,
        "H5": h5,
        "H3_pass": all(v["pass"] for v in held),
        "H5_pass": all(v["pass"] for v in h5.values()),
    }


def _decades(values, need: float, count: str, unit: str) -> dict:
    """`values`' span, ``[lo, hi]`` over the positive ones, in decades, and
    whether it reaches `need`."""
    v = [x for x in values if x > 0]
    if not v:
        return {count: 0, unit: None, "decades": None, "pass": False}
    lo, hi = min(v), max(v)
    d = math.log10(hi / lo)
    return {count: len(v), unit: [lo, hi], "decades": d, "pass": d >= need}


def h4(decode_runs: list) -> dict:
    """H4: whether the samples are the graded windows only, the decode client
    runs' drains among them."""
    graded = os.environ.get(GRADED_ONLY_ENV) == "1"
    return {
        "graded_only": graded,
        "decode_drains": len(decode_runs),
        "pass": graded,
    }

# SPDX-License-Identifier: MIT
"""`compass fit host`'s kind-block host fit, opt-in.

    COMPASS_FIT_HOST_KIND_BLOCKS=1 compass fit host RECORD... --client-run RECORD EXPORT...
        [--decode-client-run RECORD EXPORT...] --store ROOT --out DIR

With `KIND_BLOCKS_ENV` set to ``1``, `refit` tries, for the decode, prefill
and middle kinds' launch laws (`PATH_SEGMENTS`) that
`step_linear_fit.fit_host` gave, one constant per launch path
(`launch_features.path`: a decode's CUDA-graph rung or eager, a prefill's or
middle chunk's replayed or eager body and tail), the slopes on ``tokens`` and
``seqs`` shared, in place of an intercept and the launch columns a held-out
choice adopts. Each path with `PATH_MIN` forwards or more takes a constant:
the kind's commonest the law's own, ``laws[kind][segment][0]``, which prices
any path the entry's ``paths`` (`launch_features.PATHS`) does not name, any
other one under ``paths``. A path short of `PATH_MIN` forwards is left out of
the fit (`common`), too few to fit or hold out a constant on, and filled: a
decode rung of the host key's ladder linearly between the fitted rungs either
side (`fill_rungs`), any other combination of the launch features at the price
of the law `fit_host` gave, its intercept plus the columns it adopted
(`fill_flags`); the report names them under ``filled``.

The path constants are adopted only where the error they hold out is lower
than that of the law `fit_host` gave, held out in the same blocks on the same
forwards (``choice``); a tie keeps that law, the path fit then reported under
``paths_tried``. The engine loop's segments keep `fit_host`'s laws. Every law
is held out in `BLOCKS` blocks per path: each path's forwards, in step order,
are cut into `BLOCKS` contiguous blocks, and block `b` of every path is held
out together, so every path is held out in every block. The report's
``<kind>.<segment>`` gives each path's forwards and, under ``held_out``, the
error per block and per path of the law fitted on the other blocks; a path
law's also its base path, the paths left out (``short``) and the same blocks'
error of a law with no path term (``error_pooled``).

The ``tokenizer`` and ``http_intake`` terms are fitted on the steady client
runs' host-clock windows alone (`steady_windows`: each `--client-run`'s
graded span, from its first step's entry to its last step's exit), each call
or request still read among all of its record's stamps
(`step_linear_fit.within`); a `--decode-client-run` gives none.

The report's ``kind_blocks`` states the checks a calibration record's host
fit is held to, each with what it read and whether it passed:

- ``path_forwards``: every launch path of every segment has `PATH_MIN`
  forwards (a path law leaves a path short of them out and fills it);
- ``path_blocks``: every path appears in every held-out block;
- ``token_decades``: each per-token tokenizer kind's fitted calls span
  `TOKEN_DECADES` decades of tokens, per record;
- ``steady_only``: the host laws are fitted on steady windows only: the
  graded spans (`fit.samples.GRADED_ONLY_ENV`), the decode client runs'
  drains among them;
- ``body_decades``: the lone requests' bodies span `BODY_DECADES` decades of
  bytes, per record.
"""

import itertools
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from atom.compass.backends import launch_features
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
    _solve,
    tokenizer_rows,
)
from atom.compass.fit.samples import GRADED_ONLY_ENV, client_spans
from atom.compass.parity.served import served
from atom.compass.validate.runs import SIDES, named_by_prefill

#: The variable that, set to ``1``, fits the launch laws per launch path, held
#: out in blocks per path, and the tokenizer and intake terms on the steady
#: client runs' windows (module docstring).
KIND_BLOCKS_ENV = "COMPASS_FIT_HOST_KIND_BLOCKS"
#: The held-out blocks each path's forwards are cut into.
BLOCKS = 5
#: The forwards a launch path needs for a constant of its own: two per
#: held-out block.
PATH_MIN = 2 * BLOCKS
#: The relative difference in held-out error within which two laws tie: a tie
#: keeps the law `fit_host` gave, the smaller change.
TIE = 1e-6
#: The decades of tokens a per-token tokenizer kind's fitted calls span.
TOKEN_DECADES = 1.0
#: The decades of body bytes the lone requests span.
BODY_DECADES = 2.0
#: The segments a forward's launch path sets, which take one constant per path:
#: the launch. The engine loop's segments keep `fit_host`'s law: on the cell A
#: e3v2 calibration record a constant per decode rung for `E`, `W_pre` and
#: `E_out` took up what its load put at each rung, and priced the c=64 target's
#: decode `E` 23% over where the law priced it 10% over.
PATH_SEGMENTS = ("W_launch",)
#: The tokenizer kinds charged per token (`step_linear_fit.fit_tokenizer`).
PER_TOKEN = ("encode", "encode_batch", "segment")


def enabled() -> bool:
    return os.environ.get(KIND_BLOCKS_ENV) == "1"


def refit(samples, host: dict, report: dict) -> None:
    """Refit `host`'s decode, prefill and middle `PATH_SEGMENTS` laws on
    `samples` with one constant per launch path, adopted where it holds out
    lower than the law `fit_host` gave, with their `report` entries; validate
    every other law `fit_host` gave in the same held-out blocks; and state the
    report's ``kind_blocks``, its ``path_forwards`` and ``path_blocks``
    (module docstring). The forwards `fit_host` leaves out as paying a JIT
    compile, and each forward after one, are left out alike."""
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
    checks: dict = {"blocks": BLOCKS, "path_min": PATH_MIN}
    checks |= {"path_forwards": {}, "path_blocks": {}}
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
            # Both laws held out on the same forwards: those of the paths with
            # a constant of their own.
            theirs = None if given is None else _validated(common(members), list(flags))
            choice = _choice(fitted, theirs)
            if choice["kept"]:
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
    if "columns" in host and not columns:
        del host["columns"]
    for check in ("path_forwards", "path_blocks"):
        checks[check]["pass"] = all(v["pass"] for v in checks[check].values())
    report["kind_blocks"] = checks


def _choice(fitted: dict, theirs: dict | None) -> dict:
    """Whether the law `fit_host` gave a launch segment stays over its law per
    launch path, whose report is `fitted` (no ``held_out`` where that law
    could not be fitted): the path constants are adopted only where they lower
    the error held out in the same blocks, `theirs` that law's (`_validated`,
    None with no such law), a tie keeping the law `fit_host` gave; and both
    errors."""
    own = fitted.get("held_out", {}).get("error")
    other = None if theirs is None else theirs["held_out"]["error"]
    lower = own is not None and other is not None and own < other * (1 - TIE)
    keep = theirs is not None and (own is None or (other is not None and not lower))
    return {"paths": own, "fit_host": other, "kept": keep}


def _check(checks: dict, name: str, fitted: dict) -> None:
    """The checks of the segment `name` from its report entry `fitted`: its
    paths short of `PATH_MIN` forwards, and those its held-out blocks miss."""
    short = sorted(p for p, n in fitted["paths"].items() if n < PATH_MIN)
    checks["path_forwards"][name] = {"short": short, "pass": not short}
    held = fitted["held_out"]["paths"]
    missed = sorted(
        p for p in fitted["paths"] if held.get(p, {}).get("blocks", 0) < BLOCKS
    )
    checks["path_blocks"][name] = {"missed": missed, "pass": not missed}


def _validated(members: list, columns: list) -> dict:
    """A law `fit_host` gave, on ``(1, tokens, seqs)`` and the launch
    `columns` it adopted, held out in blocks per path (`_blocks`) over its
    `members` (`refit`'s rows): each path's forwards and its held-out
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
    path `base` (whose constant is `base_s`) aside: linear in the rung
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
    (`common`), with one constant per path, ``[base, per token, per seq]``
    with the commonest path's constant its base; the other paths' constants;
    and its report: `_host_law`'s, each path's forwards, the base path, the
    paths `short` of `PATH_MIN` forwards left out, and the held-out figures
    (`_blocks`)."""
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
    and the blocks it appears in; and ``error``, the mean over each block's
    rows of each path the magnitude of their p50, weighted by those rows."""
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


def steady_windows(client_runs) -> dict[Path, list[tuple[int, int]]]:
    """Per record directory, the host-clock window of each client run
    `client_runs` names as ``(record, export)``: its graded span
    (`fit.samples.client_spans`), from the earliest entry of its first step
    to the latest exit of its last over the DP ranks. A prefill side's
    exports are read as `fit.publish.inputs` reads them."""
    named: dict = {}
    for record, export in client_runs:
        named.setdefault(Path(record), []).append(export)
    out = {}
    for record, exports in named.items():
        decode = record.parent / SIDES[1]
        if record.name == SIDES[0] and decode.is_dir():
            exports = [
                named_by_prefill(read_json(e), record, decode)["requests"]
                for e in exports
            ]
        steps = served(record / "steps", exports)
        lines = [s for got in steps.values() for s in got if "t_enter_ns" in s]
        out[record] = []
        for _, (a, b) in sorted(client_spans(steps).items()):
            span = [s for s in lines if a <= s["step"] <= b]
            enter = min(s["t_enter_ns"] for s in span)
            out[record].append((enter, max(s["t_exit_ns"] for s in span)))
    return out


def spans(records, windows: list, intakes, calls) -> dict:
    """The ``token_decades`` and ``body_decades`` checks of the stamped
    `records`, each with its `windows` (as `step_linear_fit.fit_intake` takes
    them), its intake stamps in `intakes` and its tokenizer record in `calls`:
    per per-token tokenizer kind, the token span in decades of the calls
    `fit_tokenizer` fits (`step_linear_fit.tokenizer_rows`, a segment call's
    tokens the ids its own calls returned), and the lone requests' body span
    in decades (`step_linear_fit._intakes`), each with whether it reaches
    `TOKEN_DECADES` or `BODY_DECADES`; a kind with no such call is not held
    to its span."""
    tokens, bodies = {}, {}
    for record, w, intake, made in zip(records, windows, intakes, calls):
        found = made if made.is_file() else None
        lone = [b for b, _ in _intakes(intake, found, w)[0]]
        bodies[str(record)] = _decades(lone, BODY_DECADES, "requests", "bytes")
        if found is None:
            continue
        _, *lines = made.read_text().splitlines()
        rows = tokenizer_rows([json.loads(line) for line in lines], intake, w)
        fitted = {k: [r[1] for r in rows if r[0] == k and r[2] > 0] for k in PER_TOKEN}
        tokens[str(record)] = {
            k: _decades(v, TOKEN_DECADES, "calls", "tokens") for k, v in fitted.items()
        }
    held = [v for r in tokens.values() for v in r.values() if v["calls"]]
    return {
        "token_decades": tokens | {"pass": all(v["pass"] for v in held)},
        "body_decades": bodies | {"pass": all(v["pass"] for v in bodies.values())},
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


def steady_only(decode_runs: list) -> dict:
    """The ``steady_only`` check: whether the samples are the graded windows
    only, the decode client runs `decode_runs` (`fit.host.drain_cut`'s
    report) fitted on their drains."""
    graded = os.environ.get(GRADED_ONLY_ENV) == "1"
    return {"graded_only": graded, "decode_drains": len(decode_runs), "pass": graded}

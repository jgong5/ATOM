# SPDX-License-Identifier: MIT
"""A run's report, one static page for a person to read:
`python -m atom.compass.report.html OUT`

OUT is a run's directory as `split` reads it, once its split is written. The
page goes to OUT/`FILE`, and no other file changes but, as grades below
says, the reports of the other runs its cell's `GRADES` grades. Every number
on it is read from a file in OUT, or in the cell holding OUT for its grades
and its real runs' splits, and a part whose file is missing, or holds nothing
for it, says so in its place:

- the run: simulated, by which backend, or real; one instance, or a
  deployment's two sides, their KV link and the router's arguments; each
  side's ATOM arguments and tree, from its `run.json`; the machine, a
  simulated run's spec from its run file, a real side's cards, PTL state,
  stack and host CPU;
- a simulated run's law sources, from its run file: its backend's digest and
  provenance, the store, and each law's own provenance;
- the workload: the client's arguments and traces, from its export;
- headline metrics, InferenceX's as `curves.metrics` computes them, and a
  simulated run's speed ratio, from its `summary.json`;
- trust, a simulated run's, from the same summary: the steps priced, the
  steps extrapolated and those a fallback priced as shares of them, the
  refusals, whether the run is a coverage report, and each pricing method's
  share of the predicted seconds;
- the per-request split, `split.FILE`: each segment's 10th, 50th and 90th
  percentiles over the requests it split that the headline metrics count, the
  LMCache tokens those loaded and stored, and why each request it did not
  split was not; or why there is no split;
- grades, the rows `python -m atom.compass.validate` printed for the cell
  holding OUT, from the cell's `GRADES`, when OUT is a run it graded and every
  run it graded still holds the results it read (`digest`); grading writes
  that file and then renders the report of each of its runs again, and a
  report written once one of them holds other results renders the others
  again, so none still shows the grades;
- a simulated run's split beside the real runs', when the cell's `GRADES`
  grades it: each segment's percentiles of this run and of each real run
  graded with it, from their `split.FILE`, and why a run has none;
- an op-level run's op-family split, `ops.split`: each side's time per step
  phase by op family, the terms that hold no one kind and the rest of its
  predicted time, each with why, and its refused steps; or why there is none;
- the Prometheus report: a link to `prometheus.HTML`, or what
  `prometheus.TEXT` says of why there is none.
"""

import collections
import hashlib
import json
import os
import sys
from html import escape
from pathlib import Path

from atom.compass.backends.json_file import read_json
from atom.compass.report import ops, prometheus, split
from atom.compass.run import OP_GRAPH, STEP_GRID, STEP_LINEAR, STEP_STUB, SUMMARY_FILE
from atom.compass.validate import curves
from atom.compass.validate.runs import CLIENT_RUN, EXPORT, sides

FILE = "report.html"
#: The rows grading printed for a cell, in that cell, with the runs it graded.
GRADES = "grades.json"
#: The files a run writes anew each time it runs, whose `digest` `GRADES` keeps:
#: its client's export, which holds a real run's wall-clock times, and a
#: simulated run's summary, which holds its wall seconds.
RESULTS = (EXPORT, SUMMARY_FILE)
RUN = "run.json"
WORKLOAD = "artifacts/workload.json"
#: The run-file keys that name a backend, each by its name.
BACKENDS = (STEP_LINEAR, STEP_GRID, OP_GRAPH, STEP_STUB)
#: The metrics `curves.metrics` gives as a fraction.
RATES = ("GPU hit rate", "CPU hit rate")
#: The split's fields shown, each by its name on the page, all in seconds.
SEGMENTS = {
    "queue_s": "queue",
    "prefill_s": "prefill",
    "kv_transfer_s": "KV transfer",
    "decode_s": "decode",
    "forward_ttft_s": "TTFT to its forward's return",
    "ttft_s": "TTFT at the client",
    "e2el_s": "end to end",
}
PERCENTILES = (10, 50, 90)
PERCENTILE_COLUMNS = tuple(f"p{p} (s)" for p in PERCENTILES)
#: The phases of the split's requests `curves.metrics` counts: the profiling
#: phase's, and those aiperf names none for, which `from_aiperf` names unknown.
PHASES = ("profiling", "unknown")
STYLE = """
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #fff; --muted: #6b6b6b;
  --line: #d8d8d8; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e6e6; --bg: #161618; --muted: #9a9a9a; --line: #3a3a3c; }
}
body { font: 14px/1.5 system-ui, sans-serif; color: var(--fg);
  background: var(--bg); max-width: 1000px; margin: 0 auto; padding: 16px; }
table { border-collapse: collapse; margin: 8px 0 16px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px;
  text-align: left; vertical-align: top; overflow-wrap: anywhere; }
.none { color: var(--muted); font-style: italic; }
"""


def _number(v) -> str:
    """`v` as the page shows it: four significant digits, or none."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v) if isinstance(v, int) else f"{v:.4g}"


def _percent(v) -> str:
    """The fraction `v` as a percentage of four significant digits, or none."""
    return "none" if v is None else f"{v * 100:.4g}%"


def _table(rows, head=()) -> str:
    """`rows` as a table, every cell escaped, under the header `head`."""

    def cells(tag, row):
        return "".join(f"<{tag}>{escape(str(c))}</{tag}>" for c in row)

    lines = [f"<tr>{cells('th', head)}</tr>"] if head else []
    lines += [f"<tr>{cells('td', row)}</tr>" for row in rows]
    return f"<table>{''.join(lines)}</table>"


def _none(text: str) -> str:
    return f'<p class="none">{escape(text)}</p>'


def _tree(source) -> str:
    """The ATOM tree run.json's ``source`` names."""
    if not source:
        return "not stated"
    dirty = source.get("dirty")
    state = {None: "uncommitted files not known", 0: "clean"}.get(
        dirty, f"{dirty} files uncommitted"
    )
    kind, revision = source.get("revision_kind"), source.get("revision")
    return f"{kind} {revision}, {state}, at {source.get('root')}"


def _stack(stack) -> str:
    return ", ".join(f"{k} {v}" for k, v in (stack or {}).items()) or "not stated"


def _cards(own: dict) -> str:
    """A real side's machine: its cards, PTL state, stack and host CPU."""
    counted = collections.Counter(
        f"{d.get('name')} ({d.get('arch')})" for d in own.get("device") or ()
    )
    cards = ", ".join(f"{n} x {card}" for card, n in counted.items()) or "no card"
    cpu = (own.get("host_key") or {}).get("host_cpu", "host CPU not stated")
    return f"{cards}; PTL {own.get('ptl')}; stack {_stack(own.get('stack'))}; {cpu}"


def _spec(run_file: dict) -> str:
    """A simulated run's machine, as its run file's spec states it."""
    spec = run_file.get("machine") or {}
    device = spec.get("device") or {}
    return (
        f"spec {spec.get('name')}: {device.get('count_per_node')} x "
        f"{device.get('name')} ({device.get('arch')}) a node; PTL "
        f"{device.get('ptl')}; stack {_stack(device.get('software_pinned_to'))}"
    )


def _run(run: Path, meta: dict) -> str:
    run_file = meta.get("run_file")
    if run_file is None:
        rows = [("run", "real")]
    else:
        backend = ", ".join(b for b in BACKENDS if run_file.get(b))
        rows = [("run", f"simulated, {backend or 'no backend stated'}")]
        rows.append(("machine", _spec(run_file)))
    each = sides(run)
    if len(each) > 1:
        n = collections.Counter(s.name.rstrip("0123456789") for s in each)
        shape = f"{n['prefill']}P{n['decode']}D"
        rows.append(("deployment", f"{shape}, KV link {meta.get('kv_link')}"))
        rows.append(("router", " ".join(meta.get("router") or ["not stated"])))
    else:
        rows.append(("deployment", "one instance"))
    for side in each:
        own = meta if side == run else read_json(side / RUN)
        name = "" if side == run else f"{side.name} "
        argv = own.get("argv")
        if argv is not None and "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        rows.append((f"{name}ATOM arguments", " ".join(argv or ["not stated"])))
        rows.append((f"{name}ATOM tree", _tree(own.get("source"))))
        if run_file is None:
            rows.append((f"{name}machine", _cards(own)))
    return _table(rows)


def _laws(run_file) -> str:
    if run_file is None:
        return _none("None: a real run prices nothing.")
    backend = run_file.get("backend") or {}
    rows = [("backend digest", backend.get("digest", "not stated"))]
    if backend.get("provenance") is not None:
        rows.append(("backend provenance", json.dumps(backend["provenance"])))
    rows.append(("store", run_file.get("store", "not stated")))
    for name in BACKENDS:
        laws = run_file.get(name)
        for law in laws if isinstance(laws, list) else [laws]:
            if isinstance(law, dict) and isinstance(law.get("provenance"), str):
                rows.append((f"{name} law", law["provenance"]))
    return _table(rows)


def _workload(run: Path) -> str:
    given = read_json(run / CLIENT_RUN)
    arguments = given.get("arguments") or {}
    rows = [
        (k, v if isinstance(v, str) else json.dumps(v))
        for k, v in sorted(arguments.items())
    ]
    rows.append(("traces", len(given.get("traces") or ())))
    if (run / WORKLOAD).is_file():
        rows.append(("dataset sha256", read_json(run / WORKLOAD).get("dataset_sha256")))
    return _table(rows)


def _metrics(run: Path, simulated: bool, summary) -> str:
    m = curves.metrics(run, simulated)
    rows = [
        (label, _number(None if m[name] is None else m[name] * scale))
        for name, (label, scale) in curves.AXES.items()
    ]
    rows += [(name, _percent(m[name])) for name in RATES]
    if simulated:
        cost = (summary or {}).get("cost")
        if cost is None:
            ratio = f"none: no {SUMMARY_FILE} states it"
        else:
            ratio = cost["speed_refused"] or (
                f"{_number(cost['speed_ratio'])}: {_number(cost['simulated_seconds'])}"
                f" simulated seconds in {_number(cost['wall_seconds'])} wall seconds"
            )
        rows.append(("speed ratio", ratio))
    return _table(rows)


def _trust(simulated: bool, summary) -> str:
    if not simulated:
        return _none("None: a real run is measured, not priced.")
    if summary is None:
        return _none(f"None: this run's directory holds no {SUMMARY_FILE}.")
    schedule = summary.get("schedule")
    if schedule is None:
        return _none(f"None: its {SUMMARY_FILE} holds no schedule.")
    refusals = schedule["refusals"]
    steps = refusals["steps"]

    def of_steps(n):
        share = None if n is None or not steps else n / steps
        return f"{_number(n)}, {_percent(share)} of the steps priced"

    rows = [
        ("steps priced", _number(steps)),
        ("steps extrapolated", of_steps(schedule["extrapolated_steps"])),
        ("steps a fallback priced", of_steps(schedule["fallback_steps"])),
        ("refusals", _number(refusals["count"])),
        (
            "steps refused",
            f"{_percent(refusals['fraction_of_steps'])} of the steps priced",
        ),
        ("coverage report", _number(summary.get("coverage_report"))),
    ]
    out = _table(rows)
    methods = schedule["seconds_by_method"]
    if methods is None:
        out += _none("No method shares: an engine handed in no priced steps.")
    else:
        shares = [
            (g, m["steps"], _percent(m["fraction_of_predicted_seconds"]))
            for g, m in methods.items()
        ]
        out += _table(shares, ("method", "steps", "share of predicted seconds"))
    if refusals["reasons"]:
        out += _table(refusals["reasons"], ("refusal", "count"))
    return out


def _counted(made: dict) -> list[dict]:
    """The requests of the split `made` that the headline metrics count."""
    return [r for r in made["requests"] if r["phase"] in PHASES]


def _segments(counted: list[dict]) -> dict:
    """Each segment's requests and `PERCENTILES` over `counted`, by its name;
    a segment none of them holds is left out."""
    out = {}
    for key, name in SEGMENTS.items():
        values = [r[key] for r in counted if r.get(key) is not None]
        if values:
            at = [_number(curves.percentile(values, p)) for p in PERCENTILES]
            out[name] = (len(values), *at)
    return out


def _split(run: Path) -> str:
    made = read_json(run / split.FILE)
    if "refused" in made:
        return _none(f"None: {made['refused']}")
    requests, unsplit = made["requests"], made["unsplit"]
    clock = f"the {made['clock']} clock"
    if "decode_offset_s" in made:
        low, high = made["decode_offset_s"]
        clock += (
            f", the decode side's joined to it at {high:.4f} s, its bounds"
            f" {low:.4f} s to {high:.4f} s"
        )
    for node, joined in made.get("node_offsets_s", {}).items():
        if not joined:
            continue
        low, high = (
            "none" if b is None else f"{b:.4f} s"
            for b in joined["bounds_s"] or (None, None)
        )
        if joined.get("source") == "measured":
            least, most = joined["measured_s"]
            clock += (
                f", {node}'s instances joined to it by {joined['samples']} measured"
                f" offsets, {least:.4f} s to {most:.4f} s, each within"
                f" {joined['error_s'] * 1e3:.3f} ms, which put"
                f" {joined['order_violations']} requests out of order, its bounds"
                f" {low} to {high}"
            )
        else:
            clock += (
                f", {node}'s instances joined to it at {joined['offset_s']:.4f} s,"
                f" its bounds {low} to {high}"
            )
    counted = _counted(made)
    text = (
        f"{len(requests)} requests split, on {clock}; {len(unsplit)} not. Below,"
        f" the {len(counted)} of them the headline metrics count."
    )
    out = f"<p>{escape(text)}</p>"
    rows = [(name, *at) for name, at in _segments(counted).items()]
    out += _table(rows, ("segment", "requests", *PERCENTILE_COLUMNS))
    tokens = [
        sum(r[k] for r in counted)
        for k in ("lmcache_load_tokens", "lmcache_store_tokens")
    ]
    out += _table([("loaded", tokens[0]), ("stored", tokens[1])], ("LMCache", "tokens"))
    if why := collections.Counter(unsplit.values()).most_common():
        out += _table(why, ("not split because", "requests"))
    return out


def digest(run) -> str:
    """The sha256 of `run`'s `RESULTS`, a file it lacks read as empty."""
    sha = hashlib.sha256()
    for name in RESULTS:
        path = Path(run) / name
        sha.update(path.read_bytes() if path.is_file() else b"")
    return sha.hexdigest()


def _changed(cell: Path, ran: dict) -> list[str]:
    """The runs of `ran`, each name's `digest` as grading read it, that `cell`
    holds other results for now."""
    return [n for n, sha in ran.items() if digest(cell / n) != sha]


def _grades(run: Path) -> str:
    """The grades of the cell holding `run`, from its `GRADES`, while each run
    they grade holds the results grading read, by its `digest`."""
    run = Path(os.path.abspath(run))  # its cell is the directory holding it
    path = run.parent / GRADES
    if not path.is_file():
        return _none(f"None: {run.parent} holds no {GRADES}; it is not a graded cell.")
    graded = read_json(path)
    names = ", ".join(graded["runs"])
    if run.name not in graded["runs"]:
        return _none(f"None: {path} grades {names}, not this run.")
    if changed := _changed(run.parent, graded["runs"]):
        changed = ", ".join(changed)
        return _none(f"None: {path} grades {names}; {changed} changed since.")
    text = f"Cell {graded['cell']}, graded on its runs {names}."
    return f"<p>{escape(text)}</p>{_table(graded['rows'], graded['columns'])}"


def _beside(run: Path) -> str:
    """Each segment's percentiles of `run`, a simulated run, beside those of
    each real run its cell's `GRADES` grades with it, and why a run has none."""
    run = Path(os.path.abspath(run))
    path = run.parent / GRADES
    graded = read_json(path) if path.is_file() else {"runs": {}}
    if run.name not in graded["runs"]:
        return _none("None: a simulated run's directory holds no real run.")
    # Grading grades one simulated run in a cell, so the others it names are real.
    real = [n for n in graded["runs"] if n != run.name]
    shown, why = {}, []
    for name in (run.name, *real):
        try:
            made = read_json(run.parent / name / split.FILE)
        except (OSError, ValueError) as missing:
            why.append((name, str(missing)))
            continue
        if "refused" in made:
            why.append((name, made["refused"]))
        else:
            shown[name] = _segments(_counted(made))
    text = (
        f"Cell {graded['cell']}: {run.name}, this run, beside {', '.join(real)},"
        " the real runs graded with it, each over the requests its headline"
        " metrics count."
    )
    rows = [
        (segment, name, *at[segment])
        for segment in SEGMENTS.values()
        for name, at in shown.items()
        if segment in at
    ]
    out = f"<p>{escape(text)}</p>"
    if rows:
        out += _table(rows, ("segment", "run", "requests", *PERCENTILE_COLUMNS))
    if why:
        out += _table(why, ("run", "no split because"))
    return out


def _ops(run: Path) -> str:
    out = ""
    for side, ranks, phases in ops.split(run):
        for name, phase in sorted(phases.items(), key=lambda p: -p[1].predicted):
            refused = sum(phase.refused.values())
            text = (
                f"{f'{side} side, ' if side else ''}{name}: {phase.steps} steps"
                f" over {ranks} DP ranks, {refused} of them refused; seconds summed"
                " over the ranks."
            )
            share = phase.predicted or None
            rows = [
                (part, _number(s), _percent(share and s / share), why)
                for part, s, why in ops.rows(phase)
            ]
            head = ("part", "seconds", "share of predicted", "op kinds, or why")
            out += f"<p>{escape(text)}</p>{_table(rows, head)}"
            if refused:
                out += _table(phase.refused.most_common(), ("refused", "steps"))
    return out


def _prometheus(run: Path) -> str:
    if (run / prometheus.HTML).is_file():
        text = "ATOM's CI report over this run's Prometheus series"
        return f'<p><a href="{prometheus.HTML}">{escape(text)}</a></p>'
    return _none(f"None: {(run / prometheus.TEXT).read_text().strip()}")


def _part(title: str, make, *args) -> str:
    try:
        body = make(*args)
    except (OSError, ValueError) as missing:
        body = _none(f"None: {missing}")
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def page(run) -> str:
    """`run`'s report, as the module says."""
    run = Path(run)
    meta = read_json(run / RUN)
    simulated = "run_file" in meta
    path = run / SUMMARY_FILE
    summary = read_json(path) if simulated and path.is_file() else None
    title = f"{run.name}: a {'simulated' if simulated else 'real'} run"
    parts = [
        _part("Run", _run, run, meta),
        _part("Law sources", _laws, meta.get("run_file")),
        _part("Workload", _workload, run),
        _part("Headline metrics", _metrics, run, simulated, summary),
        _part("Trust", _trust, simulated, summary),
        _part("Grades", _grades, run),
        _part("Per-request split", _split, run),
    ]
    if simulated:
        parts.append(_part("A real run's per-request split", _beside, run))
    parts.append(_part("Op-family split", _ops, run))
    parts.append(_part("Prometheus report", _prometheus, run))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Run report</title><style>{STYLE}</style></head><body>"
        f"<h1>{escape(title)}</h1>{''.join(parts)}</body></html>\n"
    )


def write(run) -> Path:
    """Write `run`'s report to its `FILE`; that file. When a run its cell's
    `GRADES` grades holds other results now, render again the report of each
    other run that file grades, so none still shows the grades."""
    path = Path(run) / FILE
    path.write_text(page(run))
    here = Path(os.path.abspath(run))  # a run given as `.` too
    cell = here.parent
    if (cell / GRADES).is_file():
        ran = read_json(cell / GRADES)["runs"]
        if _changed(cell, ran):
            for name in ran:
                other = cell / name
                if name != here.name and (other / FILE).is_file():
                    (other / FILE).write_text(page(other))
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m atom.compass.report.html OUT")
    write(sys.argv[1])

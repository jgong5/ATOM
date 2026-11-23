# SPDX-License-Identifier: MIT
"""The search's dashboard, `FILE` in OUT, which `compass advice` writes when it
ends: one static page, served by no server and loading nothing from a network.

It is built from files alone and computes no result anew: OUT's results
(`results.RESULTS`), the expansion `compass advice` wrote, the head the reuse
check took (`check.REUSE`), and, for each point that ran, its GPUs as
`curves.gpus` counts them and each side's ``run.json`` for the ATOM tree it ran.
Top to bottom:

- the search: its model, machine, store, workload and SLO, and the head, its
  commit and ATOM digest, or why the check refused the tree;
- the chart of throughput per GPU against interactivity, two switches choosing
  each one's definition (`AXES`): InferenceX's, of the tokens served, or of
  the tokens computed; p90 interactivity or the E2E-normalized one. Its four
  views are drawn here and switched with no script. Each configuration is a
  line through its points by concurrency, named at its last. A point's shape
  is its grading state (`SHAPES`), its fill's slices its pricing-method shares
  under `GROUPS`' names; a point with a refusal is grey and off its line. The
  frontier the results hold is joined by p90 interactivity, and the
  recommendation and the runner-up are ringed. A point's hover text holds its
  metrics, total throughput, GPUs, ATOM tree, laws and refusal, and a click
  opens its run's `REPORT`;
- the reference layer, when one is given (`reference`): each of its curves
  dotted, under its source's name;
- the recommendation's text, its trust included;
- each point's row, grey with its reason when it has a refusal, linking its
  run's report; then each command that makes what a point lacks.
"""

import math
import os
from html import escape
from pathlib import Path

from atom.compass.advice import results
from atom.compass.advice.check import REUSE
from atom.compass.backends.cost import GROUPS
from atom.compass.backends.json_file import read_json
from atom.compass.drive.serve import RUN_JSON
from atom.compass.report.html import FILE as REPORT
from atom.compass.validate import curves
from atom.compass.validate.runs import sides

#: The dashboard, in OUT.
FILE = "dashboard.html"
#: Each switched axis: its two definitions, InferenceX's first, each with the
#: switch's label.
AXES = {
    "y": {
        "throughput per GPU": "tokens served (InferenceX)",
        "computed throughput per GPU": "tokens computed, less prefix-cache hits",
    },
    "x": {
        "p90 interactivity": "p90 interactivity (InferenceX)",
        "p90 E2E-normalized interactivity": "p90 E2E-normalized (InferenceX)",
    },
}
#: Each grading state's shape about the origin.
SHAPES = {
    "graded": '<rect x="-6" y="-6" width="12" height="12"/>',
    "ungraded": '<circle r="7"/>',
    "coverage report": '<polygon points="0,-8 7.5,6 -7.5,6"/>',
}
#: Each configuration's line, by its order in SPACE; later ones solid.
DASHES = ("", "7 4", "2 3", "9 3 2 3", "14 5", "1 5")
#: The table's columns past the results' own, and the hover text's.
COLUMNS = {
    **{k: v for k, v in results.COLUMNS.items() if k != "refusal"},
    "total": "total throughput (tok/s)",
    "gpus": "GPUs",
    "tree": "ATOM tree",
    "laws": "laws",
    "refusal": "refusal reason",
}
#: The chart's size and margins, in pixels.
W, H, LEFT, RIGHT, TOP, BOTTOM = 760, 430, 70, 120, 16, 46
STYLE = """
:root { color-scheme: light dark; --fg: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --bg: #f9f9f7; --surface: #fcfcfb; --grid: #e1e0d9; --axis: #c3c2b7;
  --refused: #c3c2b7; --m-analytical: #2a78d6; --m-declared: #eb6834;
  --m-measured: #1baf7a; --m-estimated: #eda100; --m-unvalidated: #e87ba4; }
@media (prefers-color-scheme: dark) { :root { --fg: #f0efec; --ink2: #c3c2b7;
  --bg: #0d0d0d; --surface: #1a1a19; --grid: #2c2c2a; --axis: #383835;
  --refused: #5c5b56; --m-analytical: #3987e5; --m-declared: #d95926;
  --m-measured: #199e70; --m-estimated: #c98500; --m-unvalidated: #d55181; } }
body { font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--fg); background: var(--bg); max-width: 1180px; margin: 0 auto;
  padding: 16px; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
table { border-collapse: collapse; margin: 8px 0 16px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--grid); padding: 3px 8px; text-align: left;
  vertical-align: top; overflow-wrap: anywhere; }
.wide { overflow-x: auto; } .wide td, .wide th { min-width: 4em; }
tr.refused td { color: var(--muted); }
.views > div { display: none; } .views svg { max-width: 100%; height: auto;
  background: var(--surface); }
#y0:checked ~ #x0:checked ~ .views .y0x0, #y0:checked ~ #x1:checked ~ .views .y0x1,
#y1:checked ~ #x0:checked ~ .views .y1x0, #y1:checked ~ #x1:checked ~ .views .y1x1
  { display: block; }
.switch { color: var(--ink2); margin-right: 6px; } label { margin-right: 14px; }
.grid { stroke: var(--grid); } .axis { stroke: var(--axis); }
.tick, .c { fill: var(--muted); font-size: 11px; } .name { fill: var(--ink2); font-size: 12px; }
.line { fill: none; stroke: var(--ink2); stroke-width: 1.5; }
.frontier { fill: none; stroke: var(--fg); stroke-width: 6; opacity: 0.18; }
.ref { fill: none; stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 1 4; }
.refpt { fill: var(--surface); stroke: var(--muted); stroke-width: 1.5; }
.outline { fill: none; stroke: var(--fg); stroke-width: 1.5; }
.refused { fill: var(--refused); stroke: var(--muted); stroke-width: 1.5; }
.ring { fill: none; stroke: var(--fg); stroke-width: 2; }
.ring.runner { stroke-dasharray: 3 3; }
.unpriced { fill: var(--surface); }
""" + "".join(f".m-{g} {{ fill: var(--m-{g}); }}\n" for g in GROUPS)


def reference(path: Path) -> dict:
    """The reference layer in the file at `path`: one JSON object of its
    ``source``, where its numbers come from, which labels it, and its
    ``curves``, each name to a list of points, each point an object of numbers
    by `results.METRICS`' names and ``concurrency``; refused naming the file
    and what is wrong."""
    layer = read_json(path)
    if not isinstance(layer, dict) or set(layer) != {"source", "curves"}:
        raise ValueError(f"{path}: a reference layer is an object of source and curves")
    if not isinstance(layer["source"], str) or not layer["source"].strip():
        raise ValueError(f"{path}: its source is not a non-empty string")
    curves, named = layer["curves"], {"concurrency", *results.METRICS}
    if not isinstance(curves, dict) or not all(
        isinstance(c, list) and all(isinstance(p, dict) for p in c)
        for c in curves.values()
    ):
        raise ValueError(f"{path}: its curves are not names, each to a list of objects")
    for name, points in curves.items():
        for p in points:
            if odd := sorted(set(p) - named):
                raise ValueError(f"{path}: curve {name!r} names unknown fields {odd}")
            if odd := sorted(k for k, v in p.items() if not _number(v)):
                raise ValueError(f"{path}: curve {name!r} holds no number at {odd}")
    return layer


def write(out: Path, candidates: Path, layer: dict | None = None) -> Path:
    """The dashboard of the search in `out`, whose expansion is the file
    `candidates`, with the reference `layer` when given: written into `out` as
    `FILE`, its path."""
    held, expansion = read_json(out / results.RESULTS), read_json(candidates)
    reuse = read_json(out / REUSE)
    rows = [r | _ran(r, out) for r in held["rows"]]
    rings = {held["recommendation"]: "recommended", held["runner_up"]: "runner-up"}
    rings.pop(None, None)
    page = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(Path(expansion['model']).name)}: compass advice</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>compass advice: {escape(Path(expansion['model']).name)}</h1>",
        (
            "<p>Every number here is read from the search's files in this directory "
            "and from each point's run directory; a point opens its run's report.</p>"
        ),
        _search(expansion, reuse, held["slo"]),
        "<h2>The frontier</h2>",
        _chart(rows, rings, layer),
        "<h2>The recommendation</h2>",
        *(f"<p>{escape(line)}</p>" for line in held["text"].splitlines()),
        *(
            f'<p><a href="{escape(r["report"])}">{escape(r["point"])}\'s report</a></p>'
            for r in rows
            if r["point"] in rings and r["report"]
        ),
        "<h2>Each point</h2>",
        _table(rows),
        _commands(rows),
        "</body></html>",
    ]
    path = out / FILE
    path.write_text("\n".join(page) + "\n")
    return path


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _ran(row: dict, out: Path) -> dict:
    """What `row`'s run directory adds: its GPUs (`curves.gpus`), which read
    when its metrics did; the total throughput; the ATOM tree each side's run
    record names, "not stated" where it is missing or does not say; and its
    report, relative to `out`, None where there is none. All None for a point
    that did not run."""
    if not row["run"]:
        return dict.fromkeys(("gpus", "total", "tree", "report"))
    run, named = Path(row["run"]), {}
    gpus = curves.gpus(run)
    for side in sides(run):
        try:
            source = read_json(side / RUN_JSON).get("source") or {}
        except (OSError, ValueError, AttributeError):
            source = {}
        revision, dirty = source.get("revision"), source.get("dirty")
        tree = f"{source.get('revision_kind')} {revision[:9]}" if revision else None
        if tree and dirty:
            tree += f", {dirty} files uncommitted"
        named[side.name] = tree or "not stated"
    # One tree once, else each side's by name.
    trees = list(dict.fromkeys(named.values()))
    tree = (
        trees[0] if len(trees) == 1 else "; ".join(f"{s} {t}" for s, t in named.items())
    )
    tpg = row["throughput per GPU"]
    report = run / REPORT
    return {
        "gpus": gpus,
        "total": tpg * gpus if tpg is not None else None,
        "tree": tree,
        "report": os.path.relpath(report, out) if report.is_file() else None,
    }


def _shown(row: dict, column: str) -> str:
    """`row`'s `column` as the page shows it: the total and the laws its own
    way, every other column as the result table does (`results._shown`)."""
    value = row[column]
    if column == "total" and value is not None:
        return f"{value:.1f}"
    if column == "laws":
        laws = (f"{s} {', '.join(dict.fromkeys(r))}" for s, r in value.items())
        return "; ".join(laws) or "-"
    return results._shown(row, column)


def _search(expansion: dict, reuse: dict, slo: dict) -> str:
    """The search's table: what it searched, at which head, under which SLO."""
    head, refused = reuse["head"], reuse["refused"]
    if head:
        at = f"{head['commit']} ({head['ref']}), ATOM digest {head['atom']}"
    else:
        at = f"refused: {refused['reason']}"
    load = expansion["workload"]
    workload = f"{load['traces']}, traces {load['select']}, concurrency " + ", ".join(
        map(str, load["concurrency"])
    )
    if "duration_s" in load:
        workload += f", {load['duration_s']} s"
    bounds = [
        f"{metric} {'at least' if below else 'at most'} {slo[bound]:g} {unit}"
        for bound, (metric, below, unit) in results.BOUNDS.items()
        if bound in slo
    ]
    machine = ", ".join(f"{k} {v}" for k, v in expansion["machine"].items())
    rows = [
        ("space", expansion["space"]),
        ("model", expansion["model"]),
        ("machine", machine),
        ("store", expansion["store"]),
        ("frozen commit", expansion["commit"] or "none: the integration head"),
        ("head", at),
        ("workload", workload),
        ("SLO", "; ".join(bounds) or "none"),
    ]
    cells = "".join(f"<tr><th>{k}</th><td>{escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table>{cells}</table>"


def _hover(row: dict) -> str:
    """`row`'s hover text: its name, then each column with a value."""
    said = [f"{row['configuration']} at concurrency {row['concurrency']}"]
    for column, label in list(COLUMNS.items())[2:]:
        if row.get(column) is not None:
            said.append(f"{label}: {_shown(row, column)}")
    return escape("\n".join(said))


def _pie(shares: dict | None) -> str:
    """The slices of a point's fill, each method's share of its predicted
    seconds, clockwise from the top in `GROUPS`' order."""
    out, turned = ['<circle r="10" class="unpriced"/>'], 0.0
    for g in GROUPS:
        share = (shares or {}).get(g) or 0.0
        if share >= 0.9999:
            out.append(f'<circle r="10" class="m-{g}"/>')
        elif share > 0:
            a, b = (2 * math.pi * t - math.pi / 2 for t in (turned, turned + share))
            arc = f"A10 10 0 {int(share > 0.5)} 1 {10 * math.cos(b):.2f} {10 * math.sin(b):.2f}"
            out.append(
                f'<path class="m-{g}" d="M0 0L{10 * math.cos(a):.2f} '
                f'{10 * math.sin(a):.2f}{arc}Z"/>'
            )
        turned += share
    return "".join(out)


def _dash(i: int) -> str:
    """The dashes of the `i`th configuration's line."""
    return DASHES[i] if i < len(DASHES) else ""


def _ticks(top: float) -> list[float]:
    """Round ticks from 0 to at least `top`, five or so."""
    step = 10 ** math.floor(math.log10(top / 5))
    step *= next(m for m in (1, 2, 5, 10) if top / (step * m) <= 6)
    return [i * step for i in range(math.ceil(top / step - 1e-9) + 1)]


def _view(rows: list[dict], rings: dict, layer: dict | None, x: str, y: str) -> str:
    """The chart on axes `x` and `y`, an SVG, and how many points it lacks."""
    placed = [r for r in rows if r[x] is not None and r[y] is not None]
    curves = {
        name: [p for p in points if x in p and y in p]
        for name, points in (layer["curves"].items() if layer else ())
    }
    xs = [r[x] for r in placed] + [p[x] for c in curves.values() for p in c]
    ys = [r[y] for r in placed] + [p[y] for c in curves.values() for p in c]
    tx, ty = _ticks(max(xs, default=0) * 1.05 or 1), _ticks(
        max(ys, default=0) * 1.05 or 1
    )

    def at(px, py) -> tuple[float, float]:
        return (
            LEFT + px / tx[-1] * (W - LEFT - RIGHT),
            H - BOTTOM - py / ty[-1] * (H - TOP - BOTTOM),
        )

    def line(points, cls, dash="") -> str:
        drawn = " ".join("{:.1f},{:.1f}".format(*at(*p)) for p in points)
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline class="{cls}"{dashed} points="{drawn}"/>'

    svg = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">']
    for t in tx:
        px = at(t, 0)[0]
        svg.append(
            f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>'
        )
        svg.append(
            f'<text class="tick" x="{px:.1f}" y="{H - BOTTOM + 16}" text-anchor="middle">{t:g}</text>'
        )
    for t in ty:
        py = at(0, t)[1]
        svg.append(
            f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{py:.1f}" y2="{py:.1f}"/>'
        )
        svg.append(
            f'<text class="tick" x="{LEFT - 6}" y="{py + 4:.1f}" text-anchor="end">{t:g}</text>'
        )
    svg.append(
        f'<line class="axis" x1="{LEFT}" x2="{W - RIGHT}" y1="{H - BOTTOM}" y2="{H - BOTTOM}"/>'
    )
    labels = (results.METRICS[x][0], results.METRICS[y][0])
    svg.append(
        f'<text class="name" x="{(LEFT + W - RIGHT) / 2}" y="{H - 8}" text-anchor="middle">{labels[0]}</text>'
    )
    svg.append(
        f'<text class="name" transform="translate(16 {(TOP + H - BOTTOM) / 2}) rotate(-90)" '
        f'text-anchor="middle">{labels[1]}</text>'
    )
    for name, points in curves.items():
        if points:
            svg.append(line([(p[x], p[y]) for p in points], "ref"))
        for p in points:
            px, py = at(p[x], p[y])
            said = escape(
                f"{layer['source']}: {name}"
                + "".join(f"\n{k}: {v:g}" for k, v in p.items())
            )
            svg.append(
                f'<g transform="translate({px:.1f},{py:.1f})"><title>{said}</title>'
                '<polygon class="refpt" points="0,-6 6,0 0,6 -6,0"/></g>'
            )
        if points:
            px, py = at(points[-1][x], points[-1][y])
            svg.append(
                f'<text class="name" x="{px + 9:.1f}" y="{py + 4:.1f}">{escape(name)}</text>'
            )
    frontier = sorted(
        (r for r in placed if r["frontier"]), key=lambda r: r["p90 interactivity"]
    )
    if frontier:
        svg.append(line([(r[x], r[y]) for r in frontier], "frontier"))
    configurations = list(dict.fromkeys(r["configuration"] for r in rows))
    for i, name in enumerate(configurations):
        own = sorted(
            (r for r in placed if r["configuration"] == name and not r["refusal"]),
            key=lambda r: r["concurrency"],
        )
        if own:
            svg.append(line([(r[x], r[y]) for r in own], "line", _dash(i)))
            px, py = at(own[-1][x], own[-1][y])
            svg.append(
                f'<text class="name" x="{px + 12:.1f}" y="{py + 4:.1f}">{escape(name)}</text>'
            )
    for r in placed:
        shape = SHAPES[r["grading"]]
        if r["refusal"]:
            mark = f'<g class="refused">{shape}</g>'
        else:
            mark = f'<g clip-path="url(#clip-{r["grading"].replace(" ", "-")})">{_pie(r["method_shares"])}</g>'
            mark += f'<g class="outline">{shape}</g>'
        if ring := rings.get(r["point"]):
            runner = " runner" if ring == "runner-up" else ""
            mark += f'<circle class="ring{runner}" r="13"/><text class="name" x="15" y="18">{ring}</text>'
        px, py = at(r[x], r[y])
        mark = (
            f'<g transform="translate({px:.1f},{py:.1f})"><title>{_hover(r)}</title>'
            f'<circle r="12" fill="transparent"/>{mark}'
            f'<text class="c" x="9" y="-8">c{r["concurrency"]}</text></g>'
        )
        if r["report"]:
            mark = f'<a href="{escape(r["report"])}">{mark}</a>'
        svg.append(mark)
    svg.append("</svg>")
    missing = len(rows) - len(placed)
    note = f"<p>{missing} of {len(rows)} points have no value on these axes; each is in the table below.</p>"
    return "\n".join(svg) + (note if missing else "")


def _legend(rows: list[dict], layer: dict | None) -> str:
    """What each line, shape, slice and ring on the chart stands for."""

    def glyph(inner: str, width: int = 22) -> str:
        return f'<svg width="{width}" height="20" viewBox="-11 -10 {width} 20">{inner}</svg>'

    configurations = dict.fromkeys(r["configuration"] for r in rows)
    lines = {
        "Configurations": [
            glyph(
                f'<line class="line" x1="-30" x2="30" y1="0" y2="0" stroke-dasharray="{_dash(i)}"/>',
                44,
            )
            + f" {escape(name)}"
            for i, name in enumerate(configurations)
        ],
        "Shape, the grading state": [
            glyph(f'<g class="outline">{shape}</g>') + f" {state}"
            for state, shape in SHAPES.items()
        ],
        "Fill, each pricing method's share of the predicted seconds": [
            glyph(f'<circle r="7" class="m-{g}"/>') + f" {g}" for g in GROUPS
        ],
        "Marks": [
            glyph(f'<g class="refused">{SHAPES["coverage report"]}</g>')
            + " a point with a refusal, its reason in its hover text and the table",
            glyph('<line class="frontier" x1="-30" x2="30" y1="0" y2="0"/>', 44)
            + " the frontier of throughput per GPU and p90 interactivity (InferenceX)",
            glyph('<circle class="ring" r="8"/>') + " recommended",
            glyph('<circle class="ring runner" r="8"/>') + " runner-up",
        ],
    }
    if layer:
        lines["Reference, not this search"] = [
            glyph('<polygon class="refpt" points="0,-6 6,0 0,6 -6,0"/>')
            + f" {escape(layer['source'])}"
        ]
    return "".join(
        f"<p>{name}: {' &nbsp; '.join(said)}</p>" for name, said in lines.items()
    )


def _chart(rows: list[dict], rings: dict, layer: dict | None) -> str:
    """The switches, the legend and the four views, one shown at a time."""
    clips = "".join(
        f'<clipPath id="clip-{state.replace(" ", "-")}">{shape}</clipPath>'
        for state, shape in SHAPES.items()
    )
    out = [
        f'<svg width="0" height="0" style="position:absolute"><defs>{clips}</defs></svg>'
    ]
    for axis, names in (("y", AXES["y"]), ("x", AXES["x"])):
        out.append(
            f'<span class="switch">{"throughput per GPU" if axis == "y" else "interactivity"}:</span>'
        )
        for i, label in enumerate(names.values()):
            checked = " checked" if i == 0 else ""
            out.append(
                f'<input type="radio" name="{axis}" id="{axis}{i}"{checked}>'
                f'<label for="{axis}{i}">{label}</label>'
            )
        out.append("<br>" if axis == "y" else "")
    out.append(_legend(rows, layer))
    out.append('<div class="views">')
    for i, y in enumerate(AXES["y"]):
        for j, x in enumerate(AXES["x"]):
            out.append(f'<div class="y{i}x{j}">{_view(rows, rings, layer, x, y)}</div>')
    out.append("</div>")
    return "\n".join(out)


def _table(rows: list[dict]) -> str:
    """A row per point, grey when it has a refusal, linking its run's report."""
    head = (
        "".join(f"<th>{label}</th>" for label in COLUMNS.values()) + "<th>report</th>"
    )
    body = []
    for r in rows:
        cells = "".join(f"<td>{escape(_shown(r, c))}</td>" for c in COLUMNS)
        report = (
            f'<a href="{escape(r["report"])}">report</a>' if r["report"] else "none"
        )
        refused = ' class="refused"' if r["refusal"] else ""
        body.append(f"<tr{refused}>{cells}<td>{report}</td></tr>")
    return f'<div class="wide"><table><tr>{head}</tr>{"".join(body)}</table></div>'


def _commands(rows: list[dict]) -> str:
    """Each command that makes what a point lacks, with the points it serves."""
    made = {}
    for r in rows:
        for command in r["made_by"]:
            made.setdefault((r["configuration"], command), []).append(r["concurrency"])
    if not made:
        return ""
    items = "".join(
        f"<li>{escape(name)} at concurrency {', '.join(map(str, at))}: "
        f"<code>{escape(command)}</code></li>"
        for (name, command), at in made.items()
    )
    return f"<h2>What makes what a point lacks</h2><ul>{items}</ul>"

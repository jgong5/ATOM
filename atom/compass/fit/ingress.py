# SPDX-License-Identifier: MIT
"""`compass fit ingress`: an encoding ingress's ``ingress`` term, fitted on the
stamps it wrote.

    compass fit ingress STAMPS... [--skip N] [--processes 4] [--workers 16]
        [--metrics FILE] --out FILE

Each STAMPS is a file `scripts/compass/encoding_ingress.py --stamps` wrote,
or the prefix it was given, which reads every ``<prefix>.<pid>`` beside it.
Each chat request it encoded and sent on is one sample (`samples`), the first
`--skip` of them by receipt left out (a burst that is no steady load, such as
aiperf's warmup primers). The ingress's processes x workers are the term's
``workers``, from its own flags, which no stamp states.

`fit_ingress` fits the thread a request holds, ``serialized - start`` (its
body's parse, the render, the encode and the forwarded body's serialize), on
the client's body bytes: ``fixed_s`` and ``per_byte_s`` by least squares, or
where that would put ``fixed_s`` below 0 (an encode that grows faster than its
body, as the V4.1 tokenizer's does past about 100 kB), ``fixed_s`` 0 and
``per_byte_s`` the samples' thread seconds over their bytes, which keeps their
mean, the load the threads carry. The report sets the law's thread seconds
beside the measured ones, at p50, p90 and p99, and gives the hop's parts as
the stamps time them; the queue it measured is the real arrivals', and in a
simulated run, whose requests reach the ingress in wall-clock bursts, not the
model's to match.

With `--metrics FILE`, atomesh's Prometheus text scraped after the last of
these requests (its ``/metrics``), ``router_per_byte_s`` is atomesh's whole
``json_parse`` stage seconds over the client bytes of every request the
stamps say the ingress sent it, encoded or forwarded as sent (`router_term`);
the counts must agree, so the stamps and the scrape cover one router's life.
Without it the term states no ``router_per_byte_s`` and the report says why.

FILE gets ``{"ingress": <term>, "report": <report>}``: `compass fit host
--ingress FILE` states the term in a `host_laws` entry, and a law file's
``host`` takes it as its ``ingress`` (`atom.compass.ingress`).
"""

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

from atom.compass.ingress import check

PROG = "compass fit ingress"
#: The stamps a sample needs, in their order on the request's way.
STAGES = ("recv", "body", "start", "parsed", "rendered", "encoded", "serialized")
#: Kinds of request the ingress sent on to the router with a chat body.
FORWARDED = ("encoded", "fallback")
_SERIES = re.compile(
    r'^mesh_prepare_stage_seconds_(sum|count)\{stage="json_parse"\}\s+(\S+)$',
    re.MULTILINE,
)


def stamps(paths) -> list[dict]:
    """Every stamp line of `paths`, each a file or the prefix of ``<prefix>.<pid>``
    files; a path that names neither refuses, and so does a line that is no
    JSON object but a last one cut short."""
    rows = []
    for path in paths:
        files = [path] if Path(path).is_file() else sorted(glob.glob(f"{path}.*"))
        if not files:
            raise ValueError(f"{path}: neither a stamps file nor the prefix of any")
        for name in files:
            lines = Path(name).read_text().splitlines()
            for k, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    if k < len(lines) - 1:
                        raise ValueError(f"{name}:{k + 1}: not a stamp line") from None
    return rows


def samples(rows: list[dict], skip: int = 0) -> list[dict]:
    """The chat requests of `rows` the ingress encoded and sent on, by receipt,
    less the first `skip`."""
    sent = [r for r in rows if r.get("kind") == "encoded" and "sent" in r]
    if missing := [r for r in sent if any(s not in r for s in STAGES)]:
        raise ValueError(
            f"{len(missing)} encoded requests lack a stamp of {STAGES}, as "
            f"{missing[0]}: not stamps encoding_ingress.py wrote"
        )
    return sorted(sent, key=lambda r: r["recv"])[skip:]


def _q(xs, p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def _ms(xs) -> dict:
    """p50, p90, p99 and mean of seconds `xs`, in ms."""
    return {
        **{f"p{round(100 * p)}": 1e3 * _q(xs, p) for p in (0.5, 0.9, 0.99)},
        "mean": 1e3 * statistics.fmean(xs),
    }


def fit_ingress(chosen: list[dict], workers: int) -> tuple[dict, dict]:
    """The ``ingress`` term over the `samples` `chosen`, with `workers` encode
    threads in all, but its router part (`router_term`), and its report."""
    if len(chosen) < 2:
        raise ValueError(f"{len(chosen)} encoded requests: too few to fit")
    x = [r["bytes"] for r in chosen]
    held = [(r["serialized"] - r["start"]) * 1e-9 for r in chosen]
    mx, my = statistics.fmean(x), statistics.fmean(held)
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx == 0:
        raise ValueError(f"every one of {len(x)} encoded bodies is {x[0]} bytes")
    slope = sum((v - mx) * (y - my) for v, y in zip(x, held)) / sxx
    fixed = my - slope * mx
    least = {"fixed_s": fixed, "per_byte_s": slope}
    if fixed >= 0 and slope >= 0:
        term, method = least, "least squares"
    else:
        term = {"fixed_s": 0.0, "per_byte_s": sum(held) / sum(x)}
        method = (
            f"least squares puts fixed_s at {fixed:.3g} s and per_byte_s at "
            f"{slope:.3g} s, so fixed_s is 0 and per_byte_s the thread seconds "
            "over the bytes"
        )
    term = {**term, "workers": workers, "bytes": [min(x), max(x)]}
    law = [term["fixed_s"] + term["per_byte_s"] * v for v in x]
    ns = {
        part: [(r[b] - r[a]) * 1e-9 for r in chosen]
        for part, (a, b) in {
            "body_read": ("recv", "body"),
            "queue": ("body", "start"),
            "parse": ("start", "parsed"),
            "render": ("parsed", "rendered"),
            "encode": ("rendered", "encoded"),
            "serialize": ("encoded", "serialized"),
            "dispatch": ("serialized", "sent"),
            "hop": ("recv", "sent"),
        }.items()
    }
    tokens = [r.get("tokens", 0) for r in chosen]
    report = {
        "requests": len(chosen),
        "method": method,
        "least_squares": least,
        "thread_ms": {"measured": _ms(held), "law": _ms(law)},
        "bytes": {"p50": _q(x, 0.5), "mean": mx, "max": max(x)},
        "stamps_ms": {part: _ms(v) for part, v in ns.items()},
        "most_threads_busy": _busiest(chosen),
        "wall_span_s": (chosen[-1]["recv"] - chosen[0]["recv"]) * 1e-9,
    }
    if sum(tokens) and sum(ns["encode"]):
        report["tokens"] = {"p50": _q(tokens, 0.5), "mean": statistics.fmean(tokens)}
        report["bytes_per_token"] = sum(x) / sum(tokens)
        report["encode_tokens_per_s"] = sum(tokens) / sum(ns["encode"])
    if all("sent_bytes" in r for r in chosen):
        forwarded = sum(r["sent_bytes"] for r in chosen)
        report["forwarded_bytes_per_byte"] = forwarded / sum(x)
    return check(term, PROG), report


def _busiest(chosen: list[dict]) -> dict:
    """The most encode jobs any one process of the ingress ran at once, by
    process, which its ``--workers`` bounds."""
    out = {}
    for proc in sorted({r.get("proc", 0) for r in chosen}):
        edges = sorted(
            (t, d)
            for r in chosen
            if r.get("proc", 0) == proc
            for t, d in ((r["start"], 1), (r["serialized"], -1))
        )
        busy = most = 0
        for _, d in edges:
            busy += d
            most = max(most, busy)
        out[str(proc)] = most
    return out


def router_term(rows: list[dict], metrics: str) -> tuple[float, dict]:
    """``router_per_byte_s`` from atomesh's Prometheus text `metrics` and every
    request `rows` says the ingress sent it with a chat body, and its report;
    refused where the scrape's parse count is not the stamps' count."""
    got = {kind: float(v) for kind, v in _SERIES.findall(metrics)}
    if set(got) != {"sum", "count"}:
        raise ValueError(
            "the metrics hold no mesh_prepare_stage_seconds sum and count for "
            "stage json_parse: not atomesh's /metrics"
        )
    sent = [r for r in rows if r.get("kind") in FORWARDED and "sent" in r]
    if int(got["count"]) != len(sent):
        raise ValueError(
            f"atomesh parsed {int(got['count'])} requests and the stamps say the "
            f"ingress sent it {len(sent)}: the scrape and the stamps must cover "
            "one router's life, the scrape after the last request"
        )
    nbytes = sum(r["bytes"] for r in sent)
    if nbytes == 0:
        raise ValueError(f"the {len(sent)} requests the ingress sent have no bytes")
    report = {
        "parsed": len(sent),
        "parse_s": got["sum"],
        "mean_parse_ms": 1e3 * got["sum"] / len(sent),
        "client_bytes": nbytes,
    }
    if all("sent_bytes" in r for r in sent):
        report["per_forwarded_byte_s"] = got["sum"] / sum(r["sent_bytes"] for r in sent)
    return got["sum"] / nbytes, report


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    parser.add_argument("stamps", nargs="+", metavar="STAMPS")
    parser.add_argument(
        "--skip", type=int, default=0, help="encoded requests to leave out first"
    )
    parser.add_argument(
        "--processes", type=int, default=4, help="the ingress's --processes"
    )
    parser.add_argument("--workers", type=int, default=16, help="its --workers")
    parser.add_argument(
        "--metrics", help="atomesh's /metrics text, scraped after the last request"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.skip < 0 or args.processes < 1 or args.workers < 1:
        raise ValueError(
            "--skip is a count of requests, --processes and --workers positive"
        )
    rows = stamps(args.stamps)
    term, report = fit_ingress(samples(rows, args.skip), args.processes * args.workers)
    report |= {"stamps": args.stamps, "skip": args.skip}
    if args.metrics:
        term["router_per_byte_s"], report["router"] = router_term(
            rows, Path(args.metrics).read_text()
        )
        report["router"]["metrics"] = args.metrics
    else:
        report["unfitted"] = {"router_per_byte_s": "no --metrics"}
    Path(args.out).write_text(
        json.dumps({"ingress": check(term, PROG), "report": report}, indent=1) + "\n"
    )
    print(json.dumps({"ingress": term, "thread_ms": report["thread_ms"]}))

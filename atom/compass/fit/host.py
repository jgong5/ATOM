# SPDX-License-Identifier: MIT
"""`compass fit host`: the host laws, fitted on real runs' records and published.

    compass fit host RECORD... --client-run RECORD EXPORT... --store ROOT
        [--stated-hop NAME=SECONDS...] [--ingress FILE] --out DIR

`host_laws` fits the samples' host labels by `step_linear_fit.fit_host`, under
the host key the real runner writes beside each record, and its report names
each run with every trace its steps served. On a pipeline's records it fits
each stage's samples (`Sample.stage`) apart, each `E` from its release
(`step_labels.label_run`), into one entry: each stage's ``laws``, ``admit``
and ``columns`` under ``stages``, in stage order, and the report's under its
own ``stages``, beside the frontend's terms below and the host key, once;
stages under two host keys refuse. From the tokenizer calls and HTTP
intake stamps the recorded server wrote beside its steps
(`atom.compass.run.TOKENIZER_RECORD`, `INTAKE_RECORD`) it fits the entry's
``tokenizer`` term (`step_linear_fit.fit_tokenizer`) and ``http_intake`` term
(`step_linear_fit.fit_intake`); and from those stamps, the engine's queue
handoffs and step lines, and the aiperf raw records beside each client run's
export (`client_run.RAW`, gzipped or not), its ``hops``
(`step_linear_fit.fit_hops`). A prefill side's record beside its deployment's
decode side's, both stamped on one host's clock, also gives the ``relay``
term (`step_linear_fit.fit_relay`): a prefill response's time through the
router to decode's app call, which a run delays past ``router_s``. Records of
a server that streamed its responses give the ``http_stream`` term
(`step_linear_fit.fit_stream`): its API loop's CPU per SSE event; a server
that streamed none has no such term to fit. A record whose server did not exit refuses: its
server's lines are where ``steps/<name>.local`` names (`atom.compass.run.LOCAL`),
not in its steps. A term the records cannot give, or a kind of
tokenizer call as ``tokenizer.<kind>`` or a hop as ``hops.<hop>``, is left out
and its reason listed under the report's ``unfitted``. The host key gains the
fields of the configuration's platform a `host_laws` key states, which a run's
spec must match; the entry is keyed by that host key's values and holds the run file's `host` entry as its
params.

`--stated-hop NAME=SECONDS` writes a hop no record times into the entry's
``hops``, as a committed law file states it: ``router_s``, the router's
per-request forward, which a `deploy.sh run` with a `--dp-aware` router needs
above the IPC hop it borrows for ``router_in_s``, and ``kv_write_req_s``,
Mooncake's write request (`STATED_HOPS`). The report's ``stated_hops`` names
each, so the entry says which hops were stated rather than fitted.

`--ingress FILE` states the encoding ingress in front of the deployment as the
entry's ``ingress`` term (`atom.compass.ingress`): FILE's ``ingress``, as
`compass fit ingress` writes it from the ingress's stamps, which no record
here holds. The report's ``stated_ingress`` names FILE and the term. Fit such
an entry on records taken behind that ingress: their server took each prompt's
token ids, so it rendered and tokenized nothing, and read larger bodies.
"""

import gzip
import json
import math
from pathlib import Path

from atom.compass.artifacts import Key, Kind, keyed_platform
from atom.compass.backends.step_linear_fit import (
    fit_hops,
    fit_host,
    fit_intake,
    fit_relay,
    fit_stream,
    fit_tokenizer,
    streams,
)
from atom.compass.fit import publish
from atom.compass.ingress import KEY as INGRESS
from atom.compass.ingress import check as check_ingress
from atom.compass.parity.client_run import RAW
from atom.compass.run import INTAKE_RECORD, LOCAL, TOKENIZER_RECORD
from atom.compass.validate.runs import SIDES, deployment_side

PROG = "compass fit host"


def host_laws(samples, records=(), client_runs=()) -> tuple[dict, dict]:
    """The host laws over `samples`, with the tokenizer, HTTP intake and hop
    terms of the record directories `records`, each served the client runs
    `client_runs` pairs it with by their exports, as a run file's `host` entry,
    and their report, a pipeline's stages fitted apart (`_staged`); samples
    with no host labels under a host key refuse."""
    from scipy.spatial import QhullError

    bare = sorted(
        {s.provenance["run"] for s in samples if s.provenance["config"] is None}
    )
    if bare:
        raise ValueError(
            f"{'; '.join(bare)} holds steps with no host labels under a host key, "
            "which the real runner writes beside its record; host laws are "
            "fitted on real runs only"
        )
    for name in (TOKENIZER_RECORD, INTAKE_RECORD):
        for r in records:
            if (named := Path(r) / "steps" / (name + LOCAL)).is_file():
                raise ValueError(
                    f"{r}: its server did not exit, so steps/{name} does not hold "
                    f"its lines; steps/{name + LOCAL} names the file that holds "
                    f"them, {named.read_text().strip()}"
                )
    host, report = _staged(samples)
    stamped = [r for r in records if (Path(r) / "steps" / INTAKE_RECORD).is_file()]
    steps = [Path(r) / "steps" / INTAKE_RECORD for r in stamped]
    calls = [d.with_name(TOKENIZER_RECORD) for d in steps]
    clients = [
        _client(r, [e for d, e in client_runs if Path(d) == Path(r)]) for r in stamped
    ]
    # A prefill side whose deployment's decode side stamped its intake too.
    relayed = [
        (d, sibling)
        for r, d in zip(stamped, steps)
        if Path(r).name == SIDES[0]
        and (sibling := Path(r).parent / SIDES[1] / "steps" / INTAKE_RECORD).is_file()
    ]
    # A server that streamed its responses: a non-streaming one, as a
    # prefill-decode prefill side, sends no event to price.
    streamed = any(streams(d) for d in steps)
    for term, fit in (
        ("tokenizer", lambda: fit_tokenizer(calls, steps)),
        ("http_intake", lambda: fit_intake(steps, calls)),
        ("hops", lambda: fit_hops(steps, clients)),
        *([("http_stream", lambda: fit_stream(steps))] if streamed else []),
        *([("relay", lambda: fit_relay(relayed))] if relayed else []),
    ):
        try:
            if not steps:
                raise ValueError(f"no record holds steps/{INTAKE_RECORD}")
            host[term], report[term] = fit()
        except (ValueError, QhullError) as e:
            report["unfitted"][term] = str(e).splitlines()[0]
    for term in ("tokenizer", "hops"):
        for kind, why in report.get(term, {}).pop("unfitted", {}).items():
            report["unfitted"][f"{term}.{kind}"] = why.splitlines()[0]
    if host.get("hops") == {}:
        del host["hops"]
    return host, {"provenance": publish.runs(samples), **report}


def _staged(samples) -> tuple[dict, dict]:
    """`fit_host` of `samples`, on a pipeline's each stage's apart: one entry
    under their one host key, each stage's part under ``stages``."""
    pp = 1 + max((s.stage for s in samples), default=0)
    if pp == 1:
        return fit_host(samples)
    parts = [fit_host([s for s in samples if s.stage == k]) for k in range(pp)]
    keys = [json.dumps(host["key"], sort_keys=True) for host, _ in parts]
    if len(set(keys)) > 1:
        raise ValueError(
            f"the pipeline's stages carry the host keys {' / '.join(keys)}; a "
            "pipeline's host laws are fitted under one"
        )
    stages = [{k: v for k, v in host.items() if k != "key"} for host, _ in parts]
    host = {"key": parts[0][0]["key"], "stages": stages}
    return host, {"stages": [report for _, report in parts], "unfitted": {}}


def _client(record, exports) -> dict:
    """Each request's send and ack on CLOCK_MONOTONIC, in ns, from the aiperf
    raw records beside the client runs' `exports` of `record`, as `fit_hops`
    takes them. aiperf takes ``start_perf_ns`` on that clock and states
    ``request_ack_ns`` on the wall clock it read beside it as
    ``request_start_ns``. A prefill-decode side's requests and streams pass the
    router, so its client's sends and acks time no hop of its own."""
    if deployment_side(record):
        return {"missing": f"{record} is a side of a deployment, behind its router"}
    sent, acked = [], []
    for export in exports:
        raw = Path(export).with_name(RAW)
        if not raw.is_file():
            raw = raw.with_name(f"{RAW}.gz")
        if not raw.is_file():
            return {"missing": f"no {RAW} beside {export}"}
        with (gzip.open if raw.suffix == ".gz" else open)(raw, "rb") as lines:
            for r in map(json.loads, lines):
                meta, start = r["metadata"], r["start_perf_ns"]
                sent.append(start)
                if meta.get("request_ack_ns") is not None:
                    acked.append(
                        start + meta["request_ack_ns"] - meta["request_start_ns"]
                    )
    return {"sent": sent, "acked": acked}


#: The hops no record times, each stated in a law file until now: the router's
#: per-request forward (#3009's assumed 1.5 ms) and Mooncake's KV write request
#: (#2603's p10), which need the router stamped and both sides on one clock.
STATED_HOPS = ("router_s", "kv_write_req_s")


def stated_hops(words: list[str], fitted: dict) -> dict:
    """The hops `--stated-hop NAME=SECONDS` words state, by name: each a hop
    of `STATED_HOPS`, once, a positive number of seconds; refused for any
    other hop, which the records fit or name unfitted (`fitted`, the entry's
    own ``hops``), and so is never stated over them."""
    out = {}
    for word in words:
        name, _, value = word.partition("=")
        if name not in STATED_HOPS:
            raise ValueError(
                f"--stated-hop {word}: only {', '.join(STATED_HOPS)}, which no "
                "record times, is stated; the others are fitted from the records"
            )
        if name in out or name in fitted:
            raise ValueError(
                f"--stated-hop {word}: the entry's hops state {name} already"
            )
        try:
            seconds = float(value)
        except ValueError:
            seconds = math.nan
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"--stated-hop {word}: not a positive number of seconds")
        out[name] = seconds
    return out


def stated_ingress(path: str) -> dict:
    """The ``ingress`` term `--ingress path` states: the file's ``ingress``,
    refused unless it is one (`atom.compass.ingress.check`)."""
    got = json.loads(Path(path).read_text())
    if not isinstance(got, dict) or INGRESS not in got:
        raise ValueError(f"--ingress {path}: holds no {INGRESS} term")
    return check_ingress(got[INGRESS], f"--ingress {path}: its {INGRESS}")


def main(argv: list[str]) -> None:
    parser = publish.parser(PROG, __doc__.split("\n\n")[0])
    parser.add_argument(
        "--stated-hop",
        action="append",
        default=[],
        metavar="NAME=SECONDS",
        help=f"a hop no record times, one of {', '.join(STATED_HOPS)}, written "
        "into the entry's hops and named under the report's stated_hops",
    )
    parser.add_argument(
        "--ingress",
        metavar="FILE",
        help="the encoding ingress in front of the deployment, compass fit "
        "ingress's FILE, written as the entry's ingress term and named under the "
        "report's stated_ingress",
    )
    args = parser.parse_args(argv)
    ingress = stated_ingress(args.ingress) if args.ingress else None
    samples, config = publish.inputs(args, ("record",))
    host, report = host_laws(samples, args.inputs, args.client_run)
    if stated := stated_hops(args.stated_hop, host.get("hops", {})):
        host["hops"] = {**host.get("hops", {}), **stated}
        report["stated_hops"] = stated
    if ingress is not None:
        host[INGRESS] = ingress
        report["stated_ingress"] = {"file": args.ingress, INGRESS: ingress}
    k = host["key"]
    k |= keyed_platform(Kind.HOST_LAWS, config["platform"])
    key = Key.of(
        Kind.HOST_LAWS,
        **keyed_platform(Kind.HOST_LAWS, k),
        host_cpu=k["host_cpu"],
        atom_src=k["atom"],
        model=k["model"],
        engine_config=json.dumps(k["engine"], sort_keys=True),
    )
    command = " ".join([PROG, *argv])
    publish.publish(args, command, key, publish.conditions(config), host, report)

# SPDX-License-Identifier: MIT
"""`compass fit host`: the host laws, fitted on real runs' records and published.

    compass fit host RECORD... --client-run RECORD EXPORT... --store ROOT
        [--decode-client-run RECORD EXPORT...]
        [--stated-hop NAME=SECONDS...] [--ingress FILE] --out DIR

`host_laws` fits the samples' host labels by `step_linear_fit.fit_host`, under
the host key the real runner writes beside each record, and its report names
each run with every trace its steps served. From the tokenizer calls and HTTP
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

`--decode-client-run RECORD EXPORT` names a client run RECORD's server served,
as `--client-run` does, whose host samples are its decode drains alone
(`drain_cut`); the report's ``decode_client_runs`` says what each kept and
left out. Without it, the fit is as it was.

With `host_blocks.KIND_BLOCKS_ENV` set to ``1``, the kind-block host fit
(`host_blocks`): the decode, prefill and middle launch laws take one constant
per launch path where those hold out lower than the laws `fit_host` gave, the
tokenizer and intake terms are fitted on the `--client-run`s' windows alone,
and the report's ``kind_blocks`` states the checks a calibration record's
host fit is held to. Unset, the fit is as it was.
"""

import dataclasses
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from atom.compass.artifacts import Key, Kind, keyed_platform
from atom.compass.backends.step_labels import (
    STALLS,
    WAITS,
    digest,
    group_prefills,
    launch_stall_on,
    launch_wait_on,
    record_ranks,
    stall_entry,
    stall_prices,
    wait_labels,
)
from atom.compass.backends.step_linear_fit import (
    fit_hops,
    fit_host,
    fit_intake,
    fit_relay,
    fit_stream,
    fit_tokenizer,
    streams,
)
from atom.compass.fit import host_blocks, publish
from atom.compass.fit.samples import client_spans
from atom.compass.ingress import KEY as INGRESS
from atom.compass.ingress import check as check_ingress
from atom.compass.parity.client_run import RAW
from atom.compass.parity.served import served
from atom.compass.run import INTAKE_RECORD, LOCAL, TOKENIZER_RECORD
from atom.compass.validate.runs import SIDES

PROG = "compass fit host"


def host_laws(samples, records=(), client_runs=(), steady=None) -> tuple[dict, dict]:
    """The host laws over `samples`, with the tokenizer, HTTP intake and hop
    terms of the record directories `records`, each served the client runs
    `client_runs` pairs it with by their exports, as a run file's `host` entry,
    and their report; samples with no host labels under a host key refuse.
    With `steady`, each record's steady client runs' host-clock windows
    (`host_blocks.steady_windows`), the kind-block host fit: the launch laws
    refitted per launch path (`host_blocks.refit`), the tokenizer and intake
    terms fitted within those windows, none in a record with none, and the
    report's ``kind_blocks``. Under ``COMPASS_FIT_LAUNCH_STALL=1``
    (`step_labels.STALL_ENV`) the entry also states `step_labels.STALLS`, per
    launch path what a rank's first launch of a shape costs
    (`step_labels.stall_prices`, which the report states with its counts),
    which a run charges (`runner.overrides.launch_stall`). Under
    ``COMPASS_FIT_LAUNCH_WAIT=1`` (`step_labels.WAIT_ENV`) each `W_launch`
    label in a host kind whose launches wait for the device end of the
    forward before them loses that wait (`step_labels.wait_labels`, which the
    report states under `step_labels.WAITS`), and the entry states those kinds
    under `step_labels.WAITS`, which a run's recurrence keeps
    (`runner.overrides.HostDeviceRecurrence`)."""
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
    waits = None
    if launch_wait_on():
        # `step_labels.WAIT_ENV`: launch labels less their wait on earlier work.
        samples, waited, waits = wait_labels(samples, records)
    host, report = fit_host(samples)
    if waits is not None:
        host[WAITS], report[WAITS] = waits, waited
    if steady is not None:
        host_blocks.refit(samples, host, report)
    if launch_stall_on():
        # `step_labels.STALL_ENV`: the records' launch stalls, per launch path.
        report[STALLS] = stall_prices(map(record_ranks, records))
        host[STALLS] = stall_entry(report[STALLS])
    stamped = [r for r in records if (Path(r) / "steps" / INTAKE_RECORD).is_file()]
    windows = None if steady is None else [steady.get(Path(r), []) for r in stamped]
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
        ("tokenizer", lambda: fit_tokenizer(calls, steps, windows)),
        ("http_intake", lambda: fit_intake(steps, calls, windows)),
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
    if steady is not None:
        report["kind_blocks"] |= host_blocks.spans(stamped, windows, steps, calls)
    return host, {"provenance": publish.runs(samples), **report}


def _client(record, exports) -> dict:
    """Each request's send and ack on CLOCK_MONOTONIC, in ns, from the aiperf
    raw records beside the client runs' `exports` of `record`, as `fit_hops`
    takes them. aiperf takes ``start_perf_ns`` on that clock and states
    ``request_ack_ns`` on the wall clock it read beside it as
    ``request_start_ns``. A prefill-decode side's requests and streams pass the
    router, so its client's sends and acks time no hop of its own."""
    if Path(record).name in SIDES and (Path(record).parent / SIDES[0]).is_dir():
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


def drain_cut(samples, decode) -> tuple[list, list]:
    """`samples` less, for each client run `decode` names as ``(record,
    export)``, the steps of its graded span (`fit.samples.client_spans`) in
    which some rank prefills and the dummy forwards there: a batch admitted at
    once is decoded down, so its host samples are its drains', the stretches
    of its span in which its ranks only decode. And the report's
    ``decode_client_runs``: per run its record, export, span, drains and their
    steps, and its samples left out and kept. A prefill side refuses."""
    named: dict = {}
    for record, export in decode:
        named.setdefault(Path(record), []).append(export)
    bursts, report = {}, []
    for record, exports in named.items():
        if record.name == SIDES[0] and (record.parent / SIDES[1]).is_dir():
            raise ValueError(
                f"--decode-client-run names {record}, a prefill side, which "
                "decodes nothing"
            )
        steps = served(record / "steps", exports)
        prefills = group_prefills(steps.values())
        # The run name `fit.samples.load` gives the record's samples.
        files = sorted(
            (record / "steps").glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:])
        )
        name = ", ".join(map(digest, files))
        at = Counter(s.step for s in samples if s.provenance["run"] == name)
        for k, (a, b) in sorted(client_spans(steps).items()):
            cut = {n for n in prefills if a <= n <= b}
            bursts.setdefault(name, set()).update(cut)
            # In step order: a set read in hash order is refused
            # (test_set_iteration_lint), whatever the sum.
            drains = [n for n in range(a, b + 1) if n not in cut]
            held = set(drains)
            report.append(
                {
                    "record": str(record),
                    "export": exports[k],
                    "span": [a, b],
                    "drains": sum(n - 1 not in held for n in drains),
                    "drain_steps": len(drains),
                    "dropped": sum(at[n] for n in cut),
                    "kept": sum(at[n] for n in drains),
                }
            )
    kept = []
    for s in samples:
        cut = bursts.get(s.provenance["run"], ())
        if s.step not in cut:
            dummies = tuple(d for d in s.dummy_host if d[0] not in cut)
            kept.append(dataclasses.replace(s, dummy_host=dummies))
    return kept, report


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
        "--decode-client-run",
        nargs=2,
        action="append",
        default=[],
        metavar=("RECORD", "EXPORT"),
        help="a client run RECORD's server served, by its export, whose decode "
        "drains alone give host samples: the steps of its span in which no rank "
        "prefills",
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
    records = {Path(i) for i in args.inputs if (Path(i) / "run.json").is_file()}
    for record, _ in args.decode_client_run:
        if Path(record) not in records:
            raise ValueError(
                f"--decode-client-run names {record}, which is not a RECORD"
            )
    steady_runs = args.client_run
    args.client_run = [*args.client_run, *args.decode_client_run]
    samples, config = publish.inputs(args, ("record",))
    # The --client-runs' windows, once `publish.inputs` has checked their records.
    steady = host_blocks.steady_windows(steady_runs) if host_blocks.enabled() else None
    drained = []
    if args.decode_client_run:
        samples, drained = drain_cut(samples, args.decode_client_run)
    host, report = host_laws(samples, args.inputs, args.client_run, steady)
    if args.decode_client_run:
        report["decode_client_runs"] = drained
    if steady is not None:
        report["kind_blocks"]["steady_only"] = host_blocks.steady_only(drained)
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

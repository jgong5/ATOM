# SPDX-License-Identifier: MIT
"""`compass fit host`: the host laws, fitted on real runs' records and published.

    compass fit host RECORD... --client-run RECORD EXPORT... --store ROOT
        --out DIR

`host_laws` fits the samples' host labels by `step_linear_fit.fit_host`, under
the host key the real runner writes beside each record, and its report names
each run with every trace its steps served. From the tokenizer calls and HTTP
intake stamps the recorded server wrote beside its steps
(`atom.compass.run.TOKENIZER_RECORD`, `INTAKE_RECORD`) it fits the entry's
``tokenizer`` term (`step_linear_fit.fit_tokenizer`) and ``http_intake`` term
(`step_linear_fit.fit_intake`); and from those stamps, the engine's queue
handoffs and step lines, and the aiperf raw records beside each client run's
export (`client_run.RAW`, gzipped or not), its ``hops``
(`step_linear_fit.fit_hops`). A record whose server did not exit refuses: its
server's lines are where ``steps/<name>.local`` names (`atom.compass.run.LOCAL`),
not in its steps. A term the records cannot give, or a kind of
tokenizer call as ``tokenizer.<kind>`` or a hop as ``hops.<hop>``, is left out
and its reason listed under the report's ``unfitted``. The host key gains the
fields of the configuration's platform a `host_laws` key states, which a run's
spec must match; the entry is keyed by that host key's values and holds the run file's `host` entry as its
params.
"""

import gzip
import json
from pathlib import Path

from atom.compass.artifacts import Key, Kind, keyed_platform
from atom.compass.backends.step_linear_fit import (
    fit_hops,
    fit_host,
    fit_intake,
    fit_tokenizer,
)
from atom.compass.fit import publish
from atom.compass.parity.client_run import RAW
from atom.compass.run import INTAKE_RECORD, LOCAL, TOKENIZER_RECORD
from atom.compass.validate.runs import SIDES

PROG = "compass fit host"


def host_laws(samples, records=(), client_runs=()) -> tuple[dict, dict]:
    """The host laws over `samples`, with the tokenizer, HTTP intake and hop
    terms of the record directories `records`, each served the client runs
    `client_runs` pairs it with by their exports, as a run file's `host` entry,
    and their report; samples with no host labels under a host key refuse."""
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
    host, report = fit_host(samples)
    stamped = [r for r in records if (Path(r) / "steps" / INTAKE_RECORD).is_file()]
    steps = [Path(r) / "steps" / INTAKE_RECORD for r in stamped]
    calls = [d.with_name(TOKENIZER_RECORD) for d in steps]
    clients = [
        _client(r, [e for d, e in client_runs if Path(d) == Path(r)]) for r in stamped
    ]
    for term, fit in (
        ("tokenizer", lambda: fit_tokenizer(calls, steps)),
        ("http_intake", lambda: fit_intake(steps)),
        ("hops", lambda: fit_hops(steps, clients)),
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


def main(argv: list[str]) -> None:
    args = publish.parser(PROG, __doc__.split("\n\n")[0]).parse_args(argv)
    samples, config = publish.inputs(args, ("record",))
    host, report = host_laws(samples, args.inputs, args.client_run)
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

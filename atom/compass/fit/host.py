# SPDX-License-Identifier: MIT
"""`compass fit host`: the host laws, fitted on real runs' records and published.

    compass fit host RECORD... --client-run RECORD EXPORT... --machine M
        --store ROOT --out DIR

`host_laws` fits the samples' host labels by `step_linear_fit.fit_host`, under
the host key the real runner writes beside each record, and its report names
each run with every trace its steps served. From the tokenizer calls and HTTP
intake stamps the recorded server wrote beside its steps
(`atom.compass.run.TOKENIZER_RECORD`, `INTAKE_RECORD`) it fits the entry's
``tokenizer`` term (`step_linear_fit.fit_tokenizer`) and ``http_intake`` term
(`step_linear_fit.fit_intake`); a term the records cannot give is left out
and its reason listed under the report's ``unfitted``. The host key gains the
configuration's `machine_spec`, which a run's spec must match; the entry is
keyed by that host key's values and holds the run file's `host` entry as its
params.
"""

import json
from pathlib import Path

from atom.compass.artifacts import Key, Kind
from atom.compass.backends.step_linear_fit import fit_host, fit_intake, fit_tokenizer
from atom.compass.fit import publish
from atom.compass.run import INTAKE_RECORD, TOKENIZER_RECORD

PROG = "compass fit host"


def host_laws(samples, records=()) -> tuple[dict, dict]:
    """The host laws over `samples`, with the tokenizer and HTTP intake terms
    of the record directories `records`, as a run file's `host` entry, and
    their report; samples with no host labels under a host key refuse."""
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
    host, report = fit_host(samples)
    for term, name, fit in (
        ("tokenizer", TOKENIZER_RECORD, fit_tokenizer),
        ("http_intake", INTAKE_RECORD, fit_intake),
    ):
        paths = [p for r in records if (p := Path(r) / "steps" / name).is_file()]
        try:
            if not paths:
                raise ValueError(f"no record holds steps/{name}")
            host[term], report[term] = fit(paths)
        except (ValueError, QhullError) as e:
            report["unfitted"][term] = str(e).splitlines()[0]
    return host, {"provenance": publish.runs(samples), **report}


def main(argv: list[str]) -> None:
    args = publish.parser(PROG, __doc__.split("\n\n")[0]).parse_args(argv)
    samples, config = publish.inputs(args, ("record",))
    host, report = host_laws(samples, args.inputs)
    k = host["key"]
    k["machine_spec"] = config["machine_spec"]
    key = Key.of(
        Kind.HOST_LAWS,
        host_cpu=k["host_cpu"],
        atom_src=k["atom"],
        model=k["model"],
        engine_config=json.dumps(k["engine"], sort_keys=True),
        machine_spec=k["machine_spec"],
    )
    command = " ".join([PROG, *argv])
    publish.publish(args, command, key, publish.conditions(config), host, report)

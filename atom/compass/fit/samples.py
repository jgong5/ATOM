# SPDX-License-Identifier: MIT
"""Labelled samples from `record` directories and step measurements, under one
configuration.

A record directory is `record`'s output: `run.json`, holding the run's
identity in format `FORMAT_VERSION`, and the step record in `steps/`, with the
client runs its server served (`parity.served`). Its ranks are labelled
together by `step_labels.label_run` under ATOM's two-batch-overlap split rule
(`validate.bound._split`), and a sample's provenance gains the
`traces` its step served and whether run.json says it was `profiled`.
Under `floored`, a trace run's drafter labels lose the profiler's floor
(`parity.busy.floored`). A DP rank no request reached writes a line per dummy forward it runs, and gives
no sample; its file still counts in the run's digest. An empty
rank file lost its lines, and refuses.
A record whose run.json names `aiter_jit_builds` or `ptl_changes` refuses,
naming them, and so does one whose `source` counts uncommitted files.
With `GRADED_ONLY_ENV` set to ``1``, a record's samples are its graded steps
(`graded_steps`): those within some client run's graded span, from the first
step in which a rank holds one of the requests it names as sent (its export's
``requests``, the warmup's left out) to the last, every rank's other steps
dropped after labelling, which still reads them. That drops the warmup before
the first span and the tail after the last; and, on a record serving several
client runs one after another, the steps between two spans, the drain of one
client run and the warmup of the next, which a cut at the first graded step
kept. A step inside a span stays whatever it holds: warmup and sent requests
together, or only a request its export does not name (one the client cancelled
when its run ended, in flight since). For a fit on the run it is compared
against, where the warmup's transients (a page cache filling, the first touch
of kernels, prefill on an empty KV cache) are not the graded window's: on
#3720's E3 record, one server serving six client runs, the steps between the
spans held 69% of the prefill device seconds that cut kept, and ran about 5%
faster than graded steps of the same features. With `GRADED_SECONDS_ENV`
also set to a number of seconds, they end that long after the first graded step
entered (`graded_to`): a calibration timed on a shorter window, fitted from
the same record. A sample's `dummy_host`, its rank's dummy forwards at a step
its group prefills, is cut to the same steps. Its host laws' tokenizer, intake
and hop terms still read the record's whole stamp files.

A record's client run named as a calibration ladder (`load`'s `ladders`,
`compass fit steps --ladder-run`) is served and graded as any other; each
sample of a step within its span (`client_spans`) holds as its `ladder` the
decode rungs the ladder calibrates, which the step linear fit reads
(`step_linear_fit.ladder_cut`). Every other sample's `ladder` is None.

A step measurement is a `step_measurement` entry `measure steps` published,
read by key; its member `MEMBER` holds the batch view as a step-record line
does (`rows`, `rung`, `produces_output`, and `group_needs_eager` where the
bench ran it eagerly), its device seconds (`seconds`), under a drafter the
target's, and the drafter's (`draft_seconds`), its `origin` (`replay`,
`plan` or `ragged`) and the `traces` the batch came from.
Its sample's provenance holds its `source`, the origin, its `run`, the entry's
digest, those `traces`, its `inputs`, the inputs `measure steps` read it from
(`sources`, a record named as a record's `run` is), and `profiled` false: the
bench times a replay unprofiled.

Every input states one configuration. A record reads it from its `run.json`,
its `dp0.engine.json` and the requests it served: `model`, the served
config.json's digest; `device`, each GPU's `<name> <arch>`, as a spec states
it; `engine_config`, `keyed`'s fields, the parallel layout among them, over the
level `step_linear_fit.recorded_level` reads, as is each sample's host key's;
`prefill_buckets`, those `step_linear_fit.stated_buckets` reads, null where it
states none; `source_root`, its host key's ``atom`` as `source_root` writes
it; `sampling`, the request parameters, null for a record serving no client
run's request; its host key
less its model and engine; its `stack`, the software versions run.json
names; and its `ptl`, the cards' PTL state, `unrecorded` for a run.json that
names none. A measurement reads the same fields but the host key, stack and
prefill buckets from its key, whose `engine_config` and `sampling` are JSON
objects as text. A field that differs between inputs refuses, naming each
value and the inputs holding it; so do a profiled record and a record whose
requests carry two sets of request parameters. Ladders that are non-empty
prefixes of the longest among the inputs read as it (`step_grid_plan.pooled`).

A stated engine file (`--engine`) stands in for each record's own as
`step_linear_fit --engine` takes it, in `engine_config`, the host key's
``engine`` (its graph sets kept) and a record's prefill buckets stating none.
"""

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from atom.compass.artifacts import ArtifactStore, Topology
from atom.compass.backends.launch_features import GRAPHS
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_grid_plan import PREFILL_BUCKETS, pooled
from atom.compass.backends.step_labels import Sample, digest, label_run
from atom.compass.backends.step_linear_fit import (
    recorded_buckets,
    recorded_level,
    refuse_contradicting,
    stated_buckets,
)
from atom.compass.drive import aiter_jit
from atom.compass.drive.identity import PTL_CHANGES
from atom.compass.parity import ARRIVALS
from atom.compass.parity.served import served
from atom.compass.run import source_root
from atom.compass.run.step_grid import keyed_of
from atom.compass.validate.bound import _split

#: The run.json format this reads.
FORMAT_VERSION = 2
#: The variable that, set to ``1``, keeps a record's samples from its graded
#: window on (module docstring); unset, every labelled step is a sample.
GRADED_ONLY_ENV = "COMPASS_FIT_GRADED_ONLY"
#: The variable that, with `GRADED_ONLY_ENV`, keeps only that many seconds of
#: a record's graded window, from its first step's entry (module docstring).
GRADED_SECONDS_ENV = "COMPASS_FIT_GRADED_SECONDS"
#: A step measurement's one member: the bench replies for the whole TP group.
MEMBER = ("measurement", Topology().ranks()[0], "json")
#: What names a GPU's model in run.json's `device`.
CARD = ("name", "arch")
#: The PTL state of a run.json that records none, which no machine spec states.
UNRECORDED = "unrecorded"


def load(
    records: Mapping,
    measurements: Sequence = (),
    store=None,
    floored=None,
    engine=None,
    ladders: Mapping | None = None,
) -> tuple[list[Sample], dict]:
    """The samples of `records`, {record directory: the client runs its server
    served}, labelled under `floored`, and of the `measurements` keys in the
    store at `store`, in run digest, DP rank and step order; and the
    configuration they state, the records' under the engine file `engine`, and
    each engine config at the longest ladder among them (`pooled`). `ladders`
    maps a record directory to {index of a client run in its list: the decode
    rungs that client run, a calibration ladder, calibrates}."""
    samples, configs, held = [], {}, {}
    if engine is not None:
        owns = {d: Path(d) / "steps" / "dp0.engine.json" for d in records}
        held = pooled({d: keyed_of(json.loads(o.read_text())) for d, o in owns.items()})
    for directory, client_runs in records.items():
        got, configs[str(directory)] = _record(
            Path(directory),
            client_runs,
            floored,
            engine,
            held.get(directory),
            (ladders or {}).get(directory),
        )
        samples += got
    for key in measurements:
        got, configs[str(key)] = _measurement(ArtifactStore(store), key)
        samples.append(got)
    for i, at in pooled({i: c["engine_config"] for i, c in configs.items()}).items():
        configs[i]["engine_config"] = at
    fields = sorted({f for c in configs.values() for f in c})
    differ = [
        line
        for f in fields
        for line in _differ(f, {i: c[f] for i, c in configs.items() if f in c})
    ]
    if differ:
        raise ValueError("the inputs are not one configuration:\n" + "\n".join(differ))
    one = {k: v for c in configs.values() for k, v in c.items()}
    samples.sort(key=lambda s: (s.provenance["run"], s.rank, s.step))
    return samples, one


def _record(
    directory: Path, client_runs, floored=None, stated=None, held=None, ladder=None
) -> tuple[list[Sample], dict]:
    run = json.loads((directory / "run.json").read_text())
    if run["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{directory}/run.json is format {run['format_version']}, and this "
            f"reads format {FORMAT_VERSION}"
        )
    if run["profiled"]:
        raise ValueError(
            f"{directory} was profiled, and profiling splits the steps it times, "
            "so no fit takes them"
        )
    if built := aiter_jit.refusal(run):
        raise ValueError(f"{directory} is not fitted: {built}")
    if changes := run.get(PTL_CHANGES):
        named = ", ".join(f"{c['from']} to {c['to']} at {c['at']}" for c in changes)
        raise ValueError(
            f"{directory} is not fitted: its cards' PTL state changed while it "
            f"served, {named}"
        )
    fields = run_fields(run, directory)
    steps = directory / "steps"
    if (steps / ARRIVALS).exists():
        # `parity.read` then drops the engine's start and renumbers the rest.
        raise ValueError(
            f"{steps} holds {ARRIVALS}, so its steps are not numbered as its lines"
        )
    record = served(steps, client_runs)
    spans = client_spans(record)
    # Each step within a calibration ladder's span, to the rungs it calibrates.
    laddered = {
        n: tuple(sorted(rungs))
        for k, rungs in (ladder or {}).items()
        for n in range(spans[k][0], spans[k][1] + 1)
    }
    graded = graded_to = None
    seconds = os.environ.get(GRADED_SECONDS_ENV)
    if seconds and os.environ.get(GRADED_ONLY_ENV) != "1":
        raise ValueError(
            f"{GRADED_SECONDS_ENV} cuts the graded window, which {GRADED_ONLY_ENV}=1 "
            "selects; set both"
        )
    if os.environ.get(GRADED_ONLY_ENV) == "1":
        graded = graded_steps(record)
        if not graded:
            raise ValueError(
                f"{GRADED_ONLY_ENV}=1, and no step of {directory} holds a request "
                "its client runs name as sent"
            )
        if seconds:
            graded_to = graded_to_step(record, min(graded), float(seconds))
    sampling = sorted({p for s in _steps(record) for p in s["sampling"]})
    if len(sampling) > 1:
        raise ValueError(
            f"{directory} served requests under {len(sampling)} sets of request "
            f"parameters: {' and '.join(sampling)}"
        )
    traces = {(s["dp_rank"], s["step"]): s["traces"] for s in _steps(record)}
    own, buckets = steps / "dp0.engine.json", stated_buckets(steps / "dp0.jsonl")
    level = recorded_level(steps / "dp0.jsonl")
    engine = level | keyed_of(json.loads(own.read_text()))
    if stated is not None:
        engine, buckets = _stated(own, level | held, buckets, stated)
    files = sorted(steps.glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:]))
    name, samples = ", ".join(map(digest, files)), []
    for got, _ in label_run(files, name, _split, floored):
        dummies = None
        for s in got:
            if graded is not None and s.step not in graded:
                continue
            if graded_to is not None and s.step >= graded_to:
                continue
            if dummies is None:  # the rank's dummy forwards, cut alike
                dummies = tuple(
                    d
                    for d in s.dummy_host
                    if (graded is None or d[0] in graded)
                    and (graded_to is None or d[0] < graded_to)
                )
            at = traces[s.rank, s.step]
            traced = {**s.provenance, "traces": at, "profiled": run["profiled"]}
            if stated is not None and (key := traced["config"]) is not None:
                graphs = {k: v for k, v in key["engine"].items() if k == GRAPHS}
                traced["config"] = {**key, "engine": engine | graphs}
            elif level and (key := traced["config"]) is not None:
                traced["config"] = {**key, "engine": level | key["engine"]}
            samples.append(
                dataclasses.replace(
                    s,
                    provenance=traced,
                    dummy_host=dummies,
                    ladder=laddered.get(s.step),
                )
            )
    host = run["host_key"]
    return samples, {
        **fields,
        "engine_config": engine,
        PREFILL_BUCKETS: buckets,
        "stack": run["stack"],
        "ptl": run.get("ptl", UNRECORDED),
        "sampling": json.loads(sampling[0]) if sampling else None,
        "host_key": {k: v for k, v in host.items() if k not in ("model", "engine")},
    }


def _stated(own: Path, held: dict, buckets, stated) -> tuple[dict, list | None]:
    """The `keyed` fields of the engine file `stated` and the prefill buckets a
    record with engine file `own`, `held` its keyed fields at the records'
    longest ladder, states, else the file's. A field the file omits or states
    otherwise, or other buckets, refuse."""
    raw = json.loads(Path(stated).read_text())
    engine = keyed_of(raw)
    refuse_contradicting(own, held, stated, engine)
    held_buckets = {} if buckets is None else {str(own.parent): buckets}
    stating = recorded_buckets({str(stated): raw.get(PREFILL_BUCKETS)} | held_buckets)
    return engine, None if stating is None else list(stating)


def run_fields(run: dict, record="run.json") -> dict:
    """The `model`, `device` and `source_root` a run.json's identity states.
    A run on a tree with uncommitted files refuses, naming `record`: its
    provenance names the tree's HEAD, which is not the code that ran."""
    if dirty := run["source"].get("dirty"):
        raise ValueError(
            f"{record} ran with {dirty} uncommitted files in its source tree; "
            "commit them and run again"
        )
    return {
        "model": run["host_key"]["model"],
        "device": ", ".join(
            sorted({" ".join(str(gpu[f]) for f in CARD) for gpu in run["device"]})
        ),
        "source_root": source_root(run["host_key"]["atom"]),
    }


def _steps(record: dict) -> list[dict]:
    return [s for steps in record.values() for s in steps]


def client_spans(record: dict) -> dict[int, tuple[int, int]]:
    """Each client run's graded span in `record`, as `parity.served` splits it,
    by its index: the first and last step in which some DP rank holds one of
    its sent requests."""
    spans: dict[int, tuple[int, int]] = {}
    for s in _steps(record):
        if (k := s["client_run"]) is not None:
            first, last = spans.get(k, (s["step"], s["step"]))
            spans[k] = (min(first, s["step"]), max(last, s["step"]))
    return spans


def graded_steps(record: dict) -> frozenset[int]:
    """The steps of `record`, as `parity.served` splits it, within some client
    run's graded span (`client_spans`), whatever those in between hold."""
    spans = client_spans(record).values()
    return frozenset(
        s["step"]
        for s in _steps(record)
        if any(first <= s["step"] <= last for first, last in spans)
    )


def graded_to_step(record: dict, start: int, seconds: float) -> int | None:
    """The first step that entered at least `seconds` after step `start` did,
    each step's entry the earliest of its DP ranks' (they step together); None
    when the record ends sooner. A step with no entry stamp is skipped."""
    enter: dict[int, int] = {}
    for s in _steps(record):
        if "t_enter_ns" in s:
            enter[s["step"]] = min(
                enter.get(s["step"], s["t_enter_ns"]), s["t_enter_ns"]
            )
    if start not in enter:
        raise ValueError(
            f"step {start}, where the graded window starts, has no entry stamp"
        )
    cut = enter[start] + seconds * 1e9
    later = [k for k, t in enter.items() if t >= cut]
    return min(later) if later else None


def _measurement(store: ArtifactStore, key) -> tuple[Sample, dict]:
    entry = store.read(key)
    try:
        m = json.loads(entry.read_member(*MEMBER))
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in m["rows"]),
            capture_rung=m["rung"],
            produces_output=m["produces_output"],
            group_needs_eager=m.get("group_needs_eager", False),
        )
        provenance = {
            "source": m["origin"],
            "run": entry.digest,
            "requests": [],
            "config": None,
            "traces": sorted(m["traces"]),
            "inputs": sorted(m.get("sources", ())),
            "profiled": False,
        }
        values = key.values
        config = {
            **{f: values[f] for f in ("model", "device", "source_root")},
            **{f: json.loads(values[f]) for f in ("engine_config", "sampling")},
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{key} does not load: {type(error).__name__}: {error}"
        ) from error
    draft = m.get("draft_seconds")
    sample = Sample(0, 0, view, m["seconds"], None, {}, False, provenance, draft)
    return sample, config


def _differ(field: str, held: dict) -> list[str]:
    """Where `held`, {input: value}, disagrees on `field`; a mapping field by field."""
    if all(isinstance(v, Mapping) for v in held.values()):
        names = sorted({k for v in held.values() for k in v})
        return [
            line
            for k in names
            for line in _differ(f"{field} {k}", {i: v.get(k) for i, v in held.items()})
        ]
    by: dict[str, list[str]] = {}
    for i, v in held.items():
        by.setdefault(json.dumps(v, sort_keys=True), []).append(i)
    if len(by) < 2:
        return []
    return [f"{field}: " + "; ".join(f"{v} in {', '.join(i)}" for v, i in by.items())]

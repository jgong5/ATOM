# SPDX-License-Identifier: MIT
"""`compass run --check`: every input a run lacks, listed before any server.

`lacking` resolves the run's inputs as `runfile.build`, the runner and the
server do, calling per input the function whose refusal the run would meet,
so one lacking input hides no other: the platform, its datasheet and readings
(`runfile.spec`); a step stub's file, read as `runfile.build` reads it, and its
`Coefficients` as the runner builds them; an ``lmcache_offload`` entry's file,
read as `runfile.build` reads it, or its absence, held to this deployment as
the engine holds it (`run.refuse_offload_tier`, `lmcache_offload.tier_config`);
each store key or law file a flag
names (`runfile.resolve`); a step linear law and host laws held against this
run's engine config, widths, platform and host key
(`run.pick_fitted`, `run.refuse_unfitted_law`, `run.refuse_unfitted_host`),
and to the prefill token-graph buckets it records (`run.refuse_other_buckets`),
and the step linear law's decode vectors against the rungs this configuration
schedules (`uncovered_rungs`); the host laws' tokenizer term against the
backend of the tokenizer ATOM's server loads (`run.tokenizer_term`); a step
grid bundle's load; the configuration's memory readings; an op graph's price
list; and an `--acceptance-record`. An input is missing when nothing resolves
for it, and stale when the run refuses the entry it holds, naming each artifact
axis that moved, or refuses steps it would price, as a step linear law's
coverage is where a rung the run schedules has no vector. A memory readings
entry holds every rank's member, or the store refuses it, so rank 0's load
stands for each rank's. Each is held to ATOM's config as a server resolves it
on the machine spec's device arch, whatever device this host has
(`run._config`).

Each lacking input is printed with the command that makes it. A step linear
law's is the `calibrate.sh` command that records and fits one for this
configuration (`calibration`), into `--out`, or `cal` in the current directory
without one, naming each client variable it requires that this environment
does not set, with its expected time, unknown, as no earlier calibration's
status is read for it; or, where none serves, why, as for a server that
decodes above one pipeline stage, whose stages' decode steps the fit cuts.
Memory readings whose key holds a stale entry are measured into a new store
beside that directory (`readings_store`), as the store refuses a second
publish at the key.

A prefill or decode instance's run reads a network file across nodes, as
`deploy.sh run` hands it one, and its readings command takes the ATOM
arguments a real start of it takes, where they are not the run's, and the
environment the run's check set. Its step linear law and host laws are made
by `deploy.sh record`, which records both sides together, and a fit of that
side's records (`SIDE_LAW`): this check knows one side's arguments, and the
`calibrate.sh` command that takes both sides' (`deployment_calibration`) is
`compass advice`'s, which knows both.

Printed apart, outside the exit status: the gaps `--coverage`'s check counts;
an op graph's absent price list, which prices every leaf by its analytical
law, unless `--coverage` asks for its check, which then refuses; and the
modules the step roofline leaves unclassified on the model built on the meta
device, when it prices the steps the backend refuses; and why the prefill
token-graph buckets cannot be worked out, so no law is held to them. No
`compass run` backend is the step roofline itself.
"""

import functools
import json
import multiprocessing
import os
import re
import shlex
import tempfile
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Rule,
    machine_platform,
)
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.backends.step_linear_plan import (
    TRACE_RUNS,
    records,
    scheduled_rungs,
)
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile, serve
from atom.compass.kv.lmcache_offload import ENTRY as OFFLOAD
from atom.compass.kv.lmcache_offload import tier_config
from atom.compass.memory.readings import PAGE_BOUND
from atom.compass.run import OP_GRAPH, PRICE_LIST, STEP_GRID, STEP_LINEAR, STEP_STUB
from atom.compass.runner.overrides import schedulable_bound, schedulable_ladder
from atom.compass.spec import SpecRefusal

#: Each store input by its run-file key.
NAMES = {
    STEP_LINEAR: "step linear law",
    STEP_GRID: "step grid bundle",
    "host": "host laws",
    PRICE_LIST: "price list",
}
#: The commands that make each input.
MADE_BY = {
    "platform": "compass measure device on each node, then compass fit device",
    STEP_STUB: "a JSON file of the stated fields of backends.step_stub.Coefficients",
    OFFLOAD: "python -m atom.compass.backends.step_linear_fit offload LOG... --out "
    "ENTRY, on real runs' server logs at OFFLOAD_PROFILE=1",
    STEP_GRID: "compass trace steps, compass measure steps --grid, then compass "
    "fit steps --backend step-grid",
    "host": "compass record, then compass fit host",
    "memory readings": "compass measure memory --gpu G --out D "
    "--store {store}{beside} -- {atom}",
    PRICE_LIST: "compass trace ops --population, compass measure ops --plan, then "
    "compass fit ops",
    "acceptance record": "compass record of a speculative run",
    "coverage input": "compass record or compass trace steps under these ATOM "
    "arguments",
    "coverage": "compass measure steps --grid, or compass measure ops --plan",
}
#: What `measuring` gives once the new store holds the readings at their key,
#: where a second measure would be refused.
MEASURED = (
    "none: they are at their key in {}, which a run reads once it holds the "
    "run's other inputs too, its platform readings among them"
)
#: The artifact axis each cell of an invalidation refusal names.
AXIS = re.compile(r"` x `([^`]+)`")
#: The script that takes a step linear law's records and fits it.
CALIBRATE = Path(__file__).resolve().parents[3] / "scripts/compass/calibrate.sh"
#: Its directory when `--out` names none, in the current one.
CALIBRATION_OUT = "cal"
#: The variables `calibrate.sh` requires for its clients, which no configuration
#: states: its command states each this process's environment sets, and names
#: each it does not.
CLIENT_ENV = ("TRACES", "AIPERF_PYTHON")
#: The command that makes a prefill or decode instance's law, `fit` its
#: `compass fit` command: each `deploy.sh record` deployment writes the side's
#: record in OUT/<side>.
SIDE_LAW = (
    "deploy.sh OUT record of the deployment, then compass fit {fit} on each OUT/{side}"
)
#: Each law a side fits from its own records, by its `compass fit` command.
FITS = {STEP_LINEAR: "steps --backend step-linear", "host": "host"}
#: Each side by the letter `runfile.pd_side` gives it.
SIDES = {"P": "prefill", "D": "decode"}
#: How long the child reading ATOM's attention builder may take, in seconds, far
#: past the few it takes on a CPU host; past it, the child is ended.
PROBE_SECONDS = 300


def report(args, atom_args, real=None, env=None) -> int:
    """Print each input `lacking` finds, with the run's refusal and the command
    that makes it, then what it prints apart; 1 when one is lacking, else 0."""
    lacks, apart = lacking(args, atom_args, real, env)
    for name, state, refusal, made_by in lacks:
        print(f"lacking {name}: {state}\n  refused: {refusal}\n  made by: {made_by}")
    if not lacks:
        print("nothing lacking: every input the run reads resolves and holds")
    for line in apart:
        print(f"apart: {line}")
    return 1 if lacks else 0


def lacking(args, atom_args, real=None, env=None) -> tuple[list, list]:
    """Each input the run `args`, parsed by `run`, lacks under ATOM's arguments
    `atom_args`, as ``(name, state, refusal, made by)``; and the lines printed
    apart. The readings command takes `real`, a real start's ATOM arguments,
    and runs under `env`, the variables the check set, where given."""
    from atom.compass.drive import run as drive_run
    from atom.compass.run import memory, step_grid
    from atom.model_engine.llm_engine import load_tokenizer

    lacks, apart = [], []

    def met(name, made_by, check, *given, state=None):
        """`check(*given)`'s value, True for None, or False when it refuses,
        its refusal recorded as `name` lacking, made by `made_by`, or what it
        gives when it is a callable."""
        try:
            return check(*given) or True
        except (ArtifactRefusal, SpecRefusal, ValueError) as refused:
            made_by = made_by() if callable(made_by) else made_by
            lacks.append((name, state or _state(refused), str(refused), made_by))
            return False

    store = ArtifactStore(Path(args.store).resolve())
    side = runfile.pd_side(atom_args)
    across = bool(args.network) and side is not None
    try:
        machine = runfile.spec(args, store, across)
    except (OSError, ArtifactRefusal, SpecRefusal, ValueError) as refused:
        return [("platform", "missing", str(refused), MADE_BY["platform"])], []
    arch, engine = machine.value("device.arch"), runfile._engine_args(atom_args)
    config = drive_run._config(atom_args, arch)
    run = {"machine": machine.echo(), "store": str(store.root)}
    flags = {k: getattr(args, k) for k in runfile.KINDS if getattr(args, k)}
    made_by, out = dict(MADE_BY), args.out or CALIBRATION_OUT
    if side:
        made_by |= {
            k: SIDE_LAW.format(fit=f, side=SIDES[side]) for k, f in FITS.items()
        }
    elif STEP_LINEAR in flags:
        made_by[STEP_LINEAR] = calibration(out, atom_args, config)
    if stub := getattr(args, STEP_STUB):
        try:
            Coefficients(**json.loads(Path(stub).read_text()))
        except (OSError, TypeError, ValueError) as unread:
            lacks.append(
                (f"step stub {stub}", "missing", str(unread), made_by[STEP_STUB])
            )
    entry, path = {}, args.lmcache_offload
    try:
        if path:
            entry[OFFLOAD] = json.loads(Path(path).read_text())
        compass_run.refuse_offload_tier(config, entry)
        if OFFLOAD in entry:
            tier_config(entry[OFFLOAD], config)
    except (OSError, TypeError, ValueError) as refused:
        name = f"{OFFLOAD} entry" + (f" {path}" if path else "")
        state = "stale" if entry else "missing"
        lacks.append((name, state, str(refused), made_by[OFFLOAD]))
    for name, tokens in (flags | {"host": args.host_laws or []}).items():
        tokens, held = [tokens] if isinstance(tokens, str) else tokens, []
        for token in tokens:
            made = made_by[name]
            given = store, name, token, machine, atom_args
            if found := met(f"{NAMES[name]} {token}", made, runfile.resolve, *given):
                held.append(found[0])
        if held and len(held) == len(tokens):
            run[name] = held if len(held) > 1 else held[0]
    law, host = NAMES[STEP_LINEAR], NAMES["host"]
    try:
        buckets = {STEP_LINEAR, "host"} & run.keys() and recorded_buckets(config, arch)
    except ValueError as unworked:
        buckets = ()
        apart.append(f"{unworked}, so no law is held to them")
    if STEP_LINEAR in run and met(
        law, made_by[STEP_LINEAR], compass_run.pick_fitted, run, STEP_LINEAR, config
    ):
        made = made_by[STEP_LINEAR]
        if met(law, made, compass_run.refuse_unfitted_law, run, config):
            given = config, buckets, {STEP_LINEAR: run[STEP_LINEAR]}
            met(law, made, compass_run.refuse_other_buckets, *given)
        if rungs := uncovered_rungs(run[STEP_LINEAR], config):
            whose = (
                " in the law or its draft law" if "draft" in run[STEP_LINEAR] else ""
            )
            refusal = (
                f"no decode vector{whose} for rung{'s' * (len(rungs) > 1)} "
                f"{', '.join(map(str, rungs))}, which this configuration "
                "schedules; the law refuses a decode step at each"
            )
            lacks.append((law, "stale (coverage)", refusal, made_by[STEP_LINEAR]))
    if "host" in run and met(
        host, made_by["host"], compass_run.pick_fitted, run, "host", config
    ):
        given = run["host"], config, machine_platform(machine)
        met(host, made_by["host"], compass_run.refuse_unfitted_host, *given)
        given = config, buckets, {"host": run["host"]}
        met(host, made_by["host"], compass_run.refuse_other_buckets, *given)
        tok = load_tokenizer(engine.model, engine.trust_remote_code)
        met(host, made_by["host"], compass_run.tokenizer_term, run, tok, config)
    grid, made = NAMES[STEP_GRID], MADE_BY[STEP_GRID]
    checked = STEP_GRID in run and (
        met(grid, made, compass_run.refuse_context_parallel, STEP_GRID, config)
        and met(grid, made, step_grid.load, run[STEP_GRID], config, machine)
    )

    def made() -> str:
        """The readings command, built only once they lack: into the run's
        store, else, where an entry there holds their key, the new store."""
        into, key = args.store, memory.key_of(config, machine)
        if holds(into, key):
            into = readings_store(Path(out).resolve(), compass_run.atom_digest())
        return measuring(into, key, real or atom_args, env)

    carried = runfile.carried(runfile.priced_by(args))
    given = run["store"], config, machine, 0, carried
    read = met("memory readings", made, memory.load, *given)
    if read and (why := stale_page_bound(read, config)):
        lacks.append(("memory readings", "stale", why, made()))

    if getattr(args, OP_GRAPH):
        run[OP_GRAPH], key = True, runfile._price_list_key(atom_args, machine)
        listed, made = NAMES[PRICE_LIST], MADE_BY[PRICE_LIST]
        if store.directory_for(key).exists():
            run[PRICE_LIST] = key.values
            checked = met(listed, made, compass_run._price_list, run, config, machine)
        elif args.coverage:
            given = args.coverage, run, atom_args
            met(listed, made, drive_run.check, *given, state="missing")
        else:
            apart.append(
                f"price list: none at {key}, so the op graph prices every leaf by "
                f"its analytical law; made by {made}"
            )
    if record := args.acceptance_record:
        steps = Path(record) / serve.STEPS
        drawn = any(steps.glob(compass_run.ACCEPTANCE_FILE.format("*")))
        given = record, atom_args, runfile._dp_width(atom_args)
        made, state = MADE_BY["acceptance record"], None if drawn else "missing"
        met("acceptance record", made, drive_run.measured, *given, state=state)
    if args.coverage and checked:
        made, given = MADE_BY["coverage input"], (args.coverage, run, atom_args)
        if found := met("coverage input", made, drive_run.check, *given):
            apart.append(_coverage(found))
    if args.step_roofline or STEP_GRID in flags:
        apart += _unclassified(atom_args, arch)
    return lacks, apart


def measure_memory(store, atom_args, env=None, beside=None) -> str:
    """The command that measures the memory readings of a start under ATOM's
    arguments `atom_args` into `store`, run under the variables `env`, with
    the command `beside`, a list of words, run beside the engine when given."""
    beside = f" --beside {shlex.quote(shlex.join(beside))}" if beside else ""
    atom = shlex.join(atom_args)
    made = MADE_BY["memory readings"].format(store=store, beside=beside, atom=atom)
    return " ".join([*(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items()), made])


def measuring(store, key, atom_args, env=None, beside=None) -> str:
    """`measure_memory`'s command for the readings at `key` into `store`, or,
    once an entry there holds their key, which the store refuses a second
    publish at, `MEASURED`."""
    if holds(store, key):
        return MEASURED.format(store)
    return measure_memory(store, atom_args, env, beside)


def readings_store(beside, digest: str) -> Path:
    """The new store readings are measured into where an entry holds their key
    in the store a run reads, as one an older tree measured still does:
    ``<beside>@<d>/store``, `d` the first 12 hex digits of this tree's ATOM
    digest `digest`, which no older tree's entry is in."""
    return Path(f"{beside}@{digest[:12]}") / "store"


def holds(store, key) -> bool:
    """Whether an entry is at `key` in `store`, valid or not."""
    return (ArtifactStore(store).directory_for(key) / ENTRY_FILE).is_file()


def stale_page_bound(read, config) -> str | None:
    """Why the memory readings `read` are stale for a start of `config`: it
    reads the PAGE bound (`_reads_page_bound`), and they state none; else
    None."""
    if PAGE_BOUND in read.values or not _reads_page_bound(config):
        return None
    return f"{read.source} states no {PAGE_BOUND}, which an fp4 V4.1 start reads"


def _reads_page_bound(config) -> bool:
    """Whether a start of `config` sizes by what `memory_allocated` answers on
    its device: DeepSeek-V4.1's builder under --index-cache-dtype fp4."""
    from atom.utils import get_hf_text_config
    from atom.utils.selector import Family, attn_family

    if getattr(config, "index_cache_dtype", None) != "fp4":
        return False
    return attn_family(get_hf_text_config(config.hf_config)) is Family.CSA2


def recorded_buckets(config, arch: str) -> tuple[int, ...]:
    """The prefill token-graph buckets a run of `config` on a device of `arch`
    records at capture (`state_prefill_buckets`): `ATOM_TOKEN_GRAPH_SIZES` up to
    `ModelRunner.warmup_max_tokens` where ATOM's attention builder for its model
    records token graphs; none if eager or under RapidServe, whose prefill is eager.
    Where the builder cannot be read (`_token_graphs`), a value error says why."""
    from atom.utils import envs, get_hf_text_config
    from atom.utils.selector import attn_family

    if config.enforce_eager or getattr(config, "enable_rapidserve", False):
        return ()
    dp = 1 if config.enable_dp_attention else config.parallel_config.data_parallel_size
    limit = config.max_num_batched_tokens // dp // config.prefill_context_parallel_size
    sizes = sorted({n for n in envs.ATOM_TOKEN_GRAPH_SIZES if 0 < n <= limit})
    family = attn_family(get_hf_text_config(config.hf_config))
    records = bool(sizes) and _token_graphs(family, arch)
    if isinstance(records, str):
        raise ValueError(records)  # noqa: TRY004 - why, not a wrong type
    return tuple(sizes) if records else ()


@functools.cache
def _token_graphs(family, arch: str) -> bool | str:
    """Whether ATOM's attention builder for `family` records token graphs, in a
    process declaring a device of `arch`: its module imports aiter, which needs one.
    Compass's own defect there raises as a runtime error (`check._resolving`).
    Anything else it raises, the child dying, or the child outliving
    `PROBE_SECONDS`, which ends it, gives why the buckets cannot be worked out,
    cached as an answer is, so a child that hangs is waited for once."""
    from concurrent.futures.process import BrokenProcessPool, ProcessPoolExecutor

    from atom.compass.advice.check import _resolving

    of, spawn = "attention builder", multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn) as child:
        try:
            read = child.submit(_resolving, _builder_records, family, arch, of)
            records, raised, said = read.result(PROBE_SECONDS)
        except BrokenProcessPool as died:
            raised, said = "refused", f"the child reading ATOM's {of} died: {died}"
        except TimeoutError:
            for process in list(child._processes.values()):  # 3.14: kill_workers
                process.kill()
            raised = "refused"
            said = f"the child reading ATOM's {of} outlived {PROBE_SECONDS} s"
    if raised == "defect":
        raise RuntimeError(f"Compass's own code raised reading ATOM's {of}:\n{said}")
    if raised:
        why = "the prefill token-graph buckets this run records cannot be worked out"
        return f"{why}: {said}"
    return records


def _builder_records(family, arch: str) -> bool:
    from atom.compass.capture import driverless
    from atom.utils.selector import get_attn_backend

    with tempfile.TemporaryDirectory(prefix="compass-check-") as tmpdir:
        driverless.install(tmpdir, arch)
        return get_attn_backend(family).get_builder_cls().supports_token_graph


def _state(refused) -> str:
    """``missing`` when nothing resolves for an input, else ``stale``, with each
    artifact axis that moved."""
    if isinstance(refused, SpecRefusal) or getattr(refused, "rule", None) is (
        Rule.RESOLUTION
    ):
        return "missing"
    axes = AXIS.findall(str(refused))
    return f"stale ({', '.join(axes)})" if axes else "stale"


def uncovered_rungs(law: dict, config) -> list:
    """Each decode rung `config` schedules (`scheduled_rungs`, of its capture
    ladder, `schedulable_ladder`, and the scheduler's bound) that the step
    linear `law` has no vector for, or its draft law, whose decode vectors
    price a decode's draft unless it states a `forward` vector. A step outside
    a vector's hull is priced, not refused, so no context counts.

    A prefill-decode producer (`_producer`) schedules none: its requests leave
    after their prefill, and the one decode-phase step it takes, the dummy a
    DP group runs with every rank idle, is priced without a vector
    (`cost.idle_group_dummy`)."""
    if _producer(config):
        return []
    rungs = scheduled_rungs(_ladder(config), schedulable_bound(config))
    draft = law.get("draft")
    laws = [law] + ([draft] if draft and "forward" not in draft else [])
    vectors = [{str(rung) for rung in each.get("decode", {})} for each in laws]
    return [r for r in rungs if any(str(r) not in v for v in vectors)]


def _producer(config) -> bool:
    """Whether `config` is a prefill-decode producer: every prefill-decode
    transport it runs (`KVConnectorFactory.leaf_connectors`, a `multi`
    expanded, offload tiers left out) states ``kv_role`` ``kv_producer``, as
    the recipe's prefill's Mooncake ``kv_producer`` beside its
    ``mooncake_store`` offload does. A consumer, ``kv_both``, a transport
    stating no role, or none at all decodes."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    leaves = KVConnectorFactory.leaf_connectors(config) or ()
    roles = {
        (getattr(sub, "kv_transfer_config", None) or {}).get("kv_role")
        for name, sub in leaves
        if not KVConnectorFactory.is_offload_backend(name)
    }
    return roles == {"kv_producer"}


def _ladder(config) -> list[int]:
    """The capture sizes the scheduler can produce (`schedulable_ladder`); none
    on an eager runner, which captures none."""
    return [] if config.enforce_eager else schedulable_ladder(config)


def calibrate_inputs(atom_args, config, env=None) -> dict:
    """What a `calibrate.sh` command takes of the server, or the 1P1D side,
    that ATOM's arguments `atom_args` start under its variables `env`, resolved
    as `config`, under this process's environment: its ``model``; its
    ``server`` arguments (`_server_args`); ``env``, `env` and the environment
    as a law's key states it (`keyed_env`); ``gpus``, as many as its ranks
    use; ``rungs``, a first-turn
    record of each decode rung it schedules (`step_linear_plan.records`); and
    ``untraced``, why `compass record --trace` refuses it, else None. Where
    `calibrate.sh` cannot take its arguments, or no law it fits prices the
    server's decode, ``refused`` alone says why: a server above one pipeline
    stage that decodes, unlike a producer (`_producer`), whose stages' decode
    steps every fit cuts (`step_labels.PIPELINED`)."""
    from atom.compass.drive.record import refuse_trace
    from atom.compass.run.step_grid import keyed_env

    try:
        server = _server_args(atom_args)
    except ValueError as refused:
        return {"refused": str(refused)}
    if (stages := config.pipeline_parallel_size) > 1 and not _producer(config):
        return {
            "refused": f"it decodes on {stages} pipeline stages, and the fit cuts "
            "each stage's decode steps, whose window holds the stream hold of the "
            "stage's last send and its wait for the stage before, which no label "
            "parts from its own work"
        }
    try:
        refuse_trace(atom_args)
        untraced = None
    except ValueError as refused:
        untraced = str(refused)
    engine = runfile._engine_args(atom_args)
    gpus = engine.tensor_parallel_size * engine.data_parallel_size
    gpus *= engine.pipeline_parallel_size * engine.prefill_context_parallel_size
    bound, width = schedulable_bound(config), runfile._dp_width(atom_args)
    return {
        "model": str(Path(engine.model).resolve()),
        "server": server,
        "env": dict(sorted(((env or {}) | keyed_env()).items())),
        "gpus": gpus,
        "rungs": records(_ladder(config), bound, width, False),
        "untraced": untraced,
    }


def calibration(out, atom_args, config, store=None, env=None) -> str:
    """The `calibrate.sh` command that records and fits a step linear law for
    `config`, ATOM's arguments `atom_args` resolved under the server's
    variables `env`, into `out`, publishing into `store` when given, else into
    its own, then its expected time: the records `step_linear_plan.records`
    plans, the trace runs only where `compass record --trace` takes one; on as
    many GPUs as ATOM's ranks use; under `env` and this process's environment
    as a law's key states it (`keyed_env`), and each of `CLIENT_ENV` it sets,
    then a line naming each it does not, which `calibrate.sh` refuses to run
    without. Where `calibrate.sh` cannot take ATOM's arguments, or no law it
    fits prices `config`'s decode (`calibrate_inputs`), why."""
    held = calibrate_inputs(atom_args, config, env)
    if "refused" in held:
        return f"no calibrate.sh command: {held['refused']}"
    env = {
        "MODEL": held["model"],
        "GPU": _cards(0, held["gpus"]),
        "SERVER_ARGS": " ".join(held["server"]),
    }
    traced = held["untraced"] is None
    untraced = (
        ""
        if traced
        else f"\n  no trace run, which compass record refuses: {held['untraced']}; "
        "so the law's prefill stays unfitted"
    )
    planned = held["rungs"] + (list(TRACE_RUNS) if traced else [])
    return _command(out, held["env"], env, planned, store, untraced)


def deployment_calibration(out, sides: dict, fit, router, node: int, store=None) -> str:
    """The `calibrate.sh` command that records a 1P1D configuration, each
    record one `deploy.sh record` of both sides, its router on the arguments
    `router`, and fits the laws of each side `fit` names, into `out`,
    publishing into `store` when given, else into its own, then its expected
    time, as `calibration`'s; `sides` holds each side's `calibrate_inputs`.
    Its records are a first-turn record of each decode rung the decode side
    schedules, whose prefill side the prefill side's laws are fitted on too,
    and, when it fits the prefill side, the trace runs, where `compass record
    --trace` takes the prefill side. Both sides take cards of one node, which
    holds `node`: the prefill side the first, the decode side the next. The
    variables both sides run under alike are the command's, and each side's
    others its ``PREFILL_ENV`` or ``DECODE_ENV``. A ValueError says why no
    command serves: `calibrate.sh` cannot take a side's arguments or
    variables, or no law it fits prices a side's decode (`calibrate_inputs`),
    or the sides serve two models or take more cards than one node holds, as a
    deployment across two nodes does."""
    for side, held in sides.items():
        if "refused" in held:
            raise ValueError(f"the {side} side: {held['refused']}")
    prefill, decode = sides["prefill"], sides["decode"]
    if prefill["model"] != decode["model"]:
        raise ValueError(
            "it takes one MODEL, and the prefill side "
            f"serves {prefill['model']} and the decode side {decode['model']}"
        )
    if (cards := prefill["gpus"] + decode["gpus"]) > node:
        raise ValueError(
            f"it records both sides on one node, and they take {cards} cards, "
            f"where a node holds {node}; a deployment across two nodes is not one "
            "command"
        )
    both = {k: v for k, v in prefill["env"].items() if decode["env"].get(k) == v}
    env, first, own = {"MODEL": prefill["model"]}, 0, {}
    for side in SIDES.values():
        held = sides[side]
        env[f"{side.upper()}_GPU"] = _cards(first, held["gpus"])
        first += held["gpus"]
        own[side] = [f"{k}={v}" for k, v in held["env"].items() if k not in both]
        if spaced := [word for word in own[side] if len(word.split()) != 1]:
            raise ValueError(
                f"deploy.sh splits {side.upper()}_ENV on whitespace, and "
                f"{', '.join(spaced)} holds whitespace"
            )
    for side in SIDES.values():
        env[f"{side.upper()}_ARGS"] = " ".join(sides[side]["server"])
    env |= {f"{s.upper()}_ENV": " ".join(words) for s, words in own.items() if words}
    env["FIT"] = " ".join(side for side in SIDES.values() if side in fit)
    env["ROUTER_ARGS"] = " ".join(router)
    traced = "prefill" in fit and prefill["untraced"] is None
    untraced = (
        f"\n  no trace run, which compass record refuses of the prefill side: "
        f"{prefill['untraced']}; so its law's prefill stays unfitted"
        if "prefill" in fit and not traced
        else ""
    )
    planned = decode["rungs"] + (list(TRACE_RUNS) if traced else [])
    return _command(out, both, env, planned, store, untraced)


def _cards(first: int, count: int) -> str:
    """`count` cards from card `first`, as `--gpu` takes them."""
    return ",".join(map(str, range(first, first + count)))


def _command(out, keyed: dict, env: dict, planned: list, store, untraced: str) -> str:
    """`calibrate.sh` into `out` of the records `planned`, under the keyed
    variables `keyed`, then each of `CLIENT_ENV` this process sets, then `env`
    and `store` as ``STORE`` when given; a line naming each of `CLIENT_ENV`
    unset, `untraced`, and its expected time."""
    # `calibrate.sh` refuses an empty value as it refuses an unset one.
    clients = {k: os.environ[k] for k in CLIENT_ENV if os.environ.get(k)}
    unset = [k for k in CLIENT_ENV if k not in clients]
    unheld = (
        f"\n  before it runs, set {' and '.join(unset)}, which calibrate.sh "
        "requires and this environment does not set"
        if unset
        else ""
    )
    env = keyed | clients | env
    if store is not None:
        env["STORE"] = str(store)
    words = [f"{k}={shlex.quote(v)}" for k, v in env.items()]
    words += map(shlex.quote, [str(CALIBRATE), str(Path(out).resolve()), *planned])
    return (
        " ".join(words)
        + unheld
        + untraced
        + "\n  expected time: unknown, as no earlier calibration's status is read for it"
    )


def _server_args(atom_args) -> list[str]:
    """ATOM's arguments `atom_args` less `--model`, which `calibrate.sh` takes
    apart, as its `SERVER_ARGS` takes them: split on whitespace, and its own
    default in place of none."""
    server, given = [], iter(atom_args)
    for arg in given:
        if arg == "--model":
            next(given, None)
        elif not arg.startswith("--model="):
            server.append(arg)
    if not server:
        raise ValueError(
            "it serves --max-model-len 262144 when SERVER_ARGS states nothing, and "
            "ATOM's arguments state nothing but --model"
        )
    if spaced := [arg for arg in server if len(arg.split()) != 1]:
        raise ValueError(
            "it splits SERVER_ARGS on whitespace, so it cannot pass ATOM's "
            f"argument{'s' * (len(spaced) > 1)} {', '.join(map(repr, spaced))}"
        )
    return server


def _coverage(report: dict) -> str:
    """The coverage check's `report`: its counts, and its gaps as JSON."""
    gaps = {k: v for k, v in report.items() if k not in ("inputs", "steps", "share")}
    return (
        f"coverage of {', '.join(report['inputs'])}: {gaps.pop('covered')} of "
        f"{report['steps']} steps covered; made by {MADE_BY['coverage']}\n"
        + json.dumps(gaps, indent=1)
    )


def _unclassified(atom_args, arch: str) -> list[str]:
    """The line naming each module the step roofline leaves unclassified on the
    model ATOM's arguments build on the meta device of `arch`, which refuses
    every step it is asked for; none when it classifies all. The build declares
    a device process-wide, so it runs in a process of its own."""
    from atom.compass.backends.step_roofline import UNCLASSIFIED
    from atom.compass.bench.ops import _child
    from atom.compass.validate.bound import walked

    modules = _child(walked, atom_args, arch)
    if names := [m.name for m in modules if m.kind == UNCLASSIFIED]:
        return [
            "the step roofline below the backend refuses each step it is asked "
            "for, leaving these modules unclassified: " + ", ".join(names)
        ]
    return []

"""The exact command sequence each cc-traces acceptance cell needs, written down.

A GPU lease is bought in hours, and the way to waste one is to arrive with a
protocol and improvise the commands. `CC_TRACES_PROTOCOL.md` says what a cell
must be; this says what to type, in order, per cell, with the artifacts each
step produces and the ones the next step reads. It runs nothing itself --
`cc_traces_run.py` executes this plan, and both read the same `cell_steps()`, so
what is printed for review is what is run.

The matrix is twenty-four cells: TP {1, 2, 4} x class {`clients_short`,
`clients_large`} x clients {1, 2, 4, 8}. A cell is all three coordinates, and
each one replays its own registered workload file (§1B). The earlier six-cell
`short`/`long` matrix keeps its registration and its already-run cells; it is
not re-planned here.

Three places a step can run, and they are not interchangeable:

* `gpu` -- the leased node, under the isolation audit. Only the real side.
* `device_free` -- a container with no `/dev/kfd` and no `/dev/dri`. The
  modelled side and its probe. The protocol's speedup claim is about what a
  prediction costs where no device could have helped, so this is a property of
  the container and not of `HIP_VISIBLE_DEVICES` (§5).
* `cpu` -- anywhere; reads artifacts only. Validation.

**A repeat is a process, not a request batch.** §3 registers three real and
three modelled repeats *each from a fresh process*, so every repeat here starts
its own server and stops it again. On the modelled side that is not bookkeeping:
a predicting server holds a virtual clock frozen at its epoch, and a second
replay against the same process is stamped against an origin the first replay
has already moved -- the +71% TTFT error that `replay.py` refuses a warmed
predictor for. Reuse is only correct against an engine-owned run boundary that
resets that origin, and this tree has none.

What this file will not do is guess. Where the protocol needs something the
repository does not yet provide -- the oracle's own option wiring, the
acquisition seconds -- the step says so in `gaps` rather than emitting a command
that would fail at 3 a.m. on a leased node.

    python scripts/compass/cc_traces_plan.py --root /workspace/results/cc_acceptance
    python scripts/compass/cc_traces_plan.py --root ... --shell
    python scripts/compass/cc_traces_plan.py --root ... --out plan.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """A sibling script as a module, the way `cc_traces_run.py` loads them."""
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The cells' configuration as data. The plan used to take the modelled
#: side's oracle and its dozen options as free text on the command line, which
#: is the part of a matrix run most expensive to mistype and the part no test
#: could reach. Naming `--artifact-root` resolves them from here instead.
registry = _load("cc_traces_registry")


def _cache_policy_module():
    path = ROOT / "atom/compass/core/cache_policy.py"
    spec = importlib.util.spec_from_file_location("compass_cache_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_policy = _cache_policy_module()


def _registry_oracle(args):
    return registry.ORACLE if getattr(args, "artifact_root", None) else None


def _registry_options(args, tp: int) -> list:
    root = getattr(args, "artifact_root", None)
    return registry.options(tp, root) if root else []


def _registry_target(args, tp: int):
    root = getattr(args, "artifact_root", None)
    return registry.replay_target(tp, root) if root else None


def _registry_profile(args, tp: int):
    root = getattr(args, "artifact_root", None)
    return registry.memory_model(tp, root) if root else None


def width_map(values, flag: str) -> dict:
    """`TP=PATH` pairs read into a dict keyed by width.

    A bare path is refused. Neither of these files is shared across the
    matrix: a target record is of the width it was captured or derived at --
    the replay runner takes the block and state capacities straight out of it
    and refuses a width it was not made for -- and a memory profile is of the
    width it was taken at. One path here would therefore name the wrong file
    for two of the three widths, which is exactly the mistake this plan
    exists to stop someone making by hand.
    """
    out = {}
    for value in values or ():
        width, sep, path = value.partition("=")
        if not sep or not path or not width.isdigit():
            raise SystemExit(
                f"{flag} takes TP=PATH, not {value!r}: the three widths do "
                f"not share one file, so a bare path would be handed to all "
                f"of them"
            )
        out[int(width)] = path
    return out


#: The matrix. TP=1 is the source width and its cells are a fit statistic, not
#: a prediction; they are here because the residual is reported, not because
#: they test the bet.
TPS = (1, 2, 4)
#: The clients matrix's two classes (`CC_TRACES_PROTOCOL.md` §1B). The earlier
#: `short`/`long` classes keep their own registration and their already-run
#: cells; they are not part of this matrix and are not re-planned here.
CLASSES = ("clients_short", "clients_large")
#: Offered load, as a count of root sessions. Not a cap on in-flight requests:
#: every descendant is replayed at its own offset and contributes load, so the
#: only concurrency bound in a cell is the server's own `max_num_seqs`.
#: The pools are nested (c1 ⊂ c2 ⊂ c4 ⊂ c8), so a step up this axis adds roots
#: without disturbing the requests already there.
CLIENTS = (1, 2, 4, 8)

#: Repeats per side, from the protocol's §3. Three on the modelled side too: a
#: single simulated run is not a distribution.
REPEATS = 3

# A client transport deadline, not an accuracy or latency gate. The largest
# TP1 workload can serve for more than the replay client's 600-second default.
REQUEST_TIMEOUT = 3600.0

#: What a plan is for. `--repeats` below the registered count does not build a
#: cheaper acceptance run; it builds a diagnostic, and the plan is labelled
#: that way so nothing downstream has to infer it from the file count. The
#: words are `cc_traces_run.py`'s, so the label survives into the artifacts.
ACCEPTANCE = "acceptance"
DIAGNOSTIC = "diagnostic"

ADVISORY_ISOLATION_QUALIFICATION = (
    "Node-level interference may be collected only as advisory timing evidence. "
    "Selected devices must remain clean and fully observed; this option does "
    "not establish isolated-node timing proof."
)

MODEL = "Qwen/Qwen3.8-27B"

#: The HTTP listener: what the health check polls and the replay client dials.
#: `--server-port` on the entry point's parser.
PORT = 8000

#: The engine's internal rendezvous port -- `--port` on that same parser, and
#: what `model_runner.py` exports as `MASTER_PORT` for torch.distributed. It is
#: not the listener and it is not free for the taking either: two servers alive
#: at once on one host collide on it even when their listeners differ, and at
#: TP>1 the collision is between rendezvous groups, so the second server can
#: join the first one's. Chosen explicitly per run rather than left to the
#: engine's default, and checked for a conflict beside the listener.
ENGINE_PORT = 8006

#: Seconds between `rocm-smi` samples across the real side's whole window. Five
#: is short enough that `isolation.py`'s two-sample sustained rule needs ten
#: seconds of a neighbour to call it busy, and long enough that the sampler is
#: not itself load.
SAMPLE_INTERVAL = 5.0

#: The engine configuration, unchanged from the registered matrix. Every cell
#: is served with these, and the validator reads them back out of the server's
#: own provenance rather than trusting this list.
ENGINE_ARGS = (
    "--gpu-memory-utilization",
    "0.90",
    "--max-model-len",
    "262144",
    "--no-enable_prefix_caching",
    "--max-num-seqs",
    "32",
)

def add_modelled_timing_arguments(parser):
    """Optional timing controls shared by the plan and side-run CLIs."""
    parser.add_argument(
        "--compass-request-readiness-profile", default="",
        help="source readiness profile for TP1 modelled cells only",
    )
    parser.add_argument(
        "--compass-prefill-preparation-fence", action="store_true",
        help="enable the source preparation fence for TP1 modelled cells only",
    )


def _modelled_engine_args(tp, request_readiness_profile, prefill_preparation_fence):
    args = list(ENGINE_ARGS)
    if tp == 1:
        if request_readiness_profile:
            args += ["--compass-request-readiness-profile", request_readiness_profile]
        if prefill_preparation_fence:
            args.append("--compass-prefill-preparation-fence")
    return args


#: What each cell keeps, and which step writes it. The validator refuses a cell
#: that is missing any of it, so the plan names it rather than leaving it to be
#: noticed afterwards.
EVIDENCE = {
    "real.r{n}.json": "the real side's repeat n, from its own server process",
    "modelled.r{n}.json": "the modelled side's repeat n, from its own process",
    "real.r{n}.prepare.json": "the drained preparation records, real side",
    "provenance.real.r{n}.json": "what that server said it was, read when it came up",
    "provenance.modelled.r{n}.json": "the same for the modelled repeat",
    "real.r{n}_steps.jsonl": "that repeat's step table, from that repeat's server",
    "modelled.r{n}_steps.jsonl": "that repeat's predicted steps on the virtual clock",
    "real.r{n}_memory.json": "the memory terms that repeat's card reported, and "
                             "the budget the engine made of them",
    "gpu.jsonl": "rocm-smi samples across the whole window, baseline first",
    "isolation.json": "the isolation audit over those samples",
    "costs.real.json": "the seconds the real side measured of itself",
    "costs.modelled.json": "the seconds the modelled side measured of itself",
    "costs.json": "every cost term of §5, in seconds, merged",
    "gpu_free.json": "the device-free observation, taken in the modelled container",
    "registry.json": "the calibration registry of §4",
    "cc_traces_protocol.json": "the protocol stamp, written by cc_traces_protocol.py",
    "run.real.json": "the real side's journal: every command, pid and exit status",
    "run.modelled.json": "the same for the modelled side",
    "cc_traces_cell.json": "the validator's verdict, kept pass or fail",
}

#: Things the protocol asks for that this repository does not yet hand over.
#: Listed here so a plan cannot read as more complete than it is.
GAPS = (
    (
        "the oracle's option names are a property of the oracle chosen at run "
        "time; --oracle-option here is passed through verbatim and is not "
        "validated against the oracle's constructor"
    ),
    (
        "the modelled side needs a target record of its own width to build a "
        "Config without a device, and at TP=2 and TP=4 a memory profile to "
        "size the pool from; both are inputs here, derived by the memory "
        "model, and no step in this plan creates either"
    ),
    (
        "warmup_seconds is not in the server's /compass/provenance, so 'the "
        "modelled side was given no warmup' is checked from the replay "
        "artifact's own manifest and not from the server"
    ),
    (
        "capture, calibration, derivation and load seconds are inputs to the "
        "costs step: nothing here measures them, and the cell is refused until "
        "they are supplied"
    ),
)


def _serve(
    tp: int,
    n: int,
    *,
    modelled: bool,
    oracle,
    options,
    port: int,
    engine_port: int,
    cell: str,
    target,
    memory_model=None,
    engine_args=None,
):
    """The server command for one side of one cell.

    The modelled side goes through `replay_server.py` rather than the module
    entry point: AITER asks the driver for the chip at import, which on a
    machine with no device fails before any flag could be parsed, so the
    target record has to answer that question first.
    """
    if modelled:
        cmd = [
            "python",
            "scripts/compass/replay_server.py",
            "--compass-replay-target",
            target or "$CC_TRACES_REPLAY_TARGET",
        ]
        if memory_model:
            # What the pool is actually sized from, at every width including
            # TP=1. The replay runner publishes the budget it used as
            # `source-derived` when this is given and `captured` when it is
            # not, and a cell sized from a captured count is not evidence
            # about the analytical model the acceptance is testing.
            cmd += ["--compass-memory-model", memory_model]
    else:
        cmd = ["python", "-m", "atom.entrypoints.openai.api_server"]
    cmd += [
        "--model",
        MODEL,
        # The HTTP listener, which is what the health check and the replay
        # client talk to. `--port` on this parser is a different thing -- the
        # engine's internal port -- so naming it here left the listener on its
        # own default and the harness waiting on a port nothing ever bound.
        "--server-port",
        str(port),
        # The engine's internal rendezvous port, named because leaving it at
        # the engine's default means every server on this host asks for 8006.
        "--port",
        str(engine_port),
        "-tp",
        str(tp),
        *(ENGINE_ARGS if engine_args is None else engine_args),
        "--compass",
        "--compass-mode",
        "predict" if modelled else "measure",
        "--compass-measure-out",
        f"{cell}/{'modelled' if modelled else 'real'}.r{n}_steps.jsonl",
    ]
    if modelled:
        # Named here rather than left at the parser default. One process stands
        # in for the whole group, so `rank0` would price the rank it calls
        # itself and quietly drop the TP4 rank-1 outlier -- the ranks are not
        # symmetric, and nothing in this plan establishes that they are. This
        # side is therefore evaluated on every logical rank; `rank_aggregation`
        # in each row says what the number is and in which direction it
        # approximates.
        cmd += ["--compass-rank-aggregation", registry.RANK_AGGREGATION]
        if oracle:
            cmd += ["--compass-oracle", oracle]
        for option in options:
            cmd += ["--compass-oracle-option", option]
    else:
        # Per repeat: the engine opens this path with "w", so three repeats
        # sharing one name would leave one table and two overwritten ones.
        # The terms the card actually reported, beside the budget they sized.
        # Protocol section 6 gates non-KV memory terms at 10% and the KV block
        # count at 5%, and until this flag was passed the reference side of
        # that comparison did not exist: `budget_source` publishes the block
        # count a run served but not the readings behind it, so there was
        # nothing to hold the modelled `peak_torch`, `non_torch` or graph pool
        # against. Written only here -- the modelled side derives these from a
        # profile and recording them there would compare a prediction with
        # itself.
        cmd += ["--compass-memory-out", f"{cell}/real.r{n}_memory.json"]
    return cmd


def workload(klass: str, clients: int, suffix: str = "jsonl") -> str:
    """The registered file for one (class, client count).

    One file per cell, not one file plus a client argument: the request set a
    cell replayed has to be a digest the protocol registered, and a driver flag
    that selects a subset at run time would leave the bytes the same for eight
    different workloads.
    """
    return f"atom/compass/cc_traces_{klass}_c{clients}.{suffix}"


def _replay(
    klass: str, clients: int, out: str, *, paced: bool, prepare: int, port: int,
    request_timeout: float = REQUEST_TIMEOUT,
    pretokenize: bool = False,
    workload_path: str | None = None,
    client_memory_budget_mib: int | None = None,
    diagnostic_prepare_output_cap: int | None = None,
    opening_plan=None,
):
    if client_memory_budget_mib is not None and client_memory_budget_mib <= 0:
        raise SystemExit("client memory budget must be positive")
    cmd = [
        "python",
        "scripts/compass/replay.py",
        "--port",
        str(port),
        "--model",
        MODEL,
        *(["--trace", workload(klass, clients) if workload_path is None else workload_path]
          if opening_plan is None else ["--opening-plan", opening_plan["path"],
                                       "--opening-plan-sha256", opening_plan["sha256"]]),
        "--out",
        out,
        "--check-lengths",
        "--timeout",
        str(request_timeout),
    ]
    if workload_path is not None and opening_plan is None:
        cmd += ["--num-requests", "0"]
    if client_memory_budget_mib is not None:
        cmd += ["--client-memory-budget-mib", str(client_memory_budget_mib)]
    if pretokenize:
        cmd += ["--pretokenize"]
    if paced:
        # The real engine stamps arrivals on receipt, so a declared arrival is
        # discarded there: without --pace the real side answers a burst while
        # the modelled side answers the trace, and the two are not comparable.
        cmd += ["--pace"]
    if prepare:
        cmd += [
            "--prepare",
            str(prepare),
            "--prepare-out",
            out.replace(".json", ".prepare.json"),
        ]
        if diagnostic_prepare_output_cap is not None:
            cmd += ["--diagnostic-prepare-output-cap", str(diagnostic_prepare_output_cap)]
    return cmd


def _lifecycle(
    side: str,
    n: int,
    *,
    tp: int,
    klass: str,
    clients: int,
    cell: str,
    where: str,
    port: int,
    engine_port: int,
    oracle,
    options,
    target,
    memory_model=None,
    request_timeout: float = REQUEST_TIMEOUT,
    pretokenize: bool = False,
    engine_args=None,
    workload_path: str | None = None,
    client_memory_budget_mib: int | None = None,
    diagnostic_prepare_output_cap: int | None = None,
    opening_plan=None,
):
    """One repeat: its own server, its replay, and the end of that process."""
    modelled = side == "modelled"
    engine_args = list(ENGINE_ARGS if engine_args is None else engine_args)
    return [
        {
            "id": f"serve-{side}-{n}",
            "role": "serve",
            "side": side,
            "repeat": n,
            "where": where,
            "why": (
                "the prediction, from a process that has predicted nothing yet"
                if modelled
                else "the measured side, at this cell's width"
            ),
            "command": _serve(
                tp,
                n,
                modelled=modelled,
                oracle=oracle,
                options=options,
                port=port,
                engine_port=engine_port,
                cell=cell,
                target=target,
                memory_model=memory_model,
                engine_args=engine_args,
            ),
            "engine_args": engine_args,
            "background": True,
            "health": f"http://127.0.0.1:{port}/health",
            # Named by scope, and by the flag that carries each one, so the
            # harness checks the right port for a conflict and the manifest
            # does not have to guess which of the two "port" means.
            "ports": {
                "http_listener": {"port": port, "flag": "--server-port"},
                "engine_rendezvous": {"port": engine_port, "flag": "--port"},
            },
            "produces": (
                [f"server.{side}.r{n}.log", f"provenance.{side}.r{n}.json"]
                + ([f"modelled.r{n}_steps.jsonl"] if modelled
                   else [f"real.r{n}_steps.jsonl", f"real.r{n}_memory.json"])
            ),
        },
        {
            "id": f"replay-{side}-{n}",
            "role": "replay",
            "side": side,
            "repeat": n,
            "where": where,
            "why": (
                "the same trace, declared arrivals, no pacing and no preparation"
                if modelled
                else "one repeat, from a prepared and drained engine, paced to the trace"
            ),
            "command": _replay(
                klass,
                clients,
                f"{cell}/{side}.r{n}.json",
                paced=not modelled,
                prepare=0 if modelled else 3,
                port=port,
                request_timeout=request_timeout,
                pretokenize=pretokenize,
                workload_path=workload_path,
                client_memory_budget_mib=client_memory_budget_mib,
                diagnostic_prepare_output_cap=diagnostic_prepare_output_cap,
                opening_plan=opening_plan,
            ),
            "produces": (
                [f"{side}.r{n}.json"]
                + ([] if modelled else [f"{side}.r{n}.prepare.json"])
            ),
        },
        {
            "id": f"stop-{side}-{n}",
            "role": "stop",
            "side": side,
            "repeat": n,
            "where": where,
            "stops": f"serve-{side}-{n}",
            "why": (
                "the next repeat is a fresh process: a predicting server's "
                "virtual epoch is fixed when it starts, so a second replay "
                "against it is stamped against an origin the first one moved"
                if modelled
                else "the next repeat is a fresh process, as §3 registers"
            ),
            "command": None,
            "provided_by": "the run harness, by signalling the process it started",
            "produces": [],
        },
    ]


def _real_monitoring_steps(cell, allow_advisory_isolation):
    """The same baseline, window sampler and isolation audit for any workload."""
    before = [
        {
            "id": "sample-baseline",
            "role": "command",
            "side": "real",
            "where": "gpu",
            "why": (
                "the one sample that can show a card was already somebody "
                "else's: after our server starts, every byte on our cards is "
                "ours"
            ),
            "command": [
                "python",
                "scripts/compass/gpu_sampler.py",
                f"{cell}/gpu.jsonl",
                "--once",
                "--phase",
                "baseline",
            ],
            "produces": ["gpu.jsonl"],
        },
        {
            "id": "sample",
            "role": "sample",
            "side": "real",
            "where": "gpu",
            "why": "isolation is a property of the whole window, not of two instants",
            "command": [
                "python",
                "scripts/compass/gpu_sampler.py",
                f"{cell}/gpu.jsonl",
                "--interval",
                str(SAMPLE_INTERVAL),
                "--phase-file",
                f"{cell}/phase.json",
            ],
            "background": True,
            "produces": ["gpu.jsonl"],
        },
    ]
    after = [
        {
            "id": "stop-sample",
            "role": "stop",
            "side": "real",
            "where": "gpu",
            "stops": "sample",
            "why": "the window the audit covers ends with the last real repeat",
            "command": None,
            "provided_by": "the run harness, by signalling the sampler it started",
            "produces": [],
        },
        {
            "id": "isolation",
            "role": "command",
            "side": "real",
            "where": "cpu",
            "why": "who else was on the node while this was measured",
            "command": [
                "python",
                "scripts/compass/isolation.py",
                f"{cell}/gpu.jsonl",
                "--json",
                f"{cell}/isolation.json",
            ] + (["--allow-busy-node"] if allow_advisory_isolation else []),
            "qualification": (ADVISORY_ISOLATION_QUALIFICATION
                              if allow_advisory_isolation else None),
            "produces": ["isolation.json"],
        },
    ]
    return before, after


def _gpu_free_step(cell):
    return {
        "id": "gpu-free",
        "role": "command",
        "side": "modelled",
        "where": "device_free",
        "why": "the device-free claim, observed in the container that made the prediction",
        "command": [
            "python",
            "scripts/compass/cc_traces_validate.py",
            "gpu-free",
            cell,
        ],
        "produces": ["gpu_free.json"],
    }


def cell_steps(
    tp: int,
    klass: str,
    clients: int,
    *,
    root: str,
    oracle,
    options,
    port: int,
    repeats: int,
    engine_port: int = ENGINE_PORT,
    target=None,
    memory_model=None,
    corpus: str = "$CC_TRACES_CORPUS",
    request_timeout: float = REQUEST_TIMEOUT,
    pretokenize: bool = False,
    allow_advisory_isolation: bool = False,
    request_readiness_profile: str = "",
    prefill_preparation_fence: bool = False,
    client_memory_budget_mib: int | None = None,
):
    """Every step of one cell, in the order it has to happen."""
    if port == engine_port:
        raise SystemExit(
            f"the HTTP listener and the engine's rendezvous port are both "
            f"{port}: they are two different sockets on one host and the "
            f"server cannot bind one of them"
        )
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise SystemExit("request timeout must be a positive finite number of seconds")
    if klass not in CLASSES:
        raise SystemExit(f"{klass!r} is not a registered class of this matrix")
    if clients not in CLIENTS:
        raise SystemExit(
            f"clients={clients} is not one of the registered counts {CLIENTS}: "
            f"there is no workload file for it, and inventing one here would "
            f"be choosing the load after the matrix was registered"
        )
    # All three coordinates are in the name. A cell directory that carried only
    # the width and the class would collide across client counts, and the
    # second run would overwrite the first one's evidence.
    cell = f"{root.rstrip('/')}/tp{tp}_{klass}_c{clients}"
    steps = [
        {
            "id": "stamp",
            "role": "command",
            "side": "both",
            "where": "cpu",
            "why": "a cell that does not say what it ran under is not a cell",
            "command": [
                "python",
                "scripts/compass/cc_traces_protocol.py",
                "stamp",
                cell,
            ],
            "produces": ["cc_traces_protocol.json"],
        },
        {
            "id": "verify-workload",
            "role": "command",
            "side": "both",
            "where": "cpu",
            "why": "the registered bytes, re-derived from the rule, before the lease is spent",
            "command": [
                "python",
                "scripts/compass/cc_traces_clients_workload.py",
                "verify",
                "--manifest",
                workload(klass, clients, "manifest.json"),
                "--workload",
                workload(klass, clients),
                "--corpus",
                corpus,
            ],
            "produces": [],
        },
    ]
    before, after = _real_monitoring_steps(cell, allow_advisory_isolation)
    steps += before
    for n in range(1, repeats + 1):
        steps += _lifecycle(
            "real",
            n,
            tp=tp,
            klass=klass,
            clients=clients,
            cell=cell,
            where="gpu",
            port=port,
            engine_port=engine_port,
            oracle=None,
            options=(),
            target=None,
            memory_model=None,
            request_timeout=request_timeout,
            pretokenize=pretokenize,
            client_memory_budget_mib=client_memory_budget_mib,
        )
    steps += after
    for n in range(1, repeats + 1):
        steps += _lifecycle(
            "modelled",
            n,
            tp=tp,
            klass=klass,
            clients=clients,
            cell=cell,
            where="device_free",
            port=port,
            engine_port=engine_port,
            oracle=oracle,
            options=options,
            target=target,
            memory_model=memory_model,
            request_timeout=request_timeout,
            pretokenize=pretokenize,
            client_memory_budget_mib=client_memory_budget_mib,
            engine_args=_modelled_engine_args(
                tp, request_readiness_profile, prefill_preparation_fence),
        )
    steps += [
        _gpu_free_step(cell),
        {
            "id": "costs",
            "role": "command",
            "side": "both",
            "where": "cpu",
            "why": "every term of §5, separately; the gate reads two of them and reports the rest",
            "command": [
                "python",
                "scripts/compass/cc_traces_run.py",
                "costs",
                cell,
                "--capture",
                "$CC_TRACES_CAPTURE_S",
                "--calibration",
                "$CC_TRACES_CALIBRATION_S",
                "--derivation",
                "$CC_TRACES_DERIVATION_S",
                "--load",
                "$CC_TRACES_LOAD_S",
            ],
            "produces": ["costs.json"],
        },
        {
            "id": "validate",
            "role": "command",
            "side": "both",
            "where": "cpu",
            "why": "the verdict, kept whether it passes or not",
            "command": [
                "python",
                "scripts/compass/cc_traces_validate.py",
                "cell",
                cell,
                "--class",
                klass,
                "--clients",
                str(clients),
                "--tp",
                str(tp),
                "--repeats",
                str(repeats),
                "--calibration-registry",
                f"{cell}/registry.json",
            ],
            "produces": ["cc_traces_cell.json"],
        },
    ]
    return {
        "cell": cell,
        "tp": tp,
        "class": klass,
        "clients": clients,
        "workload": workload(klass, clients),
        "request_timeout": request_timeout,
        "allow_advisory_isolation": bool(allow_advisory_isolation),
        "isolation_qualification": (ADVISORY_ISOLATION_QUALIFICATION
                                    if allow_advisory_isolation else None),
        "steps": steps,
    }


def diagnostic_steps(
    tp, case, *, cell, oracle, options, port, repeats=1,
    engine_port=ENGINE_PORT, target=None, memory_model=None,
    request_timeout=REQUEST_TIMEOUT, pretokenize=False,
    allow_advisory_isolation=False, request_readiness_profile="",
    prefill_preparation_fence=False, client_memory_budget_mib=None,
    diagnostic_prepare_output_cap=None,
    enable_prefix_caching=False,
    prompt_encoding=None,
    prompt_encoding_sha256=None,
    opening_plan=None,
):
    """Execute a pinned case through the same lifecycle, without a matrix alias."""
    klass, clients = case["case_id"], case["clients"]
    if klass in CLASSES or klass in ("short", "long"):
        raise SystemExit("a corpus diagnostic cannot alias a registered class")
    if Path(cell).name != f"tp{tp}_{klass}_c{clients}":
        raise SystemExit(f"diagnostic --cell must end in tp{tp}_{klass}_c{clients}")
    if port == engine_port:
        raise SystemExit("HTTP listener and engine rendezvous need different ports")
    if not math.isfinite(request_timeout) or request_timeout <= 0 or repeats < 1:
        raise SystemExit("diagnostic repeats and request timeout must be positive")
    if diagnostic_prepare_output_cap is not None and (
        type(diagnostic_prepare_output_cap) is not int or diagnostic_prepare_output_cap < 2
    ):
        raise SystemExit("diagnostic preparation output cap must be at least 2")
    selected_policy = case.get("cache_policy")
    if enable_prefix_caching:
        if tp != 1:
            raise SystemExit("cache-enabled diagnostics currently require TP1")
        errors = cache_policy.policy_errors(selected_policy, cache_policy.cache_on_policy())
        if errors:
            raise SystemExit("; ".join(errors))
        if opening_plan is not None:
            if (case.get("schema") != "compass.aiperf_opening_case/1"
                    or opening_plan != case.get("opening_plan") or pretokenize
                    or prompt_encoding or prompt_encoding_sha256 or clients != 1):
                raise SystemExit("opening requires its pinned chat plan, one client and no codec flags")
        else:
            encoding = case.get("prompt_encoding") or {}
            if (not prompt_encoding or not prompt_encoding_sha256
                    or Path(prompt_encoding).resolve() != Path(encoding.get("path", "")).resolve()
                    or prompt_encoding_sha256 != encoding.get("sha256")):
                raise SystemExit("cache-enabled diagnostics require the case's pinned prompt encoding")
    elif selected_policy is not None:
        raise SystemExit("a cache-policy case requires --enable-prefix-caching")
    elif prompt_encoding or prompt_encoding_sha256:
        raise SystemExit("prompt encoding requires the explicit cache-enabled diagnostic policy")
    before, after = _real_monitoring_steps(cell, allow_advisory_isolation)
    steps = before
    for side in ("real", "modelled"):
        modelled = side == "modelled"
        engine_args = (_modelled_engine_args(tp, request_readiness_profile,
                                            prefill_preparation_fence)
                       if modelled else list(ENGINE_ARGS))
        if enable_prefix_caching:
            engine_args.remove("--no-enable_prefix_caching")
            engine_args += ["--enable_prefix_caching", "--state-checkpoint-interval-tokens",
                            "8192", "--state-checkpoint-demand"]
        if modelled and opening_plan is not None:
            engine_args += ["--compass-opening-plan", opening_plan["path"],
                            "--compass-opening-plan-sha256", opening_plan["sha256"]]
        if opening_plan is not None:
            engine_args += ["--kv_cache_dtype", "bf16", "--block-size", "16",
                            "--max-num-batched-tokens", "16384", "--level", "3",
                            "--cudagraph-mode", "FULL", "--cudagraph-capture-sizes",
                            "[1,2,4,8,16,32,48,64,128,256]"]
        for n in range(1, repeats + 1):
            steps += _lifecycle(
                side, n, tp=tp, klass=klass, clients=clients, cell=cell,
                where="device_free" if modelled else "gpu", port=port,
                engine_port=engine_port, oracle=oracle if modelled else None,
                options=options if modelled else (), target=target if modelled else None,
                memory_model=memory_model if modelled else None,
                request_timeout=request_timeout, pretokenize=pretokenize,
                workload_path=case["workload"], client_memory_budget_mib=client_memory_budget_mib,
                diagnostic_prepare_output_cap=diagnostic_prepare_output_cap,
                engine_args=engine_args,
                opening_plan=opening_plan,
            )
        if not modelled:
            steps += after
    steps.append(_gpu_free_step(cell))
    if enable_prefix_caching:
        for step in steps:
            if step["role"] in ("serve", "replay"):
                step["cache_policy"] = selected_policy
            if step["role"] == "replay" and opening_plan is None:
                step["command"] += ["--prompt-encoding", str(prompt_encoding),
                                    "--prompt-encoding-sha256", prompt_encoding_sha256]
    return {
        "cell": cell, "tp": tp, "class": klass, "clients": clients,
        "workload": case["workload"], "diagnostic_case": case,
        "purpose": "diagnostic", "request_timeout": request_timeout,
        **({"cache_policy": selected_policy} if enable_prefix_caching else {}),
        **({"diagnostic_prepare_output_cap": diagnostic_prepare_output_cap}
           if diagnostic_prepare_output_cap is not None else {}),
        "allow_advisory_isolation": bool(allow_advisory_isolation),
        "isolation_qualification": (ADVISORY_ISOLATION_QUALIFICATION
                                    if allow_advisory_isolation else None),
        "steps": steps,
    }


def opening_steps(tp, case, **options):
    """Use the same owned lifecycle for an explicitly typed chat opening."""
    if tp != 1 or case.get("schema") != "compass.aiperf_opening_case/1":
        raise SystemExit("opening diagnostics require TP1 and an OpeningPlan case")
    return diagnostic_steps(tp, case, enable_prefix_caching=True,
                            opening_plan=case["opening_plan"], **options)


def build(args) -> dict:
    # Resolved per width, not once: what is handed over on the command line
    # overrides the registry, and neither can cover a width it was not named
    # for.
    targets = width_map(getattr(args, "replay_target", None), "--replay-target")
    profiles = width_map(getattr(args, "memory_model", None), "--memory-model")
    cells = [
        cell_steps(
            tp,
            klass,
            clients,
            root=args.root,
            oracle=args.oracle or _registry_oracle(args),
            options=args.oracle_option or _registry_options(args, tp),
            port=args.port,
            engine_port=args.engine_port,
            repeats=args.repeats,
            target=targets.get(tp) or _registry_target(args, tp),
            memory_model=profiles.get(tp) or _registry_profile(args, tp),
            corpus=getattr(args, "corpus", None) or "$CC_TRACES_CORPUS",
            request_timeout=getattr(args, "request_timeout", REQUEST_TIMEOUT),
            pretokenize=getattr(args, "pretokenize", False),
            allow_advisory_isolation=getattr(args, "allow_advisory_isolation", False),
            request_readiness_profile=getattr(args, "compass_request_readiness_profile", ""),
            prefill_preparation_fence=getattr(args, "compass_prefill_preparation_fence", False),
            client_memory_budget_mib=getattr(args, "client_memory_budget_mib", None),
        )
        for tp in TPS
        for klass in CLASSES
        for clients in CLIENTS
    ]
    return {
        "protocol": "atom/compass/CC_TRACES_PROTOCOL.md",
        "root": args.root,
        "model": MODEL,
        "engine_args": list(ENGINE_ARGS),
        "repeats": args.repeats,
        "request_timeout": getattr(args, "request_timeout", REQUEST_TIMEOUT),
        "prompt_encoding": "token_ids" if getattr(args, "pretokenize", False) else "text",
        "allow_advisory_isolation": bool(getattr(args, "allow_advisory_isolation", False)),
        "isolation_qualification": (ADVISORY_ISOLATION_QUALIFICATION
                                    if getattr(args, "allow_advisory_isolation", False)
                                    else None),
        "processes_per_side": args.repeats,
        # What the plan as built can be: a run of fewer than the registered
        # repeats is a legitimate thing to want, but it is a diagnostic, and
        # the plan says so here rather than letting the shortfall be
        # discovered at the verdict.
        "purpose": ACCEPTANCE if args.repeats >= REPEATS else DIAGNOSTIC,
        "repeats_registered": REPEATS,
        "cells": cells,
        "classes": list(CLASSES),
        "client_counts": list(CLIENTS),
        "ranking_groups": (
            "one per (class, clients): TP1/TP2/TP4 over the same replayed "
            "request set. Client counts are offered load and are never pooled "
            "into a rank (CC_TRACES_PROTOCOL.md §6)"
        ),
        "matrix": [
            "python",
            "scripts/compass/cc_traces_validate.py",
            "matrix",
            *[c["cell"] for c in cells],
            "--out",
            f"{args.root.rstrip('/')}/verdict.json",
        ],
        "evidence": dict(EVIDENCE),
        "gaps": list(GAPS),
        "source_width": 1,
        "means": (
            "the commands a cell needs, in order; `cc_traces_run.py` is what "
            "runs them, and no result is claimed here"
        ),
    }


def render(plan: dict) -> str:
    """The plan as something a person can read next to a terminal."""
    out = [
        (
            f"# {plan['protocol']}: {len(plan['cells'])} cells, "
            f"{plan['repeats']} repeats a side, one process each"
        ),
        "",
    ]
    if plan.get("allow_advisory_isolation"):
        out += ["# ADVISORY ISOLATION: " + ADVISORY_ISOLATION_QUALIFICATION, ""]
    if plan.get("purpose") == DIAGNOSTIC:
        out += [
            (
                f"# DIAGNOSTIC: {plan['repeats']} repeats a side is fewer than "
                f"the {plan.get('repeats_registered', REPEATS)} section 3 "
                f"registers, so no cell of this plan can pass acceptance"
            ),
            "",
        ]
    for cell in plan["cells"]:
        out.append(
            f"## {cell['cell']}  (TP={cell['tp']}, {cell['class']}, "
            f"{cell['clients']} client(s))"
        )
        for step in cell["steps"]:
            where = step["where"]
            if step.get("command"):
                line = " ".join(shlex.quote(part) for part in step["command"])
                if step.get("background"):
                    line += (
                        "   &   # the audit window's sampler"
                        if step["role"] == "sample"
                        else "   &   # this repeat's own process"
                    )
                out.append(f"  [{where}] {line}")
            else:
                out.append(f"  [{where}] ({step['id']}: {step['provided_by']})")
        out.append("")
    out.append("## the matrix, once every cell has a verdict")
    out.append("  [cpu] " + " ".join(shlex.quote(p) for p in plan["matrix"]))
    out.append("")
    out.append("## what this plan does not provide")
    for gap in plan["gaps"]:
        out.append(f"  - {gap}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="where the cells' directories live")
    ap.add_argument(
        "--oracle", default=None, help="the cost oracle the modelled side runs"
    )
    ap.add_argument(
        "--oracle-option",
        action="append",
        default=[],
        help="passed through to --compass-oracle-option, repeatable",
    )
    ap.add_argument(
        "--replay-target",
        action="append",
        default=[],
        metavar="TP=PATH",
        help=("the target record the modelled server of that width builds "
              "its Config from; repeatable, one per width, overrides the "
              "registry"),
    )
    ap.add_argument(
        "--memory-model",
        action="append",
        default=[],
        metavar="TP=PATH",
        help=("the memory profile that width sizes its pool from; "
              "repeatable, one per width, overrides the registry"),
    )
    ap.add_argument(
        "--corpus", default=None, help="the cc-traces corpus to verify against"
    )
    ap.add_argument(
        "--port", type=int, default=PORT, help="the HTTP listener (--server-port)"
    )
    ap.add_argument(
        "--engine-port",
        type=int,
        default=ENGINE_PORT,
        help="the engine's internal rendezvous port (--port on the engine)",
    )
    ap.add_argument(
        "--artifact-root",
        default=None,
        help=("resolve the modelled side's oracle and its options from "
              "cc_traces_registry.py against this directory, instead of "
              "typing them; --oracle/--oracle-option still override"),
    )
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT,
                    help="per-request transport deadline in seconds; not an SLO gate")
    ap.add_argument("--client-memory-budget-mib", type=int, default=None,
                    help="explicit replay client memory planning budget")
    ap.add_argument("--pretokenize", action="store_true",
                    help="encode prompts inside each measured window before pacing, on both sides")
    ap.add_argument("--allow-advisory-isolation", action="store_true",
                    help=ADVISORY_ISOLATION_QUALIFICATION)
    add_modelled_timing_arguments(ap)
    ap.add_argument("--out", default=None, help="write the plan as JSON here")
    ap.add_argument(
        "--shell", action="store_true", help="print the commands instead of JSON"
    )
    args = ap.parse_args(argv)
    plan = build(args)
    if args.out:
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    if args.shell:
        print(render(plan), end="")
    elif not args.out:
        print(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
"""The per-process bootstrap of a simulated run: which LP a process is, and its clock.

A run is on when `ENV` names a run file, a JSON object every process of the
deployment reads; spawned processes inherit it. The API server's
``--compass-run`` sets it. With it unset, every function here returns at once
and ATOM runs as it always does.

One engine LP at data-parallel width one and one stage, one frontend LP, and
the traffic LP the harness drives, over `single_engine_table`. The Clock
Authority is co-hosted in the frontend's process and served at the run file's
``clock_endpoint``, which every LP, the frontend's included, connects to.

- `frontend(config)` wraps the `CoreManager` construction in `LLMEngine`: it
  selects the simulated runner, serves the authority and installs the
  frontend LP's runtime; leaving it, after READY from every engine, starts
  the run.
- `engine(config)` wraps the engine's construction in its own process the
  same way, and maps every DP rank to the engine LP for the process-group
  check.
- `runner(runner)` installs the cost backend and the device readings on a
  `CompassModelRunner` where it would build its model, in the worker process.
- `tokenizer(tokenizer, config)` charges the served tokenizer's ``encode`` and
  ``decode`` to the frontend's clock, at the rates the machine spec measured.
- `engine_done(engine)` and `frontend_done(engine)` close each side after the
  finish; the frontend's writes the step table and the run summary.

Run file keys: ``clock_endpoint``, ``bound_s`` (finite), ``admission_path``,
``ipc_s``, ``stream_s``, ``coefficients`` (a `Coefficients` mapping),
``machine`` (a machine spec mapping), ``parameter_count``, and ``out_dir``,
where the step table, the run summary and the engine's command refusals go.
"""

import contextlib
import json
import math
import os
import time
from pathlib import Path

from atom.compass.clock import (
    ClockAuthority,
    LpId,
    RefusalTally,
    RunSummary,
    single_engine_table,
)
from atom.compass.clock_transport import connect, serve
from atom.compass.detect.determinism import StepTable
from atom.compass.runner import COMPASS_RUNNER_QUALNAME
from atom.utils import clock

ENV = "ATOM_COMPASS_RUN"
FRONTEND, ENGINE = LpId("frontend"), LpId("engine")
ATOM_RUNNER = "atom.model_engine.model_runner.ModelRunner"
#: Keys that differ between two runs of one configuration, kept out of its name.
PER_RUN = ("clock_endpoint", "out_dir")
COMMANDS_FILE, STEP_TABLE_FILE, SUMMARY_FILE = (
    "commands.json",
    "step_table.txt",
    "summary.json",
)

_authority: "_RecordingAuthority | None" = None


def spec() -> dict | None:
    """The run file's mapping, or None on a real run."""
    path = os.environ.get(ENV)
    if not path:
        return None
    run = json.loads(Path(path).read_text())
    bound = run.get("bound_s")
    if not isinstance(bound, (int, float)) or not math.isfinite(bound):
        raise ValueError(
            f"{path}: bound_s is {bound!r}; a run needs a finite simulated-time "
            "bound, or a timer nobody declared daemon keeps it alive forever"
        )
    return run


def channel_table(run: dict):
    return single_engine_table(
        admission_path=run["admission_path"],
        ipc_s=run["ipc_s"],
        stream_s=run["stream_s"],
    )


def configuration(run: dict) -> dict:
    """The run file without its per-run keys."""
    return {k: v for k, v in run.items() if k not in PER_RUN}


class _RecordingAuthority(ClockAuthority):
    """The co-hosted authority, recording the step table: a row per reply and a
    row per message each reply releases; and the wall seconds of the run, from
    the frontend's start to the finish."""

    def __init__(self, run: dict) -> None:
        self.steps = StepTable(json.dumps(configuration(run), sort_keys=True))
        self.started = self.finished = None
        super().__init__(channel_table(run), timeline=self, bound_s=run["bound_s"])

    def record(self, lp, time_from, time_to, kind, recovered) -> None:
        self.steps.record(lp, time_from, time_to, kind, detail="recovered" * recovered)

    def on_request(self, *args, **kwargs) -> list:
        replies = super().on_request(*args, **kwargs)
        for lp, _, released in replies:
            for ch, msgs in released.items():
                for seq, arrival in msgs:
                    self.steps.record(lp, arrival, arrival, "release", ch, seq)
        if self.final_clocks is not None and self.finished is None:
            self.finished = time.monotonic()
        return replies


def _runtime(lp: LpId, run: dict) -> clock.LPRuntime:
    rt = clock.LPRuntime(lp, channel_table(run), connect(lp, run["clock_endpoint"]))
    clock.install(rt)
    return rt


@contextlib.contextmanager
def _start_on_leaving(rt: clock.LPRuntime, authority=None):
    yield
    rt.start_run()
    if authority is not None:
        authority.started = time.monotonic()


def frontend(config):
    """Around the `CoreManager` construction; a no-op context on a real run."""
    global _authority
    run = spec()
    if run is None:
        return contextlib.nullcontext()
    widths = (config.parallel_config.data_parallel_size, config.pipeline_parallel_size)
    if widths != (1, 1) or config.enable_rapidserve:
        raise ValueError(
            f"a simulated run here is one engine LP at data-parallel and pipeline "
            f"width 1 without RapidServe, got (dp, pp)={widths}, "
            f"rapidserve={config.enable_rapidserve}"
        )
    if config.runner_qualname not in (ATOM_RUNNER, COMPASS_RUNNER_QUALNAME):
        raise ValueError(
            f"runner {config.runner_qualname!r} is named, and a simulated run "
            f"prices its steps with {COMPASS_RUNNER_QUALNAME}"
        )
    config.runner_qualname = COMPASS_RUNNER_QUALNAME
    _authority = _RecordingAuthority(run)
    serve(_authority, run["clock_endpoint"])
    return _start_on_leaving(_runtime(FRONTEND, run), _authority)


def engine(config):
    """Around the engine's construction in its process; a no-op context on a real run."""
    run = spec()
    if run is None:
        return contextlib.nullcontext()
    from atom.utils.distributed import utils

    dp = config.parallel_config.data_parallel_size
    utils.LP_OF_RANK = dict.fromkeys(range(dp), ENGINE)
    return _start_on_leaving(_runtime(ENGINE, run))


def runner(model_runner) -> None:
    """Install the cost backend and the device readings from the run file."""
    run = spec()
    if run is None:
        return
    from atom.compass.backends.geometry import Parallelism
    from atom.compass.backends.shape import Coefficients, ShapeStubBackend
    from atom.compass.memory import ModelTerms, device_readings, reserves
    from atom.compass.runner.overrides import (
        install_cost_backend,
        install_device_readings,
    )
    from atom.compass.spec import MachineSpec

    config = model_runner.config
    tp = config.tensor_parallel_size
    install_cost_backend(
        model_runner,
        ShapeStubBackend(
            Coefficients(**run["coefficients"]),
            Parallelism(tp_size=tp),
            stack_layers=config.hf_config.num_hidden_layers,
        ),
    )
    install_device_readings(
        model_runner,
        device_readings(
            MachineSpec.from_mapping(_width_keys(run["machine"])),
            tp_width=tp,
            model=ModelTerms.from_declared_config(
                config.hf_config,
                parameter_count=run["parameter_count"],
                tp_size=tp,
                warmup_tokens=config.max_num_batched_tokens,
            ),
            cudagraph_overhead=reserves(enforce_eager=True),
        ),
    )


def tokenizer(tok, config) -> None:
    """Wrap `tok`'s ``encode`` and ``decode`` with the machine spec's entry for
    the model's architecture and the backend that loaded."""
    run = spec()
    if run is None:
        return
    from atom.compass.spec import Backend, MachineSpec
    from atom.utils.compass_loop import wrap_decode, wrap_encode

    entry = MachineSpec.from_mapping(_width_keys(run["machine"])).tokenizer_for(
        config.hf_config.architectures[0],
        Backend.FAST if tok.is_fast else Backend.SLOW,
    )
    tok.encode = wrap_encode(tok.encode, entry)
    tok.decode = wrap_decode(tok.decode, entry)


def _width_keys(node):
    """JSON writes a width table's integer keys as strings; read them back."""
    if not isinstance(node, dict):
        return node
    return {int(k) if k.isdigit() else k: _width_keys(v) for k, v in node.items()}


def engine_done(engine_core) -> None:
    """After the engine's loop: leave the clock and keep the worker's command
    refusals for the run summary."""
    clock.close()
    run = spec()
    if run is not None:
        refused = engine_core.runner_mgr.call_func("refused_commands", wait_out=True)
        (Path(run["out_dir"]) / COMMANDS_FILE).write_text(json.dumps(list(refused)))


def frontend_done(llm_engine) -> bool:
    """Whether the stopped server was a simulated run's finish. If it was, stop
    the engines and write the step table and the run summary."""
    rt = clock.installed()
    if rt is None or rt.now != math.inf:
        return False
    llm_engine.close()
    run, a = spec(), _authority
    out = Path(run["out_dir"])
    reasons = (
        json.loads((out / COMMANDS_FILE).read_text())
        + rt.loop.executor.refusals
        + rt.refusals
    )
    steps = sum(r.lp == ENGINE.name and r.event == "TAR" for r in a.steps.rows)
    summary = RunSummary.of(
        a, a.finished - a.started, refusals=RefusalTally.of(reasons, steps)
    ).as_record()
    summary["configuration"] = configuration(run)
    # A refused command leaves ATOM's half of it applied, so what ran after it
    # was not the deployment simulated.
    summary["coverage_report"] = any(r.startswith("command:") for r in reasons)
    (out / STEP_TABLE_FILE).write_text(a.steps.text())
    (out / SUMMARY_FILE).write_text(json.dumps(summary, sort_keys=True, indent=1))
    return True

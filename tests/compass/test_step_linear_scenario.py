# SPDX-License-Identifier: MIT
"""A served run priced by a checked-in step linear law: Qwen3.8-27B at TP1 under
`LAW`, on `MACHINE`, the spec the law was fitted on.

The run file holds the law as ``compass run --step-linear LAW --host-laws LAW``
writes it (`runfile.resolve`): its step law, its host laws, which charge each
step's host time and the served tokenizer's, and their hops. The server starts
with the flags and environment the law's engine config states: graph mode at
ATOM's default ladder, prefix caching on, ``AITER_LOG_LEVEL`` alone of the
performance variables. As ``compass run`` starts a simulated server on a
machine, it starts under `on_arch` at the machine's arch, gfx942, whatever
device the host has, and the readings it reads are published there.

The workload is a conversation's turns: a first request at 0, and once it has
finished three more, each its whole prompt and a turn of its own. Qwen3.8-27B's
linear attention layers resume only from a state checkpoint, which ATOM anchors
at a prompt's end, so a later turn prefills on the cached prefix up to the first
one's anchor, as an agentic cell's next request does. The run is made twice,
with different `PYTHONHASHSEED`, and `check_run` holds the two to the scenario
checks; a step the law refused would make it a coverage report, which fails
them. `08` D120's levels are the step stub's and the step roofline's, so the
case carries no capability mark.

`run.runner` holds the law to the engine at install: the flags the server
starts with resolve to the law's engine config, under which its host laws load
too, and the same flags with ``--enforce-eager`` are refused, naming the ladder.
That case reads the model's checked-in config and runs anywhere. The served run
needs a driver, as the other scenarios do, and `ATOM_COMPASS_CCTRACES_MODEL`,
the Qwen3.8-27B directory `gate_gpu.sh` sets, a config and a tokenizer; no
weight is read.
"""

import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from test_vertical_slice import (
    TREE,
    WALL_S,
    Traffic,
    UnansweredRequests,
    _run_file,
    publish_entry_on_arch,
)

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.backends.json_file import read_json
from atom.compass.carriers import tracestate_with
from atom.compass.detect.scenario import check_run
from atom.compass.drive import runfile
from atom.compass.drive.serve import ON_ARCH, SERVER
from atom.compass.parity import read
from atom.compass.parity.traffic import DONE, bodies
from atom.compass.run.step_grid import READ_OFF_DEFAULT
from tests.compass.held_ports import held_ports

LAW = (
    TREE / "scripts/compass/laws/qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json.xz"
)
MACHINE = TREE / "tests/compass/machines/mi308x-hjbog39.json"
#: Qwen3.8-27B's config.json, the one the law's host key names by digest.
CONFIG = TREE / "tests/compass/qwen3_5_27b_config.json"
MODEL = os.environ.get("ATOM_COMPASS_CCTRACES_MODEL")
NEEDS_A_RUN = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL),
    reason="a run needs a driver, as the vertical slice's does, and "
    "ATOM_COMPASS_CCTRACES_MODEL naming a Qwen3.8-27B directory",
)
#: The server's flags past its model, host and port: the law's engine config
#: states graph mode at ATOM's default ladder, so no ``--enforce-eager``.
FLAGS = ["--max-model-len", "4096"]
#: The first turn's prefix words: enough that its request body, the smallest,
#: takes over 3,300 bytes whatever the model's path, past the 2,482 the law's
#: HTTP intake was fitted from at its least (#3932). The later turns are sent
#: at `LATER_S`, past the first one's end under the law.
PREFIX_WORDS = 560
REQUESTS = 4
LATER_S = 2.0


def turns() -> list[dict]:
    """The first request, and each later one its whole prompt and more."""
    first, *rest = bodies(MODEL, REQUESTS, PREFIX_WORDS)
    return [first] + [{**r, "prompt": f"{first['prompt']} {r['prompt']}"} for r in rest]


def law_env(**more) -> dict:
    """This process's environment less every performance variable but the
    device selection, and less each variable a run keys off its default, with
    the law's engine config's environment and `more`: what `keyed_env` then
    reads is the law's."""
    env = {
        k: v
        for k, v in os.environ.items()
        if (k not in PERFORMANCE_KEYS or k.endswith("_VISIBLE_DEVICES"))
        and k not in READ_OFF_DEFAULT
    }
    return env | read_json(LAW)["engine_config"]["env"] | more


# --- the law's engine config, with no driver ----------------------------------

#: In a process whose aiter answers gfx942: each flag set's config, resolved
#: as the server resolves it, held to the law as `run.runner` holds it at
#: install, and to its host laws; its engine config at the law's ladder.
INSTALL = """
import json, sys
from atom.compass.capture.driverless import import_aiter_on
import_aiter_on(sys.argv[1])
from types import SimpleNamespace
from atom.compass import run
from atom.compass.artifacts import machine_platform
from atom.compass.backends.step_grid_plan import at_ladder
from atom.compass.drive import run as drive_run
from atom.compass.run.step_grid import keyed
spec, sets = (json.loads(open(a).read()) for a in sys.argv[2:4])
model = sys.argv[4]
run.spec = lambda: spec
law, platform, out = spec[run.STEP_LINEAR], machine_platform(run._machine(spec)), {}
for name, flags in sets.items():
    config = drive_run._config(["--model", model, *flags])
    out[name] = {"keyed": at_ladder(keyed(config), law["engine_config"])}
    for check, call in (
        ("refused", lambda: run.runner(SimpleNamespace(config=config), None)),
        ("host_refused", lambda: run.refuse_unfitted_host(spec["host"], config, platform)),
    ):
        try:
            call()
        except ValueError as e:
            out[name][check] = str(e)
print(json.dumps(out))
"""


def law_run(machine: dict) -> dict:
    """The run-file keys `compass run --step-linear LAW --host-laws LAW` writes
    on `machine`: the step law and its backend, the host laws and their hops,
    less the router's, which would make the run prefill-decode."""
    law, backend = runfile.resolve(None, compass_run.STEP_LINEAR, str(LAW), None, None)
    host, _ = runfile.resolve(None, "host", str(LAW), None, None)
    hops = {k: v for k, v in host["hops"].items() if k != "router_s"}
    return {
        compass_run.STEP_LINEAR: law,
        "backend": backend,
        "host": host,
        "machine": machine,
        **hops,
    }


def test_the_scenario_s_flags_are_the_law_s_and_eager_ones_are_refused_at_install(
    tmp_path,
):
    """The scenario's flags resolve to the engine config the law states, so
    `run.runner` installs it past its checks to the run file's store, which
    this one names none of, and the law's host laws load under it; with
    ``--enforce-eager`` it refuses, naming the ladder the law was fitted at and
    the empty one an eager engine runs."""
    spec = law_run(json.loads(MACHINE.read_text()))
    law = spec[compass_run.STEP_LINEAR]
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(CONFIG.read_text())
    sets = {"graph": FLAGS, "eager": [*FLAGS, "--enforce-eager"]}
    for name, value in (("spec", spec), ("sets", sets)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    path = os.pathsep.join([str(TREE), str(TREE / "tests/compass")])
    done = subprocess.run(
        [sys.executable, "-c", INSTALL, spec["machine"]["device"]["arch"]]
        + [str(tmp_path / f"{n}.json") for n in ("spec", "sets")]
        + [str(tmp_path / "model")],
        env=law_env(PYTHONPATH=path),
        cwd=TREE,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads(done.stdout.splitlines()[-1])
    print(f"\n{json.dumps(out, indent=1)}")
    assert out["graph"]["keyed"] == law["engine_config"]
    assert out["graph"]["refused"].startswith("the run file names no store")
    assert "host_refused" not in out["graph"]
    ladder = law["engine_config"]["ladder"]
    assert out["eager"]["refused"] == (
        "the step linear law was fitted under another engine config: "
        f"ladder {ladder} in the law and [] here"
    )


# --- the served run -----------------------------------------------------------


class Later(Traffic):
    """The traffic LP: `requests`' first at 0, and the others at `later`."""

    def __init__(self, requests, later, *args) -> None:
        super().__init__(*args)
        self.requests, self.later = requests, later

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            at = self.later if k else 0.0
            while rt.now < at:
                rt.next_event(at)
                self._take()
            if k and 0 in self.open:
                raise AssertionError(f"request 0 was still open at {rt.now}")
            self.sent, seq = rt.stamp_send(self.http)
            self.open.add(k)
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()
        if self.open:
            raise UnansweredRequests(f"requests {sorted(self.open)} never finished")

    def _take(self) -> None:
        super()._take()
        done = sum(line == DONE for _, line in self.events)
        self.open = set(range(done, len(self.requests)))


def run_law(out: Path, seed: str) -> dict:
    """One run under the law in its own process tree: its summary, parity
    record and engine file."""
    out.mkdir()
    machine = json.loads(MACHINE.read_text())
    requests = turns()
    with held_ports(2) as (clock_port, port):
        run_file = _run_file(
            out,
            clock_endpoint=f"tcp://127.0.0.1:{clock_port}",
            step_stub=None,
            **law_run(machine),
        )
        env = law_env(
            PYTHONHASHSEED=seed,
            PYTHONPATH=str(TREE),
            ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
        )
        argv = ["--model", MODEL, "--host", "127.0.0.1", "--server-port", str(port)]
        argv += FLAGS
        publish_entry_on_arch(out / "store", machine, argv, env)
        arch = machine["device"]["arch"]
        with open(out / "server.log", "w") as log:
            server = subprocess.Popen(
                [sys.executable, *ON_ARCH, arch, *SERVER[1:], *argv]
                + ["--compass-run", str(run_file)],
                cwd=TREE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            watchdog = threading.Timer(WALL_S, server.kill)
            watchdog.start()
            try:
                run = json.loads(run_file.read_text())
                traffic = Later(requests, LATER_S, run, port, server)
                traffic.run()
                code = server.wait(timeout=120)
            finally:
                watchdog.cancel()
                if server.poll() is None:
                    server.kill()
    tail = (out / "server.log").read_text()[-4000:]
    assert code == 0, f"the server exited {code} after the finish:\n{tail}"
    return {
        "out": out,
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "events": traffic.events,
        "sent": [body["max_tokens"] for body in requests],
        "parity": read(out / "parity"),
        "engine": json.loads((out / "parity" / "dp0.engine.json").read_text()),
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("step_linear")
    return [run_law(root / f"seed{s}", s) for s in ("1", "2")]


@NEEDS_A_RUN
def test_a_run_under_the_law_meets_every_scenario_check_with_no_refusal(runs):
    """Both seeds' runs meet every scenario check, so no step was refused; the
    engine served the law's engine config in graph mode with prefix caching on,
    recording no prefill token-graph bucket, so the law's prefill vector, which
    states none, is not refused for them; the first turn prefills with no
    cached token and each later one on a prefix in HBM, its first prefill
    row's context less its tokens. Every request body is inside the bytes the
    law's HTTP intake was fitted on, where it states them."""
    run = runs[0]
    law = read_json(LAW)
    engine = run["engine"]
    cached = {}
    for step in run["parity"][0]:
        for key, tokens, context in step["batch"]:
            cached.setdefault(key, context - tokens)
    cached = list(cached.values())
    print(f"\n{run['summary']['schedule']['seconds_by_method']}")
    print(f"cached tokens of each request: {cached}")
    assert read_json(MODEL + "/config.json") == read_json(CONFIG)
    sizes = [len(json.dumps(body)) for body in turns()]
    low, high = law["host"]["http_intake"].get("bytes", (0, math.inf))
    print(f"request body bytes: {sizes}; the intake was fitted on {low} to {high}")
    assert all(low <= n <= high for n in sizes), (sizes, low, high)
    failed = check_run(
        [r["out"] for r in runs], run["sent"], [line for _, line in run["events"]]
    )
    assert not failed, "\n".join(failed)
    assert {k: engine[k] for k in law["engine_config"]} == law["engine_config"]
    assert engine["enable_prefix_caching"]
    assert cached[0] == 0 and all(c > 0 for c in cached[1:]), cached
    assert engine["prefill_buckets"] == []
    assert "buckets" not in law["prefill"]

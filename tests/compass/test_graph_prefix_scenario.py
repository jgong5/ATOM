# SPDX-License-Identifier: MIT
"""Two scenarios on Qwen3-0.6B at TP1, each at a default ATOM serves and every
other scenario leaves: CUDA graph mode, and a prefix hit in HBM.

The graph run starts the server without ``--enforce-eager``, so the engine
schedules the ladder `overrides.capture_cudagraph` states. Ten requests sent at
0 decode at every batch from ten rows down to one, each step at the smallest
rung of that ladder holding its rows, ATOM's default ladder padding ten and
nine rows to 16. Their prompts share less than a KV block.

The prefix run serves eager with prefix caching on, ATOM's default, and its
requests share a prefix of several KV blocks, as a system prompt is shared: the
first at 0, the rest at `LATER_S`, once the first has finished, each prefilling
on the blocks the first left in HBM. Each request's cached tokens are read from
the parity record: a request's first prefill row is (key, tokens scheduled,
context), and its context less its scheduled tokens is the prefix it found.

Each scenario runs twice, with different `PYTHONHASHSEED`, under the constant
step stub (level 0, with the hand-computed timeline) and under the step roofline
alone (level 1), and `check_run` holds the two to the scenario checks. The runs
need a driver and `ATOM_COMPASS_SLICE_MODEL`, as `test_vertical_slice`'s do.
"""

import json
import math
import os
import subprocess
import threading
from pathlib import Path

import pytest
from capabilities import ROOFLINE, ROOFLINE_MACHINE, capability, cost_overrides
from test_memory_readings import DOCUMENT
from test_vertical_slice import (
    CONSTANT,
    MODEL,
    NEEDS_A_RUN,
    TREE,
    WALL_S,
    Traffic,
    UnansweredRequests,
    _run_file,
    publish_entry,
)

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.detect.scenario import check_run
from atom.compass.drive.serve import SERVER
from atom.compass.parity import read
from atom.compass.parity.traffic import DONE, bodies
from tests.compass.held_ports import held_ports

#: The graph run's requests, by their max_tokens, all sent at 0.
GRAPH_TOKENS = tuple(range(2, 12))
#: The prefix every request of the prefix run shares: eleven whole KV blocks of
#: 16 in Qwen3's tokens.
PREFIX_WORDS = 160
PREFIX_REQUESTS = 4
#: When the prefix run's later requests are sent: past the first one's end
#: under either backend (`Later` checks it).
LATER_S = 0.5
EAGER = ["--enforce-eager"]


class Later(Traffic):
    """The traffic LP: `requests`' first at 0, and each other at `later`, or
    every one at 0 when `later` is None."""

    def __init__(self, requests, later, *args) -> None:
        super().__init__(*args)
        self.requests, self.later = requests, later

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            at = self.later if k and self.later is not None else 0.0
            while rt.now < at:
                rt.next_event(at)
                self._take()
            if k and self.later is not None and 0 in self.open:
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


def run_scenario(
    out: Path, seed: str, args, requests, later=None, backend=CONSTANT, machine=DOCUMENT
) -> dict:
    """One run in its own process tree of the server at TP1 with `args`,
    serving `requests` (`Later`), priced by `backend` on `machine`
    (`cost_overrides`): its step table, summary, parity record and engine file."""
    out.mkdir()
    with held_ports(2) as (clock_port, port):
        run_file = _run_file(
            out,
            clock_endpoint=f"tcp://127.0.0.1:{clock_port}",
            **cost_overrides(backend, machine),
        )
        env = dict(
            os.environ,
            PYTHONHASHSEED=seed,
            PYTHONPATH=str(TREE),
            AITER_LOG_LEVEL="WARNING",
            ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
        )
        argv = ["--model", MODEL, "--host", "127.0.0.1", "--server-port", str(port)]
        argv += [*args, "--max-model-len", "2048"]
        publish_entry(out / "store", machine, argv, env)
        with open(out / "server.log", "w") as log:
            server = subprocess.Popen(
                [*SERVER, *argv, "--compass-run", str(run_file)],
                cwd=TREE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            watchdog = threading.Timer(WALL_S, server.kill)
            watchdog.start()
            try:
                run = json.loads(run_file.read_text())
                traffic = Later(requests, later, run, port, server)
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
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "events": traffic.events,
        "sent": [body["max_tokens"] for body in requests],
        "parity": read(out / "parity"),
        "engine": json.loads((out / "parity" / "dp0.engine.json").read_text()),
    }


def _checked(runs, timeline: bool) -> None:
    """Both seeds' `runs` meet every scenario check; the hand-computed timeline
    only with `timeline`, which only a constant stub's run has."""
    run = runs[0]
    print(f"\n{run['summary']['schedule']['seconds_by_method']}")
    failed = check_run(
        [r["out"] for r in runs],
        run["sent"],
        [line for _, line in run["events"]],
        {"engine": run["out"] / "parity"} if timeline else None,
    )
    assert not failed, "\n".join(failed)


# --- CUDA graph mode ----------------------------------------------------------


def _graph_runs(root: Path, backend, machine) -> list[dict]:
    requests = [
        {
            "model": MODEL,
            "prompt": f"request {k} through the graph slice",
            "max_tokens": n,
            "ignore_eos": True,
            "stream": True,
        }
        for k, n in enumerate(GRAPH_TOKENS)
    ]
    return [
        run_scenario(root / f"seed{s}", s, [], requests, None, backend, machine)
        for s in ("1", "2")
    ]


def _each_decode_step_at_its_rung(run) -> None:
    """The engine replays a ladder, and every decode step runs at the smallest
    rung holding its rows; the batch shrinks through every width."""
    ladder = run["engine"]["ladder"]
    steps = [
        (len(s["batch"]), s["rung"])
        for s in run["parity"][0]
        if s["batch"] and all(decode for *_, decode in s["rows"])
    ]
    print(f"ladder {ladder}\ndecode steps (rows, rung): {steps}")
    assert ladder
    assert steps
    for rows, rung in steps:
        assert rung == min(r for r in ladder if r >= rows), (rows, rung)
    assert {rows for rows, _ in steps} == set(range(1, len(GRAPH_TOKENS) + 1))


@pytest.fixture(scope="module")
def graph_runs(tmp_path_factory):
    return _graph_runs(tmp_path_factory.mktemp("graph"), CONSTANT, DOCUMENT)


@pytest.fixture(scope="module")
def graph_roofline_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("graph_roofline")
    return _graph_runs(root, ROOFLINE, ROOFLINE_MACHINE)


@NEEDS_A_RUN
@capability("cuda_graph")
def test_a_graph_run_meets_every_scenario_check_each_decode_step_at_its_rung(
    graph_runs,
):
    _checked(graph_runs, timeline=True)
    _each_decode_step_at_its_rung(graph_runs[0])


@NEEDS_A_RUN
@capability("cuda_graph", 1)
def test_a_graph_run_meets_every_scenario_check_under_the_step_roofline(
    graph_roofline_runs,
):
    """The level-0 case's checks but one: the hand-computed timeline, which
    `check_run` computes only for a constant step stub. Every request arrives
    at 0, so the decode batches and their rungs follow no step cost."""
    _checked(graph_roofline_runs, timeline=False)
    _each_decode_step_at_its_rung(graph_roofline_runs[0])


# --- a prefix hit in HBM ------------------------------------------------------


def _prefix_runs(root: Path, backend, machine) -> list[dict]:
    requests = bodies(MODEL, PREFIX_REQUESTS, PREFIX_WORDS)
    return [
        run_scenario(root / f"seed{s}", s, EAGER, requests, LATER_S, backend, machine)
        for s in ("1", "2")
    ]


def _shared_blocks(requests, block: int) -> int:
    """The tokens of the whole KV blocks every one of `requests`' prompts
    shares with the first, as the model's tokenizer encodes them."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    first, *rest = (tokenizer(body["prompt"])["input_ids"] for body in requests)
    shared = min(
        next((i for i, (a, b) in enumerate(zip(first, ids)) if a != b), len(ids))
        for ids in rest
    )
    return shared // block * block


def _later_requests_prefill_on_the_shared_blocks(run) -> None:
    """The first request prefills with no cached token, each later one on
    every whole block of the prefix it shares with the first, read from the
    parity record: its first prefill row's context less its tokens."""
    cached = {}
    for step in run["parity"][0]:
        for key, tokens, context in step["batch"]:
            cached.setdefault(key, context - tokens)
    cached = list(cached.values())
    block = run["engine"]["kv_cache_block_size"]
    blocks = _shared_blocks(bodies(MODEL, PREFIX_REQUESTS, PREFIX_WORDS), block)
    print(f"cached tokens of each request: {cached}; shared blocks hold {blocks}")
    assert run["engine"]["enable_prefix_caching"]
    assert blocks >= 4 * block
    assert cached == [0] + [blocks] * (PREFIX_REQUESTS - 1)


@pytest.fixture(scope="module")
def prefix_runs(tmp_path_factory):
    return _prefix_runs(tmp_path_factory.mktemp("prefix"), CONSTANT, DOCUMENT)


@pytest.fixture(scope="module")
def prefix_roofline_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("prefix_roofline")
    return _prefix_runs(root, ROOFLINE, ROOFLINE_MACHINE)


@NEEDS_A_RUN
@capability("hbm_prefix_hit")
def test_a_prefix_run_meets_every_scenario_check_later_requests_hitting_in_hbm(
    prefix_runs,
):
    _checked(prefix_runs, timeline=True)
    _later_requests_prefill_on_the_shared_blocks(prefix_runs[0])


@NEEDS_A_RUN
@capability("hbm_prefix_hit", 1)
def test_a_prefix_run_meets_every_scenario_check_under_the_step_roofline(
    prefix_roofline_runs,
):
    """The level-0 case's checks but one: the hand-computed timeline, which
    `check_run` computes only for a constant step stub. The later requests are
    sent once the first has ended (`Later` checks it), so their hits follow no
    step cost."""
    _checked(prefix_roofline_runs, timeline=False)
    _later_requests_prefill_on_the_shared_blocks(prefix_roofline_runs[0])

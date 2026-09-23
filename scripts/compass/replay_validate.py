"""Calibrate, serve real, serve modelled, compare -- on a recorded trace.

`validate.py` cannot do this. It is hardwired to `run.py --num-prompts`: an
offline workload, every request arriving at once, no HTTP and so no arrival
process at all. A cc-traces replay is the opposite on all three counts, and the
arrival process is not decoration -- burstiness is what decides whether requests
batch together, and batching is most of what the cost model is being asked to
predict.

Five phases, a process each:

    calibrate  a sweep of shapes, real forward, recording how long each took
    real       the trace, --pace, against the real forward, recording steps
    modelled   the same trace, declared arrivals, forward replaced by the fit
    coverage   did the run's steps leave the table? gate before believing it
    compare    per-request TTFT and TPOT, real against modelled

The two serving phases differ in how arrivals are delivered, and they have to.
A real engine is on a real clock and discards a declared arrival, so its side
needs `--pace` -- the queue is then genuinely empty between arrivals. A
simulated engine advances a virtual clock by predicted step costs, so pacing it
against a wall clock makes the two clocks race and the simulated run performs a
different set of steps from the one it stands for. Same trace, same prompts,
same scheduler; only the forward and the clock differ.

The real side is served under `--compass-mode=measure`, which performs the real
forward and records each step. Three things follow, all wanted. Its timings stay
truthful -- `CompassConfig.__post_init__` turns the virtual clock off for any
mode but `predict`, so a wall clock stamps the readings. The arrival barrier
stays out of the way for the same reason: it opens immediately on a real clock,
so `--pace` really does pace rather than dead-lock against a 120s hold. And the
recorded steps are what `coverage.py` needs -- without a run's step table it can
only describe the sweep, never say whether the run left it.

    python scripts/compass/replay_validate.py --model M --trace trace.jsonl \\
        --out-dir out/cc --max-model-len 262144
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from atom.compass.core.artifacts import rank_path

#: Prefix every Compass warning carries. Matched on the message because ATOM's
#: log format never names the level, so filtering on "WARNING" as a field
#: matches nothing.
MARKER = "ATOMCompass WARNING:"


def _gate_path(path: Path) -> Path:
    """The file the coverage gate should read, given the name the run asked for.

    Under TP>1 every rank writes its own shard, so the unsuffixed name never
    appears on disk and the gate reported coverage as unknown -- which withholds
    the verdict on every rung of a TP2 or TP4 sweep while the step table sits
    beside it under another name. Rank 0 answers the question the gate asks: the
    ranks of a symmetric TP group run the same batches, and it is batch shapes,
    not per-rank timings, that decide whether a run left the table.
    """
    if path.exists():
        return path
    shard = Path(rank_path(str(path), {"tp": 0}))
    return shard if shard.exists() else path


def _free_port() -> int:
    """Let the OS pick. Host networking is shared with about twenty containers,
    so a fixed number collides with whoever holds it, including an earlier run
    of this script."""
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _echo_warnings(text: str) -> None:
    """Surface what Compass said, once each.

    A run emits the same extrapolation warning per step, so echoing every line
    buries the one that is new under ten thousand that are not.
    """
    seen = set()
    for line in text.splitlines():
        if MARKER in line:
            message = line[line.index(MARKER):]
            if message not in seen:
                seen.add(message)
                print(f"  ! {message}", flush=True)


def _run(cmd, label):
    """One phase, captured, with anything Compass wanted to say about it.

    Captured because an engine start-up is thousands of lines and none of them
    are the point -- but the oracle's extrapolation warnings *are* the point,
    and swallowing them leaves the one safeguard inert in the only workflow
    anybody uses.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    _echo_warnings(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"{label} failed ({proc.returncode})")
    return proc


class Served:
    """A server for the duration of a `with` block, and its log either way.

    Context-managed because the failure that matters is the one where the client
    raises: without this the engine keeps the GPU, and the next phase fails to
    allocate for a reason that looks nothing like the real one.
    """

    def __init__(self, python, model, log, extra, startup_timeout, attempts=3):
        self.python, self.model = python, model
        self.log, self.extra = Path(log), list(extra)
        self.startup_timeout, self.attempts = startup_timeout, attempts
        # Picked per attempt in `__enter__`, so read `.port` off the instance
        # rather than picking one at the call site.
        self.port = None
        self.proc = self.handle = None

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def _tail(self, n=25) -> str:
        return "\n".join(
            self.log.read_text(errors="replace").splitlines()[-n:])

    def _spawn(self, append):
        cmd = [self.python, "-m", "atom.entrypoints.openai_server",
               "--model", self.model, "--server-port", str(self.port),
               *self.extra]
        self.handle = self.log.open("a" if append else "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.handle,
                                     stderr=subprocess.STDOUT)

    def _await_health(self):
        """None once it answers /health, else why it never did."""
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return "server died during startup"
            if self._healthy():
                return None
            time.sleep(1.0)
        return (f"server never became healthy in "
                f"{self.startup_timeout:.0f}s")

    def __enter__(self):
        # `_free_port` picks a port, closes it, and the child binds it only
        # after loading the model -- about twenty containers share this host's
        # network, and that window is wide enough to lose the port to one of
        # them. sweep81_v2 lost it: the modelled c1 server loaded the 27B,
        # reached uvicorn, found its port taken, and took the whole sweep down
        # with it 34 minutes in. Losing that race says nothing about the run,
        # so pick another port and load again.
        for attempt in range(1, self.attempts + 1):
            self.port = _free_port()
            self._spawn(append=attempt > 1)
            why = self._await_health()
            if why is None:
                print(f"  server up on port {self.port}", flush=True)
                return self
            taken = "address already in use" in self._tail(60).lower()
            self.__exit__(None, None, None)
            if taken and attempt < self.attempts:
                print(f"  port {self.port} was taken between pick and bind; "
                      f"retrying on a new one "
                      f"({attempt}/{self.attempts - 1})", flush=True)
                continue
            raise SystemExit(f"{why}; see {self.log}\n" + self._tail())

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.handle:
            self.handle.close()
            self.handle = None
        # Warnings the engine emitted go to its own log, so they would never
        # reach a reader of this script's output otherwise.
        if self.log.exists():
            _echo_warnings(self.log.read_text(errors="replace"))
        return False


def _engine_flags(args):
    """Settings both sides share. Anything here that differed between them
    would come out as model error."""
    flags = ["-tp", str(args.tp), "--block-size", str(args.block_size)]
    for name, value in (("--max-model-len", args.max_model_len),
                        ("--max-num-seqs", args.max_num_seqs),
                        ("--max-num-batched-tokens", args.max_num_batched_tokens)):
        if value:
            flags += [name, str(value)]
    return flags


#: `--compass-oracle` resolves a qualname and nothing else -- there is no
#: short-name registry behind it -- so these live here, for the command line,
#: rather than in the engine's argument parser.
ORACLES = {
    "calibrated": "atom.compass.core.cost.calibrated.CalibratedCostOracle",
    "priced": "atom.compass.core.cost.priced.PricedGraphCostOracle",
    "interpolated": "atom.compass.core.cost.interpolated.InterpolatedCostOracle",
    "constant": "atom.compass.core.cost.constant.ConstantCostOracle",
}

#: Which constructor keyword each oracle takes the step table through. An
#: oracle absent from here is handed no table at all: ConstantCostOracle
#: costs every step at two fixed rates and raises on an unexpected keyword,
#: and a qualname we have never seen could do either, so it has to say.
TABLE_KEY = {
    ORACLES["calibrated"]: "table",
    ORACLES["interpolated"]: "table",
    ORACLES["priced"]: "fallback",
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle", default="calibrated",
                   help=f"what the modelled side costs its steps with: one of "
                        f"{', '.join(sorted(ORACLES))}, or a qualname. "
                        f"Default calibrated, which is what every rung so far "
                        f"was measured under. A qualname, and constant, are "
                        f"handed no step table unless you name one with "
                        f"--oracle-option.")
    p.add_argument("--oracle-option", action="append", default=[],
                   metavar="K=V",
                   help="a keyword argument for the oracle's constructor, "
                        "repeatable. The calibrated oracle is given "
                        "table=<--table> and the priced one fallback=<--table> "
                        "unless you pass those yourself. Priced also needs "
                        "prices=, graph= and prefill_graph=; it holds decode "
                        "graphs at one batch rung per captured shape, so a "
                        "rung it has no graph for is answered by another rung "
                        "and reads low -- the server log says which.")
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True,
                   help="JSONL from scripts/compass/cc_traces.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("-tp", "--tp", type=int, default=1)
    p.add_argument("--block-size", type=int, default=16,
                   help="the engine's KV block, NOT the corpus's 64-token hash "
                        "block. 64 is a multiple of 16, so shared source blocks "
                        "are whole native blocks either way, and moving this "
                        "invalidates every calibration measured at 16")
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--table", default=None,
                   help="an existing calibration table; skips phase 1")
    p.add_argument("--sweep-long-decode", type=int, default=64)
    p.add_argument("--admission-seconds", type=float, default=None,
                   help="simulated cost of becoming schedulable. validate.py "
                        "derives this from a fixed offline workload, which a "
                        "trace is not, so state it or leave it unmodelled")
    p.add_argument("--startup-timeout", type=float, default=900.0,
                   help="a 27B at 262144 context takes minutes to allocate")
    p.add_argument("--request-timeout", type=float, default=3600.0)
    p.add_argument("--skip-real", action="store_true",
                   help="reuse out-dir/real.json from an earlier run. The real "
                        "side is paced, so it costs the trace's own span in "
                        "wall clock and is worth not repeating")
    p.add_argument("--clients", type=int, default=0,
                   help="replay closed-loop as this many users instead of on "
                        "the trace's own arrivals. One point on a saturation "
                        "curve; see replay.py --clients")
    p.add_argument("--sessions-per-client", type=int, default=0,
                   help="with --clients, a cap on the instances one lane may "
                        "get through. 0 -- the default -- means the clock "
                        "decides, which is what --benchmark-duration is for")
    p.add_argument("--benchmark-duration", type=float, default=0.0,
                   help="seconds the real side keeps its lanes recycling. A "
                        "lane finishes a session instance and immediately "
                        "starts another until the clock says stop, so the "
                        "rung is bounded by time rather than by how many "
                        "sessions happened to be extracted. The modelled side "
                        "does not recycle -- it replays the schedule the real "
                        "side executed")
    p.add_argument("--startup-sampling", choices=("uniform", "none"),
                   default="uniform",
                   help="where in its recording a lane's *first* session "
                        "joins. Uniform picks an instant t* and sends the one "
                        "turn before it unmeasured, so the rung does not open "
                        "with every lane cold at turn 0")
    p.add_argument("--python", default=sys.executable)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    work = Path(args.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    table = Path(args.table) if args.table else work / "steps.jsonl"
    real_steps = work / "real_steps.jsonl"
    modelled_steps = work / "modelled_steps.jsonl"
    real_out, modelled_out = work / "real.json", work / "modelled.json"
    report = work / "compare.json"

    if args.table:
        print(f"phase 1/5  skipped; calibrated from {table}", flush=True)
    else:
        print("phase 1/5  calibrating on a sweep of shapes ...", flush=True)
        _run([args.python, "scripts/compass/run.py", "--model", args.model,
              *_engine_flags(args), "--out", str(work / "sweep.json"),
              "--sweep", "--sweep-long",
              "--sweep-long-decode", str(args.sweep_long_decode),
              "--compass", "--compass-mode", "measure",
              "--compass-measure-out", str(table),
              # One of each kind: drops the launch that autotunes, keeps every
              # other sample the sweep produced.
              "--compass-measure-warmup-steps", "1"], "calibration")

    replay = [args.python, "scripts/compass/replay.py", "--trace", args.trace,
              "--model", args.model, "--ignore-eos", "--check-lengths",
              "--timeout", str(args.request_timeout)]
    # The client count goes to both sides -- it is the rung's name, and
    # `saturation.py` reads it off each manifest. What differs is how the two
    # sides get their work: the real side recycles lanes against a clock, and
    # the modelled side replays the schedule that produced, because the
    # engine's arrival barrier holds every declared request until all of them
    # have arrived and so cannot be handed a graph that grows as it runs.
    real_only, modelled_only = [], []
    if args.clients:
        replay += ["--clients", str(args.clients)]
        if args.sessions_per_client:
            replay += ["--sessions-per-client", str(args.sessions_per_client)]
        real_only += ["--startup-sampling", args.startup_sampling]
        if args.benchmark_duration:
            real_only += ["--benchmark-duration", str(args.benchmark_duration)]
        modelled_only += ["--schedule", str(work / "real.json")]
    # --pace on the real side in *both* modes. Open loop it sleeps until each
    # recorded arrival; closed loop it sleeps each session's think time and
    # drives the next turn from the previous response. Without it the real side
    # would post a *declared* graph to an engine on a real clock, which cannot
    # honour it -- the whole trace would land as one burst while the modelled
    # side answered the recorded timeline, and the gap would be reported as
    # model error.
    real_arrivals = ["--pace"]

    if args.skip_real and real_out.exists():
        print(f"phase 2/5  skipped; reusing {real_out}", flush=True)
    else:
        print("phase 2/5  replaying the trace against the real engine ...",
              flush=True)
        real_flags = _engine_flags(args) + [
            # Real forward, real clock, steps recorded for the coverage gate.
            "--compass", "--compass-mode", "measure",
            "--compass-measure-out", str(real_steps),
        ]
        with Served(args.python, args.model, work / "real_server.log",
                    real_flags, args.startup_timeout) as srv:
            # --pace: a real clock discards a declared arrival, so without this
            # the real side answers a burst while the modelled side answers the
            # trace, and the difference comes out reported as model error.
            _run(replay + ["--port", str(srv.port), "--out", str(real_out),
                           *real_arrivals, *real_only], "real replay")

    # The step table is the default for whichever knob the chosen oracle reads
    # it through: the calibrated oracle fits it, the priced one falls back to
    # it for shapes no graph covers. Naming it in only one of those two places
    # would make `--oracle priced` silently drop the fallback, and a priced
    # run with no fallback answers an uncovered shape with a floor rather than
    # with a cost.
    oracle = ORACLES.get(args.oracle, args.oracle)
    oracle_options = list(args.oracle_option)
    default_key = TABLE_KEY.get(oracle)
    if default_key and not any(o.split("=", 1)[0] == default_key
                               for o in oracle_options):
        oracle_options.append(f"{default_key}={table}")
    print(f"phase 3/5  replaying it modelled, oracle={oracle} "
          f"{' '.join(oracle_options)} ...", flush=True)
    modelled_flags = _engine_flags(args) + [
        "--compass",
        "--compass-oracle", oracle,
        *[flag for opt in oracle_options
          for flag in ("--compass-oracle-option", opt)],
        # The predict path already records every step it ran, in the format
        # the measure path records every step it timed -- runner.py
        # `_record_measurement` is called from both. It just never got a path.
        # Without one a simulated run leaves no evidence of *which* steps it
        # chose, so a throughput gap cannot be told apart from a wrong step
        # cost, which is where the c8 diagnosis ran out of evidence.
        "--compass-measure-out", str(modelled_steps),
    ]
    if args.admission_seconds:
        modelled_flags += ["--compass-admission-seconds",
                           str(args.admission_seconds)]
    with Served(args.python, args.model, work / "modelled_server.log",
                modelled_flags, args.startup_timeout) as srv:
        # No --pace: arrivals are declared, so delivery order and socket
        # latency stop mattering against a virtual clock. `--schedule` hands
        # this side the lanes, instances and edges the real side executed, so
        # the two runs are the same graph even though only one of them could
        # have discovered it.
        _run(replay + ["--port", str(srv.port), "--out", str(modelled_out),
                       *modelled_only], "modelled replay")

    print("phase 4/5  coverage ...", flush=True)
    covered = True
    gate_table, gate_steps = _gate_path(table), _gate_path(real_steps)
    if gate_steps.exists():
        coverage = subprocess.run(
            [args.python, "scripts/compass/coverage.py",
             str(gate_table), str(gate_steps)],
            capture_output=True, text=True)
        (work / "coverage.txt").write_text(coverage.stdout + coverage.stderr)
        print(coverage.stdout.rstrip() or "  (no output)", flush=True)
        covered = coverage.returncode == 0
        if not covered:
            print("  coverage gate FAILED: the run left the calibrated table",
                  file=sys.stderr)
    else:
        # Says so rather than passing quietly: a skipped real phase leaves no
        # step table, and "no gate ran" must not read as "the gate passed".
        covered = False
        print(f"  no run step table at {gate_steps}, so coverage is unknown; "
              f"re-run phase 2 to gate on it", file=sys.stderr)

    print("phase 5/5  comparing ...", flush=True)
    # --allow-blocking so the report is written either way. The verdict is
    # withheld below; producing no report at all would mean re-running two
    # servers to find out why.
    compare = subprocess.run(
        [args.python, "scripts/compass/replay_compare.py",
         "--real", str(real_out), "--modelled", str(modelled_out),
         "--out", str(report), "--allow-blocking"],
        capture_output=True, text=True)
    print(compare.stdout.rstrip(), flush=True)
    sys.stderr.write(compare.stderr)
    if not report.exists():
        raise SystemExit(f"comparison wrote no report ({compare.returncode})")

    blocking = list(json.loads(report.read_text()).get("blocking") or [])
    if not covered:
        blocking.append("coverage did not pass; see coverage.txt")
    if blocking:
        print(f"\nverdict WITHHELD, {len(blocking)} reason(s):", file=sys.stderr)
        for reason in blocking:
            print(f"  - {reason}", file=sys.stderr)
        print(f"report: {report}", file=sys.stderr)
        return 1
    print(f"\nverdict reportable -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

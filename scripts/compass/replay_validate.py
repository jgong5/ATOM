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

#: Prefix every Compass warning carries. Matched on the message because ATOM's
#: log format never names the level, so filtering on "WARNING" as a field
#: matches nothing.
MARKER = "ATOMCompass WARNING:"


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

    def __init__(self, python, model, port, log, extra, startup_timeout):
        self.python, self.model, self.port = python, model, port
        self.log, self.extra = Path(log), list(extra)
        self.startup_timeout = startup_timeout
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

    def __enter__(self):
        cmd = [self.python, "-m", "atom.entrypoints.openai_server",
               "--model", self.model, "--server-port", str(self.port),
               *self.extra]
        self.handle = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.handle,
                                     stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.__exit__(None, None, None)
                raise SystemExit(f"server died during startup; see {self.log}\n"
                                 + self._tail())
            if self._healthy():
                print(f"  server up on port {self.port}", flush=True)
                return self
            time.sleep(1.0)
        self.__exit__(None, None, None)
        raise SystemExit(f"server never became healthy in "
                         f"{self.startup_timeout:.0f}s; see {self.log}\n"
                         + self._tail())

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


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
                   help="with --clients, sessions each user works through")
    p.add_argument("--python", default=sys.executable)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    work = Path(args.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    table = Path(args.table) if args.table else work / "steps.jsonl"
    real_steps = work / "real_steps.jsonl"
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
    # Closed loop replaces the arrival process on *both* sides, so pacing and
    # declared arrivals both fall away: there is nothing to pace to and nothing
    # to declare. Giving both sides the same client flags is what keeps the two
    # runs the same experiment.
    if args.clients:
        replay += ["--clients", str(args.clients)]
        if args.sessions_per_client:
            replay += ["--sessions-per-client", str(args.sessions_per_client)]
    real_arrivals = [] if args.clients else ["--pace"]

    if args.skip_real and real_out.exists():
        print(f"phase 2/5  skipped; reusing {real_out}", flush=True)
    else:
        print("phase 2/5  replaying the trace against the real engine ...",
              flush=True)
        port = _free_port()
        real_flags = _engine_flags(args) + [
            # Real forward, real clock, steps recorded for the coverage gate.
            "--compass", "--compass-mode", "measure",
            "--compass-measure-out", str(real_steps),
        ]
        with Served(args.python, args.model, port, work / "real_server.log",
                    real_flags, args.startup_timeout):
            # --pace: a real clock discards a declared arrival, so without this
            # the real side answers a burst while the modelled side answers the
            # trace, and the difference comes out reported as model error.
            _run(replay + ["--port", str(port), "--out", str(real_out),
                           *real_arrivals], "real replay")

    print("phase 3/5  replaying it modelled ...", flush=True)
    port = _free_port()
    modelled_flags = _engine_flags(args) + [
        "--compass",
        "--compass-oracle",
        "atom.compass.core.cost.calibrated.CalibratedCostOracle",
        "--compass-oracle-option", f"table={table}",
    ]
    if args.admission_seconds:
        modelled_flags += ["--compass-admission-seconds",
                           str(args.admission_seconds)]
    with Served(args.python, args.model, port, work / "modelled_server.log",
                modelled_flags, args.startup_timeout):
        # No --pace: arrivals are declared, so delivery order and socket
        # latency stop mattering against a virtual clock.
        _run(replay + ["--port", str(port), "--out", str(modelled_out)],
             "modelled replay")

    print("phase 4/5  coverage ...", flush=True)
    covered = True
    if real_steps.exists():
        coverage = subprocess.run(
            [args.python, "scripts/compass/coverage.py",
             str(table), str(real_steps)],
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
        print(f"  no run step table at {real_steps}, so coverage is unknown; "
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

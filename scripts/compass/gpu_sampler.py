"""Watch the cards for the whole of a run, so `isolation.py` has something to read.

`isolation.py` decides whether a measurement was alone on its devices, and it
reads `gpu.jsonl` to do it. Nothing in this repository wrote that file: the
audit existed and its input did not, which meant every cell either went
unwatched or was watched by a shell loop that lived in somebody's scrollback.

This is that file's writer, and it exists because the audit's two findings need
things a pair of snapshots cannot give:

* a **baseline** sample, taken before this run's own server starts -- the only
  sample that can show a card was *already* somebody else's, because after our
  server is up every byte on our cards is ours;
* **samples throughout**, because a neighbour who arrives for the middle nine
  minutes and leaves is invisible to a before/after pair.

So a run starts this once, before its first server, and stops it after its last
one. The phase file is how a long-lived sampler learns what is happening around
it: the harness writes `{"phase": ..., "own_pids": [...]}` and every subsequent
sample carries it, so `baseline` means what it says and our own server's pids
are never counted as a foreign tenant.

    python scripts/compass/gpu_sampler.py gpu.jsonl --once --phase baseline
    python scripts/compass/gpu_sampler.py gpu.jsonl --interval 5 \
        --phase-file phase.json --duration 7200

What it does not do is decide anything. Every reading is written as
`rocm-smi` gave it, including the readings that failed; `isolation.py` is what
reads them, and keeping the judgement there means a sampler bug cannot quietly
turn a busy node into a clean one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

#: What is asked of `rocm-smi` for a card reading. `--showmeminfo vram` is the
#: one that carries absolute bytes; `isolation.py` prefers them because a
#: percent is rounded to the integer and a 0.7 GiB neighbour on a 192 GiB card
#: reports 0%.
SMI_CARDS = ("--showuse", "--showmemuse", "--showmeminfo", "vram", "--json")

#: And for the node-level process list. Inside a container this cannot map a
#: process to a card, which is why `isolation.py` reads it as "was anyone else
#: on the box" and never as "was anyone else on card 0".
SMI_PIDS = ("--showpids", "--json")

#: Environment variables that say which cards this process may use. The first
#: one set wins, and `all` is written when none is: an unset mask means every
#: card is ours, which is what a single-tenant node run looks like.
VISIBILITY_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

#: How long one `rocm-smi` call may take before the sample is recorded as a
#: failed reading. A sampler that blocks is worse than one that misses a
#: reading: the run it was watching finishes unwatched either way, and a
#: hanging subprocess also holds the file.
SMI_TIMEOUT = 30.0


def visible() -> str:
    """Which cards this process was given, in `isolation.py`'s own spelling."""
    for name in VISIBILITY_VARS:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return "all"


def _smi_json(smi: str, arguments, timeout: float = SMI_TIMEOUT):
    """One `rocm-smi --json` call, as a dict, or the reason there is none."""
    try:
        done = subprocess.run(
            [smi, *arguments],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if done.returncode != 0:
        tail = (done.stderr or b"").decode("utf-8", "replace").strip()[-200:]
        return None, f"rocm-smi exited {done.returncode}: {tail}"
    try:
        return json.loads(done.stdout.decode("utf-8", "replace")), None
    except json.JSONDecodeError as exc:
        # A truncated blob is one lost reading, not a lost run.
        return None, f"unparsable rocm-smi output: {exc}"


def _phase(path: str | None, fallback: str, own_pids: list) -> tuple:
    """What the run says is happening, read fresh for every sample.

    Re-read each tick rather than cached: the point of the file is that a
    sampler started before the first server can be told, later, that the
    baseline is over. A file that is missing or malformed leaves the sampler on
    its command-line phase rather than stopping it.
    """
    if not path:
        return fallback, own_pids, None
    try:
        blob = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return fallback, own_pids, f"{type(exc).__name__}: {exc}"
    phase = blob.get("phase") or fallback
    pids = blob.get("own_pids")
    if isinstance(pids, list):
        own = [str(p) for p in pids]
    else:
        own = own_pids
    return str(phase), own, None


def sample(
    smi: str, phase: str, own_pids: list, *, at: float, note: str | None = None
) -> dict:
    """One reading, in the shape `isolation.py` parses.

    `t` is the wall clock, because the audit's span is reported to a person
    reading it beside a run's own log; the judgement itself is about card
    states and never about the interval between samples.
    """
    cards, cards_error = _smi_json(smi, SMI_CARDS)
    pids, pids_error = _smi_json(smi, SMI_PIDS)
    row = {
        "t": round(at, 3),
        "phase": phase,
        "visible": visible(),
        "own_pids": [str(p) for p in own_pids],
        "smi": cards or {},
        "pids": pids or {},
    }
    errors = [e for e in (cards_error, pids_error) if e]
    if errors:
        # Kept in the sample rather than dropped. A file of readings that all
        # failed must not read as a quiet node, and `isolation.py` sees a
        # sample with no cards in it, which is not a sample saying idle.
        row["error"] = "; ".join(errors)
    if note:
        row["note"] = note
    return row


class Sampler:
    """The loop, with its clock and its writer supplied.

    Injected rather than imported so the tests can run a hundred samples
    without a device, a subprocess or a second of sleeping. The loop's
    behaviour -- when it stops, what it stamps, what it does when `rocm-smi`
    fails -- is what the tests are about.
    """

    def __init__(
        self,
        out,
        *,
        smi: str = "rocm-smi",
        interval: float = 5.0,
        phase: str = "run",
        phase_file: str | None = None,
        own_pids=(),
        now=time.monotonic,
        wall=time.time,
        sleep=time.sleep,
    ):
        self.out = out
        self.smi = smi
        self.interval = max(0.0, float(interval))
        self.phase = phase
        self.phase_file = phase_file
        self.own_pids = [str(p) for p in own_pids]
        self.now = now
        self.wall = wall
        self.sleep = sleep
        self.stopping = False
        self.written = 0
        self.failed = 0

    def stop(self, *_args) -> None:
        """Finish the sample in hand and leave. Bound to SIGTERM and SIGINT."""
        self.stopping = True

    def one(self, handle) -> dict:
        phase, own, phase_error = _phase(self.phase_file, self.phase, self.own_pids)
        self.phase, self.own_pids = phase, own
        row = sample(self.smi, phase, own, at=self.wall(), note=phase_error)
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        self.written += 1
        if row.get("error"):
            self.failed += 1
        return row

    def run(self, duration: float | None = None, limit: int | None = None) -> int:
        """Sample until told to stop, or until `duration` or `limit` runs out.

        Appends. A cell's real side is three server lifetimes and the audit is
        over the whole window, so a sampler restarted in the middle of it must
        not truncate what the first one saw.
        """
        began = self.now()
        with open(self.out, "a", encoding="utf-8") as handle:
            while not self.stopping:
                self.one(handle)
                if limit is not None and self.written >= limit:
                    break
                if duration is not None and self.now() - began >= duration:
                    break
                if self.interval:
                    self.sleep(self.interval)
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="the gpu.jsonl to append to")
    ap.add_argument("--smi", default="rocm-smi", help="the rocm-smi to call")
    ap.add_argument(
        "--interval", type=float, default=5.0, help="seconds between samples"
    )
    ap.add_argument(
        "--phase",
        default="run",
        help="what to stamp samples with when no phase file says "
        "otherwise; 'baseline' is the one isolation.py reads "
        "as predating this run's own server",
    )
    ap.add_argument(
        "--phase-file",
        default=None,
        help="JSON of {phase, own_pids}, re-read every sample",
    )
    ap.add_argument(
        "--own-pid",
        action="append",
        default=[],
        help="a pid of ours, so it is not read as a tenant's; "
        "repeatable, and superseded by the phase file",
    )
    ap.add_argument(
        "--duration", type=float, default=None, help="stop after this many seconds"
    )
    ap.add_argument(
        "--samples", type=int, default=None, help="stop after this many samples"
    )
    ap.add_argument(
        "--once", action="store_true", help="take exactly one sample and leave"
    )
    args = ap.parse_args(argv)

    sampler = Sampler(
        args.out,
        smi=args.smi,
        interval=args.interval,
        phase=args.phase,
        phase_file=args.phase_file,
        own_pids=args.own_pid,
    )
    if not args.once:
        # A run harness stops this with a signal; the sample in hand is
        # finished and the file is closed, rather than the last line being
        # half a JSON object the audit then skips.
        for name in (signal.SIGTERM, signal.SIGINT):
            signal.signal(name, sampler.stop)
    limit = 1 if args.once else args.samples
    code = sampler.run(duration=None if args.once else args.duration, limit=limit)
    print(
        f"  {sampler.written} sample(s) -> {args.out}"
        + (f", {sampler.failed} with no reading" if sampler.failed else ""),
        file=sys.stderr,
    )
    if sampler.written and sampler.failed == sampler.written:
        # Every reading failed: the file exists and says nothing. Reported
        # here, because the audit would otherwise call it an unwatched run
        # long after the node was handed back.
        print("  rocm-smi never answered: this run is unwatched", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())

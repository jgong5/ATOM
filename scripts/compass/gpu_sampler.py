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

PROC_ROOT = Path("/proc")
KFD_PROC_ROOT = Path("/sys/class/kfd/kfd/proc")
EMPTY_PID_WARNING = b"WARNING: No JSON data to report\n"


def visible() -> str:
    """Which cards this process was given, in `isolation.py`'s own spelling."""
    for name in VISIBILITY_VARS:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return "all"


def _kfd_pid_snapshot() -> dict:
    observation = {"path": str(KFD_PROC_ROOT), "observed_at": time.time()}
    try:
        observation["entries"] = sorted(path.name for path in KFD_PROC_ROOT.iterdir())
    except OSError as exc:
        observation["error"] = f"{type(exc).__name__}: {exc}"
    return observation


def _smi_json(smi: str, arguments, timeout: float = SMI_TIMEOUT, *, evidence=None):
    """One `rocm-smi --json` call, as a dict, or the reason there is none."""
    pid_query = tuple(arguments) == SMI_PIDS
    before = _kfd_pid_snapshot() if pid_query else None
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
    if pid_query and done.stdout == b"" and done.stderr == EMPTY_PID_WARNING:
        # ROCm 7.2 emits no JSON for an empty PID list, but can also reset an
        # enumeration error to exit 0. The CLI warning alone is not evidence
        # that no process exists: require independent empty kernel snapshots.
        after = _kfd_pid_snapshot()
        verified = all(snapshot.get("entries") == [] and not snapshot.get("error")
                       for snapshot in (before, after))
        if evidence is not None:
            evidence.update({
                "basis": "sysfs_kfd_empty_bracket",
                "command": [smi, *arguments], "returncode": done.returncode,
                "stdout": "", "stderr": done.stderr.decode("ascii"),
                "before": before, "after": after, "verified_empty": verified,
            })
        if verified:
            return {}, None
        return None, "rocm-smi emitted no PID JSON; empty KFD state was not independently verified"
    try:
        return json.loads(done.stdout.decode("utf-8", "replace")), None
    except json.JSONDecodeError as exc:
        # A truncated blob is one lost reading, not a lost run.
        return None, f"unparsable rocm-smi output: {exc}"


def _process_identity(pid):
    """Parent and start time in the sampler's PID namespace, or no witness."""
    try:
        raw = (PROC_ROOT / str(pid) / "stat").read_text()
        fields = raw[raw.rindex(")") + 1:].split()
        return int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _owned_processes(roots) -> list[dict]:
    """Translate witnessed server descendants to the kernel PIDs SMI reports.

    A container PID is not a ROCm PID. Linux's /proc/<pid>/sched header names
    the kernel task even when stat and children are viewed in a container PID
    namespace. Keep that mapping beside the observation; an unreadable mapping
    stays unclassified, and a new SMI PID is never evidence of ownership.
    """
    if not roots:
        return []
    processes = {}
    for path in PROC_ROOT.iterdir():
        if path.name.isdigit():
            identity = _process_identity(int(path.name))
            if identity is not None:
                processes[int(path.name)] = identity
    owned = {}
    for root in roots or []:
        pid, ticks = root.get("pid"), root.get("start_ticks")
        if pid in processes and ticks is not None and processes[pid][1] == ticks:
            owned[pid] = {"root_pid": pid, "root_start_ticks": ticks}
    changed = True
    while changed:
        changed = False
        for pid, (parent, ticks) in processes.items():
            if (pid not in owned and parent in owned
                    and ticks >= processes[parent][1]):
                owned[pid] = owned[parent]
                changed = True
    witnessed = []
    for pid in sorted(owned):
        try:
            header = (PROC_ROOT / str(pid) / "sched").read_text().splitlines()[0]
            kernel_pid = int(header.rsplit(" (", 1)[1].split(", #threads:", 1)[0])
        except (OSError, ValueError, IndexError):
            continue
        # Every link must still identify the process whose ancestry we
        # walked. Reusing a parent PID must not confer ownership on a later
        # process or on that later process's children.
        chain = [pid]
        while chain[-1] != owned[pid]["root_pid"]:
            chain.append(processes[chain[-1]][0])
        if kernel_pid <= 0 or any(_process_identity(p) != processes[p] for p in chain):
            continue
        parent, ticks = processes[pid]
        witnessed.append({"pid": pid, "ppid": parent, "start_ticks": ticks,
                          "kernel_pid": kernel_pid, **owned[pid],
                          "kernel_pid_source": "/proc/<pid>/sched"})
    return witnessed


def _phase(path: str | None, fallback: str, own_pids: list) -> tuple:
    """What the run says is happening, read fresh for every sample.

    Re-read each tick rather than cached: the point of the file is that a
    sampler started before the first server can be told, later, that the
    baseline is over. A file that is missing or malformed leaves the sampler on
    its command-line phase rather than stopping it.
    """
    if not path:
        return fallback, own_pids, None, None
    try:
        blob = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return fallback, own_pids, f"{type(exc).__name__}: {exc}", None
    phase = blob.get("phase") or fallback
    pids = blob.get("own_pids")
    if isinstance(pids, list):
        own = [str(p) for p in pids]
    else:
        own = own_pids
    witnessed = None
    if "process_roots" in blob:
        witnessed = _owned_processes(blob["process_roots"])
        own = sorted(set(own) | {str(p["kernel_pid"]) for p in witnessed})
    return str(phase), own, None, witnessed


def sample(
    smi: str, phase: str, own_pids: list, *, at: float, note: str | None = None
) -> dict:
    """One reading, in the shape `isolation.py` parses.

    `t` is the wall clock, because the audit's span is reported to a person
    reading it beside a run's own log; the judgement itself is about card
    states and never about the interval between samples.
    """
    cards, cards_error = _smi_json(smi, SMI_CARDS)
    pid_evidence = {}
    pids, pids_error = _smi_json(smi, SMI_PIDS, evidence=pid_evidence)
    row = {
        "t": at,
        "phase": phase,
        "visible": visible(),
        "own_pids": [str(p) for p in own_pids],
        "smi": cards or {},
        "pids": pids or {},
    }
    if pid_evidence:
        row["pid_observation"] = pid_evidence
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
        """Finish in-flight work, then take a closing sample. Bound to signals."""
        self.stopping = True

    def one(self, handle) -> dict:
        phase, own, phase_error, witnessed = _phase(
            self.phase_file, self.phase, self.own_pids)
        self.phase, self.own_pids = phase, own
        row = sample(self.smi, phase, own, at=self.wall(), note=phase_error)
        if witnessed is not None:
            row["own_processes"] = witnessed
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
                if self.interval and not self.stopping:
                    self.sleep(self.interval)
            if self.stopping:
                # The harness signals only after the last server exits. A
                # sample started before that signal cannot close its window,
                # even if its probes finished later. Take a fresh observation;
                # keep any probe failure so the audit can refuse it.
                self.one(handle)
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
        # finished, followed by one fresh closing observation. Neither an
        # interrupted JSON line nor a pre-stop timestamp can close the window.
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

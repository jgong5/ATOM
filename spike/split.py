"""Split a profiled simulated run's wall by what its processes spent it on.

usage: python3 split.py OUT TREE [--rate HZ] [--json F]

OUT is the run's directory; OUT.prof holds sampler.sh's procs.txt and one py-spy
raw file per process (``--threads --idle --full-filenames``). TREE is the ATOM
tree the run ran, whose sources name each frame's enclosing function.

Each sample of a thread is put in one part, the first that holds:

- lazy trace: a frame in `_op_graphs`'s ``lazily``;
- lookup: any other frame inside `_op_graphs` (store reads, JSON parse, caches);
- pricing: a frame in ``OpGraphBackend.price``;
- host recurrence: a frame in a ``HostDeviceRecurrence`` method;
- wait: the innermost frame blocks (poll, select, a queue or lock wait, a read),
  or a frame spins for another process (a worker's doorbell, a DP group's
  shared-memory exchange);
- clock/IPC shim: a frame in ATOM's clock, ZMQ or loop shims or Compass's clock;
- rest: anything else.

A wait, a shim or the rest is named by its innermost three ATOM frames, so a DP
group's exchange, a worker waiting for its engine core and an engine core waiting
for its worker read apart. Seconds are samples over the rate.
"""

import argparse
import ast
import collections
import json
import re
from pathlib import Path

BLOCKING = {
    "wait",
    "poll",
    "select",
    "recv",
    "recv_multipart",
    "recv_pyobj",
    "recv_into",
    "readinto",
    "read",
    "accept",
    "sleep",
    "acquire",
    "_recv",
    "epoll",
    "get",
    "_poll",
    "recvfrom",
    "sock_recv",
    "wait_for",
    "_wait_for_tstate_lock",
    "join",
    "barrier",
    "all_gather",
    "all_reduce",
    "broadcast",
    "all_gather_object",
}
SHIMS = (
    "atom/utils/clock.py",
    "atom/utils/zmq_shim.py",
    "atom/utils/compass_loop.py",
    "atom/compass/clock/",
    "atom/compass/clock_transport/",
)
#: Waits that spin rather than block: a worker's doorbell, the shared-memory
#: collectives of a DP group.
SPINS = {
    ("atom/compass/runner/doorbell.py", "spin"),
    ("atom/compass/runner/shm_collectives.py", "exchange"),
}
FRAME = re.compile(r"^(?P<func>.*) \((?P<file>[^()]*):(?P<line>\d+)\)$")


class Names:
    """A frame's enclosing qualified name from the tree's source, or its own."""

    def __init__(self, tree):
        self.tree = Path(tree).resolve()
        self.files = {}
        self.seen = {}
        self.rels = {}

    def _index(self, path):
        spans = []
        try:
            source = Path(path).read_text()
            module = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            return spans

        def walk(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    name = prefix + child.name
                    if not isinstance(child, ast.ClassDef):
                        spans.append((child.lineno, child.end_lineno, name))
                        walk(child, name + ".<locals>.")
                    else:
                        walk(child, name + ".")
                else:
                    walk(child, prefix)

        walk(module, "")
        return spans

    def of(self, func, file, line):
        key = (file, line)
        if key not in self.seen:
            self.seen[key] = self._of(func, file, line)
        return self.seen[key]

    def _of(self, func, file, line):
        if file not in self.files:
            self.files[file] = self._index(file) if file.endswith(".py") else []
        best = None
        for lo, hi, name in self.files[file]:
            if lo <= line <= (hi or lo) and (best is None or lo >= best[0]):
                best = (lo, name)
        return best[1] if best else func

    def rel(self, file):
        if file not in self.rels:
            try:
                self.rels[file] = str(Path(file).resolve().relative_to(self.tree))
            except ValueError:
                self.rels[file] = file
        return self.rels[file]


def frames(stack, names):
    """The stack's frames, outermost first, as (qualname, file relative to the
    tree, func); the thread frame apart."""
    thread, out = None, []
    for f in stack.split(";"):
        m = FRAME.match(f.strip())
        if not m:
            if f.startswith(("thread", "process")):
                thread = f
            continue
        file, line = m["file"], int(m["line"])
        out.append((names.of(m["func"], file, line), names.rel(file), m["func"]))
    return thread, out


def part(fs):
    """(part, detail) of one sample's frames."""
    q = [f[0] for f in fs]
    files = [f[1] for f in fs]
    in_run = [
        i
        for i, (qn, fl, _) in enumerate(fs)
        if fl.endswith("atom/compass/run/__init__.py")
        and qn.startswith("_op_graphs.<locals>.")
    ]
    if any(fs[i][0].startswith("_op_graphs.<locals>.lazily") for i in in_run):
        return "lazy trace", ""
    if in_run:
        return "lookup", fs[in_run[0]][0].split(".<locals>.")[-1]
    if any(x == "OpGraphBackend.price" for x in q):
        return "pricing", ""
    if any(x.startswith("HostDeviceRecurrence.") for x in q):
        return "host recurrence", next(
            x for x in q if x.startswith("HostDeviceRecurrence.")
        )
    atom = [f for f in fs if f[1].startswith("atom/")]
    where = " < ".join(f[0] for f in reversed(atom[-3:])) if atom else "?"
    if (fs and fs[-1][2] in BLOCKING) or any((fl, fn) in SPINS for _, fl, fn in fs):
        return "wait", where
    if any(fl.startswith(SHIMS) for fl in files):
        return "clock/IPC shim", where
    return "rest", where


def role(pid, procs):
    """A process's role from procs.txt: authority, launcher, frontend, core or
    worker of a side and DP rank, client, or other."""
    _, ppid, comm, cmd = procs[pid]
    if "--compass-run" in cmd:
        return "authority"
    if "run --join" in cmd:
        return "launcher-" + (
            "prefill" if "/prefill" in cmd.split("--out", 1)[1].split()[0] else "decode"
        )
    if "api_server" in cmd:
        return (
            "frontend-" + role(ppid, procs).split("-", 1)[1]
            if ppid in procs
            else "frontend"
        )
    m = re.search(r"EngineCore_DP(\d+)", cmd)
    if m:
        side = role(ppid, procs).split("-", 1)[1] if ppid in procs else "?"
        return f"core-{side}-dp{m[1]}"
    m = re.search(r"ATOM::DP(\d+)TP\d+", comm + " " + cmd)
    if m:
        up = role(ppid, procs) if ppid in procs else "core-?-"
        return f"worker-{up.split('-')[1]}-dp{m[1]}"
    if cmd.startswith("aiperf") or "aiperf" in comm:
        return "client"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("tree")
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--json")
    a = p.parse_args()
    prof = Path(a.out + ".prof")
    procs = {}
    for line in (prof / "procs.txt").read_text().splitlines():
        pid, ppid, comm, *cmd = line.split(" ")
        procs[pid] = (pid, ppid, comm, " ".join(cmd))
    names = Names(a.tree)
    result = {}
    for f in sorted((prof / "pyspy").glob("*.txt")):
        pid = f.stem
        if pid not in procs:
            continue
        r = role(pid, procs)
        threads = collections.defaultdict(collections.Counter)
        details = collections.defaultdict(collections.Counter)
        for line in f.read_text(errors="replace").splitlines():
            stack, _, n = line.rpartition(" ")
            if not stack or not n.isdigit():
                continue
            thread, fs = frames(stack, names)
            k, d = part(fs)
            threads[thread][k] += int(n)
            details[thread][(k, d)] += int(n)

        # the main thread: the one whose samples are most often not a wait
        def busy(t, threads=threads):
            return sum(c for k, c in threads[t].items() if k != "wait")

        named = [t for t in threads if t and t.endswith(": MainThread")]
        main_t = named[0] if named else (max(threads, key=busy) if threads else None)
        result[pid] = {
            "role": r,
            "main_thread": main_t,
            "main": (
                {k: c / a.rate for k, c in threads[main_t].items()} if main_t else {}
            ),
            "main_top": (
                [[k, d, c / a.rate] for (k, d), c in details[main_t].most_common(12)]
                if main_t
                else []
            ),
            "main_all": (
                [[k, d, c / a.rate] for (k, d), c in details[main_t].items()]
                if main_t
                else []
            ),
            "threads": {t: sum(c.values()) / a.rate for t, c in threads.items()},
        }
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
    order = [
        "lazy trace",
        "lookup",
        "pricing",
        "host recurrence",
        "clock/IPC shim",
        "rest",
        "wait",
    ]
    groups = collections.defaultdict(list)
    for pid, x in result.items():
        g = re.sub(r"-dp\d+$", "", x["role"])
        groups[g].append(x)
    print(
        f"{'role (main thread)':<24} {'n':>3} {'seconds':>9} "
        + " ".join(f"{k[:10]:>10}" for k in order)
    )
    for g, xs in sorted(groups.items()):
        tot = [sum(x["main"].values()) for x in xs]
        for label, pick in (
            ("mean", None),
            (
                "max busy",
                max(
                    xs,
                    key=lambda x: sum(v for k, v in x["main"].items() if k != "wait"),
                ),
            ),
        ):
            if pick is None:
                shares = [
                    sum(x["main"].get(k, 0) for x in xs) / max(sum(tot), 1e-9)
                    for k in order
                ]
            else:
                t = sum(pick["main"].values())
                shares = [pick["main"].get(k, 0) / max(t, 1e-9) for k in order]
            print(
                f"{g + ' ' + label:<24} {len(xs):>3} {max(tot):>9.0f} "
                + " ".join(f"{s:>10.1%}" for s in shares)
            )
    print()
    for g, xs in sorted(groups.items()):
        pick = max(
            xs, key=lambda x: sum(v for k, v in x["main"].items() if k != "wait")
        )
        print(f"{g}: busiest {pick['role']}, top parts of its main thread:")
        for k, d, s in pick["main_top"]:
            print(f"  {s:>9.1f} s  {k:<16} {d}")


if __name__ == "__main__":
    main()

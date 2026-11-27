# SPDX-License-Identifier: MIT
"""Advice points rerun on the tree now checked out, from a finished point of
the same candidate: to check a simulator fix on the points a search left failed,
which have no point.json of their own, without rerunning the whole search.

    PYTHONPATH=$PIT2_TREE python3 rerun_points.py OUT NAME=BASE_POINT_DIR:C... [--tries N]

Each NAME=BASE_POINT_DIR:C runs into OUT/runs/NAME: BASE_POINT_DIR's recorded
server and client commands and variables (a finished point's point.json), its
directory swapped for the new one, the client's --concurrency set to C and
the tree read afresh. A candidate's server arguments must not vary with the
concurrency (no `per_concurrency` in its space file blocks); the M3.1 space's
do not. Each is tried as advice tries a point, TRIES times (default 1, so a
failure shows at once). AT_ONCE at a time (default 3). Run it inside the CPU
container with env.sh sourced, from /, at nice 19.

The rerun is not advice's own result: run `compass advice SPACE OUT` again on
the fixed tree for the search's answer.
"""

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path


def plan(name: str, base: Path, c: int, out: Path, tree=None) -> dict:
    """`base`'s point rerun at concurrency `c` as `name` into `out`'s runs;
    `tree`, a (revision, dirty) pair, read from the tree on PYTHONPATH when
    None."""
    from atom.compass.advice import simulate

    ran = json.loads((base / simulate.RAN).read_text())["ran"]
    at = out / simulate.RUNS / name
    old = str(base)
    if old not in ran["server"] or old not in ran["client"]:
        raise ValueError(f"{base}: its point.json runs another directory")

    def swap(args):
        return [str(at) if a == old else a for a in args]

    client = swap(ran["client"])
    client[client.index("--concurrency") + 1] = str(c)
    if tree is None:
        from atom.compass.artifacts.provenance import git_tree_root
        from atom.compass.drive import serve

        root = git_tree_root("atom", str(serve.TREE))
        tree = (root.revision, root.dirty)
    ran = dict(ran, server=swap(ran["server"]), client=client)
    ran["tree"] = {"revision": tree[0], "dirty": tree[1]}
    return {"name": name, "at": at, "ran": ran, "ranks": 1}


def parse(spec: str) -> tuple[str, Path, int]:
    """NAME=BASE_POINT_DIR:C as its three parts."""
    name, eq, rest = spec.partition("=")
    base, colon, c = rest.rpartition(":")
    if not (name and eq and base and colon and c.isdigit()):
        raise ValueError(f"{spec!r} is not NAME=BASE_POINT_DIR:C")
    return name, Path(base), int(c)


def main() -> None:
    from atom.compass.advice import simulate

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("specs", nargs="+", metavar="NAME=BASE_POINT_DIR:C")
    parser.add_argument("--tries", type=int, default=1)
    args = parser.parse_args()
    plans = [plan(*parse(s), args.out) for s in args.specs]
    simulate.TRIES = args.tries
    stopping = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(
        int(os.environ.get("AT_ONCE", "3"))
    ) as pool:
        futures = {
            pool.submit(simulate._simulated, p, stopping, None): p for p in plans
        }
        for f in concurrent.futures.as_completed(futures):
            print(futures[f]["name"], json.dumps(f.result()), flush=True)


if __name__ == "__main__":
    main()

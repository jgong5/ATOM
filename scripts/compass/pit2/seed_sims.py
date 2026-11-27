# SPDX-License-Identifier: MIT
"""Finished advice points rerun under client send-order seeds, H1's "three send
orders" (LLM-Booster #766, #778): one real run graded against the mean of
simulated runs that differ only in the order the traffic LP numbers the
requests aiperf sends at one instant (COMPASS_HARNESS_SEND_ORDER_SEED).

    PYTHONPATH=$PIT2_TREE python3 seed_sims.py OUT POINT_DIR... [--seeds 0,1,2]

Each POINT_DIR is a point `compass advice` finished (OUT/runs/<point>, its
point.json). Each seed's run goes into OUT/runs/<point>-s<seed>: the point's
recorded server and client commands and variables, its directory swapped for
the new one, the seed in the client's variables, and the tree that runs it
read afresh, as advice records it. Each is tried as advice tries a point
(`simulate._simulated`: up to TRIES from a fresh directory, point.json written
once finished); a run whose point.json records what it would run is skipped.
AT_ONCE runs at a time (default 3). Run it inside the CPU container with
env.sh sourced, from /, at nice 19. Grade each seed against the real run with
scripts/compass/one_real.py REAL SIM...: it prints each seed's errors.
"""

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path


def plan(point_dir: Path, seed: int, out: Path, tree=None) -> dict:
    """The run of `point_dir`'s point under `seed` into `out`'s runs, as
    `simulate._simulated` takes it; `tree`, a (revision, dirty) pair, read
    from the tree on PYTHONPATH when None."""
    from atom.compass.advice import simulate

    ran = json.loads((point_dir / simulate.RAN).read_text())["ran"]
    name = f"{point_dir.name}-s{seed}"
    at = out / simulate.RUNS / name
    old = str(point_dir)
    if old not in ran["server"] or old not in ran["client"]:
        raise ValueError(f"{point_dir}: its point.json runs another directory")

    def swap(args):
        return [str(at) if a == old else a for a in args]

    ran = dict(ran, server=swap(ran["server"]), client=swap(ran["client"]))
    ran["client_env"] = dict(
        ran["client_env"], COMPASS_HARNESS_SEND_ORDER_SEED=str(seed)
    )
    if tree is None:
        from atom.compass.artifacts.provenance import git_tree_root
        from atom.compass.drive import serve

        root = git_tree_root("atom", str(serve.TREE))
        tree = (root.revision, root.dirty)
    ran["tree"] = {"revision": tree[0], "dirty": tree[1]}
    return {"name": name, "at": at, "ran": ran, "ranks": 1}


def main() -> None:
    from atom.compass.advice import simulate

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("points", nargs="+", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    plans = [plan(p, s, args.out) for s in seeds for p in args.points]
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

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A simulated `--dp-aware` router's start, checked against its deployment:
`python3 -m atom.compass.drive.router_workers RUN_FILE ROUTER_LOG`.

Atomesh reads each prefill worker's TP size once, as it builds its routes,
from the workers registered then. Its wait for them ends once each configured
URL has one, but a `--dp-aware` URL registers its DP ranks one at a time, so a
rank can register after the wait. The router then places requests on that
rank and fails each once its prefill has answered, its relay already stamped,
for want of the rank's TP size. So `deploy.sh` and `pd_sim.sh` serve a router
only once it has counted every worker of the run file's deployments, each
instance's DP ranks, in the `Workers initialized: <n> total` line it logs just
before it builds its routes. Otherwise this exits 1, naming the count, or the
missing line.
"""

import json
import re
import sys

from atom.compass.run import INSTANCES, _dp

#: atomesh's count of the workers it registered, logged before it builds its routes.
_COUNT = re.compile(r"Workers initialized: (\d+) total")


def refusal(run: dict, log: str) -> str | None:
    """Why the router that wrote `log` may not serve the deployments of run
    file `run`, or None when it counted each of their workers."""
    workers = sum(run.get(n, 1) * _dp(run, side) for side, n in INSTANCES.items())
    counts = _COUNT.findall(log)
    if not counts:
        return (
            "the router logged no 'Workers initialized: <n> total' line, so "
            f"whether it counted the deployment's {workers} workers is unknown"
        )
    if int(counts[-1]) != workers:
        return (
            f"the router counted {counts[-1]} of the deployment's {workers} "
            "workers as it built its routes: atomesh's start ended its wait "
            "for them before every DP rank registered, and it fails each "
            "request on a rank registered later after the request's relay is "
            "stamped, for want of that rank's TP size; start the deployment again"
        )
    return None


def main(argv: list[str]) -> None:
    run_file, log = argv
    with open(run_file) as f:
        run = json.load(f)
    with open(log, errors="replace") as f:
        why = refusal(run, f.read())
    if why:
        sys.exit(why)


if __name__ == "__main__":
    main(sys.argv[1:])

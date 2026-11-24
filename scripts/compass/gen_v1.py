#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The calibration design of a config (#739 v1.1), as the traces its lanes send.

    PYTHONPATH=$T python3 scripts/compass/gen_v1.py CONFIG OUT [--record RECORD]

CONFIG is a design config (`scripts/compass/calib/*.json`,
`atom.compass.calib.deploy`): the deployment, the law's columns, the noise and
the acceptance criteria, and the segments. The design reads no law and no
target workload: each segment is generated from the deployment alone
(`atom.compass.calib.design`). Writes OUT/<segment>.<lane>.jsonl, one weka
trace a session as `client.sh --traces` sends them, and OUT/design.json (the
config, and per segment its lanes, pins, concurrency and what its generator
planned; its deployment as plain fields), then prints one line a lane.
With RECORD (a record directory, or `cal_v1.sh`'s OUT), the deployment is
the one that record ran (`deploy.from_record`), in place of the config's own
fields wherever the record states one: a smoke record under a new deployment
gives the design for it with no edit.
"""

import argparse
from pathlib import Path

from atom.compass.calib import deploy, design


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--record",
        type=Path,
        help="the deployment this record ran, over the config's own fields",
    )
    a = p.parse_args(argv)
    cfg = deploy.load(a.config, a.record)
    doc = design.write(design.generate(cfg), cfg, a.out)
    print(f"{cfg.name}: {len(doc['segments'])} segments in {a.out}")
    for name, need in doc["skipped"].items():
        print(f"  {name}: skipped, it needs {need} DP ranks")
    for seg in doc["segments"]:
        for lane in seg["lanes"]:
            pin = "router" if lane["pin"] is None else f"rank {lane['pin']}"
            print(
                f"  {seg['name']:12s} {lane['name']:7s} {pin:7s} c={lane['concurrency']:<4d} "
                f"{lane['traces']:5d} traces {lane['requests']:5d} requests "
                f"{lane['prompt_tokens'] / 1e6:7.2f}M prompt {lane['output_tokens'] / 1e3:7.1f}k out"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The calibration design of a config (#739 v1.1), as the traces its lanes send.

    PYTHONPATH=$T python3 scripts/compass/gen_v1.py CONFIG OUT

CONFIG is a design config (`scripts/compass/calib/*.json`,
`atom.compass.calib.deploy`): the deployment, the law's columns, the noise and
the acceptance criteria, and the segments. The design reads no law and no
target workload: each segment is generated from the deployment alone
(`atom.compass.calib.design`). Writes OUT/<segment>.<lane>.jsonl, one weka
trace a session as `client.sh --traces` sends them, and OUT/design.json (the
config, and per segment its lanes, pins, concurrency and what its generator
planned), then prints one line a lane.
"""

import argparse
from pathlib import Path

from atom.compass.calib import deploy, design


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args(argv)
    cfg = deploy.load(a.config)
    doc = design.write(design.generate(cfg), cfg, a.out)
    print(f"{cfg.name}: {len(doc['segments'])} segments in {a.out}")
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

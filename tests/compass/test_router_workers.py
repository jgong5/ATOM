# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A simulated `--dp-aware` router serves an 8P16D deployment only once it has
counted all 24 of its workers, each instance's DP ranks, as atomesh logs the
count just before it reads the prefills' TP sizes. The log lines are an
8P16D C1228 start's (#4258), whose router counted 23."""

import json

import pytest

from atom.compass.drive import router_workers

#: Four prefill instances at DP 2 and four decode instances at DP 4.
RUN = {
    "router_s": 1e-5,
    "prefill_instances": 4,
    "decode_instances": 4,
    "prefill_data_parallel_size": 2,
    "decode_data_parallel_size": 4,
}
LOGGED = (
    "\x1b[2m2026-11-26 17:30:45\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mmesh::server\x1b[0m"
    "\x1b[2m:\x1b[0m \x1b[2m692:\x1b[0m Workers initialized: {n} total, {n} healthy\n"
)


@pytest.mark.parametrize(
    "log, refusal",
    [
        (LOGGED.format(n=24), None),
        (LOGGED.format(n=23), "the router counted 23 of the deployment's 24 workers"),
        (
            "Router ready\n",
            "the router logged no 'Workers initialized: <n> total' line",
        ),
    ],
)
def test_an_8p16d_router_serves_once_it_counted_its_24_workers(tmp_path, log, refusal):
    run_file, log_file = tmp_path / "run_file.json", tmp_path / "router.log"
    run_file.write_text(json.dumps(RUN))
    log_file.write_text(log)
    if refusal is None:
        router_workers.main([str(run_file), str(log_file)])
        return
    with pytest.raises(SystemExit) as refused:
        router_workers.main([str(run_file), str(log_file)])
    assert str(refused.value.code).startswith(refusal)

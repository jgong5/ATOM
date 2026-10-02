# SPDX-License-Identifier: MIT
"""Each step's scheduling decision, recorded on a real or a simulated runner.

A simulated run reuses ATOM's scheduler, so it should build the batches a real
run builds. This records them where both runs can be read the same way, at the
model runner's `forward`: one JSON line per call, written before the call runs,
because a real forward rewrites some batch fields in place. `compare` reads two
records and names, per DP rank, the first step where they part.

Setting `ATOM_COMPASS_PARITY_RECORD` to a directory turns recording on: TP rank 0
of each DP rank writes `dp<rank>.jsonl` there, replacing any earlier file.
Unset, nothing is recorded. The simulated runner records from
`NonAllocatingRunner.forward`; a real one records through `StepRecording`,
composed into `atom.compass.parity.runner.RecordingModelRunner`, which
`--runner-qualname` names.

A request is named by its key, a digest of the tokens it was first scheduled
with, because request ids are numbered per run. Two requests of one run with the
same key could not be told apart in either record, so the second is refused.
"""

import hashlib
import json
import os
import pathlib
from itertools import zip_longest

import numpy as np

ENV = "ATOM_COMPASS_PARITY_RECORD"


def request_key(tokens) -> str:
    """The digest a request is named by across runs."""
    data = np.asarray(tokens, dtype=np.int32).tobytes()
    return hashlib.blake2b(data, digest_size=8).hexdigest()


class StepRecord:
    """One DP rank's record: a JSON line per forward call."""

    def __init__(self, directory, dp_rank: int) -> None:
        self.path = pathlib.Path(directory) / f"dp{dp_rank}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.dp_rank = dp_rank
        self.step = 0
        self.keys: dict[int, str] = {}  # request id -> key
        self.owners: dict[str, int] = {}  # key -> request id

    def add(self, batch) -> None:
        """Append `batch` as (key, scheduled tokens, context length) per request.

        A dummy batch is fabricated for DP synchronisation and names no
        request, so it is recorded with an empty batch.
        """
        rows = []
        if not batch.is_dummy_run:
            ends = np.cumsum(batch.num_scheduled_tokens)
            for i, req_id in enumerate(batch.req_ids):
                num = int(batch.num_scheduled_tokens[i])
                if req_id not in self.keys:
                    key = request_key(batch.scheduled_tokens[ends[i] - num : ends[i]])
                    if key in self.owners:
                        raise ValueError(
                            f"requests {self.owners[key]} and {req_id} were first "
                            f"scheduled with the same tokens (key {key}), so no "
                            "record can tell them apart; give each request a "
                            "distinct prompt."
                        )
                    self.keys[req_id], self.owners[key] = key, req_id
                rows.append([self.keys[req_id], num, int(batch.context_lens[i])])
        line = {
            "step": self.step,
            "dp_rank": self.dp_rank,
            "is_dummy_run": bool(batch.is_dummy_run),
            "produces_output": bool(batch.produces_output()),
            "batch": rows,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        self.step += 1


def record_step(runner, batch) -> None:
    """Record `batch` if `ENV` names a directory and `runner` is TP rank 0."""
    if not hasattr(runner, "_parity_record"):
        directory = os.environ.get(ENV)
        record = None
        if directory and runner.rank == 0:
            if runner.config.pipeline_parallel_size > 1:
                # Every stage's TP rank 0 would write the same file.
                raise ValueError(
                    f"{ENV} records one file per DP rank, and under pipeline "
                    "parallelism each stage would write it."
                )
            dp_rank = runner.config.parallel_config.data_parallel_rank
            record = StepRecord(directory, dp_rank)
        runner._parity_record = record
    if runner._parity_record is not None:
        runner._parity_record.add(batch)


class StepRecording:
    """Records each forward call; put it before `ModelRunner` in the bases.

    ATOM's `ModelRunner.__init__` warms up with a forward over a fabricated
    batch, which the simulated runner declines, so recording starts once
    construction returns.
    """

    _recording = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._recording = True

    def forward(self, batch):
        if self._recording:
            record_step(self, batch)
        return super().forward(batch)


def read(directory) -> dict[int, list[dict]]:
    """A record directory as {DP rank: steps}."""
    return {
        int(path.stem[2:]): [json.loads(line) for line in path.read_text().splitlines()]
        for path in sorted(pathlib.Path(directory).glob("dp*.jsonl"))
    }


def _request_ranks(name: str, record: dict[int, list[dict]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, steps in record.items():
        for step in steps:
            for key, _num, _context in step["batch"]:
                if ranks.setdefault(key, rank) != rank:
                    raise ValueError(
                        f"the {name} record schedules request {key} on DP ranks "
                        f"{ranks[key]} and {rank}; a request lives on one rank, "
                        "so the key joins two requests."
                    )
    return ranks


def compare(real, simulated) -> dict:
    """Two record directories: per DP rank, the first step whose decision differs.

    A rank's entry is None where its steps all agree, else the step index and
    both steps, None for a record that has already ended. A record with no
    steps is refused: two runs that scheduled nothing agree, and that means
    nothing.
    """
    records = {"real": read(real), "simulated": read(simulated)}
    for name, record in records.items():
        if not any(record.values()):
            raise ValueError(f"the {name} record has no steps to compare.")
    first = {}
    for rank in sorted(records["real"].keys() | records["simulated"].keys()):
        pairs = zip_longest(
            records["real"].get(rank, []), records["simulated"].get(rank, [])
        )
        first[rank] = next(
            (
                {"step": i, "real": a, "simulated": b}
                for i, (a, b) in enumerate(pairs)
                if a != b
            ),
            None,
        )
    return {
        "first_divergence": first,
        "request_dp_rank": {
            name: _request_ranks(name, record) for name, record in records.items()
        },
    }

# SPDX-License-Identifier: MIT
"""What a forward's `E` spends admitting requests, labelled, fitted and charged.

ATOM's scheduler allocates a request's whole prompt the first time it schedules
it: it walks the prefix-cache hit and takes a fresh block for each block past
it, so that forward's `E` grows with the prompt. A real record labels it per
step (`step_labels.admissions`); `fit_host` prices the rest of `E` by its kind's
law and the admissions by `admit`; a run's runner reads them off ATOM's own
batches and `HostDeviceRecurrence` adds them to `E`.
"""

import numpy as np
import pytest
from conftest import MockConfig
from test_dp_step_max import runner
from test_parity_recurrence import _labelled, _law, line
from test_runner_step_semantics import _drive

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import Sample
from atom.compass.backends.step_linear_fit import fit_host
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.parity import recurrence
from atom.compass.runner import overrides
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    HostDeviceRecurrence,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

MS = 1_000_000
BACKEND = StepStubBackend()
#: Per admitted request, per prompt token past the hit and per token in it.
ADMIT = [1.0e-3, 26e-9, 10e-9]


def _record() -> list[dict]:
    """Request 7 prefills 100 tokens in two chunks and decodes; request 8
    arrives while it decodes, with 300 of its 400 prompt tokens cached, and
    prefills its last 100 in one; then both decode."""
    steps = [
        line(0, [[50, 50, False]], False, 0, (0, 0), (1, 2), None, 3),
        line(1, [[50, 100, False]], True, 4, (4, 4), (5, 6), (6, 6), 7),
        line(2, [[1, 101, True]], True, 8, (8, 8), (9, 10), (10, 18), 19),
        line(3, [[100, 400, False]], True, 23, (23, 23), (24, 25), (25, 29), 30),
        line(
            4,
            [[1, 102, True], [1, 401, True]],
            True,
            31,
            (31, 31),
            (32, 33),
            (33, 41),
            42,
        ),
    ]
    steps[3]["batch"] = [[8, 100, 400]]
    steps[4]["batch"] = [[7, 1, 102], [8, 1, 401]]
    for s in steps:
        s["t_pull_ns"] = s["t_exit_ns"] + MS
    return steps


def test_a_request_s_first_forward_admits_its_whole_prompt_past_its_hit(tmp_path):
    samples = _labelled(tmp_path, _record())
    assert [s.admitted for s in samples] == [
        (1, 100, 0),  # request 7's first chunk, its whole prompt uncached
        (0, 0, 0),  # its last chunk
        (0, 0, 0),
        (1, 100, 300),  # request 8: 300 cached, 100 past them
        (0, 0, 0),
    ]


def _sample(step, kind, tokens, e, admitted=(0, 0, 0), run="a"):
    """A busy forward with every host segment but `E` at 0.1 ms."""
    rows = [RequestShape(tokens, tokens, kind == "decode")]
    view = BatchView(tuple(rows), produces_output=kind != "middle")
    host = {"E": e} | {g: 1e-4 for g in HOST_SEGMENTS[1:]}
    if kind == "middle":
        del host["W_post"]
    provenance = {"source": "record", "run": run, "requests": [step], "config": {}}
    return Sample(step, 0, view, 0.01, None, host, False, provenance, admitted=admitted)


def _cell(rng, n=80):
    """Forwards of each kind; a prompt's last chunk and a middle chunk that
    continue a request cost 0.6 ms plus 4 ns a token of `E`, a decode 0.3 ms,
    and a forward that admits a request `ADMIT` on top of its kind's."""
    out = []
    for i in range(n):
        out.append(_sample(4 * i, "decode", 1, 3e-4 * rng.uniform(0.98, 1.02)))
        for j, kind in enumerate(("prefill", "middle")):
            tokens = int(rng.integers(1000, 16000))
            e = (6e-4 + 4e-9 * tokens) * rng.uniform(0.98, 1.02)
            out.append(_sample(4 * i + 1 + j, kind, tokens, e))
        cached, uncached = int(rng.integers(0, 200_000)), int(rng.integers(1000, 16000))
        e = 6e-4 + 4e-9 * uncached + np.dot(ADMIT, (1, uncached, cached))
        a = (1, uncached, cached)
        out.append(
            _sample(4 * i + 3, "prefill", uncached, e * rng.uniform(0.98, 1.02), a)
        )
    return out


def test_a_kind_s_e_law_prices_the_forwards_that_admit_nothing_and_admit_the_rest():
    host, report = fit_host(_cell(np.random.default_rng(2620)))
    laws, x = host["laws"], (1, 8000, 1)

    # A prompt's last chunk of 8000 tokens keeps its own cost, and so does a
    # middle chunk that continues a request.
    for kind in ("prefill", "middle"):
        assert np.dot(laws[kind]["E"], x) == pytest.approx(6.32e-4, rel=0.02)
    # The first forward of a prompt of 100k cached tokens and 8000 past them.
    first = np.dot(laws["prefill"]["E"], x) + np.dot(host["admit"], (1, 8000, 100_000))
    assert first == pytest.approx(6.32e-4 + np.dot(ADMIT, (1, 8000, 100_000)), rel=0.02)
    assert host["admit"] == pytest.approx(ADMIT, rel=0.1)
    assert report["admit"]["samples"] + report["admit"]["dropped"] == 80
    assert "admit" not in report["unfitted"]


def test_records_with_no_admitting_forward_say_so_and_fit_no_admission():
    rng = np.random.default_rng(1)
    host, report = fit_host([s for s in _cell(rng) if not s.admitted[0]])
    assert "admit" not in host
    assert report["unfitted"]["admit"] == "0 forwards for 3 coefficients"


#: The pricing `NonAllocatingRunner.forward` calls; `Priced` stands in for it.
PRICE = overrides._group_step_seconds


class Priced:
    """A runner with host laws of 0.1 ms per segment, pricing each batch as the
    engine hands it over; `out` holds each forward's `E` and the admissions it
    read off the batch."""

    def __init__(self, admit, stage=None, skip=()):
        self.r, self.clock, self.out = runner(BACKEND), [0.0], []
        self.r.config.kv_cache_block_size = 16  # the block of ATOM's batches here
        law = {k: {g: [1e-4, 0, 0] for g in HOST_SEGMENTS} for k in FORWARD_KINDS}
        now = lambda: self.clock[0]
        self.r.compass_host = HostDeviceRecurrence(law, skip, now, stage, admit=admit)

    def __call__(self, batch, *_):
        self.clock[0] += PRICE(self.r, batch, BACKEND)
        e = self.r.compass_host.segments["E"]
        self.out.append(
            (list(batch.req_ids), e, getattr(batch, "compass_admitted", None))
        )
        return 0.0


def driven(admit, stage=None, skip=()):
    """`Priced`'s record of ATOM's own run of four 200-token prompts, each
    prefilled in four chunks of 64 tokens and decoded."""
    priced = Priced(admit, stage, skip)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda _, b, *a: priced(b))
        _drive()
    return priced.out


def test_a_run_charges_each_request_s_admission_on_its_first_forward():
    seen, admitting = set(), 0
    for ids, e, admitted in driven(ADMIT):
        new = [r for r in ids if r not in seen]
        seen.update(ids)
        # Each prompt of 200 tokens fills 13 blocks of 16; none is cached.
        assert admitted == (len(new), 208 * len(new), 0)
        assert e == pytest.approx(1e-4 + np.dot(ADMIT, admitted), rel=1e-12)
        admitting += len(new)
    assert admitting == len(seen) == 4
    bare = [e for _, e, _ in driven(None)]
    assert bare == pytest.approx([1e-4] * len(bare), rel=1e-12)


def test_skipping_e_skips_its_admission_and_a_later_stage_admits_nothing():
    assert {e for _, e, _ in driven(ADMIT, skip=("E",))} == {0.0}
    later = [e for _, e, _ in driven(ADMIT, stage=(1, 2))]
    assert later == pytest.approx([1e-4] * len(later), rel=1e-12)


def test_the_hit_a_run_reads_is_the_tokens_the_first_row_starts_from():
    """A 64-token prompt, finished; then that prompt and 32 more tokens, whose
    first 64 hit the prefix cache."""
    sched = Scheduler(
        MockConfig(
            enable_prefix_caching=True,
            kv_cache_block_size=16,
            num_kvcache_blocks=64,
            max_num_batched_tokens=256,
            max_model_len=256,
        )
    )
    priced = Priced(ADMIT)
    for prompt in (list(range(5, 69)), list(range(5, 101))):
        sched.add(Sequence(prompt, 16, sampling_params=SamplingParams(max_tokens=1)))
        batch, seqs = sched.schedule()
        priced(batch)
        sched.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(2,)] * len(batch.req_ids),
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
    assert [a for *_, a in priced.out] == [(1, 64, 0), (1, 32, 64)]


def test_the_recurrence_grade_charges_a_busy_admission_and_measures_an_idle_one(
    tmp_path,
):
    """Request 8's 2 ms admission joins step 3's `E`: its host reaches the
    launch 2 ms later, so 2 ms less device work is queued there, and no other
    step moves. Step 0 admits request 7 after no forward: its `E` stays measured."""
    steps = _record()
    samples = _labelled(tmp_path, steps)
    law = _law()
    bare = recurrence.priced(steps, samples, law)
    law["host"]["admit"] = [2e-3, 0, 0]
    admitted = recurrence.priced(steps, samples, law)

    def queued(report):
        return {r[0]: r[4] for r in report["failures"]}

    moved = {k: queued(admitted)[k] - v for k, v in queued(bare).items()}
    assert moved == pytest.approx({2: 0, 3: -2.0, 4: 0})
    assert admitted["kinds"]["idle"] == bare["kinds"]["idle"]

# SPDX-License-Identifier: MIT
"""A runner that does not defer its output reads its own draft ids.

ATOM's non-deferred `prepare_draft_ids` blocks in `draft_token_ids.cpu()`
until the forward's own draft is done, after its host queued the draft, so
such a runner's output forward leaves no earlier than its device end. Its
record stamps that read as the draft copy wait, `t_draft_read_ns`, which the
gate and the labels take out of `W_post`; a record stamping none keeps it there.
"""

import copy
import statistics
from types import SimpleNamespace

import pytest
from test_parity_device_window import _Processor, _atom_processor_calls, _recorded
from test_parity_record import _config, _drive
from test_parity_recurrence import MS, _labelled, _law, line

from atom.compass.backends.step_labels import HOST_SEGMENTS, label
from atom.compass.parity import ENV, StepRecording, read, recurrence
from atom.compass.runner.overrides import MTP_SEGMENT, HostDeviceRecurrence

OUTPUT = SimpleNamespace(
    total_seqs_num_prefill=0,
    total_tokens_num=1,
    total_seqs_num=1,
    produces_output=lambda: True,
)


@pytest.mark.parametrize("defer, exits", [(False, [3, 6, 9]), (True, [0, 2, 5])])
def test_a_runner_that_reads_its_own_draft_ids_leaves_at_its_device_end(
    defer, exits
):
    """Three 2 s output forwards with 1 s of draft each on one stage, no host
    laws, each called at the exit before it: the device ends at 3, 6 and 9 s
    either way, and a runner that defers leaves at the previous output's."""
    clock, got = [0.0], []
    host = HostDeviceRecurrence({}, (), lambda: clock[0], defer=defer)
    for _ in exits:
        clock[0] += host.charge(OUTPUT, 2.0, 1.0)
        got.append((host.ends[-1], clock[0]))
    assert got == list(zip([3, 6, 9], exits))


def own_drafts(stamped=True):
    """Decodes of a runner that reads its own tokens and draft ids, in ms: each
    enters 1 after the last exit, drains its empty MTP status at 1 and launches
    from 1 to 2; its 9 of target run from 1 and it reads its tokens until they
    end, at 10. It queues its 3 of draft at 11 and reads them, `stamped` or
    not, until they end, at 14, and leaves 1 later. The device waits for the
    host from 10 to 11 and from 14 to the next launch, 2 after the exit event."""
    steps, enter = [], 0
    for n in range(4):
        rows, launch = [[1, 151 + n, True]], (enter + 1, enter + 2)
        s = line(n, rows, True, enter, (enter,) * 2, launch, None, enter + 15)
        s["t_mtp_ns"] = [(enter + 1) * MS] * 2
        s["t_read_ns"] = [(enter + 2) * MS, (enter + 10) * MS]
        if stamped:
            s["t_draft_read_ns"] = [(enter + 11) * MS, (enter + 14) * MS]
        s |= {"d_busy_ns": 14 * MS, "d_target_ns": 9 * MS, "d_propose_ns": 3 * MS}
        s |= {"d_gap_ns": 2 * MS} if n else {}
        steps.append(s)
        enter += 16
    return steps


def test_the_gate_takes_a_record_s_own_draft_read_as_its_draft_copy_wait():
    """Stamped, the read leaves `W_post` at 2 ms and the draft is that read,
    3 ms: each forward leaves at its draft's end, as it ran. Unstamped, `W_post`
    holds it, 5 ms, so each forward leaves 5 after its target ends, and its
    draft, 5 ms of busy time past the target, runs after that: exits at 15,
    36, 57 and 78 against the 15, 31, 47 and 63 measured."""
    report = recurrence.check(own_drafts())
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    old = recurrence.check(own_drafts(stamped=False))
    assert old["total"] == pytest.approx(100 * (78 / 63 - 1))
    assert [r[:2] for r in old["failures"]] == [(n, "decode") for n in (1, 2, 3)]


@pytest.mark.parametrize("h", [0.25, 0.5, 1])
def test_the_gate_charges_a_record_s_own_draft_from_its_read(h):
    """ATOM's recorder starts `d_propose_ns` at `propose_draft_token_ids`'s
    entry, where the device has idled since the token read: with the host
    preparing the draft there for `h` ms before it queues it, the pass reads
    3 + `h`, and `W_post` holds those `h` ms too. Each forward still leaves
    at its draft's end, as it ran; a record stamping no read reads as before."""
    steps, old = own_drafts(), own_drafts(stamped=False)
    for s in steps + old:
        s["d_propose_ns"] = round((3 + h) * MS)
    report = recurrence.check(steps)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    assert recurrence.check(old)["total"] == pytest.approx(100 * (78 / 63 - 1))


@pytest.mark.parametrize("stamped, post", [(True, 2), (False, 5)])
def test_a_record_s_own_draft_read_is_not_in_its_post(tmp_path, stamped, post):
    """From its token read to its exit, each decode spends 5 ms, 3 of them
    reading its draft ids."""
    samples = _labelled(tmp_path, own_drafts(stamped))
    assert [s.host["W_post"] for s in samples] == pytest.approx([post * 1e-3] * 4)


def _fitted(samples, draft_ms) -> dict:
    """`_law` with its host laws at `samples`' median labels, and its decode
    vector at 9 ms of target and `draft_ms` of draft. Every decode's device
    start precedes its launch end, so none has a device label."""
    law = _law()
    law["decode"]["eager"]["coefficients"]["step"] = 9e-3
    law["draft"] = {"provenance": "stated", "decode": copy.deepcopy(law["decode"])}
    law["draft"]["decode"]["eager"]["coefficients"]["step"] = draft_ms * 1e-3
    law["host"]["laws"] = {
        "decode": {
            g: [statistics.median(s.host[g] for s in samples if g in s.host), 0, 0]
            for g in (*HOST_SEGMENTS, MTP_SEGMENT)
        }
    }
    return law


@pytest.mark.parametrize("stamped, draft_ms, last", [(True, 3, 63), (False, 5, 72)])
def test_a_law_fitted_on_a_record_s_own_draft_reads_prices_it_as_the_gate_reads_it(
    tmp_path, stamped, draft_ms, last
):
    """A law at each record's labels, its draft the propose pass where the
    read is stamped and the busy time past the target where not: stamped, it
    grades the record as the gate does; unstamped, each forward leaves 5 after
    its target ends and the next target waits for its draft, the gap before
    its launch unpriced: exits at 15, 34, 53 and 72."""
    steps = own_drafts(stamped)
    samples = _labelled(tmp_path, steps)
    got = recurrence.priced(steps, samples, _fitted(samples, draft_ms))
    assert got["total"] == pytest.approx(100 * (last / 63 - 1), abs=1e-9)
    if stamped:
        gate = recurrence.check(steps)
        assert got["pass"] and gate["pass"]
        assert got["kinds"].keys() == gate["kinds"].keys()
        # Each kind's count and percentiles; the gate alone reports the MTP
        # status wait's release.
        for k, v in gate["kinds"].items():
            assert got["kinds"][k][:4] == pytest.approx(v[:4], abs=1e-9)


def test_a_line_stamps_its_own_draft_read(tmp_path, monkeypatch):
    """A drafter's runner that does not defer: each output forward reads its
    own tokens, queues its draft and reads the draft ids for 7000 ns. Its line
    stamps that read under `t_draft_read_ns`, and `W_post` leaves it out."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1], drafter="eagle")
    device.tick = 10**12
    late = [0]

    def read_drafts(self, batch, ids):
        late[0] += 7000
        return ids

    def propose(self, batch, align_only=False):
        self.queued.append("propose")
        if not align_only:
            return self.tokenID_processor.prepare_draft_ids(batch, [])

    class Reading(recorded):
        def step_clock_ns(self):
            return recorded.step_clock_ns() + late[0]

    monkeypatch.setattr(recorded.__mro__[2], "propose_draft_token_ids", propose)
    monkeypatch.setattr(_Processor, "is_deferred_out", False, raising=False)
    monkeypatch.setattr(_Processor, "prepare_sampled_ids", lambda self, batch: [])
    monkeypatch.setattr(_Processor, "prepare_draft_ids", read_drafts, raising=False)
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = Reading(0, _config())
    # ATOM's non-deferred path receives no earlier forward's draft ids.
    runner.tokenID_processor.recv_draft_token_ids = lambda: []
    for batch in batches:
        runner.forward(batch)
    runner.exit()

    lines = read(tmp_path / "real")[0]
    outputs = [b.produces_output() for b in batches]
    assert any(outputs) and not all(outputs)
    assert [("t_draft_read_ns" in x) for x in lines] == outputs
    assert not any("t_draft_recv_ns" in x for x in lines)
    posts = []
    for s, x in zip(label(tmp_path / "real" / "dp0.jsonl")[0], lines):
        if "t_draft_read_ns" in x:
            (_, r1), (d0, d1) = x["t_read_ns"], x["t_draft_read_ns"]
            assert r1 < d0 and d1 - d0 > 7000 and d1 < x["t_exit_ns"]
            posts.append((s.host["W_post"], (x["t_exit_ns"] - r1 - (d1 - d0)) * 1e-9))
    assert len(posts) == sum(outputs)
    assert [got for got, _ in posts] == pytest.approx([want for _, want in posts])


def test_a_recorder_that_does_not_defer_stamps_atom_s_own_draft_read():
    """Over a processor with exactly ATOM's methods that does not defer, a
    recorder with a drafter builds, and its `prepare_draft_ids` is stamped as
    the forward's own draft read and nothing else."""
    calls = _atom_processor_calls()
    processor = SimpleNamespace(**{name: lambda *a: None for name in calls})
    processor.is_deferred_out = False
    clock = iter(range(1, 100))

    class _Base:
        def __init__(self):
            self.tokenID_processor = processor
            self.drafter = SimpleNamespace(compute_draft_kv=None)

    class _Recorded(StepRecording, _Base):
        @staticmethod
        def step_clock_ns():
            return next(clock)

    runner = _Recorded()
    runner._parity_record = object()
    processor.prepare_draft_ids(None, None)
    assert set(runner._stamps) == {"t_draft_read_ns"}

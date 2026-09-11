"""What the sampler has to get right, checked without a device or a second.

`isolation.py` is the judge; this is its only witness. So the tests are about
the two ways a witness fails: saying something the judge cannot read, and
staying quiet about a reading it never got. The clock, the sleep and `rocm-smi`
are all injected, so a hundred samples cost nothing and the loop's behaviour --
when it stops, what it stamps, what it does when the tool fails -- is what is
actually under test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sampler_mod = _load("gpu_sampler")
isolation = _load("isolation")


def _cards(used_bytes=0, use=0.0, vram=0):
    return {
        "card0": {
            "GPU use (%)": str(use),
            "GPU Memory Allocated (VRAM%)": str(vram),
            "VRAM Total Used Memory (B)": str(used_bytes),
            "GUID": "4123",
            "Device ID": "0x74a1",
        }
    }


class FakeSmi:
    """`rocm-smi`, as a dict per call, or an exception where one is wanted."""

    def __init__(self, cards=None, pids=None, fail=None):
        self.cards = cards if cards is not None else _cards()
        self.pids = pids if pids is not None else {"system": {}}
        self.fail = fail
        self.calls = []

    def __call__(self, smi, arguments, timeout=30.0):
        self.calls.append(tuple(arguments))
        if self.fail:
            return None, self.fail
        if "--showpids" in arguments:
            return dict(self.pids), None
        return dict(self.cards), None


class Clock:
    """A monotonic clock that only moves when something sleeps."""

    def __init__(self, start=1000.0):
        self.t = start
        self.slept = []

    def now(self):
        return self.t

    def wall(self):
        return 1_700_000_000.0 + (self.t - 1000.0)

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


@pytest.fixture
def smi(monkeypatch):
    fake = FakeSmi()
    monkeypatch.setattr(sampler_mod, "_smi_json", fake)
    return fake


@pytest.fixture
def clock():
    return Clock()


def _sampler(tmp_path, clock, **kwargs):
    return sampler_mod.Sampler(
        str(tmp_path / "gpu.jsonl"),
        now=clock.now,
        wall=clock.wall,
        sleep=clock.sleep,
        **kwargs,
    )


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


class TestTheFileTheAuditReads:
    def test_a_sample_carries_every_field_isolation_parses(self, tmp_path, smi, clock):
        s = _sampler(tmp_path, clock, interval=0.0)
        s.run(limit=1)
        (row,) = _rows(s.out)
        assert set(row) >= {"t", "phase", "visible", "own_pids", "smi", "pids"}
        assert isinstance(row["t"], float)

    def test_the_audit_reads_back_what_the_sampler_wrote(self, tmp_path, smi, clock):
        """The contract that matters: these two files are one interface."""
        s = _sampler(tmp_path, clock, interval=1.0, phase="baseline")
        s.run(limit=3)
        samples = isolation.read(s.out)
        assert len(samples) == 3
        verdict = isolation.audit(samples)
        assert verdict["verdict"] == "clean"
        assert verdict["baseline_provenance"] == "phase-stamped"

    def test_a_neighbour_in_the_baseline_is_seen_through_the_sampler(
        self, tmp_path, monkeypatch, clock
    ):
        """A card already holding a gigabyte before our server starts.

        The mask is set here rather than inherited: whether card0 is a
        neighbour or one of ours is the whole difference between this verdict
        and the next test's, and the harness that runs this suite sets a mask
        of its own.
        """
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
        busy = FakeSmi(cards=_cards(used_bytes=1 << 30, use=44.0, vram=1))
        monkeypatch.setattr(sampler_mod, "_smi_json", busy)
        s = _sampler(tmp_path, clock, interval=1.0, phase="baseline")
        s.run(limit=2)
        verdict = isolation.audit(isolation.read(s.out))
        assert verdict["verdict"] == "node_busy"

    def test_the_same_busy_card_is_worse_news_when_it_is_ours(
        self, tmp_path, monkeypatch, clock
    ):
        """Identical readings, opposite meanings.

        A busy neighbour costs bandwidth we might notice in a timing. A busy
        card of our own puts somebody else's bytes inside our device-wide
        memory readings, which no later subtraction recovers.
        """
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        busy = FakeSmi(cards=_cards(used_bytes=1 << 30, use=44.0, vram=1))
        monkeypatch.setattr(sampler_mod, "_smi_json", busy)
        s = _sampler(tmp_path, clock, interval=1.0, phase="baseline")
        s.run(limit=2)
        verdict = isolation.audit(isolation.read(s.out))
        assert verdict["verdict"] == "own_contaminated"
        assert "already in use at the baseline" in verdict["problems"][0]

    def test_the_visible_mask_is_recorded_as_isolation_spells_it(
        self, tmp_path, smi, clock, monkeypatch
    ):
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        s = _sampler(tmp_path, clock, interval=0.0)
        s.run(limit=1)
        assert _rows(s.out)[0]["visible"] == "all"
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
        s.run(limit=2)
        assert _rows(s.out)[-1]["visible"] == "2,3"

    def test_our_own_pids_are_not_left_to_look_like_a_tenant(
        self, tmp_path, monkeypatch, clock
    ):
        ours = FakeSmi(pids={"system": {"PID4242": "server"}})
        monkeypatch.setattr(sampler_mod, "_smi_json", ours)
        s = _sampler(tmp_path, clock, interval=0.0, own_pids=["4242"])
        s.run(limit=1)
        verdict = isolation.audit(isolation.read(s.out))
        assert verdict["foreign_pids"] == []
        assert verdict["own_pids"] == ["4242"]

    def test_a_pid_that_is_not_ours_is_reported(self, tmp_path, monkeypatch, clock):
        theirs = FakeSmi(pids={"system": {"PID99": "someone"}})
        monkeypatch.setattr(sampler_mod, "_smi_json", theirs)
        s = _sampler(tmp_path, clock, interval=0.0, own_pids=["4242"])
        s.run(limit=1)
        verdict = isolation.audit(isolation.read(s.out))
        assert verdict["foreign_pids"] == ["99"]


class TestAReadingThatFailedIsStillARecord:
    def test_a_failed_reading_is_written_with_its_reason(
        self, tmp_path, monkeypatch, clock
    ):
        """Dropped, it would leave a gap that reads as a quiet node."""
        broken = FakeSmi(fail="rocm-smi exited 1: unable to open /dev/kfd")
        monkeypatch.setattr(sampler_mod, "_smi_json", broken)
        s = _sampler(tmp_path, clock, interval=0.0)
        s.run(limit=2)
        rows = _rows(s.out)
        assert len(rows) == 2
        assert all("/dev/kfd" in row["error"] for row in rows)
        assert all(row["smi"] == {} for row in rows)

    def test_a_file_of_failed_readings_is_not_an_idle_node(
        self, tmp_path, monkeypatch, clock
    ):
        broken = FakeSmi(fail="timeout")
        monkeypatch.setattr(sampler_mod, "_smi_json", broken)
        s = _sampler(tmp_path, clock, interval=0.0)
        s.run(limit=2)
        verdict = isolation.audit(isolation.read(s.out))
        assert verdict["verdict"] == "unwatched"
        assert "unwatched" in verdict["problems"][0]

    def test_some_failed_readings_leave_the_window_partly_unobserved(
        self, tmp_path, monkeypatch, clock
    ):
        good = FakeSmi()
        monkeypatch.setattr(sampler_mod, "_smi_json", good)
        s = _sampler(tmp_path, clock, interval=0.0, phase="baseline")
        s.run(limit=2)
        good.fail = "timeout"
        s.run(limit=3)
        verdict = isolation.audit(isolation.read(s.out))
        assert verdict["verdict"] == "unknown"
        assert any("unobserved" in u for u in verdict["unknowns"])

    def test_it_says_so_when_nothing_ever_answered(self, tmp_path, monkeypatch, capsys):
        broken = FakeSmi(fail="no such file")
        monkeypatch.setattr(sampler_mod, "_smi_json", broken)
        out = str(tmp_path / "gpu.jsonl")
        assert sampler_mod.main([out, "--once"]) == 1
        assert "unwatched" in capsys.readouterr().err

    def test_one_good_reading_is_not_an_unwatched_run(self, tmp_path, smi, capsys):
        out = str(tmp_path / "gpu.jsonl")
        assert sampler_mod.main([out, "--once"]) == 0


class TestThePhaseIsWhatTheRunSays:
    def test_the_phase_file_is_re_read_every_sample(self, tmp_path, smi, clock):
        """A sampler started before the first server has to be told, later,
        that the baseline is over -- it cannot be told at launch."""
        phase_file = tmp_path / "phase.json"
        phase_file.write_text(json.dumps({"phase": "baseline", "own_pids": []}))
        s = _sampler(
            tmp_path, clock, interval=0.0, phase="run", phase_file=str(phase_file)
        )
        with open(s.out, "a", encoding="utf-8") as handle:
            s.one(handle)
        phase_file.write_text(json.dumps({"phase": "serving", "own_pids": ["7"]}))
        s.run(limit=2)
        rows = _rows(s.out)
        assert [r["phase"] for r in rows] == ["baseline", "serving"]
        assert rows[-1]["own_pids"] == ["7"]

    def test_a_broken_phase_file_leaves_the_sampler_sampling(
        self, tmp_path, smi, clock
    ):
        phase_file = tmp_path / "phase.json"
        phase_file.write_text("{not json")
        s = _sampler(
            tmp_path, clock, interval=0.0, phase="run", phase_file=str(phase_file)
        )
        s.run(limit=1)
        (row,) = _rows(s.out)
        assert row["phase"] == "run"
        assert "note" in row

    def test_a_missing_phase_file_is_not_a_stopped_sampler(self, tmp_path, smi, clock):
        s = _sampler(
            tmp_path,
            clock,
            interval=0.0,
            phase="run",
            phase_file=str(tmp_path / "never"),
        )
        s.run(limit=2)
        assert len(_rows(s.out)) == 2


class TestTheLoop:
    def test_it_sleeps_the_interval_between_samples(self, tmp_path, smi, clock):
        s = _sampler(tmp_path, clock, interval=5.0)
        s.run(limit=3)
        assert clock.slept == [5.0, 5.0]

    def test_a_duration_ends_it(self, tmp_path, smi, clock):
        s = _sampler(tmp_path, clock, interval=5.0)
        s.run(duration=12.0)
        assert len(_rows(s.out)) == 4

    def test_being_told_to_stop_finishes_the_sample_in_hand(self, tmp_path, smi, clock):
        """The harness signals it; a half-written last line is one the audit
        skips, which is a silently shorter window."""
        s = _sampler(tmp_path, clock, interval=1.0)
        original = s.one

        def one(handle):
            row = original(handle)
            if s.written == 2:
                s.stop()
            return row

        s.one = one
        s.run()
        rows = _rows(s.out)
        assert len(rows) == 2
        assert rows[-1]["smi"]

    def test_it_appends_rather_than_truncating(self, tmp_path, smi, clock):
        """A cell's real side is three server lifetimes and one audit window;
        a sampler restarted inside it must not erase the baseline."""
        first = _sampler(tmp_path, clock, interval=0.0, phase="baseline")
        first.run(limit=1)
        second = _sampler(tmp_path, clock, interval=0.0, phase="run")
        second.run(limit=2)
        assert [r["phase"] for r in _rows(first.out)] == ["baseline", "run", "run"]

    def test_once_takes_exactly_one_sample(self, tmp_path, smi):
        out = str(tmp_path / "gpu.jsonl")
        assert sampler_mod.main([out, "--once", "--phase", "baseline"]) == 0
        rows = _rows(out)
        assert len(rows) == 1 and rows[0]["phase"] == "baseline"


class TestWhatItAsksRocmSmiFor:
    def test_it_asks_for_the_absolute_bytes_the_audit_prefers(
        self, tmp_path, smi, clock
    ):
        """A percent is rounded to the integer, and a 0.7 GiB neighbour on a
        192 GiB card reports 0%."""
        _sampler(tmp_path, clock, interval=0.0).run(limit=1)
        asked = set(smi.calls)
        assert any("--showmeminfo" in call and "vram" in call for call in asked)
        assert any("--showpids" in call for call in asked)
        assert all("--json" in call for call in asked)

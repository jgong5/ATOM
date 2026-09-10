"""The check that justifies the deferred-output clock rule.

`_defers_output` in the engine core charges a deferred step's cost after the
drain, so the previous step's tokens are stamped at the previous step's
completion. That is only right if the real engine publishes them there, and the
`is_deferred_out` flag does not say so -- it describes the buffer, not when the
host got it. This module is the measurement that does, and these tests cover
the readings that would change the answer.

Two things have to be right for the measurement to mean anything.

The device timeline. A step row carries a host launch stamp (`started_at`) and
the CUDA-event span of the forward (`seconds`); adding them describes a launch,
not an execution, and on a real cell the sum has consecutive steps overlapping
because the host runs ahead. Work on one stream is serialised, so a step ends
at `max(previous end, its launch) + its cost`.

The producing step. Asking only "which step completed last before this
publication" is true of every publication and would pass whatever the runtime
did. So the producer comes from the lifecycle witness -- the step that
*returned* a sequence's tokens, minus one when that step's output was deferred
-- and a cell without a witness is reported as not checked rather than passed.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


publication = _load("publication")


def _cell(tmp_path, steps, pubs, side="real", deferred=True):
    """A cell directory holding both artifacts the check needs.

    `steps` is [(started_at, seconds)]. `pubs` is [(seq, returned_at_step,
    first_token_time)] -- the step index whose output *carried* the token,
    which with a deferred output is one after the step that produced it.
    """
    (tmp_path / f"{side}_steps.jsonl").write_text(
        "\n".join(json.dumps({"started_at": at, "seconds": cost})
                  for at, cost in steps) + "\n")
    events = []
    for i in range(max([len(steps)] + [r + 1 for _, r, _ in pubs])):
        events.append({"event": "step_output", "deferred": deferred,
                       "returned": [s for s, r, _ in pubs if r == i]})
    for seq, _, t in pubs:
        events.append({"event": "seq_update", "seq": seq,
                       "first_token_published": True, "first_token_time": t})
    (tmp_path / f"{side}.lifecycle.1234.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n")
    return str(tmp_path)


class TestTheDeviceTimeline:
    def test_a_step_launched_early_still_waits_for_the_stream(self):
        """The reading that made the first pass of this wrong.

        Measured on the 27B TP=1 cell: step 2 was launched at +7.12 while step
        1's forward ran to +11.99. `started_at + seconds` puts step 2 finishing
        at +11.59, before the step it queues behind.
        """
        steps = [{"started_at": 0.0, "seconds": 7.005},
                 {"started_at": 6.7465, "seconds": 4.982},
                 {"started_at": 7.1225, "seconds": 4.4694}]
        _, ends = publication.timeline(steps)
        assert ends == pytest.approx([7.005, 11.987, 16.4564])

    def test_an_idle_device_starts_when_the_host_says(self):
        steps = [{"started_at": 0.0, "seconds": 1.0},
                 {"started_at": 10.0, "seconds": 1.0}]
        _, ends = publication.timeline(steps)
        assert ends == [1.0, 11.0]

    def test_a_step_launched_at_zero_is_not_dropped(self):
        """A virtual-clock run stamps its first launch at exactly 0.0.

        Testing that stamp for truth rather than for None silently renumbers
        every step after it, which moves each publication one producer along.
        """
        steps = [{"started_at": 0.0, "seconds": 2.0},
                 {"started_at": 2.0, "seconds": 3.0}]
        kept, ends = publication.timeline(steps)
        assert len(kept) == 2 and ends == [2.0, 5.0]


class TestTheProducingStep:
    def test_a_deferred_step_returns_the_previous_steps_tokens(self):
        events = [{"event": "step_output", "deferred": True, "returned": []},
                  {"event": "step_output", "deferred": True, "returned": ["0"]},
                  {"event": "step_output", "deferred": True,
                   "returned": ["1", "2"]}]
        assert publication.producers(events) == {"0": 0, "1": 1, "2": 1}

    def test_an_undeferred_step_produces_its_own(self):
        events = [{"event": "step_output", "deferred": False,
                   "returned": ["0"]}]
        assert publication.producers(events) == {"0": 0}

    def test_only_the_first_return_counts(self):
        """Every decode step returns the sequence again; the first token has
        one producer."""
        events = [{"event": "step_output", "deferred": True, "returned": []},
                  {"event": "step_output", "deferred": True, "returned": ["0"]},
                  {"event": "step_output", "deferred": True, "returned": ["0"]}]
        assert publication.producers(events)["0"] == 0


class TestTheVerdict:
    def test_publishing_at_the_producing_steps_completion_passes(self, tmp_path):
        """What the real cells do: ~0.12s of host drain latency, no more.

        Step 0 ends at 7.005 and step 1 at 11.987. The token step 1 returned
        was produced by step 0, and it is published at 7.1203.
        """
        cell = _cell(tmp_path,
                     [(0.0, 7.005), (6.7465, 4.982), (7.1225, 4.4694)],
                     [("0", 1, 7.1203)])
        assert publication.main(cell) == 0

    def test_publishing_a_step_late_fails(self, tmp_path):
        """A runtime that synchronised would look like this: the token step 1
        returned appears only once step 1's own device work has finished, and
        charging the drain at step 0's completion would then be a whole step
        too early."""
        cell = _cell(tmp_path,
                     [(0.0, 7.005), (6.7465, 4.982), (7.1225, 4.4694)],
                     [("0", 1, 11.99)])
        assert publication.main(cell) == 1

    def test_a_token_returned_by_the_first_step_cannot_be_placed(self, tmp_path):
        """Nothing produced it, so there is no completion to compare against;
        that is a broken witness, not a pass."""
        cell = _cell(tmp_path, [(0.0, 7.005), (6.7465, 4.982)],
                     [("0", 0, 7.02)])
        assert publication.main(cell) == 1

    def test_a_cell_without_a_witness_is_not_checked(self, tmp_path, capsys):
        """Placement alone is true of any publication, so it must not pass."""
        (tmp_path / "real_steps.jsonl").write_text(
            json.dumps({"started_at": 0.0, "seconds": 1.0}) + "\n")
        assert publication.main(str(tmp_path)) == 2
        assert "NOT CHECKED" in capsys.readouterr().out

    def test_a_witness_from_another_run_is_not_checked(self, tmp_path, capsys):
        """More witnessed steps than step rows: the two artifacts disagree
        about which run they describe, and aligning them by index would place
        every publication against the wrong step."""
        cell = _cell(tmp_path, [(0.0, 1.0)],
                     [("0", 1, 1.05), ("1", 2, 2.05)])
        assert publication.main(cell) == 2
        assert "do not describe the same run" in capsys.readouterr().out

    def test_per_rank_step_files_are_read(self, tmp_path):
        """At TP>1 the sink writes one file per rank; rank 0 stands in."""
        (tmp_path / "real_steps.tp0.jsonl").write_text(
            json.dumps({"started_at": 0.0, "seconds": 1.0}) + "\n")
        (tmp_path / "real_steps.tp1.jsonl").write_text(
            json.dumps({"started_at": 0.0, "seconds": 1.0}) + "\n")
        _, name = publication.steps_of(str(tmp_path))
        assert name == "real_steps.tp0.jsonl"

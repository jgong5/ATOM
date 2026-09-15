"""Which step's completion does a real publication land on?

The check behind the engine's deferred-output clock rule (``_defers_output`` in
``atom/model_engine/engine_core.py``). That rule charges a deferred step's cost
after the drain, so the previous step's tokens are stamped at the previous
step's completion. `is_deferred_out` does not justify it: the flag describes
the buffer, not when the host received it, and a runtime that synchronised
would hand the tokens over only once the current step's device work had
finished. Then charging the drain a step earlier would be wrong in the other
direction. So the rule is taken from what the run recorded.

Two artifacts are needed and they answer different halves.

The step table gives the device timeline. `started_at` is a host stamp taken
before the forward is submitted and `seconds` is the CUDA-event span of the
forward itself, so adding them describes a launch, not an execution: on the 27B
TP=1 cell the host launched step 2 at +7.12 while step 1's forward ran to
+11.99, and the naive sum has consecutive steps overlapping. Work on one stream
is serialised, so

    end[i] = max(end[i-1], started_at[i]) + seconds[i]

The lifecycle witness gives the producing step. Without it the question cannot
be asked at all -- "the last step that completed before this publication" is
true of any publication and proves nothing. The witness records, per step, the
sequences whose tokens that step *returned* and whether the output was
deferred, so the step that produced a sequence's first token is the step before
the one that returned it. The publication instant is then compared against that
step's completion and against the next one's, and the nearer wins.

    python scripts/compass/publication.py <cell-dir> [real|modelled]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            # The sink is killed mid-write when a run ends; a torn final line
            # costs one step, not the artifact.
            pass
    return out


def steps_of(cell, side="real"):
    for name in (f"{side}_steps.jsonl", f"{side}_steps.tp0.jsonl"):
        path = Path(cell) / name
        if path.exists():
            return rows(path), name
    raise SystemExit(f"no step file in {cell}")


def lifecycle_of(cell, side="real"):
    found = sorted(Path(cell).glob(f"{side}.lifecycle.*.jsonl"))
    if not found:
        return None, None
    return rows(found[0]), found[0].name


def timeline(steps: list[dict]) -> tuple[list[dict], list[float]]:
    """Device completion of each step, in the order they were launched.

    ``started_at`` is tested against None rather than for truth: the first step
    of a virtual-clock run is stamped at exactly 0.0, and dropping it silently
    renumbers every step after it.
    """
    steps = sorted((s for s in steps if s.get("started_at") is not None),
                   key=lambda r: r["started_at"])
    ends, prev = [], None
    for step in steps:
        begin = (step["started_at"] if prev is None
                 else max(prev, step["started_at"]))
        prev = begin + float(step["seconds"])
        ends.append(prev)
    return steps, ends


def witnessed_steps(events: list[dict]) -> int:
    return sum(1 for e in events if e.get("event") == "step_output")


def producers(events: list[dict]) -> dict[str, int]:
    """Step index that produced each sequence's first returned token.

    A deferred step returns the *previous* step's tokens, so the producer is
    the step before the one that returned them. A step that returns its own is
    its own producer.
    """
    out, step = {}, -1
    for event in events:
        if event.get("event") != "step_output":
            continue
        step += 1
        for seq in event.get("returned") or []:
            out.setdefault(str(seq),
                           step - 1 if event.get("deferred") else step)
    return out


def published(events: list[dict]) -> dict[str, float]:
    out = {}
    for event in events:
        if (event.get("event") == "seq_update"
                and event.get("first_token_published")
                and event.get("first_token_time")):
            out.setdefault(str(event["seq"]), float(event["first_token_time"]))
    return out


def main(cell, side="real") -> int:
    steps, name = steps_of(cell, side)
    steps, ends = timeline(steps)
    events, witness = lifecycle_of(cell, side)

    print(f"{cell} [{name}] {len(steps)} steps")
    for i in range(min(4, len(steps))):
        print(f"    step {i}: launch {steps[i]['started_at'] - ends[0]:+9.4f}"
              f"  end {ends[i] - ends[0]:+9.4f}"
              f"  cost {float(steps[i]['seconds']):.4f}")

    if events is None:
        print("  NOT CHECKED: no lifecycle witness in this cell, so the step "
              "that produced each token is unknown and placement alone would "
              "be true of any publication. Re-run the cell with the witness "
              "enabled to check the rule.")
        return 2

    produced, instants = producers(events), published(events)
    print(f"  witness {witness}: {len(produced)} sequences returned, "
          f"{len(instants)} first tokens published")
    witnessed = witnessed_steps(events)
    if witnessed != len(ends):
        print(f"  NOT CHECKED: {len(ends)} step rows against {witnessed} "
              f"witnessed steps -- the two artifacts do not describe the same "
              f"run, and aligning them by index would place every publication "
              f"against the wrong step")
        return 2

    own, late, unplaceable, slacks = 0, 0, 0, []
    for seq, instant in sorted(instants.items()):
        producer = produced.get(seq)
        if producer is None or not 0 <= producer < len(ends):
            unplaceable += 1
            continue
        here = instant - ends[producer]
        slacks.append(here)
        if producer + 1 < len(ends):
            there = abs(instant - ends[producer + 1])
            if abs(here) <= there:
                own += 1
            else:
                late += 1
        else:
            own += 1

    print(f"  publications: {own} on the producing step's completion, "
          f"{late} nearer the next step's, {unplaceable} unplaceable")
    if slacks:
        print(f"  slack after the producing step's completion: "
              f"median {statistics.median(slacks):.4f}s  "
              f"min {min(slacks):.4f}s  max {max(slacks):.4f}s")
    if late or unplaceable or not slacks:
        print("  RULE VIOLATED: a publication did not land on the completion "
              "of the step that produced it, so the deferred-output clock "
              "rule does not hold for this runtime")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], *sys.argv[2:]))

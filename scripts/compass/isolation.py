"""Was this measurement alone on the devices it used, for the whole of it?

A pair of `rocm-smi` snapshots taken before and after a run says what the box
looked like at two instants. It cannot distinguish a quiet run from one that
shared its cards with a neighbour for the middle nine minutes, and free memory
says nothing about who was computing. So the run samples `rocm-smi` throughout,
and this reads the samples:

    python scripts/compass/isolation.py <cell>/gpu.jsonl [--json out.json]

Two findings, deliberately separate, because they invalidate different things:

  * **The selected devices were not ours.** A card this run owned was already
    occupied at the baseline -- a sample taken before this run's server
    started. `torch.cuda.mem_get_info` is *device-wide*: everything resident on
    that one card is charged to this configuration's budget, whoever put it
    there. Timings and memory from such a run are both unusable. Exit 1.

  * **The node was not quiet.** A card this run did not own was busy. That
    card's memory is *not* in this run's readings -- device-wide is not
    node-wide, and nothing on card 4 can enter card 0's `mem_get_info`. What it
    shares is the host: CPU, PCIe, the memory bus, the power envelope. Under
    the conservative all-node-quiet timing policy this run's timings are
    advisory rather than gate evidence. Exit 2. The artifact is kept and
    labelled, not discarded.

On 2026-09-10 a 0.6B constants probe recorded 112.9 GB of "non-torch" memory.
That probe was placed on devices 4--7 and another tenant was already on 4 and
5: the contamination was of the selected devices themselves, which is the first
finding, not the second.
"""

from __future__ import annotations

import argparse
import json
import sys

#: What counts as occupied. `rocm-smi` reports VRAM as a rounded integer
#: percent, so on a 192 GiB card a process holding 0.7 GiB reports 0% -- which
#: is exactly the compute-bound small-allocation case. Absolute bytes are
#: therefore preferred when the sample carries them (`--showmeminfo vram`), and
#: the percent is the fallback for samples written before it was recorded.
VRAM_IDLE_PCT = 3
#: The floor has to clear the driver's own per-card reservation, or every card
#: on the node reads occupied and no run can ever be clean. Sampled on an idle
#: MI308X node with nothing of ours started: 297,779,200 B (284 MiB) on seven
#: cards and 297,783,296 B on the eighth, byte-identical across cards nobody
#: was using -- a reservation, not a tenant. 512 MiB sits above it with room to
#: spare and below the 0.7 GiB compute-bound neighbour this exists to catch,
#: and is still an order of magnitude finer than the 3% percent path (5.9 GiB
#: on a 192 GiB card).
VRAM_IDLE_BYTES = 512 << 20
USE_IDLE_PCT = 5

#: How many samples an activity reading with no visible memory must span before
#: it is believed. One is not enough to call busy and not enough to call idle:
#: `rocm-smi` flickers a few percent while it queries, and a rounded-zero VRAM
#: does not mean no process. Such a sample is reported `unknown`.
SUSTAINED_SAMPLES = 2

BUSY, SPIKE, IDLE = "busy", "spike", "idle"

#: Phases a sample may be stamped with. Only `baseline` is taken before this
#: run's own server exists, so only `baseline` can prove a card was another
#: tenant's. A sample during our own startup, or while our own predecessor is
#: still releasing memory, shows our bytes and says nothing about neighbours.
BASELINE_PHASE = "baseline"


def _state(card: dict) -> str:
    """Busy, an unexplained spike, or idle.

    `busy` is evidence of somebody working: bytes on the card above the idle
    floor, or a percent reading high enough that rounding cannot explain it.
    `spike` is activity with no memory *visible* -- which is not the same as no
    memory, because the percent is rounded and this sample may carry no byte
    count at all. It is the evidence limit, and it is reported as such rather
    than resolved in either direction.
    """
    used = card.get("used_bytes")
    if used is not None:
        if used >= VRAM_IDLE_BYTES:
            return BUSY
        return SPIKE if card["use"] >= USE_IDLE_PCT else IDLE
    if card["vram"] >= VRAM_IDLE_PCT:
        return BUSY
    return SPIKE if card["use"] >= USE_IDLE_PCT else IDLE


def _cards(sample: dict) -> dict:
    """Per-card readings from one `rocm-smi --json` blob, keyed by index.

    The index is the physical card number `rocm-smi` reports, and `guid`/`did`
    carry the device's own identity beside it, so a reading can be tied to a
    device rather than to a position in a list that renumbers under
    `HIP_VISIBLE_DEVICES`.
    """
    out = {}
    for name, values in (sample.get("smi") or {}).items():
        if not name.startswith("card") or not isinstance(values, dict):
            continue
        try:
            index = int(name[4:])
        except ValueError:
            continue

        def number(key):
            try:
                return float(values.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        used = None
        for key in ("VRAM Total Used Memory (B)", "VRAM Total Used Memory (b)",
                    "vram_total_used_memory"):
            if values.get(key) not in (None, ""):
                try:
                    used = float(values[key])
                except (TypeError, ValueError):
                    used = None
                break
        out[index] = {"use": number("GPU use (%)"),
                      "vram": number("GPU Memory Allocated (VRAM%)"),
                      "used_bytes": used,
                      "guid": values.get("GUID"),
                      "did": values.get("Device ID")}
    return out


def _pids(sample: dict) -> set:
    """KFD process ids in one sample.

    `rocm-smi --showpids` inside a container cannot map a process to a card, so
    this is a node-level signal only: a process id present while we ran that is
    not one of ours belongs to somebody. It answers "was anyone else on the
    box", never "was anyone else on card 0".
    """
    pids = sample.get("pids")
    if isinstance(pids, dict):
        # `rocm-smi --showpids --json` nests under "system"; an empty process
        # list is `{"system": {}}`, which is not the same as no such key.
        if "system" in pids:
            pids = pids["system"] or {}
        return {str(k)[3:] if str(k).startswith("PID") else str(k) for k in pids}
    if isinstance(pids, list):
        return {str(p) for p in pids}
    return set()


def read(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                # A sample written while `rocm-smi` was interrupted. Skipping it
                # loses one reading; refusing the file loses the run.
                continue
    return samples


def _unwatched() -> dict:
    # Not a pass. A run with no samples is a run nobody watched, which is the
    # state this exists to stop being reported as a clean one.
    return {"samples": 0, "span": [None, None], "owned": [], "owned_peak": {},
            "cards_seen": [], "busy_neighbours": {}, "unexplained": {},
            "occupied_at_start": [], "baseline_samples": 0,
            "baseline_provenance": "none", "foreign_pids": [], "own_pids": [],
            "own_clean": False, "node_quiet": False, "verdict": "unwatched",
            "isolated": False,
            "problems": ["no samples were taken: nothing watched this run"]}


def _tally(store: dict, index: int, card: dict, sample: dict) -> None:
    seen = store.setdefault(index, {
        "samples": 0, "max_use": 0.0, "max_vram": 0.0, "max_used_bytes": None,
        "guid": card.get("guid"), "first": sample.get("t"), "last": None})
    seen["samples"] += 1
    seen["max_use"] = max(seen["max_use"], card["use"])
    seen["max_vram"] = max(seen["max_vram"], card["vram"])
    if card.get("used_bytes") is not None:
        seen["max_used_bytes"] = max(seen["max_used_bytes"] or 0.0,
                                     card["used_bytes"])
    seen["last"] = sample.get("t")


def _describe(store: dict, total: int) -> str:
    return ", ".join(
        "%d (%d/%d samples, up to %.0f%% use, %s)"
        % (i, v["samples"], total, v["max_use"],
           ("%.1f GiB" % (v["max_used_bytes"] / (1 << 30)))
           if v["max_used_bytes"] is not None else "%.0f%% VRAM" % v["max_vram"])
        for i, v in sorted(store.items()))


def audit(samples: list[dict]) -> dict:
    if not samples:
        return _unwatched()

    owned = set()
    own_pids = set()
    for sample in samples:
        visible = str(sample.get("visible") or "")
        if visible and visible != "all":
            owned |= {int(d) for d in visible.split(",") if d.strip().isdigit()}
        own_pids |= {str(p) for p in (sample.get("own_pids") or [])}

    # Which samples predate our own server. Explicit stamps when the run wrote
    # them; otherwise the first sample, which is weaker evidence and is
    # labelled as such rather than quietly treated as equivalent.
    stamped = [s for s in samples if s.get("phase") == BASELINE_PHASE]
    if stamped:
        baseline, provenance = stamped, "phase-stamped"
    else:
        baseline, provenance = samples[:1], "first-sample"

    dirty_own: dict = {}
    unclear_own: dict = {}
    for sample in baseline:
        for index, card in _cards(sample).items():
            if index not in owned:
                continue
            state = _state(card)
            if state == BUSY:
                dirty_own[index] = card
            elif state == SPIKE:
                unclear_own[index] = card

    busy_neighbours: dict = {}
    unclear_neighbours: dict = {}
    own_peak: dict = {}
    foreign_pids: dict = {}
    for sample in samples:
        for index, card in _cards(sample).items():
            state = _state(card)
            if index in owned:
                peak = own_peak.setdefault(
                    index, {"use": 0.0, "vram": 0.0, "guid": card.get("guid")})
                peak["use"] = max(peak["use"], card["use"])
                peak["vram"] = max(peak["vram"], card["vram"])
            elif state == BUSY:
                _tally(busy_neighbours, index, card, sample)
            elif state == SPIKE:
                _tally(unclear_neighbours, index, card, sample)
        for pid in _pids(sample) - own_pids:
            entry = foreign_pids.setdefault(pid, {"samples": 0,
                                                  "first": sample.get("t")})
            entry["samples"] += 1

    # Activity with no memory visible, sustained, is somebody working with a
    # small allocation -- the 0.7 GiB compute-bound case a rounded percent
    # hides. One isolated sample stays unexplained.
    for index in [i for i, v in unclear_neighbours.items()
                  if v["samples"] >= SUSTAINED_SAMPLES]:
        busy_neighbours[index] = unclear_neighbours.pop(index)

    own_problems, node_problems, unknowns = [], [], []
    if dirty_own:
        own_problems.append(
            "cards %s were already in use at the baseline (%s): this run's "
            "device-wide memory readings include somebody else's bytes"
            % (", ".join(str(i) for i in sorted(dirty_own)), provenance))
    if busy_neighbours:
        node_problems.append("cards %s were busy alongside it"
                             % _describe(busy_neighbours, len(samples)))
    if foreign_pids:
        node_problems.append(
            "%d process(es) on the node were not ours (%s)"
            % (len(foreign_pids), ", ".join(sorted(foreign_pids)[:6])))
    if unclear_own:
        unknowns.append(
            "own cards %s showed activity at the baseline with no memory "
            "visible; a rounded 0%% VRAM does not mean no process"
            % ", ".join(str(i) for i in sorted(unclear_own)))
    if unclear_neighbours:
        unknowns.append(
            "cards %s showed a single unexplained activity sample"
            % _describe(unclear_neighbours, len(samples)))

    own_clean = not own_problems
    node_quiet = not node_problems
    if not own_clean:
        verdict = "own_contaminated"
    elif not node_quiet:
        verdict = "node_busy"
    elif unknowns:
        verdict = "unknown"
    else:
        verdict = "clean"
    return {
        "samples": len(samples),
        "span": [samples[0].get("t"), samples[-1].get("t")],
        "owned": sorted(owned),
        "own_pids": sorted(own_pids),
        "owned_peak": {str(k): v for k, v in sorted(own_peak.items())},
        "cards_seen": sorted(_cards(samples[0])),
        "baseline_samples": len(baseline),
        "baseline_provenance": provenance,
        "busy_neighbours": {str(k): v for k, v in sorted(busy_neighbours.items())},
        "unexplained": {str(k): v for k, v in sorted(unclear_neighbours.items())},
        "occupied_at_start": sorted(dirty_own),
        "unclear_at_start": sorted(unclear_own),
        "foreign_pids": sorted(foreign_pids),
        "own_clean": own_clean,
        "node_quiet": node_quiet,
        "verdict": verdict,
        # Kept for readers of older audits: the strict all-node-quiet reading.
        "isolated": verdict == "clean",
        "problems": own_problems + node_problems,
        "unknowns": unknowns,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples", help="gpu.jsonl written alongside the run")
    ap.add_argument("--json", dest="json_out", help="write the audit here")
    ap.add_argument("--allow-busy-node", action="store_true",
                    help="exit 0 when only unowned cards were busy or "
                         "unexplained. The selected devices' memory is "
                         "unaffected by them; what they share is the host. Use "
                         "when the run is reported for memory, not for timing.")
    args = ap.parse_args(argv)

    report = audit(read(args.samples))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)

    print("  %d samples over %s .. %s, cards %s owned of %s seen"
          % (report.get("samples", 0), *(report.get("span") or [None, None]),
             report.get("owned"), report.get("cards_seen")))
    print("  baseline: %d sample(s), %s"
          % (report["baseline_samples"], report["baseline_provenance"]))
    for index, peak in (report.get("owned_peak") or {}).items():
        print("    card %s: up to %.0f%% use, %.0f%% VRAM  (its own work)"
              % (index, peak["use"], peak["vram"]))
    for note in report.get("unknowns") or []:
        print("  UNKNOWN: %s" % note)

    if report["verdict"] == "unwatched":
        print("  UNWATCHED: %s" % report["problems"][0])
        print("  A run nobody sampled cannot be reported as an isolated one.")
        return 1

    if report["verdict"] == "clean":
        print("  CLEAN: the selected devices were this run's alone and no "
              "other card was busy")
        return 0

    if not report["own_clean"]:
        for problem in report["problems"]:
            print("  SELECTED DEVICES CONTAMINATED: %s" % problem)
        print("  Neither timings nor memory from this run describe this "
              "configuration.")
        return 1

    if report["verdict"] == "unknown":
        print("  NOT CLASSIFIED: the selected devices were this run's alone, "
              "so its memory\n  readings stand, but the node cannot be called "
              "quiet on this evidence.\n  The raw samples are kept beside the "
              "run.")
        return 0 if args.allow_busy_node else 3

    for problem in report["problems"]:
        print("  NODE NOT QUIET: %s" % problem)
    print("  The selected devices were this run's alone, so its memory "
          "readings stand.\n  Its timings shared the host, so under the "
          "all-node-quiet policy they are\n  advisory rather than gate "
          "evidence.")
    return 0 if args.allow_busy_node else 2


if __name__ == "__main__":
    sys.exit(main())

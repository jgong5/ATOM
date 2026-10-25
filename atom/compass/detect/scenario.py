# SPDX-License-Identifier: MIT
"""The acceptance checks every simulated deployment's runs are held to.

    python -m atom.compass.detect.scenario OUT_DIR [OUT_DIR ...] [--record LP=DIR ...]

Each OUT_DIR holds one run of one run file, each under its own
`PYTHONHASHSEED`, with the step table and run summary the authority wrote.
`check_run` returns one line per check that fails, and none when all hold:

- each run reached its finish with no refusal: every LP's last row ends at
  ``inf``, the summary counts no refusal and is no coverage report;
- every run's step table is the first's, byte for byte;
- each channel's messages were released to its receiver exactly once, seqs 0
  to n-1, and none before the receiver's clock: a release row runs from the
  receiver's clock as the grant found it to the message's arrival;
- with `sent`, the max_tokens of each request the traffic LP sent, and
  `taken`, the SSE data lines it took: each request has its ``[DONE]`` and a
  usage chunk counting every token it asked for;
- with `records`, an engine LP's name to the parity record its run wrote
  (`atom.compass.parity`): the LP's TAR rows are `expected_timeline`'s.
- above pipeline width one, the order of the stages' forwards and sends: a
  stage starts a forward no earlier than its input is released to it, and
  sends on no earlier than that forward starts and its previous send is done
  (`pipeline_timeline`, which recomputes no time);
- with `records` naming a pipeline's engine and `pipeline`, its model's layer
  count and send parts: each stage's TAR rows and the pipeline's releases are
  `stage_timelines`'.

The CLI checks the runs and the records it is given; the traffic's view is a
harness's, so `sent` is for callers that hold one.
"""

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace

from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import TAR
from atom.compass.detect.determinism import CONFIGURATION_PREFIX, compare_step_tables
from atom.compass.parity import read
from atom.compass.parity.recurrence import _waited
from atom.compass.run import (
    RAPIDSERVE,
    STEP_TABLE_FILE,
    SUMMARY_FILE,
    channel_table,
    entries,
)
from atom.compass.runner.pp_send import (
    carrier_models,
    gloo_wake,
    is_rendezvous,
    recv_done,
    split,
    woken,
)
from atom.compass.spec import MachineSpec

DONE = "data: [DONE]"


def expected_timeline(
    record,
    prefill_s: float,
    decode_s: float,
    tars,
    sync: bool = False,
    draft_s: float = 0.0,
) -> list:
    """One engine LP's TAR rows, ``(from, to)``, under a constant step stub.

    `record` maps each DP rank to its steps (`atom.compass.parity.read`).
    `tars` is the LP's TAR rows as the step table has them, read only for the
    clock each step starts at; everything after the start is computed here.

    A forward's device seconds are `prefill_s` when a row prefills plus
    `decode_s` when a row decodes. With no host laws it enters at its start or
    once the device work two forwards back ends, whichever is later, and
    returns then, or, when it produces output, once the previous output
    forward's device work ends. Above DP width one the group's device end is
    the latest over ranks, which each rank takes as its last device end, and
    as its output end on an output forward; each rank returns at its own exit,
    and the step is one TAR per distinct return, ascending. With `sync`, every
    forward returns at the group's device end, as one that synchronizes its
    stream before it returns does.

    With `draft_s`, a speculative engine's: each forward's device work runs
    `draft_s` of draft after its target, and an output forward enters no
    earlier than the previous output forward's target ends, the group's
    latest above DP width one.

    Raises ValueError when the ranks hold different step counts, or a step
    starts after the TAR rows end.
    """
    ranks = sorted(record)
    counts = {r: len(record[r]) for r in ranks}
    if len(set(counts.values())) > 1:
        raise ValueError(f"the record's ranks hold {counts} steps")
    # Per rank: device end of forwards n-2 and n-1, and of the last output one,
    # its target's under a drafter.
    device = {r: (-math.inf, -math.inf, -math.inf) for r in ranks}
    want, i = [], 0
    for k in range(len(record[ranks[0]])):
        if i >= len(tars):
            raise ValueError(f"record step {k} has no TAR row")
        now = tars[i][0]
        returns, ends, targets = set(), {}, {}
        for r in ranks:
            step = record[r][k]
            rows = [row[-1] for row in step["rows"]]
            busy = prefill_s * (not all(rows)) + decode_s * any(rows)
            two_back, last, output = device[r]
            host = max(now, two_back)
            if draft_s and step["produces_output"]:
                host = max(host, output)
            targets[r] = max(last, host) + busy
            ends[r] = targets[r] + draft_s
            returns.add((max(host, output) if step["produces_output"] else host) - now)
        end = max(ends.values())
        mark = max(targets.values()) if draft_s else end
        if sync:
            returns = {end - now}
        for r in ranks:
            _, last, output = device[r]
            device[r] = (last, end, mark if record[r][k]["produces_output"] else output)
        start = now
        for t in sorted(returns):
            want.append((start, now + t))
            start, i = now + t, i + 1
    return want


def _run_properties(table: str, summary: dict) -> list[str]:
    lines = table.splitlines()
    channels = channel_table(json.loads(lines[0].removeprefix(CONFIGURATION_PREFIX)))
    rows = [line.split() for line in lines[1:]]
    failed = []
    last = {r[0]: r for r in rows}  # each LP's last row
    for lp in sorted(lp.name for lp in channels.registry.ids()):
        if lp not in last or last[lp][2] != "inf":
            failed.append(
                f"{lp} did not reach the finish: its last row is {last.get(lp)}"
            )
    refusals = summary["schedule"]["refusals"]
    if refusals["count"] or summary["coverage_report"]:
        failed.append(f"the run refused: {refusals['reasons']}")
    seqs = defaultdict(list)
    for lp, clock, arrival, event, ch, seq, _ in rows:
        if event != "release":
            continue
        seqs[ch].append(int(seq))
        target = channels.channel(ch).target.name
        if lp != target:
            failed.append(f"{ch} seq {seq} was released to {lp}, not to {target}")
        if float(arrival) < float(clock):
            failed.append(
                f"{ch} seq {seq} arrived at {arrival}, behind {lp}'s clock at {clock}"
            )
    for ch, got in sorted(seqs.items()):
        if sorted(got) != list(range(len(got))):
            failed.append(f"{ch} released seqs {sorted(got)}, not each of 0..n-1 once")
    return failed


def _rows(table: str):
    """A step table's run file, its TAR rows by LP, its release ends by channel
    and seq."""
    lines = table.splitlines()
    run = json.loads(lines[0].removeprefix(CONFIGURATION_PREFIX))
    tars, arrived = defaultdict(list), defaultdict(dict)
    for lp, t0, t1, event, ch, seq, _ in map(str.split, lines[1:]):
        if event == TAR:
            tars[lp].append((float(t0), float(t1)))
        elif event == "release":
            arrived[ch][int(seq)] = float(t1)
    return run, tars, arrived


def _first_diff(row: str, want, got, model="the pipeline") -> list[str]:
    """The first of `got`'s `row`s that `want` does not match, as a line."""
    for i, (w, g) in enumerate(zip_longest(want, got)):
        if w is None or g is None or not all(map(math.isclose, w, g)):
            return [f"{row} {i} is {g}; {model} implies {w}"]
    return []


def pipeline_timeline(table: str) -> list[str]:
    """Each stage's forward n against the stage before's send n, one failure
    a line; none at pipeline width one. It checks order on the step table's
    rows and recomputes no time: a receive or a forward of the wrong length
    passes.

    The head's forward n starts when it sends ``meta#pp1`` seq n. A stage's
    send n leaves no earlier than its forward n starts and its send n-1
    completes, read from the ``pp_ack`` the stage after returned. The stage
    after takes its batch n on ``meta#pp<k>`` and then receives its
    ``pp_data`` n, so it starts its forward n where the later of the two is
    released, or, when that falls inside one of its TARs, at that TAR's end;
    the forward's own waits for the metadata and the tensors come after.
    """
    run, tars, arrived = _rows(table)
    channels = channel_table(run)
    lps = [lp.name for lp in channels.registry.ids()]
    failed = []
    for head in (lp for lp in lps if lp.endswith("-pp0")):
        engine = head.removesuffix("-pp0")
        pp = sum(lp.startswith(f"{engine}-pp") for lp in lps)
        failed += _pipeline_order(engine, pp, channels, tars, arrived)
    return failed


def _pipeline_order(engine: str, pp: int, channels, tars, arrived) -> list[str]:
    """`pipeline_timeline`'s checks on the pipeline ``<engine>-pp<k>``."""
    meta = f"{engine}-pp0->{engine}-pp1:meta#pp1"
    starts = {n: a - channels.lookahead(meta) for n, a in arrived[meta].items()}
    failed = []
    for k in range(pp):
        lp = f"{engine}-pp{k}"
        data_ch = f"{engine}-pp{k}->{engine}-pp{k + 1}:pp_data#dp0"
        data = arrived[data_ch]
        ack = arrived[f"{engine}-pp{k + 1}->{engine}-pp{k}:pp_ack#dp0"]
        for n, start in sorted(starts.items()):
            span = (
                b for a, b in tars[lp] if a < start < b and not math.isclose(a, start)
            )
            start = next(span, start)
            if not any(math.isclose(a, start) for a, _ in tars[lp]):
                failed.append(f"{lp} starts no forward {n} at {start}")
            elif k < pp - 1 and n not in data:
                failed.append(f"{lp} never sent forward {n}'s output on {data_ch}")
            elif k < pp - 1:
                least = max(start, ack.get(n - 1, -math.inf))
                sent = data[n] - channels.lookahead(data_ch)
                if sent < least and not math.isclose(sent, least):
                    failed.append(f"{lp} sent forward {n} at {sent}, before {least}")
        batch = arrived[f"{engine}-pp0->{engine}-pp{k + 1}:meta#pp{k + 1}"]
        starts = {n: max(a, batch.get(n, a)) for n, a in data.items()}
    return failed


def stage_timelines(table: str, engine: str, stages, layers: int, parts) -> list[str]:
    """The pipeline ``<engine>-pp<k>``'s TAR rows and its ``meta``, ``pp_data``,
    ``pp_ack`` and ``tokens`` releases, computed under a constant step stub
    with no host laws, against the step table's: the first that differs, a
    line per stage and channel.

    `stages` holds each stage's steps on DP rank 0 (`atom.compass.parity.read`),
    `layers` the model's layer count, and `parts(tokens)` a stage's send of a
    `tokens`-row forward as `pp_send` parts. Stage k's forward is the stub's at
    its share of the layers (`get_pp_indices`); its host takes no time, so it
    returns at its launch, which waits for its device work `pp` forwards back
    and on a later stage for its metadata, and on the last stage at its device
    end. Its sends, receives and acks are `pp_send.StageSends`', each priced by
    the machine's carriers. The stage loops are ATOM's:

    - the head launches while fewer than `pp` forwards are in flight and the
      tokens its batch waits for (`recurrence._waited`) are taken. A batch with
      a request the head has not run starts where the table starts it: when
      the request reached the head is the traffic's. Launching none, the head
      flushes its last send, then waits for the first in flight's tokens, or
      that start;
    - a later stage takes its batch at its ``meta`` release; each arrival
      before it ends the stage's poll, which flushes its last send;
    - a send or flush waits for the last send's ack; a host that blocked on it
      goes on at gloo's wake (`pp_send.woken`), a TAR of its own.
    """
    from atom.models.utils import get_pp_indices

    run, got, arrived = _rows(table)
    stub, channels = Coefficients(**run.get("step_stub", {})), channel_table(run)
    pp, lp = len(stages), [f"{engine}-pp{k}" for k in range(len(stages))]
    if not stub.shape_blind or any(h.get("laws") for h in entries(run, "host")):
        return [
            f"{engine}: a timeline is computed for a constant step stub with no host laws"
        ]
    if len({len(s) for s in stages}) > 1:
        return [f"{engine}: the stages hold {[len(s) for s in stages]} steps"]
    spec = MachineSpec.from_mapping(run["machine"])
    models, wake = carrier_models(spec), gloo_wake(spec)
    tars, rel, sent, acks = defaultdict(list), defaultdict(dict), {}, {}
    st = [
        SimpleNamespace(
            now=0.0,
            ends=deque([-math.inf] * pp, maxlen=pp),
            pending=None,
            held=-math.inf,
            acked=-math.inf,
            done=-math.inf,
        )
        for _ in lp
    ]

    def ch(a, b, kind):
        return f"{lp[a]}->{lp[b]}:{kind}"

    def stamp(name, n, at):
        rel[name][n] = at
        return at

    def settle(k):  # wait for the last send's ack
        s = st[k]
        if s.pending is not None:
            gloo, s.held = acks[k, s.pending]
            s.pending, last, s.acked = None, s.acked, gloo
            if gloo > s.now:
                late = woken(wake, gloo - last)
                if late:
                    tars[lp[k]].append((gloo, gloo + late))
                s.now = gloo + late

    def forward(k, n, called, meta, release):
        s, step = st[k], stages[k][n]
        decode, (first, last) = [r[-1] for r in step["rows"]], get_pp_indices(
            layers, k, pp
        )
        busy = stub.prefill_step * (not all(decode)) + stub.decode_step * any(decode)
        host = max(called, s.ends[0], meta)
        s.ends.append(max(s.ends[-1], host, release) + busy * (last - first) / layers)
        out = step["produces_output"]
        t = max(host, s.ends[-1]) if k == pp - 1 and out else host
        tars[lp[k]].append((s.now, t))
        s.now = t
        if k < pp - 1:
            settle(k)
            sent[k, n], s.pending = (s.now, s.ends[-1]), n
            data = ch(k, k + 1, "pp_data#dp0")
            stamp(data, n, s.now + channels.lookahead(data))
        elif out:
            tokens[n] = stamp(
                ch(k, 0, "tokens"), n, s.now + channels.lookahead(ch(k, 0, "tokens"))
            )

    def downstream(k, n):
        s = st[k]
        meta_at = rel[ch(0, k, f"meta#pp{k}")][n]
        data_at = rel[ch(k - 1, k, "pp_data#dp0")][n]
        # Each arrival ends the meta poll, 100 ms while a send is held.
        while meta_at > s.now:
            ack = (acks[k, s.pending][0], s.now + 0.1) if s.pending is not None else ()
            s.now = min(t for t in (meta_at, data_at, *ack) if t > s.now)
            if meta_at > s.now:
                settle(k)
        called, s.now = s.now, max(s.now, data_at)
        t_send, t_ready = sent[k - 1, n]
        gloo, tensors = split(parts(sum(r[0] for r in stages[k][n]["rows"])))
        if not is_rendezvous(models, gloo):
            raise ValueError("a send's gloo parts are eager, so no ack ends them")
        done, meta = recv_done(models, t_send, max(called, s.ends[0]), gloo)
        posted = max(meta, s.ends[-1], s.held)
        s.done, end = recv_done(models, max(t_ready, s.done), posted, tensors)
        acks[k - 1, n] = stamp(ch(k, k - 1, "pp_ack#dp0"), n, done), s.done
        forward(k, n, called, meta, end)

    head, s, tokens = stages[0], st[0], {}
    flight, taken, seen, n = deque(), set(), set(), 0

    def arrival(n):  # where the table starts a forward with a request not run before
        if n == len(head):
            return math.inf
        if {r[0] for r in head[n]["batch"]} <= seen:
            return -math.inf
        i = len(tars[lp[0]])
        return got[lp[0]][i][0] if i < len(got[lp[0]]) else math.inf

    try:
        while n < len(head) or flight:
            launched = 0
            while (
                n < len(head)
                and len(flight) < pp
                and _waited(head, n, pp) in {None, *taken}
                and (arrival(n) <= s.now or math.isclose(arrival(n), s.now))
            ):
                for k in range(1, pp):
                    meta = ch(0, k, f"meta#pp{k}")
                    stamp(meta, n, s.now + channels.lookahead(meta))
                forward(0, n, s.now, -math.inf, s.held)
                for k in range(1, pp):
                    downstream(k, n)
                seen |= {r[0] for r in head[n]["batch"]}
                flight.append(n)
                n, launched = n + 1, launched + 1
            if not launched:
                settle(0)
            # Take the tokens out, waiting for the first only if none launched.
            while flight:
                m = flight[0]
                if head[m]["produces_output"]:
                    if tokens[m] > s.now:
                        if launched:
                            break
                        if s.now < (t := arrival(n)) < tokens[m]:
                            s.now = t
                            break
                        s.now = tokens[m]
                    launched = 1
                taken.add(flight.popleft())
            else:
                if not launched and n < len(head):  # idle until the request
                    if (t := arrival(n)) == math.inf:
                        raise ValueError(f"no TAR row starts forward {n}")
                    s.now = max(s.now, t)
    except ValueError as e:
        return [f"{engine}: {e}"]
    failed = [
        f for name in lp for f in _first_diff(f"{name}: TAR row", tars[name], got[name])
    ]
    for name, want in rel.items():
        ends = ([(d[n],) for n in sorted(d)] for d in (want, arrived[name]))
        failed += _first_diff(f"{name} seq", *ends)
    return failed


def _outputs(sent, taken) -> list[str]:
    chunks = [json.loads(line[len("data: ") :]) for line in taken if line != DONE]
    tokens = sorted(c["usage"]["completion_tokens"] for c in chunks if "usage" in c)
    done = sum(line == DONE for line in taken)
    if done == len(sent) and tokens == sorted(sent):
        return []
    took = f"took {done} [DONE] and usage counts {tokens}"
    return [f"the traffic sent requests for {sorted(sent)} tokens and {took}"]


def _timeline(table: str, lp: str, record) -> list[str]:
    run, tars, _ = _rows(table)
    stub, tars = Coefficients(**run.get("step_stub", {})), tars[lp]
    if not stub.shape_blind or any(h.get("laws") for h in entries(run, "host")):
        return [
            f"{lp}: a timeline is computed for a constant step stub with no host laws"
        ]
    # RapidServe's prefill_forward synchronizes its stream before it returns.
    sync = run.get(RAPIDSERVE, False) and lp == "engine-P"
    try:
        want = expected_timeline(
            record, stub.prefill_step, stub.decode_step, tars, sync, stub.draft_step
        )
    except ValueError as e:
        return [f"{lp}: {e}"]
    return _first_diff(f"{lp}: TAR row", want, tars, "the constant stub")


def check_run(out_dirs, sent=None, taken=(), records=None, pipeline=None) -> list[str]:
    """Every failed check over the runs in `out_dirs`, one line each. A record
    holding stages above 0 is a pipeline's: `pipeline` is `stage_timelines`'
    ``(layers, parts)``."""
    out_dirs = [Path(d) for d in out_dirs]
    tables = [(d / STEP_TABLE_FILE).read_text() for d in out_dirs]
    failed = []
    for d, table in zip(out_dirs, tables):
        summary = json.loads((d / SUMMARY_FILE).read_text())
        failed += [f"{d}: {f}" for f in _run_properties(table, summary)]
        failed += [f"{d}: {f}" for f in pipeline_timeline(table)]
    for d, table in zip(out_dirs[1:], tables[1:]):
        code, report = compare_step_tables(tables[0], table, str(out_dirs[0]), str(d))
        if code:
            failed.append(report)
    if sent is not None:
        failed += _outputs(sent, taken)
    for lp, directory in (records or {}).items():
        stages = [read(directory)]
        while more := read(directory, len(stages)):
            stages.append(more)
        if len(stages) == 1:
            failed += _timeline(tables[0], lp, stages[0])
        elif pipeline is None or any(len(s) > 1 for s in stages):
            failed.append(
                f"{lp}: a pipeline's timeline is computed at DP width 1, from its "
                "model's layers and send parts"
            )
        else:
            stages = [s[0] for s in stages]
            failed += stage_timelines(tables[0], lp, stages, *pipeline)
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.detect.scenario", description=__doc__
    )
    parser.add_argument("out_dirs", nargs="+", help="each run's out_dir")
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="LP=DIR",
        help="an engine LP and the parity record of the first run",
    )
    args = parser.parse_args(argv)
    records = dict(r.split("=", 1) for r in args.record)
    failed = check_run(args.out_dirs, records=records)
    for line in failed:
        print(f"scenario: FAIL {line}")
    if not failed:
        print(f"scenario: every check holds over {len(args.out_dirs)} run(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

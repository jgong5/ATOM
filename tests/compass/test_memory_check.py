# SPDX-License-Identifier: MIT
"""A simulated engine's hand-in, its memory held against a real server's log per term.

`run.runner` records the readings it installs on the worker, `engine_done`
hands them in to the run's out_dir, and `python -m atom.compass.memory.check`
compares them with the lines a real ATOM server logs, through `compare` and
`compare_graph_pool`. Each case installs through `run.runner` on a Qwen3.5-27B
config at TP1 on the suite's machine document, as `test_run_graph_pool.py`
does.
"""

import json
import math
import shutil
from types import SimpleNamespace

import pytest
from test_memory_readings import DOCUMENT, MAX_NUM_BATCHED_TOKENS, READINGS
from test_run_graph_pool import _installed
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.memory import MemoryRefusal, check
from atom.compass.memory.check import main
from atom.compass.runner.overrides import NonAllocatingRunner
from atom.compass.validate import __main__ as validate
from atom.utils import clock
from atom.utils.clock import LPRuntime

GIB = 1 << 30
#: The suite's machine document as probed, so its spec terms are measured.
PROBED = DOCUMENT | {"provenance": DOCUMENT["provenance"] | {"origin": "probed"}}
#: The entry's readings of a start that captured graphs, and the pool it read.
POOL = 420_000_000
CAPTURED = dict(READINGS[1], cudagraph_pool_bytes=POOL)


def _memory(monkeypatch, tmp_path, **kwargs) -> dict:
    """The memory section of the engine's hand-in, with the reservation
    `run.runner` installed on the worker, from the probed document."""
    runner = _installed(monkeypatch, tmp_path, machine=PROBED, **kwargs)
    engine = LPRuntime(
        compass_run.ENGINE, compass_run.channel_table(compass_run.spec()), None
    )
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)
    worker = SimpleNamespace(
        call_func=lambda name, wait_out: (
            NonAllocatingRunner.predicted_memory(runner)
            if name == "predicted_memory"
            else ()
        )
    )
    compass_run.engine_done(SimpleNamespace(runner_mgr=worker))
    saved = json.loads((tmp_path / compass_run.LP_FILE.format("engine")).read_text())
    reserved = NonAllocatingRunner._estimate_cudagraph_overhead(runner)
    return saved["memory"] | {"reserved": reserved}


def _total(saved, name) -> int:
    return sum(row["nbytes"] for row in saved["readings"][name])


def _log(path, *, peak, non_torch, tokens=MAX_NUM_BATCHED_TOKENS, pool=None, ranks=1):
    """A real server's log: the lines ATOM writes, at the given byte counts."""
    warmup = (
        "[atom 08:31:41] Model Runner0/1: warmup_model 40.35 seconds with 1 "
        f"reqs {tokens} tokens"
    )
    budget = (
        "[atom 08:32:21] Memory budget: total_gpu=191.98GB, free=139.49GB, "
        f"utilization=0.9, budget=172.79GB, peak_torch={peak / GIB:.2f}GB, "
        f"non_torch={non_torch / GIB:.2f}GB, cudagraph_est=0.55GB, "
        "safety=3.84GB, available_for_kv=113.24GB, block_bytes=1056768, "
        "num_kvcache_blocks=77049"
    )
    lines = [warmup, budget] * ranks
    if pool is not None:
        lines.append(
            "[atom 08:32:24] CUDA graph capture memory: 11 graphs | "
            f"pool(reserved)={pool / GIB:.2f}GB allocated=0.57GB"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _check(capsys, tmp_path, log) -> dict[str, str]:
    """The command's output, and the first row naming each term."""
    main([str(tmp_path / compass_run.LP_FILE.format("engine")), str(log)])
    out = capsys.readouterr().out
    rows = {
        line.split()[0]: line
        for line in reversed(out.splitlines())
        if line.startswith("  ")
    }
    return {"out": out} | rows


@pytest.mark.parametrize(
    "eager,piecewise,readings",
    [(True, False, READINGS[1]), (False, False, CAPTURED), (False, True, CAPTURED)],
    ids=["eager", "whole-graph", "piecewise"],
)
def test_a_simulated_run_writes_the_readings_it_sized_the_pool_from(
    monkeypatch, tmp_path, eager, piecewise, readings
):
    saved = _memory(
        monkeypatch, tmp_path, eager=eager, piecewise=piecewise, readings=readings
    )
    assert saved["tokens"] == MAX_NUM_BATCHED_TOKENS
    assert _total(saved, "cudagraph_overhead") == saved["reserved"]
    assert set(saved["readings"]) == {"peak_torch", "non_torch", "cudagraph_overhead"}
    if eager:
        assert saved["predicts"] is None
    else:
        assert sum(row["nbytes"] for row in saved["predicts"]) == POOL


@pytest.mark.parametrize("factor,verdict", [(1.0, "pass"), (2.0, "fail")])
def test_each_reading_is_one_row_with_its_own_verdict(
    monkeypatch, tmp_path, capsys, factor, verdict
):
    saved = _memory(monkeypatch, tmp_path, eager=False, readings=CAPTURED)
    log = _log(
        tmp_path / "real.log",
        peak=_total(saved, "peak_torch"),
        non_torch=_total(saved, "non_torch") * factor,
        pool=600_000_000,
    )
    rows = _check(capsys, tmp_path, log)
    # Within the gate, and none of its terms declared: the activations are
    # the entry's.
    assert rows["peak_torch"].endswith("pass")
    assert rows["non_torch"].endswith(verdict)
    # The predicted readings' own terms are printed under the table.
    assert rows["weights"].split()[2] == "analytical"
    assert "graph pool against pool(reserved)" in rows["out"]
    assert rows["reserves()"].split()[-2] == f"{saved['reserved']:,}"


def test_a_real_pool_logged_as_zero_is_not_compared_and_both_numbers_are_shown(
    monkeypatch, tmp_path, capsys
):
    saved = _memory(monkeypatch, tmp_path, eager=False, readings=CAPTURED)
    peak, non_torch = _total(saved, "peak_torch"), _total(saved, "non_torch")
    log = _log(tmp_path / "real.log", peak=peak, non_torch=non_torch, pool=0)
    rows = _check(capsys, tmp_path, log)
    assert rows["non_torch"].endswith("pass")
    assert (
        "graph pool not compared: the real run's capture reserved 0.00GB; "
        f"reserves() {saved['reserved']:,} B, predicts() {POOL:,} B"
    ) in rows["out"]


def test_a_real_run_at_another_warmup_shape_refuses_peak_torch_only(
    monkeypatch, tmp_path, capsys
):
    saved = _memory(monkeypatch, tmp_path, eager=True)
    log = _log(
        tmp_path / "real.log",
        peak=_total(saved, "peak_torch"),
        non_torch=_total(saved, "non_torch"),
        tokens=MAX_NUM_BATCHED_TOKENS * 2,
    )
    rows = _check(capsys, tmp_path, log)
    assert "! peak_torch: this term was taken at a shape on each side" in rows["out"]
    assert rows["non_torch"].endswith("pass")
    assert "graph pool not compared: the simulated run captures no graph and " in (
        rows["out"]
    )


def test_a_log_of_two_ranks_is_refused(monkeypatch, tmp_path):
    _memory(monkeypatch, tmp_path, eager=True)
    log = _log(tmp_path / "real.log", peak=GIB, non_torch=GIB, ranks=2)
    with pytest.raises(MemoryRefusal, match="has 2 lines matching"):
        main([str(tmp_path / compass_run.LP_FILE.format("engine")), str(log)])


def _ranks(tmp_path) -> list[str]:
    """The engine's hand-in as each of two DP ranks hands it in."""
    engine = tmp_path / compass_run.LP_FILE.format("engine")
    ranks = [engine.with_suffix(f".dp{r}.json") for r in (0, 1)]
    for rank in ranks:
        shutil.copy(engine, rank)
    return [str(r) for r in ranks]


def test_a_log_of_two_ranks_is_checked_against_each_ranks_hand_in(
    monkeypatch, tmp_path, capsys
):
    saved = _memory(monkeypatch, tmp_path, eager=True)
    real = {
        "peak": _total(saved, "peak_torch"),
        "non_torch": _total(saved, "non_torch"),
    }
    log = _log(tmp_path / "real.log", **real)
    main([str(tmp_path / compass_run.LP_FILE.format("engine")), str(log)])
    one = capsys.readouterr().out
    assert "non_torch" in one
    main([*_ranks(tmp_path), str(_log(log, **real, ranks=2))])
    assert capsys.readouterr().out == one * 2


def test_a_log_whose_ranks_disagree_is_refused(monkeypatch, tmp_path):
    _memory(monkeypatch, tmp_path, eager=True)
    log = _log(tmp_path / "real.log", peak=GIB, non_torch=GIB, ranks=2)
    log.write_text(log.read_text().replace("non_torch=1.00GB", "non_torch=2.00GB", 1))
    with pytest.raises(
        MemoryRefusal, match="which name no rank to pair with a hand-in"
    ):
        main([*_ranks(tmp_path), str(log)])


def _dp_rows(monkeypatch, tmp_path, records, lines) -> dict:
    """validate's memory rows for DP ranks whose real records hold `records`
    KV blocks each, against a server log of `lines`, one (blocks, factor on
    the predicted non_torch) per rank in the order logged."""
    saved = _memory(monkeypatch, tmp_path, eager=True)
    engine = tmp_path / compass_run.LP_FILE.format("engine")
    real = tmp_path / "real"
    (real / "steps").mkdir(parents=True)
    for r, blocks in enumerate(records):
        shutil.copy(engine, engine.with_suffix(f".dp{r}.json"))
        record = {"num_kvcache_blocks": blocks}
        (real / "steps" / f"dp{r}.engine.json").write_text(json.dumps(record))
    peak, non_torch = _total(saved, "peak_torch"), _total(saved, "non_torch")
    one = tmp_path / "one.log"
    logged = [
        _log(one, peak=peak, non_torch=non_torch * f)
        .read_text()
        .replace("num_kvcache_blocks=77049", f"num_kvcache_blocks={b}")
        for b, f in lines
    ]
    (real / "server.log").write_text("".join(logged))
    rows = validate._memory("real", tmp_path, "engine", real)
    return {(where, name): (value, verdict) for where, name, value, verdict in rows}


def test_a_dp_log_whose_ranks_disagree_pairs_each_rank_by_its_kv_blocks(
    monkeypatch, tmp_path
):
    # dp1 and dp2 share a block count and log alike; dp0's line is not first.
    lines = [(200, 2.0), (100, 1.0), (200, 2.0)]
    rows = _dp_rows(monkeypatch, tmp_path, [100, 200, 200], lines)
    assert rows["memory real dp0", "non_torch"][1] == "pass"
    assert rows["memory real dp1", "non_torch"][1] == "fail"
    assert rows["memory real dp2", "non_torch"][1] == "fail"
    assert rows["memory real dp0", "peak_torch"][1] == "pass"


@pytest.mark.parametrize(
    "records,lines,refused",
    [
        (
            [100, 200, 200],
            [(100, 1.0), (200, 1.0), (200, 2.0)],
            "which name no rank to pair with a hand-in",
        ),
        (
            [100, 200, 300],
            [(100, 1.0), (200, 2.0), (200, 2.0)],
            "logs num_kvcache_blocks=300, the rank's",
        ),
    ],
    ids=["one-count-two-readings", "count-on-no-line"],
)
def test_a_dp_rank_whose_kv_blocks_name_no_one_reading_is_refused(
    monkeypatch, tmp_path, records, lines, refused
):
    rows = _dp_rows(monkeypatch, tmp_path, records, lines)
    assert rows["memory real dp0", "non_torch"][1] == "pass"
    value, verdict = rows["memory real dp2", "memory"]
    assert verdict == "refused" and refused in value


def test_a_hand_in_with_no_memory_section_is_refused_by_name(tmp_path):
    compass_run._hand_in({"out_dir": str(tmp_path)}, "frontend", [])
    frontend = tmp_path / compass_run.LP_FILE.format("frontend")
    log = _log(tmp_path / "real.log", peak=GIB, non_torch=GIB)
    with pytest.raises(MemoryRefusal, match=f"{frontend} has no memory section"):
        main([str(frontend), str(log)])


def test_the_authority_removes_an_earlier_runs_hand_in(tmp_path):
    stale = tmp_path / compass_run.LP_FILE.format("engine-P")
    stale.write_text("{}")
    compass_run._RecordingAuthority(json.loads(_run_file(tmp_path).read_text()))
    assert not stale.exists()


def _measured(tmp_path, *storages, **log):
    """A `compass measure memory` ``--out``: its engine's log, and a file per
    rank holding `storages` beside its readings, None for none."""
    out = tmp_path / "measured"
    (out / check.READINGS).mkdir(parents=True)
    _log(out / check.LOG, **log)
    for rank, held in enumerate(storages):
        written = {"readings": {}} | ({} if held is None else {check.STORAGE: held})
        (out / check.READINGS / f"readings.dp{rank}.json").write_text(
            json.dumps(written)
        )
    return out


def _term(saved, name) -> int:
    (row,) = [row for row in saved["readings"]["peak_torch"] if row["name"] == name]
    return row["nbytes"]


@pytest.mark.parametrize("factor,verdict", [(1.0, "pass"), (1.2, "fail")])
def test_a_measured_start_grades_weights_and_buffers_one_row_each(
    monkeypatch, tmp_path, capsys, factor, verdict
):
    saved = _memory(monkeypatch, tmp_path, eager=True)
    stored = {
        "weights": round(_term(saved, "weights") * factor),
        "buffers": _term(saved, "buffers"),
    }
    real = {
        "peak": _total(saved, "peak_torch"),
        "non_torch": _total(saved, "non_torch"),
    }

    main([*_ranks(tmp_path), str(_measured(tmp_path, stored, stored, **real, ranks=2))])

    out = capsys.readouterr().out
    lines = [line for line in reversed(out.splitlines()) if line.startswith("  ")]
    rows = {line.split()[0]: line for line in lines}
    assert rows["weights"].split()[1] == "analytical"
    assert rows["weights"].endswith(verdict)
    assert rows["buffers"].endswith("pass")
    assert rows["peak_torch"].endswith("pass")


@pytest.mark.parametrize(
    "second,refused",
    [(None, "2 rank files and 1 hold"), ({"weights": 1, "buffers": 1}, "name no rank")],
)
def test_a_measured_start_with_a_rank_holding_no_storages_or_others_is_refused(
    monkeypatch, tmp_path, second, refused
):
    _memory(monkeypatch, tmp_path, eager=True)
    first = {"weights": GIB, "buffers": GIB}
    out = _measured(tmp_path, first, second, peak=GIB, non_torch=GIB, ranks=2)
    with pytest.raises(MemoryRefusal, match=refused):
        main([*_ranks(tmp_path), str(out)])

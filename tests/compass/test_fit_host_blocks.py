# SPDX-License-Identifier: MIT
"""`compass fit host --decode-client-run`: a client run whose host samples are
its decode drains alone, the steps of its span in which no rank prefills
(`fit.host.drain_cut`); and, under `fit.host_blocks.KIND_BLOCKS_ENV`, the
kind-block host fit: one launch constant per launch path, held out in blocks
per path and adopted where it holds out lower than `fit_host`'s law, and the
tokenizer and intake terms on the steady client runs' windows.

The record is `test_fit_steps.record`'s, alternating prefill and decode steps
but at `LONG`, a decode, served by two client runs one after another:
`steady.json` sends the requests of the steps before `SPLIT`, `decode.json`
the rest.

The kind-block fit's planted decodes replay a graph per rung, each rung's
launch its own constant (`RUNG_S`): rung 1 the commonest, rung 2 too rare for a
constant of its own, rung 8 unseen. A planted provider's prefill past `BUCKET`
tokens runs its body eagerly, `EAGER_S` dearer; another's middle chunks take
a constant per path that no intercept and columns add up to (`MIDDLE_S`), one
path too rare for a constant of its own.
"""

import dataclasses
import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _steps, _timed_record
from test_fit_host import host_laws_key, stamped
from test_fit_steps import (  # noqa: F401
    HOST,
    PLATFORM,
    as_record,
    fit,
    member,
    planted_roots,
)

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind, keyed_platform
from atom.compass.backends.launch_features import (
    GRAPHS,
    PATHS,
    REGISTRY,
    LaunchFeatures,
    path,
    stated,
)
from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.backends.step_labels import HOST_SEGMENTS, Sample
from atom.compass.backends.step_linear_fit import (
    fit_host,
    fit_intake,
    fit_tokenizer,
    label_records,
)
from atom.compass.fit.host import drain_cut
from atom.compass.fit.host_blocks import (
    BLOCKS,
    KIND_BLOCKS_ENV,
    PATH_MIN,
    fill_flags,
    fill_rungs,
    refit,
    steady_windows,
)
from atom.compass.validate.runs import SIDES

SPLIT = 120
#: A step of the decode client run's span that decodes where the others
#: prefill: its drain runs from the step before it to the step after.
LONG = SPLIT + 2


def _two_runs(directory):
    steps = directory / "steps"
    steps.mkdir(parents=True)
    shapes = _steps(19)
    shapes[LONG] = shapes[LONG - 1]
    _timed_record(steps / "dp0.jsonl", shapes, idle_after=(SPLIT - 1,), key=HOST)
    as_record(directory)
    sent = json.loads((directory / "client.json").read_text())["requests"]
    lines = [json.loads(x) for x in (steps / "dp0.jsonl").read_text().splitlines()]
    late = {f"r{r[0]}" for line in lines[SPLIT:] for r in line["batch"]}
    for name, part in (("steady", sent.keys() - late), ("decode", late)):
        exported = {"requests": {k: sent[k] for k in part}}
        (directory / f"{name}.json").write_text(json.dumps(exported))
    return directory


def _drained(samples) -> list:
    """`samples` less the decode client run's prefill steps, its even ones
    but `LONG`."""
    return [s for s in samples if s.step < SPLIT or s.step % 2 or s.step == LONG]


def test_a_decode_client_run_keeps_only_its_drains_and_their_dummies(tmp_path):
    a = _two_runs(tmp_path / "a")
    samples = label_records([a / "steps/dp0.jsonl"])[0]
    dummies = ((2, None, {}, {}), (SPLIT, None, {}, {}))
    samples = [dataclasses.replace(s, dummy_host=dummies) for s in samples]
    kept, report = drain_cut(samples, [(str(a), str(a / "decode.json"))])

    assert kept == [
        dataclasses.replace(s, dummy_host=dummies[:1]) for s in _drained(samples)
    ]
    # Each odd step of the span is a drain of its own, between two prefills,
    # but the three from LONG - 1 to LONG + 1.
    drained = [s for s in _drained(samples) if s.step >= SPLIT]
    assert report == [
        {
            "record": str(a),
            "export": str(a / "decode.json"),
            "span": [SPLIT, 239],
            "drains": 59,
            "drain_steps": 61,
            "dropped": len(samples) - len(_drained(samples)),
            "kept": len(drained),
        }
    ]


def test_the_command_fits_a_decode_client_run_on_its_drains(tmp_path):
    a = _two_runs(tmp_path / "a")
    fit(tmp_path, "fit host", a, "--client-run", a, a / "steady.json")
    whole = member(tmp_path / "store", host_laws_key(HOST), "params")
    runs = ("--client-run", a, a / "steady.json", "--decode-client-run")
    fit(tmp_path / "d", "fit host", a, *runs, a, a / "decode.json")

    host = member(tmp_path / "d" / "store", host_laws_key(HOST), "params")
    laws, _ = fit_host(_drained(label_records([a / "steps/dp0.jsonl"])[0]))
    platform = keyed_platform(Kind.HOST_LAWS, PLATFORM)
    assert host == {**laws, "key": {**laws["key"], **platform}}
    assert host["laws"] != whole["laws"]
    report = member(tmp_path / "d" / "store", host_laws_key(HOST), "report")
    (run,) = report["decode_client_runs"]
    assert run["export"] == str(a / "decode.json") and run["span"] == [SPLIT, 239]


def test_a_decode_client_run_of_no_record_is_refused_by_its_own_flag(tmp_path):
    a = _two_runs(tmp_path / "a")
    runs = ("--client-run", a, a / "steady.json", "--decode-client-run")
    with pytest.raises(SystemExit, match="--decode-client-run names .*b, which"):
        fit(tmp_path, "fit host", a, *runs, tmp_path / "b", a / "decode.json")


def test_a_prefill_side_refuses_as_a_decode_client_run(tmp_path):
    for side in SIDES:
        (tmp_path / side).mkdir()
    with pytest.raises(ValueError, match="a prefill side, which decodes nothing"):
        drain_cut([], [(str(tmp_path / SIDES[0]), "client.json")])


ARCH = "PathsForCausalLM"
MIDDLES = "MiddlesForCausalLM"
BUCKET = 4000
SETS = {"architecture": ARCH, "token": {"prefill": [BUCKET], "decode": []}}
EAGER_S = 0.04
#: Each decode rung's launch constant; rung 8 runs no forward.
RUNG_S = {1: 2.0e-3, 2: 2.2e-3, 4: 2.4e-3, 16: 2.9e-3}
LADDER = [1, 2, 4, 8, 16]
#: The forwards each rung runs: rung 2 fewer than `PATH_MIN`, and fewer than
#: `BLOCKS`, so some held-out block lacks it.
RUNS = {1: 160, 2: 3, 4: 80, 16: 60}
#: Each middle chunk path's launch constant and forwards: both eager parts
#: cost less than their sum, and a split one is too rare for a constant.
MIDDLE_S = {
    "replay": (2e-3, 30),
    "eager_body": (30e-3, 40),
    "eager_tail": (25e-3, 30),
    "eager_body+eager_tail": (40e-3, 60),
    "eager_body+split": (34e-3, 6),
}


class Paths(LaunchFeatures):
    features = ("eager",)

    def predict(self, view, graphs):
        bucket = max(graphs["token"]["prefill"])
        return {
            "eager": float(bool(view.prefill) and sum_tokens(view.requests) > bucket)
        }


class Middles(LaunchFeatures):
    """A middle chunk past `BUCKET` tokens runs its body eagerly, one whose
    `target_bs` has bit 0 set its tail, and bit 1 split."""

    features = ("eager_body", "eager_tail", "split")

    def predict(self, view, graphs):
        if not view.prefill:
            return dict.fromkeys(self.features, 0.0)
        bs = view.target_bs or 0
        return {
            "eager_body": float(sum_tokens(view.requests) > BUCKET),
            "eager_tail": float(bs & 1),
            "split": float(bs >> 1 & 1),
        }


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    monkeypatch.setitem(REGISTRY, ARCH, Paths())
    monkeypatch.setitem(REGISTRY, MIDDLES, Middles())


def _key(ladder=LADDER, sets=SETS):
    return {"engine": {"ladder": ladder, GRAPHS: sets}}


def _sample(n, view, w_launch, rng, key):
    """Forward `n` of `view`, its launch `w_launch` within 1% and every other
    segment 0.1 ms within 1%, its launch features those its key's provider
    predicts."""
    host = {g: 1e-4 * rng.uniform(0.99, 1.01) for g in HOST_SEGMENTS}
    host["W_launch"] = w_launch * rng.uniform(0.99, 1.01)
    provenance = {"source": "record", "run": "a", "requests": [n], "config": key}
    provider, graphs = stated(key)
    launch = provider.predict(view, graphs)
    return Sample(n, 0, view, 0.01, None, host, False, provenance, launch=launch)


def _decodes(seed=3720):
    """Decode forwards in a random order of `RUNS`' rungs, each launch its
    rung's constant plus 1 us a token."""
    rng, out = np.random.default_rng(seed), []
    order = [r for r, k in RUNS.items() for _ in range(k)]
    rng.shuffle(order)
    for n, rung in enumerate(order):
        seqs = int(rng.integers(max(rung // 2, 1), rung + 1))
        view = BatchView(
            (RequestShape(6, 1000, True),) * seqs,
            capture_rung=rung,
            produces_output=True,
        )
        w = RUNG_S[rung] + 1e-6 * sum_tokens(view.requests)
        out.append(_sample(n, view, w, rng, _key()))
    return out


def _prefills(seed=3721):
    """Output prefills, a third past `BUCKET` tokens, `EAGER_S` dearer to launch."""
    rng, out = np.random.default_rng(seed), []
    for i in range(90):
        tokens = int(
            rng.integers(BUCKET + 1, 6000) if i % 3 == 0 else rng.integers(256, BUCKET)
        )
        view = BatchView((RequestShape(tokens, tokens, False),), produces_output=True)
        w = 2e-3 + 1e-6 * tokens + EAGER_S * (tokens > BUCKET)
        out.append(_sample(i, view, w, rng, _key()))
    return out


def _middles(seed=3722):
    """Middle chunks in a random order of `MIDDLE_S`' paths, each launch its
    path's constant plus 1 us a token."""
    rng, out = np.random.default_rng(seed), []
    order = [p for p, (_, k) in MIDDLE_S.items() for _ in range(k)]
    rng.shuffle(order)
    key = _key((), {**SETS, "architecture": MIDDLES})
    for n, p in enumerate(order):
        eager = "eager_body" in p
        tokens = int(
            rng.integers(BUCKET + 1, 6000) if eager else rng.integers(256, BUCKET)
        )
        target = 4 * int(rng.integers(1, 8)) + ("eager_tail" in p) + 2 * ("split" in p)
        view = BatchView(
            (RequestShape(tokens, tokens, False),),
            target_bs=target,
            produces_output=False,
        )
        w = MIDDLE_S[p][0] + 1e-6 * tokens
        out.append(_sample(n, view, w, rng, key))
    return out


def test_a_forward_s_launch_path_is_its_rung_or_its_eager_features():
    decode = BatchView(
        (RequestShape(6, 100, True),), capture_rung=4, produces_output=True
    )
    assert path(decode, {}) == "rung4"
    eager = BatchView((RequestShape(6, 100, True),), produces_output=True)
    assert path(eager, {"split": 0.0}) == "eager"
    assert path(eager, {"split": 1.0}) == "eager+split"
    prefill = BatchView((RequestShape(900, 900, False),), produces_output=True)
    assert path(prefill, {"eager_body": 0.0, "eager_tail": 0.0}) == "replay"
    assert (
        path(prefill, {"eager_tail": 1.0, "eager_body": 1.0}) == "eager_body+eager_tail"
    )
    middle = BatchView((RequestShape(900, 900, False),), produces_output=False)
    assert path(middle, {"eager_body": 1.0}) == "eager_body"


def test_each_rung_takes_its_own_launch_constant_and_a_rare_or_unseen_one_is_filled():
    samples = _decodes()
    host, report = fit_host(samples)
    loop = {g: host["laws"]["decode"][g] for g in HOST_SEGMENTS if g != "W_launch"}
    refit(samples, host, report)

    law, own = host["laws"]["decode"]["W_launch"], host[PATHS]["decode"]["W_launch"]
    fitted = report["decode.W_launch"]
    assert fitted["base"] == "rung1" and fitted["short"] == ["rung2"]
    assert law[0] == pytest.approx(RUNG_S[1], rel=0.02)
    for rung in (4, 16):
        assert own[f"rung{rung}"] == pytest.approx(RUNG_S[rung], rel=0.02)
    # Rung 2 lies a third of the way from rung 1 to rung 4; rung 8 between 4 and 16.
    assert own["rung2"] == pytest.approx(law[0] + (own["rung4"] - law[0]) / 3)
    assert own["rung8"] == pytest.approx(
        own["rung4"] + (own["rung16"] - own["rung4"]) / 3
    )
    assert fitted["filled"] == {"rung2": own["rung2"], "rung8": own["rung8"]}
    # Every path is held out in every block; rung 2's three forwards are left
    # out of the fit and its blocks alike.
    held = fitted["held_out"]
    assert len(held["blocks"]) == BLOCKS
    assert {p: v["blocks"] for p, v in held["paths"].items()} == {
        "rung1": BLOCKS,
        "rung4": BLOCKS,
        "rung16": BLOCKS,
    }
    assert held["error"] < 0.01 < held["error_pooled"]
    choice = fitted["choice"]
    assert choice["paths"] == held["error"] < 0.01 < choice["fit_host"]
    assert not choice["kept"]
    checks = report["kind_blocks"]
    assert checks["path_forwards"]["decode.W_launch"] == {
        "short": ["rung2"],
        "pass": False,
    }
    assert checks["path_blocks"]["decode.W_launch"] == {
        "missed": ["rung2"],
        "pass": False,
    }
    assert not checks["path_forwards"]["pass"] and PATH_MIN > RUNS[2]
    # The engine loop's segments keep their law, validated in the same blocks.
    assert {g: host["laws"]["decode"][g] for g in loop} == loop
    assert report["decode.E"]["held_out"]["paths"]["rung16"]["blocks"] == BLOCKS
    assert set(host[PATHS]) == {"decode"} and set(host[PATHS]["decode"]) == {"W_launch"}


def test_middle_chunks_take_their_path_s_constants_and_a_rare_path_its_additive_price():
    samples = _middles()
    host, report = fit_host(samples)
    given = host["laws"]["middle"]["W_launch"]
    columns = host.get("columns", {}).get("middle", {}).get("W_launch", {})
    refit(samples, host, report)

    fitted = report["middle.W_launch"]
    assert not fitted["choice"]["kept"]
    assert fitted["choice"]["paths"] < 0.01 < fitted["choice"]["fit_host"]
    law, own = host["laws"]["middle"]["W_launch"], host[PATHS]["middle"]["W_launch"]
    assert fitted["base"] == "eager_body+eager_tail"
    assert fitted["short"] == ["eager_body+split"]
    assert law[0] == pytest.approx(MIDDLE_S["eager_body+eager_tail"][0], rel=0.02)
    for p in ("replay", "eager_body", "eager_tail"):
        assert own[p] == pytest.approx(MIDDLE_S[p][0], rel=0.02)
    # Each split path, too rare for a constant or never run, keeps what
    # fit_host's law priced it at: its intercept plus its features' columns.
    split = ["split", "eager_body+split", "eager_tail+split"]
    split += ["eager_body+eager_tail+split"]
    assert fitted["filled"] == {
        p: given[0] + sum(columns.get(f, 0.0) for f in p.split("+")) for p in split
    }
    assert {p: own[p] for p in split} == fitted["filled"]
    assert "W_launch" not in host.get("columns", {}).get("middle", {})


def test_a_prefill_s_eager_path_ties_with_its_column_and_keeps_fit_host_s_law():
    """One feature: a constant per path spans what an intercept and the
    column do, so both laws hold out alike; the tie keeps the law `fit_host`
    gave, its column."""
    samples = _prefills()
    host, report = fit_host(samples)
    law = host["laws"]["prefill"]["W_launch"]
    assert host["columns"]["prefill"]["W_launch"] == {
        "eager": pytest.approx(EAGER_S, rel=0.05)
    }
    refit(samples, host, report)

    choice = report["prefill.W_launch"]["choice"]
    assert choice["kept"] and choice["paths"] == pytest.approx(choice["fit_host"])
    assert PATHS not in host and host["laws"]["prefill"]["W_launch"] == law
    assert "eager" in host["columns"]["prefill"]["W_launch"]
    assert report["kind_blocks"]["path_forwards"]["prefill.W_launch"]["pass"]


def test_a_launch_no_rung_moves_keeps_the_law_fit_host_gave():
    samples = [
        dataclasses.replace(s, host=s.host | {"W_launch": s.host["W_pre"] * 20})
        for s in _decodes()
    ]
    host, report = fit_host(samples)
    law = host["laws"]["decode"]["W_launch"]
    refit(samples, host, report)

    assert host["laws"]["decode"]["W_launch"] == law and PATHS not in host
    entry = report["decode.W_launch"]
    assert entry["choice"]["kept"] and "held_out" in entry["paths_tried"]
    assert entry["held_out"]["paths"]["rung4"]["blocks"] == BLOCKS


def test_a_path_too_rare_for_its_own_constant_keeps_the_additive_price():
    own = {"eager_tail": 0.05}
    filled = fill_flags(
        0.01,
        {"eager_body": 0.03},
        ("eager_body", "eager_tail"),
        "eager_body+eager_tail",
        own,
    )
    assert filled == {"replay": 0.01, "eager_body": 0.04}
    assert own == {"eager_tail": 0.05, "replay": 0.01, "eager_body": 0.04}
    rungs = {"rung4": 4.0}
    assert fill_rungs(1.0, "rung1", rungs, [1, 2, 4, 8]) == {"rung2": 2.0, "rung8": 4.0}


#: When the steady client run's last step exits on the host clock: past
#: `test_fit_host.stamped`'s fourth lone request, a second before its fifth.
STEADY_END = 10**12 + 45 * 10**8


def _on_the_stamps_clock(record) -> tuple[int, int]:
    """Move `record`'s step stamps so its steady client run ends at
    `STEADY_END`, and return that run's window: its first step's entry to its
    last step's exit. The decode client run then starts a second later."""
    steps = record / "steps" / "dp0.jsonl"
    lines = [json.loads(x) for x in steps.read_text().splitlines()]
    shift = STEADY_END - lines[SPLIT - 1]["t_exit_ns"]
    for line in lines:
        for k, v in line.items():
            if k.startswith("t_"):
                line[k] = [t + shift for t in v] if isinstance(v, list) else v + shift
    steps.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return lines[0]["t_enter_ns"], STEADY_END


def test_the_switch_fits_the_paths_and_the_terms_within_the_steady_runs_windows(
    tmp_path, monkeypatch
):
    a = _two_runs(tmp_path / "a")
    window = _on_the_stamps_clock(a)
    stamped(a, burst=(40_000, 600_000, 50_000))
    assert steady_windows([(a, a / "steady.json")]) == {a: [window]}
    monkeypatch.setenv(KIND_BLOCKS_ENV, "1")
    runs = ("--client-run", a, a / "steady.json", "--decode-client-run")
    fit(tmp_path, "fit host", a, *runs, a, a / "decode.json")

    host = member(tmp_path / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "store", host_laws_key(HOST), "report")
    samples = _drained(label_records([a / "steps/dp0.jsonl"])[0])
    laws, fitted = fit_host(samples)
    refit(samples, laws, fitted)
    assert host["laws"] == laws["laws"] and host.get(PATHS) == laws.get(PATHS)
    # The tokenizer and intake terms read the steady run's window alone: four
    # lone requests, none of the decode run's, nor its burst.
    intake = a / "steps" / compass_run.INTAKE_RECORD
    calls = intake.with_name(compass_run.TOKENIZER_RECORD)
    term, _ = fit_intake([intake], [calls], [[window]])
    assert host["http_intake"] == term != fit_intake([intake], [calls])[0]
    assert host["tokenizer"] == fit_tokenizer([calls], [intake], [[window]])[0]
    checks = report["kind_blocks"]
    assert checks["path_forwards"] == fitted["kind_blocks"]["path_forwards"]
    assert checks["steady_only"] == {
        "graded_only": False,
        "decode_drains": 1,
        "pass": False,
    }
    sizes = np.linspace(30_000, 400_000, 8).astype(int).tolist()
    bodies = checks["body_decades"][str(a)]
    assert (bodies["requests"], bodies["bytes"]) == (4, [sizes[0], sizes[3]])
    assert not bodies["pass"] and not checks["body_decades"]["pass"]
    # The first four requests' calls alone: the fifth's, between the two
    # runs' windows, and the later ones', in the decode run's, are left out.
    tokens = checks["token_decades"][str(a)]
    assert {k: (v["calls"], v["tokens"]) for k, v in tokens.items()} == {
        "encode": (2, [9000, 88_768]),
        "encode_batch": (2, [5000, 90_000]),
        "segment": (1, [5000, 5000]),
    }
    assert not checks["token_decades"]["pass"]


def test_a_record_with_no_steady_run_gives_no_tokenizer_or_intake_term(
    tmp_path, monkeypatch
):
    a = _two_runs(tmp_path / "a")
    _on_the_stamps_clock(a)
    stamped(a, burst=(40_000, 600_000, 50_000))
    fit(tmp_path, "fit host", a, "--decode-client-run", a, a / "client.json")
    whole = member(tmp_path / "store", host_laws_key(HOST), "params")
    monkeypatch.setenv(KIND_BLOCKS_ENV, "1")
    fit(tmp_path / "k", "fit host", a, "--decode-client-run", a, a / "client.json")

    host = member(tmp_path / "k" / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "k" / "store", host_laws_key(HOST), "report")
    assert "http_intake" in whole and "http_intake" not in host
    assert "http_intake" in report["unfitted"]
    assert report["kind_blocks"]["body_decades"][str(a)]["requests"] == 0

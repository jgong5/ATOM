# SPDX-License-Identifier: MIT
"""`compass advice`'s reuse check holds a candidate's step linear law and host
laws to the prefill token-graph buckets its run records, as the runner does at
capture: a DeepSeek-V4.1 law stating none or others, or host laws whose key
states other prefill buckets, is ``refused`` with the runner's refusal, before
its coverage and figures are judged, where the law's records ran at the head;
where they ran at another ATOM digest or state none, it is ``calibrate``,
requirement 1, naming the refusal, since a calibration of the config records
the run's buckets (the Ruling on #3792). A law stating the run's buckets is
reused; a model whose attention builder records no token graph keeps its
verdicts. A pipeline's law is held so stage by stage, the stage stating others
named, and its host laws, a part per stage, by their key's graph sets as one
stage's are. Buckets the check cannot work out, ATOM's attention builder
failing to load in the child that reads it or the child dying, refuse by name;
a defect of Compass's own code there fails `compass advice` with its traceback.

The candidate is `test_advice_reuse`'s, its config DeepSeek-V4.1's. No device:
which attention builder records token graphs is stated, V4.1's alone, as
`test_check_token_graph_buckets` reads them where a device is declared, but
where the child that reads it fails.
"""

import os
import tempfile

import pytest
from test_advice_reuse import (  # noqa: F401  fixtures
    advise,
    at_head,
    law,
    law_file,
    searched,
    staged,
    stored,
    verdicts,
)
from test_check_token_graph_buckets import OTHERS, V41
from test_key_token_graph_sizes import BUCKETS, FORMER
from test_law_token_graph_buckets import RECORDED, environment  # noqa: F401  fixture

from atom.compass.backends.launch_features import GRAPHS
from atom.compass.drive import lacking
from atom.utils.selector import Family

#: Whether a builder records token graphs, read in a child of its own.
TOKEN_GRAPHS = lacking._token_graphs.__wrapped__
#: Why the check cannot work out the buckets.
UNWORKED = "the prefill token-graph buckets this run records cannot be worked out: "

#: The runner's refusals at capture: of a step linear law stating no buckets,
#: or others as its prefill vector's, and of host laws whose key states others.
NO_BUCKETS, OTHER_VECTOR = (
    f"the step linear law states no ATOM_TOKEN_GRAPH_SIZES in its engine config and "
    f"{stated} as its prefill vector's buckets, and this run's runner records the "
    f"prefill token-graph buckets {RECORDED}"
    for stated in ("none", OTHERS)
)
OTHER_HOST = (
    f"the host laws' key states the prefill token-graph buckets {OTHERS}, and this "
    f"run's runner records {RECORDED}"
)
#: The runner's refusal of a pipeline's law one stage of which states others as
#: its prefill vector's buckets, that stage formatted in.
OTHER_STAGE = (
    "the step linear law states no ATOM_TOKEN_GRAPH_SIZES in its engine config and "
    f"{OTHERS} as pipeline stage {{}}'s prefill vector's buckets, and this run's "
    f"runner records the prefill token-graph buckets {RECORDED}"
)


def held(r, prefill=None, host=None) -> dict:
    """`law` re-keyed to `r`, its prefill vector stating the buckets `prefill`,
    or none, and its host key the graph sets of a V4.1 runner that recorded the
    prefill buckets `host`, or none."""
    stated = law(r)
    if prefill is not None:
        stated["prefill"]["buckets"] = prefill
    if host is not None:
        stated["host"]["key"]["engine"][GRAPHS] = recorded(host)
    return stated


def recorded(prefill) -> dict:
    """The graph sets of a V4.1 runner that recorded the prefill buckets
    `prefill`."""
    token = {"prefill": prefill, "decode": []}
    return {"architecture": V41, "token": token, "request": [], "decoder": None}


def pipelined(tmp_path, r, *prefill, host=None) -> dict:
    """`staged`'s law on `r`'s config, graphed, on as many pipeline stages as
    `prefill`, stage k's prefill vector stating the buckets `prefill[k]`, and
    its host key the graph sets of a V4.1 runner that recorded the prefill
    buckets `host`, or none."""
    stated = staged(tmp_path, r, *[-0.02] * len(prefill), eager=False)
    for part, buckets in zip(stated["stages"], prefill):
        part["prefill"] = {**part["prefill"], "buckets": buckets}
    if host is not None:
        stated["host"]["key"]["engine"][GRAPHS] = recorded(host)
    return stated


def judged(found: list[dict]) -> list[tuple]:
    """Each of `found`'s verdicts, the requirement it names and its reason."""
    return [(v["verdict"], v["requirement"], v["reason"]) for v in found]


@pytest.fixture(autouse=True)
def builders(monkeypatch):
    """V4.1's attention builder records token graphs, and no other does."""
    monkeypatch.setattr(lacking, "_token_graphs", lambda f, arch: f is Family.CSA2)


@pytest.fixture
def v41(searched):  # noqa: F811
    """The candidate's config DeepSeek-V4.1's, whose builder records token graphs."""
    searched.config.hf_config.architectures = [V41]
    return searched


@pytest.mark.parametrize(
    "case, refused",
    [
        ("a law file stating no buckets", NO_BUCKETS),
        ("a store law stating no buckets", NO_BUCKETS),
        ("a prefill vector stating others", OTHER_VECTOR),
        ("a host key stating others", OTHER_HOST),
    ],
    ids=["law-file-no-buckets", "store-law-no-buckets", "vector-others", "host-others"],
)
def test_a_v41_law_its_runner_refuses_at_capture_is_refused_with_its_refusal(
    tmp_path, v41, case, refused
):
    if case == "a store law stating no buckets":
        stored(tmp_path, v41, held=held(v41))
    elif case == "a prefill vector stating others":
        law_file(tmp_path, held(v41, OTHERS))
    elif case == "a host key stating others":
        law_file(tmp_path, held(v41, RECORDED, OTHERS))
    else:
        law_file(tmp_path, held(v41))

    found = verdicts(advise(tmp_path), "law")

    assert judged(found) == [("refused", None, refused)] * 2


@pytest.mark.parametrize(
    "records, named",
    [
        ("unstated", "stating no ATOM digest"),
        ("at another digest", f"from ATOM digest {'0' * 64}"),
    ],
)
@pytest.mark.parametrize(
    "case, refused",
    [
        ("a law file stating no buckets", NO_BUCKETS),
        ("a host key stating others", OTHER_HOST),
    ],
    ids=["law-file-no-buckets", "host-others"],
)
def test_a_v41_law_its_runner_refuses_on_records_from_another_tree_is_calibrate(
    tmp_path, v41, case, refused, records, named
):
    if case == "a host key stating others":
        stated = held(v41, RECORDED, OTHERS)
    else:
        stated = held(v41)
    stated["records_atom"] = "0" * 64 if records == "at another digest" else None
    where = law_file(tmp_path, stated)

    found = verdicts(advise(tmp_path), "law")

    why = f"{where}: {refused}, on records re-keyed {named}"
    assert judged(found) == [("calibrate", 1, why)] * 2
    assert all(v["made_by"] for v in found)


@pytest.mark.parametrize(
    "states", ["prefill-vector", "engine-config"], ids=lambda s: f"buckets-in-{s}"
)
def test_a_v41_law_stating_its_run_s_buckets_is_reused(
    tmp_path, monkeypatch, v41, states
):
    if states == "engine-config":
        monkeypatch.setenv(BUCKETS, FORMER)
        where = law_file(tmp_path, held(v41, host=OTHERS))
    else:
        where = law_file(tmp_path, held(v41, RECORDED, RECORDED))

    found = verdicts(advise(tmp_path), "law")

    assert judged(found) == [("reuse", None, where)] * 2


@pytest.mark.parametrize(
    "prefill, host, refused",
    [
        ((RECORDED, RECORDED), None, None),
        ((OTHERS, RECORDED), None, OTHER_STAGE.format(0)),
        ((RECORDED, OTHERS), None, OTHER_STAGE.format(1)),
        ((RECORDED, RECORDED), OTHERS, OTHER_HOST),
    ],
    ids=["every-stage-the-run-s", "stage-0-others", "stage-1-others", "host-others"],
)
def test_a_pipeline_s_v41_law_is_held_to_its_run_s_buckets_stage_by_stage(
    tmp_path, v41, prefill, host, refused
):
    where = law_file(tmp_path, pipelined(tmp_path, v41, *prefill, host=host))

    found = verdicts(advise(tmp_path), "law")

    verdict = ("reuse", None, where) if refused is None else ("refused", None, refused)
    assert judged(found) == [verdict] * 2


def test_a_model_whose_builder_records_no_token_graph_keeps_its_verdict(
    tmp_path, searched  # noqa: F811
):
    where = law_file(tmp_path, held(searched, host=OTHERS))

    found = verdicts(advise(tmp_path), "law")

    assert judged(found) == [("reuse", None, where)] * 2


def unimportable(family, arch):
    """In the child: ATOM's attention builder failing to load, as without aiter."""
    raise ModuleNotFoundError("No module named 'aiter'")


def dying(family, arch):
    """In the child: the child dying."""
    os._exit(1)


def compass_failing(family, arch):
    """In the child: Compass's own code failing, on a card reading with no
    ``props``."""
    from atom.compass.capture import driverless

    with tempfile.TemporaryDirectory() as tmpdir:
        driverless.install(tmpdir, arch, read={"card": 0})


@pytest.mark.parametrize(
    "reads, why",
    [
        (
            unimportable,
            "ATOM's attention builder: ModuleNotFoundError: No module named",
        ),
        (dying, "the child reading ATOM's attention builder died: A process in the"),
    ],
    ids=["atom-fails", "child-dies"],
)
def test_buckets_that_cannot_be_worked_out_refuse_by_name(
    tmp_path, monkeypatch, v41, reads, why
):
    monkeypatch.setattr(lacking, "_token_graphs", TOKEN_GRAPHS)
    monkeypatch.setattr(lacking, "_builder_records", reads)
    law_file(tmp_path, held(v41, RECORDED))

    found = verdicts(advise(tmp_path), "law")

    assert [(v["verdict"], v["requirement"]) for v in found] == [("refused", None)] * 2
    assert found[0]["reason"].startswith(UNWORKED + why)


def test_a_compass_defect_reading_the_builder_fails_compass_advice_with_its_traceback(
    tmp_path, monkeypatch, v41
):
    monkeypatch.setattr(lacking, "_token_graphs", TOKEN_GRAPHS)
    monkeypatch.setattr(lacking, "_builder_records", compass_failing)
    law_file(tmp_path, held(v41, RECORDED))

    with pytest.raises(RuntimeError) as failed:
        advise(tmp_path)

    said = str(failed.value)
    assert said.startswith(
        "Compass's own code raised reading ATOM's attention builder:\n"
        "Traceback (most recent call last):\n"
    )
    assert ", in _props\n" in said
    assert said.endswith("\nKeyError: 'props'\n")

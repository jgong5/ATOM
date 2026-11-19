# SPDX-License-Identifier: MIT
"""`compass advice`'s reuse check holds a candidate's step linear law and host
laws to the prefill token-graph buckets its run records, as the runner does at
capture: a DeepSeek-V4.1 law stating none or others, or host laws whose key
states other prefill buckets, is ``refused`` with the runner's refusal, before
its coverage and figures are judged. A law stating the run's buckets is
reused; a model whose attention builder records no token graph keeps its
verdicts. Buckets the check cannot work out, ATOM's attention builder unread,
refuse by name.

The candidate is `test_advice_reuse`'s, its config DeepSeek-V4.1's. No device:
which attention builder records token graphs is stated, V4.1's alone, as
`test_check_token_graph_buckets` reads them where a device is declared.
"""

import pytest
from test_advice_reuse import (  # noqa: F401  fixtures
    advise,
    at_head,
    law,
    law_file,
    searched,
    stored,
    verdicts,
)
from test_check_token_graph_buckets import OTHERS, V41
from test_key_token_graph_sizes import BUCKETS, FORMER
from test_law_token_graph_buckets import RECORDED, environment  # noqa: F401  fixture

from atom.compass.backends.launch_features import GRAPHS
from atom.compass.drive import lacking
from atom.utils.selector import Family

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


def held(r, prefill=None, host=None) -> dict:
    """`law` re-keyed to `r`, its prefill vector stating the buckets `prefill`,
    or none, and its host key the graph sets of a V4.1 runner that recorded the
    prefill buckets `host`, or none."""
    stated = law(r)
    if prefill is not None:
        stated["prefill"]["buckets"] = prefill
    if host is not None:
        token = {"prefill": host, "decode": []}
        graphs = {"architecture": V41, "token": token, "request": [], "decoder": None}
        stated["host"]["key"]["engine"][GRAPHS] = graphs
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


def test_a_model_whose_builder_records_no_token_graph_keeps_its_verdict(
    tmp_path, searched  # noqa: F811
):
    where = law_file(tmp_path, held(searched, host=OTHERS))

    found = verdicts(advise(tmp_path), "law")

    assert judged(found) == [("reuse", None, where)] * 2


def test_buckets_that_cannot_be_worked_out_refuse_by_name(tmp_path, monkeypatch, v41):
    def unread(family, arch):
        raise ModuleNotFoundError("No module named 'aiter'")

    monkeypatch.setattr(lacking, "_token_graphs", unread)
    law_file(tmp_path, held(v41, RECORDED))

    found = verdicts(advise(tmp_path), "law")

    reason = (
        "the prefill token-graph buckets this run records cannot be worked out: "
        "ModuleNotFoundError: No module named 'aiter'"
    )
    assert judged(found) == [("refused", None, reason)] * 2

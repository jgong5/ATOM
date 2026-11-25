# SPDX-License-Identifier: MIT
"""The child that reads whether ATOM's attention builder records token graphs,
bounded.

`compass run --check` and `compass advice`'s reuse check work out the prefill
token-graph buckets a run records by reading the builder in a child of their
own. A child that outlives its bound is ended there and the buckets cannot be
worked out, as where ATOM's builder fails to load in it: `--check` prints why
apart, outside its exit status, and holds no law to the buckets; the reuse
check refuses the law by name. The answer is cached as a builder's answer is,
so a search over many points waits the bound once. A defect of Compass's own
code there still fails `--check` with its traceback.

The config is DeepSeek-V4.1's, as `test_check_token_graph_buckets` and
`test_advice_reuse_token_graph_buckets` set it up. No device: the child runs
the stand-ins below in place of the builder's read.
"""

import os
import time
from pathlib import Path

import pytest
import test_check_token_graph_buckets as check_buckets
from test_advice_reuse import (  # noqa: F401  fixtures
    advise,
    at_head,
    law_file,
    searched,
    verdicts,
)
from test_advice_reuse_token_graph_buckets import (  # noqa: F401  fixture
    UNWORKED,
    compass_failing,
    held,
    unimportable,
    v41,
)
from test_drive_runfile import step_linear
from test_law_token_graph_buckets import RECORDED, environment  # noqa: F401  fixture
from test_run_lacking import checked, lacked, served

from atom.compass.drive import lacking

#: The bound in place of `lacking.PROBE_SECONDS`: well past the time a child
#: takes to start and import this module, so `outliving` writes its pid first.
BOUND = 40
#: How long `outliving` sleeps: past the bound, and finite, so a check with no
#: bound ends.
SLEEP = 4 * BOUND
#: The variable naming the file `outliving` appends its pid to.
PIDS = "COMPASS_TEST_PROBE_PIDS"
#: Why a child that outlives the bound gives no buckets.
OUTLIVED = f"the child reading ATOM's attention builder outlived {BOUND} s"
#: Why a child in which ATOM's builder fails to load gives none.
ATOM_FAILS = "ATOM's attention builder: ModuleNotFoundError: No module named 'aiter'"
#: The candidate's dimensions: an argument that varies with the concurrency, so
#: each of its points is an instance of its own, judged apart.
SCALED = {"config": {"a": {"per_concurrency": {"--max-num-seqs": 8}}}}


def outliving(family, arch):
    """In the child: its pid appended to `PIDS`' file, then a sleep past the
    bound; an answer at its end, that the builder records token graphs."""
    with open(os.environ[PIDS], "a") as pids:
        pids.write(f"{os.getpid()}\n")
    time.sleep(SLEEP)
    return True


@pytest.fixture(autouse=True)
def bounded(monkeypatch, tmp_path):
    """The bound `BOUND`, the stand-ins' pids in `tmp_path`, and no builder's
    answer cached before or after."""
    monkeypatch.setattr(lacking, "PROBE_SECONDS", BOUND, raising=False)
    monkeypatch.setenv(PIDS, str(tmp_path / "pids"))
    lacking._token_graphs.cache_clear()
    yield
    lacking._token_graphs.cache_clear()


def ended(tmp_path) -> list[int]:
    """The pid of each child `outliving` ran in, each asserted gone."""
    pids = [int(p) for p in (tmp_path / "pids").read_text().split()]
    assert pids and not [p for p in pids if Path(f"/proc/{p}").exists()]
    return pids


@pytest.mark.parametrize(
    "reads, why",
    [(outliving, OUTLIVED), (unimportable, ATOM_FAILS)],
    ids=["outlives-the-bound", "atom-fails"],
)
def test_check_prints_apart_buckets_it_cannot_work_out_and_holds_no_law_to_them(
    monkeypatch, tmp_path, capsys, reads, why
):
    """A V4.1 law stating no buckets, which the run refuses at capture once its
    buckets are worked out (`test_check_token_graph_buckets`), lacks nothing."""
    r = check_buckets.v41(tmp_path)
    law = step_linear(tmp_path, r).key.token()
    served(monkeypatch, r.config)
    monkeypatch.setattr(lacking, "_builder_records", reads)
    started = time.monotonic()

    assert checked(tmp_path, "--step-linear", law) == 0

    out = capsys.readouterr().out
    assert time.monotonic() - started < SLEEP
    assert lacked(out) == []
    assert f"apart: {UNWORKED}{why}, so no law is held to them\n" in out
    if reads is outliving:
        assert len(ended(tmp_path)) == 1


def test_a_compass_defect_reading_the_builder_fails_check_with_its_traceback(
    monkeypatch, tmp_path
):
    r = check_buckets.v41(tmp_path)
    law = step_linear(tmp_path, r).key.token()
    served(monkeypatch, r.config)
    monkeypatch.setattr(lacking, "_builder_records", compass_failing)

    with pytest.raises(RuntimeError) as failed:
        checked(tmp_path, "--step-linear", law)

    said = str(failed.value)
    assert said.startswith(
        "Compass's own code raised reading ATOM's attention builder:\n"
        "Traceback (most recent call last):\n"
    )
    assert said.endswith("\nKeyError: 'props'\n")


def test_the_reuse_check_refuses_by_name_a_law_whose_buckets_outlive_the_bound_once(
    tmp_path, monkeypatch, v41  # noqa: F811
):
    """Each of the candidate's two points, judged apart, refuses; the builder's
    child is read once."""
    monkeypatch.setattr(lacking, "_builder_records", outliving)
    law_file(tmp_path, held(v41, RECORDED))
    started = time.monotonic()

    found = verdicts(advise(tmp_path, dimensions=SCALED), "law")

    assert time.monotonic() - started < SLEEP
    assert [(v["verdict"], v["requirement"]) for v in found] == [("refused", None)] * 2
    assert [v["reason"] for v in found] == [UNWORKED + OUTLIVED] * 2
    assert len(ended(tmp_path)) == 1

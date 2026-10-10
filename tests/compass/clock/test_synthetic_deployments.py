# SPDX-License-Identifier: MIT
"""The two shipped channel tables driven end to end against the Clock Authority.

What each group defends:

* **The collapse.** A single deployment is three LPs and a prefill-decode
  deployment behind the router is five; the router is not one of them.
* **Every deployment runs** to the finish with nothing in flight, every step
  charged, no grant from the recovery branch, each LP answered once per
  request, and sends only on declared channels. The engine's metrics push and
  the traffic LP's scrape are daemon deadlines, and the finish leaves both
  held.
* **Every mechanism is driven**: steps (TAR), idle points (NER), registered
  sends, released receives, some handled below their grant, and local timers.
* **The request order decides nothing**: on the full-length trace each LP's
  replies are the same under four orders of submission, which do interleave
  differently.
* **The checks fire**: a message held back, or handed one grant late, is a
  step-over; a final response never sent lets the run finish, and the traffic
  LP's end-of-run check names that request; an engine push left essential
  stops the run at the simulated-time bound.
* **The run ends with a scrape**: one scrape follows the last response even
  with no periodic scrape, and its answer is delivered before the finish.
"""

import dataclasses
import math

import pytest

from atom.compass.clock import ClockAbort

from .deployments import DEPLOYMENTS
from .harness import SteppedOverEvent, SyntheticRun
from .participants import (
    DESIGN_WORKLOAD,
    ENCODE_TOKENS_PER_S,
    Engine,
    Traffic,
    UnansweredRequests,
)

BRIEF = dataclasses.replace(DESIGN_WORKLOAD, requests=8, decode_steps=3)

LPS = {
    "single-deployment": ("engine", "frontend", "traffic"),
    "prefill-decode-1p1d": (
        "engine-D",
        "engine-P",
        "frontend-D",
        "frontend-P",
        "traffic",
    ),
}


def orders(name):
    """Four priorities over the LP names: name order, reversed, two rotations."""
    names = list(LPS[name])
    return [names, names[::-1], names[1:] + names[:1], names[2:] + names[:2]]


def steps(workload):
    return workload.requests * (workload.prefill_steps + workload.decode_steps)


@pytest.fixture(scope="module", params=sorted(DEPLOYMENTS))
def report(request):
    return SyntheticRun(request.param, BRIEF).run()


def test_each_deployment_has_exactly_the_lps_its_table_registers():
    for name, table in DEPLOYMENTS.items():
        assert tuple(map(str, table().registry.ids())) == LPS[name]


class TestEveryDeploymentRuns:
    def test_the_run_ends_by_the_finish_with_every_lp_finished(self, report):
        assert report.stopped_by == "the finish"
        assert report.unfinished == ()

    def test_every_step_of_the_trace_is_charged(self, report):
        assert report.steps == steps(BRIEF)

    def test_each_lp_receives_one_reply_per_request(self, report):
        issued = {lp: sum(kinds.values()) for lp, kinds in report.requests.items()}
        assert issued == report.replies

    def test_no_grant_takes_the_recovery_branch(self, report):
        assert report.recovered == 0

    def test_lps_send_on_the_declared_channels_and_no_other(self, report):
        table = DEPLOYMENTS[report.deployment]()
        declared = [
            c.name for lp in table.registry.ids() for c in table.channels_from(lp)
        ]
        # Control commands come from client disconnects and stats calls, which
        # this workload has none of.
        assert sorted(report.channels) == sorted(
            c for c in declared if ":control#" not in c
        )


def test_every_mechanism_is_driven(report):
    assert report.requests["traffic"]["NER"] > 0
    assert sum(kinds.get("TAR", 0) for kinds in report.requests.values()) > 0
    assert report.messages > 0
    assert report.handled == report.messages
    assert report.handled_before_grant > 0
    assert report.timers > 0


def test_a_pair_from_one_arrival_is_tokenized_side_by_side(report):
    first, second = sorted(
        a
        for log in report.reply_log.values()
        for _, m in log
        for c, _, a in m
        if ":request#" in c
    )[:2]
    short, long = BRIEF.prompt_tokens
    assert second - first == (long - short) / ENCODE_TOKENS_PER_S


@pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
def test_each_lps_replies_do_not_depend_on_the_request_order(name):
    runs = [SyntheticRun(name, DESIGN_WORKLOAD, order) for order in orders(name)]
    reports = [run.run() for run in runs]
    assert len({r.submitted for r in reports}) == len(reports)
    for run, r in zip(runs, reports, strict=True):
        assert r.stopped_by == "the finish"
        assert r.unfinished == ()
        # Daemon deadlines fired during the run, and every one was still held
        # when it finished.
        lps = run.lps.values()
        assert all(p.scrapes > 0 for p in lps if isinstance(p, Traffic))
        held = [p.next_scrape for p in lps if isinstance(p, Traffic)]
        held += [p.next_push for p in lps if isinstance(p, Engine)]
        assert min(held) > r.modelled_seconds
        assert r.steps == steps(DESIGN_WORKLOAD)
        assert r.recovered == 0
        assert r.handled == r.messages
        assert {lp: sum(k.values()) for lp, k in r.requests.items()} == r.replies
        assert r.reply_log == reports[0].reply_log


class TestTheChecksFire:
    def test_a_message_held_back_is_reported_as_a_step_over(self):
        run = SyntheticRun("single-deployment", BRIEF, grant_cap=2000)
        run._hand_over = lambda lp, g, released: []
        with pytest.raises(SteppedOverEvent, match="never handed"):
            run.run()

    @pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
    def test_a_message_handed_one_grant_late_is_a_step_over(self, name):
        """Granting at `N` equal to its bound hands a tied send over a grant late."""
        run = SyntheticRun(name, DESIGN_WORKLOAD, orders(name)[1])
        lbts = run.clock._lbts
        run.clock._lbts = lambda i: math.nextafter(lbts(i), math.inf)
        with pytest.raises(SteppedOverEvent, match="previous grant"):
            run.run()

    @pytest.mark.parametrize("scrape_s", [BRIEF.scrape_interval_seconds, math.inf])
    def test_a_dropped_response_is_named_by_the_end_of_run_check(self, scrape_s):
        """The engine's metrics push does not keep the run going."""
        workload = dataclasses.replace(BRIEF, scrape_interval_seconds=scrape_s)
        run = SyntheticRun("single-deployment", workload, grant_cap=2000)
        frontend = next(lp for lp in run.lps.values() if str(lp.name) == "frontend")
        send = frontend.send
        frontend.send = lambda channel, payload: (
            None if payload == ("chunk", 3, True) else send(channel, payload)
        )
        with pytest.raises(UnansweredRequests, match=r"requests \[3\] sent but"):
            run.run()
        assert run.clock.final_clocks is not None

    @pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
    def test_an_essential_engine_push_stops_at_the_bound(self, name):
        def essential(run):
            def push_joins_t(now):
                kind, t, t_daemon = run(now)
                return kind, min(t, t_daemon), math.inf

            return push_joins_t

        assert SyntheticRun(name, BRIEF, bound_s=64.0).run().unfinished == ()
        run = SyntheticRun(name, BRIEF, bound_s=64.0)
        for lp in run.lps.values():
            if isinstance(lp, Engine):
                lp.run = essential(lp.run)
        with pytest.raises(ClockAbort, match="passes the simulated-time bound 64.0"):
            run.run()


def test_one_scrape_follows_the_last_response():
    """With no periodic scrape, the closing scrape still runs and is answered."""
    workload = dataclasses.replace(BRIEF, scrape_interval_seconds=math.inf)
    report = SyntheticRun("single-deployment", workload).run()
    assert report.stopped_by == "the finish"
    assert report.channels["traffic->frontend:http"] == BRIEF.requests + 1
    assert report.handled == report.messages

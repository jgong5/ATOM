# SPDX-License-Identifier: MIT
"""The two shipped channel tables driven end to end against the Clock Authority.

What each group defends:

* **The collapse.** A single deployment is three LPs and a prefill-decode
  deployment behind the router is five; the router is not one of them.
* **Every deployment runs** to the traffic LP's `END`, every step charged, no
  grant from the recovery branch, each LP answered once per request, and sends
  only on declared channels.
* **Every mechanism is driven**: steps (TAR), idle points (NER), registered
  sends, released receives, some handled below their grant, and local timers.
* **The request order decides nothing**: on the full-length trace each LP's
  replies are the same under four orders of submission, which do interleave
  differently.
* **The checks fire**: a message held back is a step-over, and a message never
  sent leaves its receiver unfinished while the scrape keeps the run going.
"""

import dataclasses
import math

import pytest

from .deployments import DEPLOYMENTS
from .harness import SteppedOverEvent, SyntheticRun
from .participants import DESIGN_WORKLOAD, Station, scaled

BRIEF = scaled(DESIGN_WORKLOAD, requests=8, decode_steps=3)

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
    def test_the_run_ends_by_end_with_every_lp_finished(self, report):
        assert report.stopped_by == "END from traffic"
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


@pytest.mark.parametrize("name", sorted(DEPLOYMENTS))
def test_each_lps_replies_do_not_depend_on_the_request_order(name):
    reports = [
        SyntheticRun(name, DESIGN_WORKLOAD, order).run() for order in orders(name)
    ]
    assert len({r.submitted for r in reports}) == len(reports)
    for r in reports:
        assert r.stopped_by == "END from traffic"
        assert r.steps == steps(DESIGN_WORKLOAD)
        assert r.recovered == 0
        assert {lp: sum(k.values()) for lp, k in r.requests.items()} == r.replies
        assert r.reply_log == reports[0].reply_log


def test_a_pair_tokenized_together_completes_apart_only_through_its_lengths():
    wide = Station(2)
    assert (wide.admit(0.0, 1.0), wide.admit(0.0, 2.0)) == (1.0, 2.0)
    wide = Station(2)
    assert (wide.admit(0.0, 1.0), wide.admit(0.0, 1.0)) == (1.0, 1.0)


class TestTheChecksFire:
    def test_a_message_held_back_is_reported_as_a_step_over(self):
        run = SyntheticRun("single-deployment", BRIEF, grant_cap=2000)
        run._hand_over = lambda lp, g, released: []
        with pytest.raises(SteppedOverEvent, match="never handed"):
            run.run()

    @pytest.mark.parametrize(
        "scrape_s, stopped_by",
        [
            (BRIEF.scrape_interval_seconds, "the grant cap of 2000 was reached"),
            (math.inf, "every N infinite"),
        ],
    )
    def test_a_message_never_sent_leaves_traffic_unfinished(self, scrape_s, stopped_by):
        """Only the scrape keeps a run with no END from ending on its own."""
        workload = dataclasses.replace(BRIEF, scrape_interval_seconds=scrape_s)
        run = SyntheticRun("single-deployment", workload, grant_cap=2000)
        frontend = next(lp for lp in run.lps.values() if str(lp.name) == "frontend")
        send = frontend.send
        frontend.send = lambda channel, payload: (
            None if channel == "frontend->traffic:stream" else send(channel, payload)
        )
        report = run.run()
        assert report.stopped_by == stopped_by
        assert report.unfinished == ("traffic",)

"""Finite-root completeness, causal release order and persistent ingress."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.core.loaded_input import load_json
from atom.compass.fixed_absolute import (
    DEPENDENCY_BASIS, PROFILE, SCHEMA, FixedAbsolutePlan, FixedAbsoluteReleases,
)
from atom.compass.prefix_workload import token_digest
from atom.compass.replay_plan import RESPONSE_DELIVERY
from atom.compass.runtime.native_ingress import NativeWriterReceiver
from atom.compass.runtime.request_readiness import RegisteredRequest, UnsupportedReadiness


def bundle(*, root_times=(0., 100.), child_times=((10., 30.),), join=False, clients=1):
    """A source root with a future continuation and finite child chains."""
    roots, rows, conversations, branches = [], [], [], []
    for client in range(clients):
        root_id = f"root{client}"
        root_indices = []
        root = {"root_id": root_id, "client_index": client,
                "source": {"path": f"/source/{root_id}.json", "sha256": "a" * 64},
                "source_paths": [], "_source_fixture": {"id": root_id, "hash_id_scope": "local",
                                                         "block_size": 64, "requests": []}}
        roots.append(root)

        def add_conversation(name, times, depth, parent):
            indices = []
            for turn, arrived in enumerate(times):
                index = len(rows)
                source_path = f"/requests/{len(root['source_paths'])}"
                root["source_paths"].append(source_path)
                root["_source_fixture"]["requests"].append({"type": "n", "t": arrived, "in": 2, "out": 2})
                tokens = [100 + index, 200 + index]
                rows.append({"index": index, "root_id": root_id, "client_index": client,
                    "source_path": source_path, "source_time_s": arrived, "arrival_s": arrived,
                    "source_input_tokens": 2,
                    "conversation_id": name, "turn_index": turn, "depends_on": indices[-1:],
                    "input_tokens": len(tokens), "output_tokens": 2,
                    "prompt_token_ids": tokens, "prompt_token_sha256": token_digest(tokens),
                    "body": {"model": "fixture", "messages": [{"role": "user", "content": f"request {index}"}],
                             "max_completion_tokens": 2, "ignore_eos": True, "stream": True,
                             "stream_options": {"include_usage": True}, "temperature": 0}})
                indices.append(index)
            conversations.append({"conversation_id": name, "root_id": root_id,
                                  "agent_depth": depth, "parent_conversation_id": parent,
                                  "request_indices": indices})
            return indices

        root_indices = add_conversation(root_id, root_times, 0, None)
        children = []
        for child, times in enumerate(child_times):
            child_name = f"{root_id}/child{child}"
            indices = add_conversation(child_name, times, 1, root_id)
            rows[indices[0]]["depends_on"] = [root_indices[0]]
            children.append(child_name)
            if join:
                rows[root_indices[-1]]["depends_on"].append(indices[-1])
        if children:
            branches.append({"branch_id": root_id + "/spawn", "parent_conversation_id": root_id,
                "after_request": root_indices[0], "child_conversation_ids": children,
                "mode": "spawn", "dispatch_timing": "post", "is_background": not join,
                "join_before_request": root_indices[-1] if join else None})
    return {"schema": SCHEMA, "profile": PROFILE, "dependency_basis": DEPENDENCY_BASIS,
            "clients": clients, "roots": roots, "conversations": conversations, "branches": branches,
            "source_time_origin_s": 0., "time_scale": 1, "response_delivery": RESPONSE_DELIVERY,
            "initial_cache": "acknowledged_empty", "cache_policy": cache_on_policy(), "requests": rows}


def plan_file(tmp_path, data=None):
    data = copy.deepcopy(bundle() if data is None else data)
    for root in data["roots"]:
        source = tmp_path / (root["root_id"] + ".json")
        source.write_text(json.dumps(root.pop("_source_fixture")))
        root["source"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, FixedAbsolutePlan.load(path, digest)


class FixtureIngress(NativeWriterReceiver):
    """Synthetic service coefficients, using the actual persistent queue code."""

    def __init__(self, source):
        data, loaded = load_json(source, role="runtime.request_readiness.fixture")
        self.loaded_inputs = (loaded,)
        self.min_bytes, self.max_bytes = 0, 10**9
        self._services = {"writer": (data["writer_us"], 0), "receiver": (data["receiver_us"], 0)}


def ingress_profile(tmp_path, writer_seconds=.1, receiver_seconds=.2):
    source = tmp_path / "service.json"
    source.write_text(json.dumps({"writer_us": writer_seconds * 1e6, "receiver_us": receiver_seconds * 1e6}))
    profile = tmp_path / "readiness.json"
    profile.write_text(json.dumps({"schema": "compass.request_readiness_profile/1",
        "resolver": f"{__name__}.FixtureIngress", "options": {"source": str(source)},
        "source_law": "synthetic fixture", "support": {"kind": "test-only"},
        "origin_contract": {"declared_arrival_event": "causal release",
                            "writer_eligibility_event": "causal release",
                            "transition": "synthetic source service"}}))
    return profile, FixtureIngress(str(source))


def request(index, arrival):
    return RegisteredRequest(index, arrival, 2, index, SimpleNamespace(
        frame_request_count=1, pickle_protocol=4, token_typecode="i", token_itemsize=4,
        reconstructed_add_bytes=100))


def test_complete_root_membership_and_bodies_are_immutable(tmp_path):
    _, _, plan = plan_file(tmp_path)
    escaped = plan.rows
    escaped[0]["body"]["messages"][0]["content"] = "changed"
    escaped[0]["prompt_token_ids"][0] = 0
    assert plan.rows[0]["body"]["messages"][0]["content"] == "request 0"
    assert plan.rows[0]["prompt_token_ids"][0] == 100
    payloads = [json.loads(body) for body in plan.encode_payloads(declared=True)]
    assert [body["compass_workload_index"] for body in payloads] == list(range(4))
    assert all(body["compass_workload_size"] == 4 for body in payloads)
    assert plan.evidence()["dependency_basis"] == DEPENDENCY_BASIS


@pytest.mark.parametrize("damage, message", [
    ("duplicate", "identity"), ("missing_leaf", "every declared source leaf"),
    ("unowned_leaf", "omit requests"), ("fork", "SPAWN"), ("pre", "SPAWN"),
    ("nested", "conversation"), ("extra_prerequisite", "prerequisites"),
    ("changed_gap", "source offset"), ("changed_tokens", "token identity"),
    ("ground_truth_claim", "qualification"),
])
def test_invalid_or_unsupported_bundle_refuses(tmp_path, damage, message):
    data = bundle()
    if damage == "duplicate":
        data["requests"][1]["source_path"] = data["requests"][0]["source_path"]
    elif damage == "missing_leaf":
        data["requests"].pop()
    elif damage == "unowned_leaf":
        data["conversations"].pop()
    elif damage == "fork":
        data["branches"][0]["mode"] = "fork"
    elif damage == "pre":
        data["branches"][0]["dispatch_timing"] = "pre"
    elif damage == "nested":
        data["conversations"][1]["agent_depth"] = 2
    elif damage == "extra_prerequisite":
        data["requests"][1]["depends_on"].append(2)
    elif damage == "changed_gap":
        data["requests"][1]["arrival_s"] = 50.
    elif damage == "changed_tokens":
        data["requests"][0]["prompt_token_ids"][0] += 1
    elif damage == "ground_truth_claim":
        data["dependency_basis"] = "observed causality"
    with pytest.raises(ValueError, match=message):
        plan_file(tmp_path, data)


@pytest.mark.parametrize("damage, message", [
    ("dropped_row_and_roster", "source-leaf roster"),
    ("zero_output", "zero-output"), ("changed_time", "source leaf"),
    ("changed_output", "source leaf"),
])
def test_source_bytes_independently_bind_complete_roster_and_original_times(tmp_path, damage, message):
    data = bundle()
    if damage == "dropped_row_and_roster":
        data["requests"].pop()
        data["roots"][0]["source_paths"].pop()
        data["conversations"][1]["request_indices"].pop()
    elif damage == "zero_output":
        data["roots"][0]["_source_fixture"]["requests"][-1]["out"] = 0
    elif damage == "changed_time":
        data["requests"][1].update(source_time_s=90., arrival_s=90.)
    elif damage == "changed_output":
        data["requests"][1]["output_tokens"] = 3
        data["requests"][1]["body"]["max_completion_tokens"] = 3
    with pytest.raises(ValueError, match=message):
        plan_file(tmp_path, data)


@pytest.mark.parametrize("child_finished, expected", [(15., 20.), (30., 30.)])
def test_active_join_always_keeps_original_absolute_due(tmp_path, child_finished, expected):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., 20.), child_times=((10.,),), join=True))
    state = FixedAbsoluteReleases(plan)
    assert state.pop_due(0.) == [(0, 0.)]
    state.complete(0, .01)
    assert state.pop_due(9.99) == []
    assert state.pop_due(10.) == [(2, 10.)]
    state.complete(2, child_finished)
    assert state.next_due == expected
    assert state.pop_due(expected) == [(1, expected)]
    state.complete(1, expected + .01)
    assert state.done and state.root_completion_times() == {"root0": expected + .01}


def test_new_earlier_branch_beats_a_previously_known_future_due_request(tmp_path):
    _, _, plan = plan_file(tmp_path)
    _, source = ingress_profile(tmp_path, 1., 2.)
    queue = source.new_release_queue()
    state = FixedAbsoluteReleases(plan)

    def mature(now):
        return [(i, queue.resolve_release(request(i, when))) for i, when in state.pop_due(now)]

    assert [(i, e.ready_at) for i, e in mature(0.)] == [(0, 3.)]
    state.complete(0, 5.)  # Root continuation is known now, but due at 100.
    assert mature(5.) == []
    assert queue.writer_available == 1. and queue.receiver_available == 3.
    assert [(i, e.ready_at) for i, e in mature(10.)] == [(2, 13.)]
    state.complete(2, 20.)  # A new earlier child continuation becomes eligible.
    assert state.next_due == 30.
    assert [(i, e.ready_at) for i, e in mature(30.)] == [(3, 33.)]
    assert [(i, e.ready_at) for i, e in mature(100.)] == [(1, 103.)]


def test_background_leaves_hold_root_completion_and_never_duplicate(tmp_path):
    _, _, plan = plan_file(tmp_path, bundle(root_times=(0., 1.), child_times=((10.,),)))
    state = FixedAbsoluteReleases(plan)
    assert state.pop_due(0.) == [(0, 0.)]
    state.complete(0, .01)
    assert state.pop_due(1.) == [(1, 1.)]
    state.complete(1, 2.)
    assert not state.done and state.root_completion_times() == {}
    assert state.pop_due(10.) == [(2, 10.)]
    assert state.pop_due(10.) == []
    state.complete(2, 11.)
    assert state.done and state.root_completion_times() == {"root0": 11.}
    with pytest.raises(ValueError, match="duplicated"):
        state.complete(2, 12.)


def test_root_clients_do_not_cap_outstanding_requests_and_ties_are_stable(tmp_path):
    _, _, plan = plan_file(tmp_path, bundle(
        root_times=(0., 1.), child_times=((1.,), (1.,)), clients=2))
    state = FixedAbsoluteReleases(plan)
    assert state.pop_due(0.) == [(0, 0.), (4, 0.)]
    # Completion callback order differs from frozen request order.
    state.complete(4, .5)
    state.complete(0, .5)
    due = state.pop_due(1.)
    assert due == [(i, 1.) for i in (1, 2, 3, 5, 6, 7)]
    assert len(state.released) - len(state.completed) == 6 > plan.evidence()["clients"]


def test_persistent_ingress_retains_writer_receiver_overlap_and_fifo(tmp_path):
    _, source = ingress_profile(tmp_path, 1., 2.)
    queue = source.new_release_queue()
    a, b, c = [queue.resolve_release(request(i, when)) for i, when in enumerate((0., 0., 1.5))]
    assert [e.source_service_started_at for e in (a, b, c)] == [0., 1., 2.]
    assert [e.ready_at for e in (a, b, c)] == [3., 5., 7.]
    assert [e.receipt_order for e in (a, b, c)] == [0, 1, 2]
    assert queue.evidence() == {"released_requests": 3, "writer_available_at": 3., "receiver_available_at": 7.}
    with pytest.raises(UnsupportedReadiness, match="chronological"):
        queue.resolve_release(request(3, 1.))

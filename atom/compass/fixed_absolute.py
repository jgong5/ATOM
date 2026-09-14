"""Immutable finite root bundles for corrected fixed-absolute chat replay."""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import re

from atom.compass.core.cache_policy import cache_on_policy, policy_errors
from atom.compass.core.loaded_input import load_json
from atom.compass.prefix_workload import token_digest, tokenizer_identity
from atom.compass.replay_plan import RESPONSE_DELIVERY


SCHEMA = "compass.fixed_absolute_root_bundle/1"
PROFILE = "aiperf_corrected_fixed_absolute_ignore_eos_v1"
DEPENDENCY_BASIS = "pinned_aiperf_loader_inferred"
QUALIFICATION = (
    "Corrected fixed-absolute replay; loader-inferred dependencies, "
    "not recovered ground-truth causality; zero-response-delivery approximation"
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class FixedAbsolutePlan:
    """Validate closed source membership and derive the supported dependency edges.

    Only same-chain continuations and one-level, post-response SPAWN branches
    are supported. A foreground branch may gate one later parent turn through
    SPAWN_JOIN. Several branches may share that parent gate. Nothing samples or
    replaces roots at runtime: every selected root is one original client.
    """

    def __init__(self, payload, loaded_input):
        self.loaded_input = loaded_input
        self._data = copy.deepcopy(payload)
        data = self._data
        if data.get("schema") != SCHEMA or data.get("profile") != PROFILE:
            raise ValueError("unsupported corrected fixed-absolute profile")
        if (data.get("dependency_basis") != DEPENDENCY_BASIS
                or data.get("initial_cache") != "acknowledged_empty"
                or data.get("time_scale") != 1
                or data.get("response_delivery") != RESPONSE_DELIVERY
                or policy_errors(data.get("cache_policy"), cache_on_policy())):
            raise ValueError("fixed-absolute policy, dependency qualification or source-time contract differs")
        roots, rows, conversations = data.get("roots"), data.get("requests"), data.get("conversations")
        if (not isinstance(roots, list) or len(roots) not in (1, 2, 4, 8)
                or type(data.get("clients")) is not int or data["clients"] != len(roots)
                or not isinstance(rows, list) or not rows or not isinstance(conversations, list)):
            raise ValueError("fixed-absolute requires a finite C1/C2/C4/C8 selected-root bundle")
        origin = data.get("source_time_origin_s")
        if not _finite(origin):
            raise ValueError("fixed-absolute requires a finite shared source time origin")
        root_by_id, source_members = {}, set()
        for client, root in enumerate(roots):
            root_id, paths = root.get("root_id"), root.get("source_paths")
            source = root.get("source") or {}
            if (not isinstance(root_id, str) or not root_id or root_id in root_by_id
                    or root.get("client_index") != client
                    or not isinstance(paths, list) or not paths or len(set(paths)) != len(paths)
                    or not all(isinstance(path, str) and re.fullmatch(r"/requests/\d+(?:/requests/\d+)*", path) for path in paths)
                    or not isinstance(source.get("path"), str) or not source["path"]
                    or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))):
                raise ValueError("root identity or complete source-leaf declaration is invalid")
            root_by_id[root_id] = root
            source_members.update((root_id, path) for path in paths)
        models, observed_members = set(), set()
        for index, row in enumerate(rows):
            root = root_by_id.get(row.get("root_id"))
            identity = (row.get("root_id"), row.get("source_path"))
            arrival, source_time = row.get("arrival_s"), row.get("source_time_s")
            if (type(row.get("index")) is not int or row["index"] != index or root is None
                    or row.get("client_index") != root["client_index"]
                    or identity in observed_members or identity not in source_members
                    or not _finite(arrival) or arrival < 0 or not _finite(source_time)
                    or not math.isclose(arrival, source_time - origin, rel_tol=0, abs_tol=1e-9)):
                raise ValueError("request identity or original absolute source offset differs")
            observed_members.add(identity)
            tokens = row.get("prompt_token_ids")
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens)
                    or len(tokens) != row.get("input_tokens")
                    or token_digest(tokens) != row.get("prompt_token_sha256")):
                raise ValueError("fixed-absolute prompt token identity differs")
            body, output = row.get("body") or {}, row.get("output_tokens")
            messages = body.get("messages")
            if (type(output) is not int or output <= 0
                    or body.get("max_completion_tokens") != output or body.get("ignore_eos") is not True
                    or body.get("stream") is not True or body.get("temperature") != 0
                    or body.get("stream_options") != {"include_usage": True}
                    or not isinstance(messages, list) or not messages
                    or any(not isinstance(message, dict)
                           or message.get("role") not in ("system", "user", "assistant")
                           or not isinstance(message.get("content"), str) for message in messages)
                    or not isinstance(body.get("model"), str) or not body["model"]):
                raise ValueError("fixed-absolute requires pinned text-chat and exact-output streaming bodies")
            models.add(body["model"])
        if observed_members != source_members or len(models) != 1 or min(row["arrival_s"] for row in rows) != 0:
            raise ValueError("fixed-absolute must retain every declared source leaf under one model and origin")

        by_conversation, owners, root_chains = {}, {}, {}
        dependencies = [set() for _ in rows]
        for conversation in conversations:
            name = conversation.get("conversation_id")
            indices = conversation.get("request_indices")
            root_id, depth = conversation.get("root_id"), conversation.get("agent_depth")
            if (not isinstance(name, str) or not name or name in by_conversation
                    or root_id not in root_by_id or type(depth) is not int or depth not in (0, 1)
                    or not isinstance(indices, list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("unsupported or incomplete conversation declaration")
            for turn, index in enumerate(indices):
                row = rows[index]
                if (index in owners or row.get("conversation_id") != name or row.get("turn_index") != turn
                        or row["root_id"] != root_id):
                    raise ValueError("request belongs to a different or duplicate conversation")
                if turn:
                    previous = indices[turn - 1]
                    if row["arrival_s"] < rows[previous]["arrival_s"]:
                        raise ValueError("conversation source timestamps run backwards")
                    dependencies[index].add(previous)
                owners[index] = name
            if depth == 0:
                if root_id in root_chains or conversation.get("parent_conversation_id") is not None:
                    raise ValueError("each original client requires exactly one root chain")
                root_chains[root_id] = name
            by_conversation[name] = conversation
        if set(owners) != set(range(len(rows))) or set(root_chains) != set(root_by_id):
            raise ValueError("conversation declarations omit requests or original roots")

        branches = data.get("branches")
        if not isinstance(branches, list):
            raise ValueError("fixed-absolute requires explicit branch declarations")
        branch_ids, spawned = set(), set()
        for branch in branches:
            name, parent_name = branch.get("branch_id"), branch.get("parent_conversation_id")
            parent = by_conversation.get(parent_name)
            trigger, gate = branch.get("after_request"), branch.get("join_before_request")
            children = branch.get("child_conversation_ids")
            if (not isinstance(name, str) or not name or name in branch_ids or parent is None
                    or parent["agent_depth"] != 0 or branch.get("mode") != "spawn"
                    or branch.get("dispatch_timing") != "post" or type(branch.get("is_background")) is not bool
                    or type(trigger) is not int or trigger not in parent["request_indices"]
                    or not isinstance(children, list) or not children or len(set(children)) != len(children)):
                raise ValueError("only single-level post-response SPAWN branches are supported")
            branch_ids.add(name)
            if gate is not None:
                if (type(gate) is not int or gate not in parent["request_indices"]
                        or parent["request_indices"].index(gate) <= parent["request_indices"].index(trigger)
                        or branch["is_background"]):
                    raise ValueError("SPAWN_JOIN must gate a later parent turn of a foreground branch")
            for child_name in children:
                child = by_conversation.get(child_name)
                if (child is None or child_name in spawned or child["agent_depth"] != 1
                        or child.get("parent_conversation_id") != parent_name
                        or child["root_id"] != parent["root_id"]):
                    raise ValueError("child must belong to exactly one branch of its original root")
                spawned.add(child_name)
                dependencies[child["request_indices"][0]].add(trigger)
                if gate is not None:
                    dependencies[gate].add(child["request_indices"][-1])
        if spawned != {name for name, c in by_conversation.items() if c["agent_depth"] == 1}:
            raise ValueError("a child conversation has no spawning branch")
        for index, row in enumerate(rows):
            if row.get("depends_on") != sorted(dependencies[index]):
                raise ValueError("request prerequisites differ from the declared loader-inferred chains/branches/joins")
        # Check reachability, including joins that would point back into their
        # own prerequisites. Merely having unique source identities is not enough.
        remaining = {i: set(deps) for i, deps in enumerate(dependencies)}
        while remaining:
            ready = {i for i, deps in remaining.items() if not deps}
            if not ready:
                raise ValueError("fixed-absolute dependency cycle")
            remaining = {i: deps - ready for i, deps in remaining.items() if i not in ready}
        self._dependencies = tuple(tuple(sorted(deps)) for deps in dependencies)

    @classmethod
    def load(cls, path, sha256):
        payload, loaded = load_json(path, role="runtime.fixed_absolute")
        if loaded.sha256 != sha256:
            raise ValueError("fixed-absolute artifact digest changed")
        return cls(payload, loaded)

    @property
    def rows(self):
        return copy.deepcopy(self._data["requests"])

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def roots(self):
        return copy.deepcopy(self._data["roots"])

    @property
    def model(self):
        return self._data["requests"][0]["body"]["model"]

    @property
    def cache_policy(self):
        return copy.deepcopy(self._data["cache_policy"])

    @property
    def response_delivery_seconds(self):
        return float(self._data["response_delivery"]["seconds"])

    def evidence(self):
        return {"schema": SCHEMA, "profile": PROFILE, "input": self.loaded_input.as_dict(),
                "clients": len(self._data["roots"]), "requests": len(self._data["requests"]),
                "dependency_basis": DEPENDENCY_BASIS, "qualification": QUALIFICATION,
                "response_delivery": copy.deepcopy(RESPONSE_DELIVERY),
                "root_ids": [root["root_id"] for root in self._data["roots"]],
                "prompt_token_sha256": [r["prompt_token_sha256"] for r in self._data["requests"]]}

    def workload(self):
        keys = ("index", "root_id", "client_index", "source_path", "conversation_id", "turn_index",
                "arrival_s", "input_tokens", "output_tokens", "prompt_token_sha256", "depends_on")
        return [{key: copy.deepcopy(row[key]) for key in keys} for row in self._data["requests"]]

    def verify_tokenizer(self, tokenizer):
        if tokenizer_identity(tokenizer, self.model) != self._data.get("tokenizer"):
            raise ValueError("fixed-absolute tokenizer identity changed")
        template = getattr(tokenizer, "chat_template", None)
        if (not isinstance(template, str)
                or hashlib.sha256(template.encode()).hexdigest() != self._data.get("chat_template_sha256")):
            raise ValueError("fixed-absolute chat template identity changed")
        for row in self._data["requests"]:
            text = tokenizer.apply_chat_template(row["body"]["messages"], tokenize=False,
                add_generation_prompt=True, **row["body"].get("chat_template_kwargs", {}))
            if tokenizer.encode(text) != row["prompt_token_ids"]:
                raise ValueError("fixed-absolute chat rendering differs from pinned tokens")

    def encode_payloads(self, *, declared):
        payloads = []
        for row in self._data["requests"]:
            body = copy.deepcopy(row["body"])
            body["compass_prompt_token_sha256"] = row["prompt_token_sha256"]
            if declared:
                body.update(compass_arrival=row["arrival_s"], compass_workload_size=len(self._data["requests"]),
                            compass_workload_index=row["index"])
            payloads.append(json.dumps(body).encode())
        return payloads


class FixedAbsoluteReleases:
    """Shared finite release state for real transport and the virtual calendar.

    Prerequisite completion queues eligibility; only pop_due matures it. This
    object never reserves ingress service for a future source timestamp.
    """

    def __init__(self, plan, epoch=0.0):
        if not _finite(epoch):
            raise ValueError("release epoch must be finite")
        self.epoch = float(epoch)
        self.rows = plan.rows
        self.dependencies = plan.dependencies
        self.released, self.completed = {}, {}
        self._pending = []
        self._queued = set()
        self._dependents = [[] for _ in self.rows]
        for index, deps in enumerate(self.dependencies):
            for predecessor in deps:
                self._dependents[predecessor].append(index)
            if not deps:
                self._queue(index)

    def _queue(self, index):
        if index in self._queued:
            return
        when = max([self.epoch + self.rows[index]["arrival_s"],
                    *(self.completed[p] for p in self.dependencies[index])])
        heapq.heappush(self._pending, (when, index))
        self._queued.add(index)

    @property
    def next_due(self):
        return self._pending[0][0] if self._pending else math.inf

    @property
    def done(self):
        return len(self.completed) == len(self.rows)

    def pop_due(self, now):
        if not _finite(now) or now < self.epoch:
            raise ValueError("release time must be finite and at or after the epoch")
        due = []
        while self._pending and self._pending[0][0] <= now:
            when, index = heapq.heappop(self._pending)
            self.released[index] = when
            due.append((index, when))
        return due

    def complete(self, index, response_at):
        if (index not in self.released or index in self.completed
                or not _finite(response_at) or response_at < self.released[index]):
            raise ValueError("request completion is duplicated, unreleased or before its release")
        self.completed[index] = float(response_at)
        for child in self._dependents[index]:
            if all(predecessor in self.completed for predecessor in self.dependencies[child]):
                self._queue(child)

    def root_completion_times(self):
        groups = {}
        for row in self.rows:
            groups.setdefault(row["root_id"], []).append(row["index"])
        return {root: max(self.completed[i] for i in indices)
                for root, indices in groups.items() if all(i in self.completed for i in indices)}

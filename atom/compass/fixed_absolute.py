"""Immutable finite root bundles for corrected fixed-absolute chat replay."""

from __future__ import annotations

import copy
from collections import Counter
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


class UnsupportedPredecessors(ValueError):
    def __init__(self, missing):
        self.missing = missing
        super().__init__(f"{len(missing)} loader replay predecessor(s) are not implied by the supported dependencies")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_leaves(data):
    """Current bounded source form; refuse unsupported/zero-output leaves whole.

    Flattened Weka requests can still reconstruct one-level SPAWN/joins. Raw
    nested wrappers need an independently witnessed source-path mapping before
    this exporter/profile can accept them; they are never silently pruned.
    """
    if (not isinstance(data, dict) or data.get("hash_id_scope") != "local"
            or data.get("block_size") != 64 or not isinstance(data.get("requests"), list)
            or not data["requests"]):
        raise ValueError("fixed-absolute source requires the pinned local 64-token Weka format")
    leaves = {}
    for index, row in enumerate(data["requests"]):
        if (not isinstance(row, dict) or row.get("type") not in ("n", "s")
                or row.get("requests") or type(row.get("out")) is not int or row["out"] <= 0
                or type(row.get("in")) is not int or row["in"] < 0 or not _finite(row.get("t"))):
            raise ValueError("fixed-absolute source contains an unsupported or zero-output leaf; root refused intact")
        leaves[f"/requests/{index}"] = row
    return leaves


def sequential_preparation_rows(plan):
    """Bound the reviewed exact-chat preparation to seven serial N1 requests."""
    rows = plan.rows
    reserved = {"compass_arrival", "compass_workload_size", "compass_workload_index"}
    if (len(plan.roots) != 1 or len(rows) != 7 or plan.dependencies[0]
            or any(index - 1 not in plan.dependencies[index] for index in range(1, 7))
            or any(row["output_tokens"] < 2 or reserved.intersection(row["body"]) for row in rows)):
        raise ValueError("exact-chat preparation supports only seven causally serial N1 requests with cap2")
    return rows


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
        root_by_id, source_members, originals, source_inputs = {}, set(), {}, []
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
            source_data, source_input = load_json(source["path"], role="runtime.fixed_absolute.source_root")
            if source_input.sha256 != source["sha256"] or source_data.get("id") != root_id:
                raise ValueError("fixed-absolute source-root bytes or root identity changed")
            actual = source_leaves(source_data)
            if paths != list(actual):
                raise ValueError("complete source-leaf roster differs from pinned source-root bytes")
            originals.update(((root_id, path), row) for path, row in actual.items())
            source_inputs.append(source_input)
        if origin != min(row["t"] for row in originals.values()):
            raise ValueError("shared source origin differs from pinned source-root times")
        self.source_inputs = tuple(source_inputs)
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
            original = originals[identity]
            if (source_time != original["t"] or row.get("source_input_tokens") != original["in"]
                    or row.get("output_tokens") != original["out"]):
                raise ValueError("request time/input/output differs from its pinned source leaf")
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
        if [(row["root_id"], row["source_path"]) for row in rows] != list(originals):
            raise ValueError("fixed-absolute tie order must retain selected-root/source-leaf order")

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
        missing = []
        for index, row in enumerate(rows):
            predecessors = row.get("loader_replay_predecessors")
            if (not isinstance(predecessors, list) or predecessors != sorted(set(predecessors))
                    or any(type(p) is not int or not 0 <= p < len(rows) for p in predecessors)):
                raise ValueError("each request must retain explicit loader replay predecessors")
            for predecessor in predecessors:
                seen, pending = set(), list(dependencies[index])
                while pending and predecessor not in seen:
                    node = pending.pop()
                    if node not in seen:
                        seen.add(node)
                        pending.extend(dependencies[node])
                if predecessor not in seen:
                    missing.append({"request_index": index, "predecessor_index": predecessor})
        if missing:
            raise UnsupportedPredecessors(missing)
        self._dependencies = tuple(tuple(sorted(deps)) for deps in dependencies)

    @classmethod
    def load(cls, path, sha256):
        payload, loaded = load_json(str(path), role="runtime.fixed_absolute")
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
                "source_roots": [record.as_dict() for record in self.source_inputs],
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

    def observation_errors(self, server, engine, results):
        """Require every source leaf, used plan, causal gate and ingress receipt."""
        # Source files are part of the validation closure, not declarations
        # accepted merely because they were copied into the submitted plan.
        FixedAbsolutePlan.load(self.loaded_input.requested, self.loaded_input.sha256)
        rows = self._data["requests"]
        errors = []
        by_index = {result.get("index"): result for result in results}
        engine_rows = engine.get("requests") or []
        observed = {row.get("request_id"): row for row in engine_rows}
        response_ids = [(result.get("response") or {}).get("id") for result in results]
        if (len(results) != len(rows) or set(by_index) != set(range(len(rows)))
                or len(set(response_ids)) != len(rows) or None in response_ids
                or len(engine_rows) != len(rows) or set(observed) != set(response_ids)):
            return ["fixed-absolute results/engine records do not cover each unique source request exactly once"]
        records = {}
        for index, wanted in enumerate(rows):
            result = by_index[index]
            record = observed[result["response"]["id"]]
            records[index] = record
            receipt, usage = record.get("shared_preprocessing") or {}, result["response"].get("usage") or {}
            if (result.get("ok") is not True
                    or receipt.get("prompt_token_sha256") != wanted["prompt_token_sha256"]
                    or receipt.get("input_tokens") != wanted["input_tokens"]
                    or usage.get("prompt_tokens") != wanted["input_tokens"]
                    or usage.get("completion_tokens") != wanted["output_tokens"]):
                errors.append(f"request {index} lacks matching consumed tokens and complete output")
        seq_ids = [record.get("seq_id") for record in records.values()]
        if None in seq_ids or len(set(seq_ids)) != len(rows):
            return errors + ["fixed-absolute engine sequence bindings are incomplete or duplicated"]
        if engine.get("clock") == "wall":
            for index, wanted in enumerate(rows):
                timing = by_index[index].get("send_timing") or {}
                predecessors = [(by_index[p].get("send_timing") or {}).get("finished_offset_s")
                                for p in self.dependencies[index]]
                if any(not _finite(value) for value in predecessors):
                    errors.append(f"request {index} has no complete prerequisite response time")
                    continue
                expected = max([wanted["arrival_s"], *predecessors])
                started, finished = timing.get("request_started_offset_s"), timing.get("finished_offset_s")
                if (timing.get("causal_release_offset_s") != expected or not _finite(started)
                        or not _finite(finished) or started < expected or finished < started):
                    errors.append(f"request {index} violates its real source/response release gate")
            return errors
        if engine.get("clock") != "virtual":
            return errors + ["fixed-absolute engine clock is not identified"]

        compass = server.get("compass") or {}
        ranks = (compass.get("loaded_inputs") or {}).get("ranks") or []
        if len(ranks) != 1:
            return errors + ["fixed-absolute requires one owning core receipt"]
        core = ranks[0].get("core_inputs") or {}
        reader = core.get("reader") or {}
        if reader.get("component") != "EngineCore.Scheduler" or type(reader.get("pid")) is not int or reader["pid"] <= 0:
            errors.append("fixed-absolute core receipt has no owning reader")
        inputs = core.get("inputs") or []
        plan_reads = [row for row in inputs if row.get("role") == "runtime.fixed_absolute"]
        if (compass.get("fixed_absolute_plan_sha256") != self.loaded_input.sha256
                or len(plan_reads) != 1 or plan_reads[0].get("sha256") != self.loaded_input.sha256
                or plan_reads[0].get("requested") != compass.get("fixed_absolute_plan")):
            errors.append("core did not load the configured fixed-absolute plan")
        actual_sources = Counter((row.get("requested"), row.get("sha256"), row.get("size"))
                                 for row in inputs if row.get("role") == "runtime.fixed_absolute.source_root")
        expected_sources = Counter((row.requested, row.sha256, row.size) for row in self.source_inputs)
        if actual_sources != expected_sources:
            errors.append("core loaded source-root inputs differ from the independently verified source bytes")
        calendar = core.get("release_calendar") or {}
        epoch = calendar.get("epoch")
        if (not _finite(epoch) or calendar.get("schema") != SCHEMA or calendar.get("profile") != PROFILE
                or (calendar.get("input") or {}).get("sha256") != self.loaded_input.sha256
                or calendar.get("response_delivery") != RESPONSE_DELIVERY
                or calendar.get("dependency_basis") != DEPENDENCY_BASIS
                or calendar.get("registered") is not True or calendar.get("complete") is not True):
            return errors + ["core did not complete the pinned corrected fixed-absolute calendar"]
        releases = {row.get("index"): row for row in calendar.get("releases") or []}
        completions = {row.get("index"): row for row in calendar.get("completions") or []}
        services = (core.get("request_readiness") or {}).get("causal_releases") or []
        by_seq = {row.get("seq_id"): row for row in services}
        if (len(calendar.get("releases") or []) != len(rows) or set(releases) != set(range(len(rows)))
                or len(calendar.get("completions") or []) != len(rows) or set(completions) != set(releases)
                or len(services) != len(rows) or set(by_seq) != set(seq_ids)):
            return errors + ["calendar/readiness does not cover every source leaf exactly once"]
        root_done = {}
        for index, wanted in enumerate(rows):
            release, completion, record = releases[index], completions[index], records[index]
            service = by_seq[record["seq_id"]]
            responses = [completions[p].get("modelled_client_response_available_at") for p in self.dependencies[index]]
            if any(not _finite(value) for value in responses):
                errors.append(f"request {index} has no completed prerequisite")
                continue
            expected = max([epoch + wanted["arrival_s"], *responses])
            ready, started, finished = service.get("ready_at"), service.get("source_service_started_at"), record.get("finish_time")
            if (release.get("seq_id") != record["seq_id"] or completion.get("seq_id") != record["seq_id"]
                    or release.get("root_id") != wanted["root_id"] or release.get("released_at") != expected
                    or release.get("source_earliest_at") != epoch + wanted["arrival_s"]
                    or record.get("arrive_time") != expected or service.get("arrived_at") != expected
                    or release.get("ready_at") != ready or release.get("source_service_started_at") != started
                    or release.get("receipt_order") != service.get("receipt_order")
                    or not _finite(ready) or not _finite(started) or not expected <= started <= ready
                    or not _finite(finished) or finished < ready
                    or completion.get("native_engine_finished_at") != finished
                    or completion.get("modelled_client_response_available_at") != finished + self.response_delivery_seconds
                    or completion.get("completion_tokens") != wanted["output_tokens"]):
                errors.append(f"request {index} release/completion/readiness binding differs")
            if _finite(finished):
                root_done[wanted["root_id"]] = max(root_done.get(wanted["root_id"], -math.inf), finished + self.response_delivery_seconds)
            ingress = service.get("ingress") or {}
            if ingress.get("token_bytes") != 4 * wanted["input_tokens"]:
                errors.append(f"request {index} ingress token storage differs")
        if calendar.get("root_completed_at") != root_done or set(root_done) != {root["root_id"] for root in self.roots}:
            errors.append("original root clients did not drain through all their leaves")
        if sorted(service.get("receipt_order", -1) for service in services) != list(range(len(rows))):
            return errors + ["causal ingress receipt order is incomplete or duplicated"]
        ordered = sorted(services, key=lambda row: row["receipt_order"])
        source_order = sorted(releases, key=lambda i: (releases[i]["released_at"], i))
        if [row["seq_id"] for row in ordered] != [releases[i]["seq_id"] for i in source_order]:
            errors.append("ingress service order differs from chronological causal release order")
        # Rebuild the independent source queue from its actual loaded profile,
        # not from reported elapsed request times or evaluated-pair residuals.
        try:
            from atom.compass.runtime.request_readiness import IngressDescriptor, RegisteredRequest, RequestReadiness
            source = RequestReadiness(compass.get("request_readiness_profile", ""))
            expected_inputs = Counter((row.role, row.requested, row.sha256) for row in source._inputs)
            actual_inputs = Counter((row.get("role"), row.get("requested"), row.get("sha256"))
                                   for row in inputs if str(row.get("role", "")).startswith("runtime.request_readiness."))
            if expected_inputs != actual_inputs:
                errors.append("source ingress profile/fit inputs differ from the core's actual reads")
            queue = source._provider.new_release_queue()
            for row in ordered:
                descriptor = IngressDescriptor(**row["ingress"])
                event = queue.resolve_release(RegisteredRequest(row["seq_id"], row["arrived_at"],
                    descriptor.token_bytes // 4, row["receipt_order"], descriptor))
                if (event.ready_at != row["ready_at"] or event.source_service_started_at != row["source_service_started_at"]
                        or event.receipt_order != row["receipt_order"]):
                    errors.append(f"source ingress queue re-derivation differs for sequence {row['seq_id']}")
            state = queue.evidence()
            if (core["request_readiness"].get("causal_queue") != state
                    or state["released_requests"] != len(rows)
                    or max(state["writer_available_at"], state["receiver_available_at"]) > max(root_done.values())):
                errors.append("source ingress queues did not drain consistently with completed roots")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"cannot verify source ingress queue: {exc}")
        return errors


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

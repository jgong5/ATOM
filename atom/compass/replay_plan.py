"""Pinned chat payloads and release rules for a two-turn AIPerf opening."""

from __future__ import annotations

import copy
import hashlib
import json
import math

from atom.compass.core.loaded_input import load_json
from atom.compass.core.cache_policy import cache_on_policy, policy_errors
from atom.compass.prefix_workload import token_digest, tokenizer_identity


SCHEMA = "compass.aiperf_opening/1"
PROFILE = "aiperf_fixed_opening_ignore_eos_v1"
RESPONSE_DELIVERY = {
    "kind": "assumed_zero",
    "seconds": 0.0,
    "from": "native_engine_finish",
    "to": "client_response_available",
}


class OpeningPlan:
    """Read identity at load time; callers receive copies of mutable payloads."""

    def __init__(self, payload, loaded_input):
        self.loaded_input = loaded_input
        self._data = copy.deepcopy(payload)
        data = self._data
        if data.get("schema") != SCHEMA or data.get("profile") != PROFILE:
            raise ValueError("unsupported AIPerf opening profile")
        if (data.get("clients") != 1 or data.get("branches") != []
                or data.get("initial_cache") != "acknowledged_empty"):
            raise ValueError("opening profile requires one client, no branches and empty cache")
        if data.get("response_delivery") != RESPONSE_DELIVERY:
            raise ValueError("opening v1 requires the explicit zero response-delivery approximation")
        if data.get("time_scale") != 1:
            raise ValueError("opening profile preserves source gaps without scaling")
        if policy_errors(data.get("cache_policy"), cache_on_policy()):
            raise ValueError("opening requires its complete cache/checkpoint policy")
        rows = data.get("requests")
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError("opening profile requires exactly two requests")
        models = set()
        for index, row in enumerate(rows):
            arrival = row.get("arrival_s")
            if (row.get("index") != index
                    or row.get("depends_on") != ([] if index == 0 else [0])
                    or row.get("source_path") != f"/requests/{index}"):
                raise ValueError("opening requests must be the first two source turns in one chain")
            if not isinstance(arrival, (int, float)) or not math.isfinite(arrival) or arrival < 0:
                raise ValueError("opening arrival must be finite and nonnegative")
            if index == 0 and arrival != 0:
                raise ValueError("opening must start at the original first request")
            tokens = row.get("prompt_token_ids")
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError("opening requires explicit integer prompt tokens")
            if (len(tokens) != row.get("input_tokens")
                    or token_digest(tokens) != row.get("prompt_token_sha256")):
                raise ValueError("opening token identity does not match its pinned input")
            body = row.get("body", {})
            output = row.get("output_tokens")
            if (type(output) is not int or output < 1
                    or body.get("max_completion_tokens") != output
                    or body.get("ignore_eos") is not True
                    or body.get("stream") is not True or body.get("temperature") != 0):
                raise ValueError("opening requires explicit exact-output streaming policy")
            messages = body.get("messages")
            if (not isinstance(messages, list) or not messages
                    or any(message.get("role") not in ("system", "user", "assistant")
                           or not isinstance(message.get("content"), str)
                           for message in messages)):
                raise ValueError("opening requires loader-exported text chat messages")
            models.add(body.get("model"))
        if len(models) != 1 or not next(iter(models)):
            raise ValueError("opening requests must target one named model")

    @classmethod
    def load(cls, path, sha256):
        payload, loaded = load_json(path, role="runtime.aiperf_opening")
        if loaded.sha256 != sha256:
            raise ValueError("AIPerf opening artifact digest changed")
        return cls(payload, loaded)

    @property
    def model(self):
        return self._data["requests"][0]["body"]["model"]

    @property
    def response_delivery_seconds(self):
        return float(self._data["response_delivery"]["seconds"])

    @property
    def rows(self):
        return copy.deepcopy(self._data["requests"])

    @property
    def cache_policy(self):
        return copy.deepcopy(self._data["cache_policy"])

    def evidence(self):
        return {
            "schema": SCHEMA,
            "profile": PROFILE,
            "input": self.loaded_input.as_dict(),
            "response_delivery": copy.deepcopy(RESPONSE_DELIVERY),
            "qualification": "AIPerf-aligned opening, zero-response-delivery approximation",
            "prompt_token_sha256": [row["prompt_token_sha256"] for row in self._data["requests"]],
        }

    def workload(self):
        return [{key: copy.deepcopy(row[key]) for key in (
            "arrival_s", "input_tokens", "output_tokens", "prompt_token_sha256",
            "source_path", "depends_on")} for row in self._data["requests"]]

    def verify_tokenizer(self, tokenizer):
        if tokenizer_identity(tokenizer, self.model) != self._data.get("tokenizer"):
            raise ValueError("opening tokenizer identity changed")
        template = getattr(tokenizer, "chat_template", None)
        if (not isinstance(template, str)
                or hashlib.sha256(template.encode()).hexdigest() != self._data.get("chat_template_sha256")):
            raise ValueError("opening chat template identity changed")
        for row in self._data["requests"]:
            # This is the server's Jinja render followed by IOProcessor tokenization.
            text = tokenizer.apply_chat_template(
                row["body"]["messages"], tokenize=False, add_generation_prompt=True,
                **row["body"].get("chat_template_kwargs", {}))
            tokens = tokenizer.encode(text)
            if tokens != row["prompt_token_ids"]:
                raise ValueError("server-equivalent chat rendering differs from the opening export")

    def encode_payloads(self, *, declared):
        payloads = []
        for row in self._data["requests"]:
            body = copy.deepcopy(row["body"])
            body["compass_prompt_token_sha256"] = row["prompt_token_sha256"]
            if declared:
                body.update(compass_arrival=row["arrival_s"], compass_workload_size=2,
                            compass_workload_index=row["index"])
            payloads.append(json.dumps(body).encode())
        return payloads

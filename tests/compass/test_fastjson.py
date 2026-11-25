# SPDX-License-Identifier: MIT
"""atom.compass.run.fastjson gives json's and pydantic's values, bytes and refusals."""

import enum
import json
import random

import pytest
from pydantic import NonNegativeInt, TypeAdapter, ValidationError

from atom.compass import carriers
from atom.compass.run import fastjson


def _dumps(content):
    return json.dumps(
        content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")
    )


class _Small(enum.IntEnum):
    ONE = 1


def _leaf(rng):
    return rng.choice(
        [
            rng.randrange(-(10**6), 10**6),
            rng.random() * 10 ** rng.randint(-30, 30),
            "".join(rng.choice("ab\"\\\n\t\x01é中  ") for _ in range(rng.randrange(6))),
            True,
            None,
            2**70,
            _Small.ONE,
        ]
    )


def _doc(rng, depth=0):
    kind = rng.random()
    if depth > 3 or kind < 0.3:
        return _leaf(rng)
    if kind < 0.5:
        return [rng.randrange(200000) for _ in range(rng.choice([0, 3, 511, 512, 2000]))]
    if kind < 0.6:
        big = [rng.randrange(200000) for _ in range(600)]
        big[rng.randrange(600)] = rng.choice([True, 1.5, "x", None, -3, 2**80, _Small.ONE])
        return big
    if kind < 0.8:
        return {f"k{i}é": _doc(rng, depth + 1) for i in range(rng.randrange(5))}
    return rng.choice([list, tuple])(_doc(rng, depth + 1) for _ in range(rng.randrange(5)))


def test_dumps_writes_the_bytes_json_dumps_writes():
    rng = random.Random(7)
    for _ in range(3000):
        doc = _doc(rng)
        assert fastjson.dumps(doc) == _dumps(doc)
    for doc in ({1: "a"}, {"a": {True: 1}}, [[1] * 600, [2] * 600], {"x": (1,) * 700}):
        assert fastjson.dumps(doc) == _dumps(doc)


def test_dumps_refuses_what_json_dumps_refuses():
    for doc in ([float("nan")], {"a": [1.0] * 600 + [float("inf")]}, [object()]):
        with pytest.raises((ValueError, TypeError)):
            _dumps(doc)
        with pytest.raises((ValueError, TypeError)):
            fastjson.dumps(doc)


def test_loads_reads_what_json_loads_reads_and_refuses_alike():
    rng = random.Random(11)
    for _ in range(2000):
        doc = _doc(rng)
        try:
            body = _dumps(doc).encode()
        except (ValueError, TypeError):
            continue
        assert repr(fastjson.loads(body)) == repr(json.loads(body))
    for body in (b"[NaN]", b"[1e400]", b'["\\ud800"]', b'\xef\xbb\xbf{"x":1}', b'{"a":1,"a":2}'):
        assert repr(fastjson.loads(body)) == repr(json.loads(body))
    for body in (b"", b"{", b"[1,]"):
        with pytest.raises(json.JSONDecodeError):
            json.loads(body)
        with pytest.raises(json.JSONDecodeError):
            fastjson.loads(body)


def test_prompt_ids_take_pydantics_values_and_refusals():
    adapter = TypeAdapter(list[NonNegativeInt])
    fast = fastjson._PromptIds(adapter)
    for value in ([1, 2, 3], [0] * 1000, [2**70], [True, 1], [1.0, 2], ["3"]):
        assert fast.validate_python(value) == adapter.validate_python(value)
    for value in ([-1], [1.5], ["x"], None, {"a": 1}):
        with pytest.raises(ValidationError) as theirs:
            adapter.validate_python(value)
        with pytest.raises(ValidationError) as ours:
            fast.validate_python(value)
        assert ours.value.errors() == theirs.value.errors()


def test_installed_render_and_request_json():
    import asyncio

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    doc = {"id": "x", "kv_transfer_params": {"prompt_token_ids": list(range(5000))}}
    before = JSONResponse(doc).body
    fastjson.install()
    assert JSONResponse(doc).body == before
    with pytest.raises(ValueError):
        JSONResponse({"a": float("nan")})

    async def read(body):
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return await Request({"type": "http", "headers": []}, receive).json()

    assert asyncio.run(read(before)) == doc


def test_relay_fields_read_once_give_what_each_carrier_read():
    bodies = [
        b'{"kv_transfer_params":{"compass":"a:1.5;s:7","compass_relay_s":0.25,"prompt_token_ids":[1,2]}}',
        b'{"kv_transfer_params":{"compass":"a:2.0;s:1"}}',
        b'{"kv_transfer_params":null}',
        b'{"other":1}',
        b"not json",
    ]
    for body in bodies:
        params = carriers.relay_params(body)
        assert carriers.stamp_of(params) == carriers.relay_stamp(body)
        assert carriers.service_of(params) == carriers.relay_service(body)

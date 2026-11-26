# SPDX-License-Identifier: MIT
"""A simulated prefill-decode deployment's hand-off of a prompt's token ids.

A prefill server's response carries its prompt's token ids, and the router
puts them in decode's request as ``kv_transfer_params["prompt_token_ids"]``,
where its cache-aware decode policy reads them. The prefill frontend writes
them with ``JSONResponse.render``; the decode frontend reads them with
``Request.json`` and checks them in `resolve_prompt_token_ids`. In a simulated
server each runs while its LP's clock stands still and every other LP waits
on its wall time: the standard library's JSON and pydantic held 14.4% of the
8P16D C1228 sim's critical path (jgong5/ATOM#4114).

`install` does the same work through msgspec, and keeps every byte and verdict:

- A decode frontend's request body that msgspec decodes is that decoding,
  which for every body it decodes is ``json.loads``'s; ``json.loads`` reads or
  refuses any other. A relayed body is decoded once: the decoding
  `atom.utils.compass_loop.HttpChannel` made of it to read its stamp
  (`atom.compass.carriers.decode_relayed`) comes in the request's scope, and
  the first reading of the same bytes takes it. A list of ids msgspec takes as
  non-negative ints is those ints, as pydantic returns them; pydantic coerces
  or refuses any other.
- A prefill frontend's response whose top-level ``prompt_token_ids`` holds
  ints alone is the original rendering of the rest with msgspec's text of
  them in their place; the original renders any other.

What a run is charged is unchanged: no charge reads the wall time these take.

With `DIR_ENV` naming a directory, the prefill side hands ids an int32 array
holds to decode by reference: it writes them to a file there and puts
``[REF, pid, n, bytes]`` in their place, the file's name and the bytes of the
ids' text, which the router carries as it carries the ids. The decode side
puts the same ids back, removes the file and skips their check, so a
reference is read once. Every count of a body's bytes adds what its reference
stands for (`inline_extra`): a run is charged as with the ids inline, and only
the wall time of writing, carrying and reading them goes. The run's prefill
and decode frontends share the directory, so they run on one host; it is the
run's own, on tmpfs, and whoever starts the run removes it afterwards. Unset,
every id goes inline, and a decode side without it refuses a reference.
"""

import array
import collections
import contextlib
import functools
import itertools
import os
from typing import Annotated

import msgspec

from atom.compass.carriers import RELAYED_JSON

#: The response key whose integers the prefill side writes with msgspec.
KEY = "prompt_token_ids"
#: Takes the integers' place while the original renders the rest.
_STAND_IN = "\0compass:prompt_token_ids\0"
#: The attribute marking a function this module installed.
_INSTALLED = "compass_handoff"
_IDS = list[Annotated[int, msgspec.Meta(ge=0)]]
_DECODE = msgspec.json.Decoder()
_ENCODE = msgspec.json.Encoder()

#: Names the directory through which ids go by reference; unset or empty, none do.
DIR_ENV = "COMPASS_IDS_BY_REF_DIR"
_DIR = os.environ.get(DIR_ENV) or None
#: A reference's first item: no id, so a decode side reading ids inline refuses it.
REF = "compass:ids"
#: Ids that may go by reference: what an int32 array holds, >= 0.
_IDS32 = list[Annotated[int, msgspec.Meta(ge=0, lt=2**31)]]
_FILES = itertools.count()
_KEY_TEXT = b'"' + KEY.encode() + b'":'
#: A reference's start, as msgspec and the router write it.
_PATTERN = _KEY_TEXT + _ENCODE.encode([REF])[:-1] + b","
#: Ids read back from a file, by `id`, each held until `CheckedIds` takes it.
_READ_BACK: collections.OrderedDict = collections.OrderedDict()
_READ_BACK_HELD = 64


def inline_extra(body: bytes) -> int:
    """The bytes the ids `body`'s hand-off reference stands for would add to
    it inline, less the reference's own; 0 for a body with none, as with ids
    by reference off. A body with two is refused."""
    if _DIR is None or (k := body.find(_PATTERN)) < 0:
        return 0
    if body.find(_PATTERN, k + 1) >= 0:
        raise ValueError("a body carries two hand-off references")
    start = k + len(_KEY_TEXT)
    end = body.index(b"]", start) + 1
    return _checked(_DECODE.decode(body[start:end]))[3] - (end - start)


def _checked(ref: list) -> list:
    """`ref`, a list whose first item is `REF`, refused unless it is a reference."""
    if len(ref) != 4 or not all(type(v) is int and v >= 0 for v in ref[1:]):
        raise ValueError(f"a malformed hand-off reference {ref!r}")
    return ref


def _keep(ids: list, nbytes: int) -> bytes:
    """Write `ids`, checked as `_IDS32`, to a file and return the text of the
    reference that takes the place of their text, `nbytes` bytes."""
    n = next(_FILES)
    with open(f"{_DIR}/{os.getpid()}_{n}", "wb") as f:
        f.write(array.array("i", ids))
    return _ENCODE.encode([REF, os.getpid(), n, nbytes])


def _take(doc, body: bytes) -> None:
    """Put back in `doc`, `body`'s decoding, the ids its hand-off reference
    stands for, and remove their file."""
    kv = doc.get("kv_transfer_params") if type(doc) is dict else None
    ref = kv.get(KEY) if type(kv) is dict else None
    if type(ref) is not list or not ref or ref[0] != REF:
        return
    _checked(ref)
    if body.find(_PATTERN) < 0:  # its bytes would be counted short
        raise ValueError(f"a hand-off reference {ref} in a body not written compactly")
    path = f"{_DIR}/{ref[1]}_{ref[2]}"
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise ValueError(
            f"no ids for hand-off reference {ref}: it was read already, or "
            f"written to another directory than {_DIR}"
        ) from None
    os.unlink(path)
    ids = array.array("i")
    ids.frombytes(data)
    ids = kv[KEY] = ids.tolist()
    _READ_BACK[id(ids)] = ids
    while len(_READ_BACK) > _READ_BACK_HELD:
        _READ_BACK.popitem(last=False)


def install(side: str | None) -> None:
    """Install the decode side's reading of the hand-off (``D``) or the
    prefill side's writing of it (``P``), once a process; nothing for any
    other side."""
    if side == "D":
        from starlette.requests import Request

        from atom.entrypoints.openai import protocol

        _once(Request, "json", _reading)
        if not isinstance(protocol._PROMPT_TOKEN_IDS_ADAPTER, CheckedIds):
            protocol._PROMPT_TOKEN_IDS_ADAPTER = CheckedIds(
                protocol._PROMPT_TOKEN_IDS_ADAPTER
            )
    elif side == "P":
        from starlette.responses import JSONResponse

        if _DIR is not None:
            os.makedirs(_DIR, exist_ok=True)
        _once(JSONResponse, "render", _writing)


def _once(owner, name: str, wrapper) -> None:
    """Replace `owner`'s `name` with `wrapper` of it, unless this did already."""
    current = getattr(owner, name)
    if not getattr(current, _INSTALLED, False):
        wrapped = wrapper(current)
        setattr(wrapped, _INSTALLED, True)
        setattr(owner, name, wrapped)


def _reading(json_of):
    """`json_of`, Starlette's ``Request.json``, with a body msgspec decodes
    taken as that decoding: for a relayed body, the one its scope hands on
    under `RELAYED_JSON`, taken once and for the same bytes. The ids a hand-off
    reference in it stands for take its place."""

    @functools.wraps(json_of)
    async def json(request):
        if not hasattr(request, "_json"):
            body = await request.body()
            relayed = request.scope.pop(RELAYED_JSON, None)
            if relayed is not None and relayed[0] == body:
                request._json = relayed[1]
            else:
                # A body msgspec refuses, json_of reads or refuses as it always has.
                with contextlib.suppress(
                    msgspec.DecodeError, ValueError, RecursionError
                ):
                    request._json = _DECODE.decode(body)
            if _DIR is not None and hasattr(request, "_json"):
                _take(request._json, body)
        return await json_of(request)

    return json


class CheckedIds:
    """`adapter`, a pydantic ``TypeAdapter`` of a list of non-negative ints,
    taking a list that msgspec checks as one: strict, so a bool, a float or a
    string goes to `adapter`, which coerces or refuses it as before. Ids read
    back from a hand-off reference's file are taken as they are."""

    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def validate_python(self, ids, *args, **kwargs):
        if type(ids) is list and not args and not kwargs:
            if _READ_BACK.pop(id(ids), None) is ids:
                return ids
            with contextlib.suppress(msgspec.ValidationError):
                return msgspec.convert(ids, _IDS)
        return self.adapter.validate_python(ids, *args, **kwargs)


def _writing(render):
    """`render`, Starlette's ``JSONResponse.render``, writing a dict's
    top-level `KEY` list with msgspec where it holds ints alone, by reference
    where it holds ids that may go so."""

    @functools.wraps(render)
    def write(response, content):
        ids = content.get(KEY) if type(content) is dict else None
        if type(ids) is list:
            held = None  # the ids checked once as `_IDS32`, to go by reference
            if _DIR is not None:
                with contextlib.suppress(msgspec.ValidationError):
                    held = msgspec.convert(ids, _IDS32)
            # Ints alone are written alike; anything else, render writes or
            # refuses as it always has.
            try:
                text = _ENCODE.encode(
                    held if held is not None else msgspec.convert(ids, list[int])
                )
            except (msgspec.ValidationError, ValueError):
                text = None
            if text is not None:
                stand_in = render(response, _STAND_IN)
                rest = render(response, {**content, KEY: _STAND_IN})
                if rest.count(stand_in) == 1:
                    if held:
                        text = _keep(held, len(text))
                    return rest.replace(stand_in, text)
        return render(response, content)

    return write

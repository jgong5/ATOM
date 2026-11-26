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
- That decoding holds a ``kv_transfer_params["prompt_token_ids"]`` list as a
  tuple of its items. Starlette keeps a request's decoding until its response
  ends, so an in-flight stream holds its prompt's ids all along, and every
  full collection of the cyclic garbage collector walked each id of a list;
  a tuple of ints leaves the collector's tracking at the first collection
  that meets it. No reader of the decoding mutates the ids, and only
  `resolve_prompt_token_ids` tells a tuple from a list, in its check, which
  takes the tuple as the list it was; JSON writes a tuple as that list.
- A prefill frontend's response whose top-level ``prompt_token_ids`` holds
  ints alone is the original rendering of the rest with msgspec's text of
  them in their place; the original renders any other.

What a run is charged is unchanged: no charge reads the wall time these take.
"""

import contextlib
import functools
from typing import Annotated

import msgspec

from atom.compass.carriers import JSON_DECODER, RELAYED_JSON

#: The response key whose integers the prefill side writes with msgspec.
KEY = "prompt_token_ids"
#: Takes the integers' place while the original renders the rest.
_STAND_IN = "\0compass:prompt_token_ids\0"
#: The attribute marking a function this module installed.
_INSTALLED = "compass_handoff"
_IDS = list[Annotated[int, msgspec.Meta(ge=0)]]
_ENCODE = msgspec.json.Encoder()


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
    under `RELAYED_JSON`, taken once and for the same bytes."""

    @functools.wraps(json_of)
    async def json(request):
        if not hasattr(request, "_json"):
            body = await request.body()
            relayed = request.scope.pop(RELAYED_JSON, None)
            if relayed is not None and relayed[0] == body:
                request._json = _held(relayed[1])
            else:
                # A body msgspec refuses, json_of reads or refuses as it always has.
                with contextlib.suppress(
                    msgspec.DecodeError, ValueError, RecursionError
                ):
                    request._json = _held(JSON_DECODER.decode(body))
        return await json_of(request)

    return json


def _held(doc):
    """`doc` with the list its ``kv_transfer_params`` holds under `KEY`, where
    it holds one, as a tuple of the same items: a request holds its decoding
    while its response streams, and the collector walks a list on every full
    collection but stops tracking a tuple of ints. In place, as `HttpChannel`
    holds a relayed body's decoding while it serves the request."""
    params = doc.get("kv_transfer_params") if type(doc) is dict else None
    ids = params.get(KEY) if type(params) is dict else None
    if type(ids) is list:
        params[KEY] = tuple(ids)
    return doc


class CheckedIds:
    """`adapter`, a pydantic ``TypeAdapter`` of a list of non-negative ints,
    taking a list that msgspec checks as one: strict, so a bool, a float or a
    string goes to `adapter`, which coerces or refuses it as before. A tuple
    is the list `_held` made one of, and is taken as that list."""

    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def validate_python(self, ids, *args, **kwargs):
        if type(ids) in (list, tuple) and not args and not kwargs:
            with contextlib.suppress(msgspec.ValidationError):
                return msgspec.convert(ids, _IDS)
            if type(ids) is tuple:
                ids = list(ids)
        return self.adapter.validate_python(ids, *args, **kwargs)


def _writing(render):
    """`render`, Starlette's ``JSONResponse.render``, writing a dict's
    top-level `KEY` list with msgspec where it holds ints alone."""

    @functools.wraps(render)
    def write(response, content):
        ids = content.get(KEY) if type(content) is dict else None
        if type(ids) is list:
            # Ints alone are written alike; anything else, render writes or
            # refuses as it always has.
            try:
                text = _ENCODE.encode(msgspec.convert(ids, list[int]))
            except (msgspec.ValidationError, ValueError):
                text = None
            if text is not None:
                stand_in = render(response, _STAND_IN)
                rest = render(response, {**content, KEY: _STAND_IN})
                if rest.count(stand_in) == 1:
                    return rest.replace(stand_in, text)
        return render(response, content)

    return write

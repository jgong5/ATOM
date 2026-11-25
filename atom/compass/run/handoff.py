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
  refuses any other. A list of ids msgspec takes as non-negative ints is those
  ints, as pydantic returns them; pydantic coerces or refuses any other.
- A prefill frontend's response whose top-level ``prompt_token_ids`` holds
  ints alone is the original rendering of the rest with msgspec's text of
  them in their place; the original renders any other.

What a run is charged is unchanged: no charge reads the wall time these take.
"""

import contextlib
import functools
from typing import Annotated

import msgspec

#: The response key whose integers the prefill side writes with msgspec.
KEY = "prompt_token_ids"
#: Takes the integers' place while the original renders the rest.
_STAND_IN = "\0compass:prompt_token_ids\0"
#: The attribute marking a function this module installed.
_INSTALLED = "compass_handoff"
_IDS = list[Annotated[int, msgspec.Meta(ge=0)]]
_DECODE = msgspec.json.Decoder()
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
    taken as that decoding."""

    @functools.wraps(json_of)
    async def json(request):
        if not hasattr(request, "_json"):
            body = await request.body()
            # A body msgspec refuses, json_of reads or refuses as it always has.
            with contextlib.suppress(msgspec.DecodeError, ValueError, RecursionError):
                request._json = _DECODE.decode(body)
        return await json_of(request)

    return json


class CheckedIds:
    """`adapter`, a pydantic ``TypeAdapter`` of a list of non-negative ints,
    taking a list that msgspec checks as one: strict, so a bool, a float or a
    string goes to `adapter`, which coerces or refuses it as before."""

    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def validate_python(self, ids, *args, **kwargs):
        if type(ids) is list and not args and not kwargs:
            with contextlib.suppress(msgspec.ValidationError):
                return msgspec.convert(ids, _IDS)
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

# SPDX-License-Identifier: MIT
"""A simulated frontend's JSON, decoded and encoded faster to the same values and bytes.

A prefill-decode hand-off carries the prompt's token ids, beside the response
prefill sends and inside the request decode takes in: about 110k ids at the
C1228 recipe's later turns. A simulated server runs that path for real while its
clock stands still, so every other LP waits on its wall time; what it costs in
simulated time is the host laws' charge on the body's bytes, never its wall time.
`install`, called in a simulated frontend, replaces three steps of it:

* `starlette.requests.Request.json` decodes with msgspec, and with ``json.loads``
  where msgspec refuses the body. msgspec refuses, not misreads, what ``json``
  reads otherwise (``NaN``, an out-of-range float, a lone surrogate, a BOM, a
  body that is not UTF-8), and gives the same values for the rest.
* ``atom.entrypoints.openai.protocol``'s prompt-id adapter takes a list of plain
  non-negative ints through ``msgspec.convert`` in strict mode, which accepts
  exactly those; anything else, and every refusal, goes to pydantic as before.
* `starlette.responses.JSONResponse.render` writes the same bytes as its
  ``json.dumps``: a dict with str keys and a list are written member by member
  with that encoder, and a list of `BIG` or more members by msgspec, kept only
  when its text holds nothing but digits, ``-``, ``,``, ``[`` and ``]``, a list
  of ints, which both write alike; anything else falls back to the original.

Outside a simulated run nothing is installed.
"""

import json
from typing import Annotated

import msgspec

#: A list this long or longer is tried through msgspec.
BIG = 512
_INT_TEXT = b"0123456789,-[]"
_ENCODE = json.JSONEncoder(
    ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")
).encode
_MSG_DECODE = msgspec.json.decode
_MSG_ENCODE = msgspec.json.Encoder().encode
_PROMPT_IDS = list[Annotated[int, msgspec.Meta(ge=0)]]
_installed = False


def loads(body):
    """``json.loads(body)``'s value, decoded by msgspec where it reads the body."""
    try:
        return _MSG_DECODE(body)
    except (msgspec.DecodeError, msgspec.ValidationError):
        return json.loads(body)


def dumps(content) -> str:
    """``json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None,
    separators=(",", ":"))``, the same text."""
    t = type(content)
    if t is dict:
        if not all(type(k) is str for k in content):
            return _ENCODE(content)
        if not content:
            return "{}"
        return "{" + ",".join([f"{_ENCODE(k)}:{dumps(v)}" for k, v in content.items()]) + "}"
    if t is list or t is tuple:
        if not content:
            return "[]"
        if len(content) >= BIG:
            try:
                raw = _MSG_ENCODE(content)
            except (TypeError, ValueError, OverflowError, msgspec.EncodeError):
                return _ENCODE(content)
            if not raw.translate(None, _INT_TEXT):
                return raw.decode("ascii")
        if any(type(x) in (dict, list, tuple) for x in content):
            return "[" + ",".join([dumps(x) for x in content]) + "]"
    return _ENCODE(content)


class _PromptIds:
    """The prompt-id adapter, a list of plain non-negative ints through msgspec."""

    def __init__(self, adapter) -> None:
        self._adapter = adapter

    def validate_python(self, value, *args, **kwargs):
        if not args and not kwargs and type(value) is list:
            try:
                return msgspec.convert(value, _PROMPT_IDS, strict=True)
            except msgspec.ValidationError:
                pass
        return self._adapter.validate_python(value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._adapter, name)


def install() -> None:
    """Install the three replacements in this process, once."""
    global _installed
    if _installed:
        return
    _installed = True
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    async def request_json(self):
        if not hasattr(self, "_json"):
            body = await self.body()
            self._json = loads(body)
        return self._json

    Request.json = request_json
    render = JSONResponse.render

    def fast_render(self, content) -> bytes:
        try:
            return dumps(content).encode("utf-8")
        except (RecursionError, TypeError, ValueError):
            return render(self, content)

    JSONResponse.render = fast_render
    try:
        from atom.entrypoints.openai import protocol
    except ImportError:
        return
    adapter = getattr(protocol, "_PROMPT_TOKEN_IDS_ADAPTER", None)
    if adapter is not None and not isinstance(adapter, _PromptIds):
        protocol._PROMPT_TOKEN_IDS_ADAPTER = _PromptIds(adapter)

# SPDX-License-Identifier: MIT
"""JSON and JSON-lines files read whole, a malformed one refused naming its path."""

import json
from pathlib import Path


def read_json(path):
    """The JSON document in the file at `path`."""
    return _parsed(_text(path), path)


def read_jsonl(path) -> list:
    """The JSON document on each line of the file at `path`."""
    # Bytes end a line at \n or \r only, which JSON escapes inside strings;
    # str.splitlines would also cut a string holding a raw U+2028.
    lines = _text(path).encode().splitlines()
    return [_parsed(text.decode(), path, n) for n, text in enumerate(lines)]


def _text(path) -> str:
    """The file at `path` decoded, refused naming it when its bytes do not decode."""
    try:
        return Path(path).read_text()
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not JSON: {error}") from error


def _parsed(text: str, path, before: int = 0):
    """`text`, the file at `path` after its first `before` lines, parsed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        at = f"line {before + error.lineno} column {error.colno}"
        raise ValueError(f"{path} is not JSON: {error.msg}: {at}") from error

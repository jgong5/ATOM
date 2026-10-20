# SPDX-License-Identifier: MIT
"""JSON and JSON-lines files read whole, a malformed one refused naming its path."""

import json
from pathlib import Path


def read_json(path):
    """The JSON document in the file at `path`."""
    return _parsed(Path(path).read_text(), path)


def read_jsonl(path) -> list:
    """The JSON document on each line of the file at `path`."""
    lines = Path(path).read_text().splitlines()
    return [_parsed(text, path, n) for n, text in enumerate(lines)]


def _parsed(text: str, path, before: int = 0):
    """`text`, the file at `path` after its first `before` lines, parsed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        at = f"line {before + error.lineno} column {error.colno}"
        raise ValueError(f"{path} is not JSON: {error.msg}: {at}") from error

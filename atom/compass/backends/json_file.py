# SPDX-License-Identifier: MIT
"""JSON and JSON-lines files read whole, a malformed one refused naming its path.

A file whose name ends in `XZ` holds its text xz-compressed, as `write_json`
writes it; the checked-in law files are stored so."""

import json
import lzma
from pathlib import Path

#: The name suffix of a file whose text is xz-compressed.
XZ = ".xz"


def read_json(path):
    """The JSON document in the file at `path`."""
    return _parsed(_text(path), path)


def read_jsonl(path) -> list:
    """The JSON document on each line of the file at `path`."""
    # Bytes end a line at \n or \r only, which JSON escapes inside strings;
    # str.splitlines would also cut a string holding a raw U+2028.
    lines = _text(path).encode().splitlines()
    return [_parsed(text.decode(), path, n) for n, text in enumerate(lines)]


def write_json(document, path) -> None:
    """Write `document` to `path` as one line of JSON, xz-compressed when the
    name ends in `XZ`: at preset 9, whose 64 MiB dictionary spans a whole law,
    and with no timestamp, so the same document always gives the same bytes."""
    text = (json.dumps(document) + "\n").encode()
    Path(path).write_bytes(lzma.compress(text, preset=9) if _xz(path) else text)


def _xz(path) -> bool:
    return str(path).endswith(XZ)


def _text(path) -> str:
    """The file at `path` decoded, refused naming it when its bytes do not decode."""
    try:
        if _xz(path):
            return lzma.decompress(Path(path).read_bytes()).decode()
        return Path(path).read_text()
    except (UnicodeDecodeError, lzma.LZMAError) as error:
        raise ValueError(f"{path} is not JSON: {error}") from error


def _parsed(text: str, path, before: int = 0):
    """`text`, the file at `path` after its first `before` lines, parsed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        at = f"line {before + error.lineno} column {error.colno}"
        raise ValueError(f"{path} is not JSON: {error.msg}: {at}") from error

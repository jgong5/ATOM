# SPDX-License-Identifier: MIT
"""A simulated deployment's token ids by text, shared by its frontends.

ATOM's segment tokenizer (`atom.model_engine.segment_tokenizer`) keeps each
API server's own cache of a prompt's segments. In a multi-instance deployment
a conversation's next turn often reaches another prefill instance, and at high
concurrency a server's own cache evicts histories it encoded, so the same text
is encoded again (jgong5/ATOM#3993 measured it). In a simulated server that
encode runs while its clock stands still, and every other LP waits on it.

`install` wraps a Rust backend's ``encode_batch`` so that each text of a
segment tokenizer call, ``encode_batch(texts, add_special_tokens=False)``,
that some frontend of the deployment has encoded before is read back instead
of encoded. Its ids are the encoding's own: the file is the ids of exactly
that text, under a directory named by the ``tokenizers`` library's version and
the backend's serialized state, so neither another vocabulary nor another
library build reads them. A call
in any other shape, or with the backend's truncation or padding set, goes to
the backend untouched, as does a text shorter than `MIN_CHARS`. The wrapper
returns one object per text with the ids as ``ids``, which is all the segment
tokenizer and `atom.utils.compass_loop.wrap_encode_batch` read, so what a run
is charged is unchanged: it goes beneath that charge.

Where: ``<out_dir>/token_ids`` of the run file, which every instance of a
deployment names and every node sees at one path, or the directory `ENV`
names, so that runs over the same traces can share one; `ENV` set to `OFF`
installs nothing. A file is written to a temporary name and renamed into
place, so a reader finds it whole or not at all, and two servers writing one
text write the same bytes.
"""

import hashlib
import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("atom")

#: A directory to keep the token ids in, in place of the run's; `OFF` for none.
ENV = "ATOM_COMPASS_TOKEN_IDS"
OFF = "off"
#: Under the run file's ``out_dir``, where the ids go by default.
SUBDIR = "token_ids"
#: Texts shorter than this are encoded: a file round trip costs about what
#: encoding a thousand characters does.
MIN_CHARS = 1024
#: The ids' file format: little-endian int32, as the segment tokenizer keeps them.
_DTYPE = np.dtype("<i4")


class _Ids:
    """The ``ids`` of one text, the one attribute of an `Encoding` read here."""

    __slots__ = ("ids",)

    def __init__(self, ids: np.ndarray) -> None:
        self.ids = ids


def directory(run: dict) -> Path | None:
    """Where the run keeps its token ids, or None for nowhere."""
    named = os.environ.get(ENV)
    if named == OFF:
        return None
    if named:
        return Path(named)
    out = run.get("out_dir")
    return None if not out else Path(out) / SUBDIR


def install(backend, root: Path) -> None:
    """Wrap `backend`'s ``encode_batch`` with the ids kept under `root`, in a
    directory of their own for the ``tokenizers`` version and the backend's
    serialized state."""
    import tokenizers

    keyed = f"{tokenizers.__version__}\0{backend.to_str()}"
    state = hashlib.sha256(keyed.encode()).hexdigest()[:16]
    backend.encode_batch = shared(backend.encode_batch, backend, root / state)


def shared(encode_batch, backend, where: Path):
    """`encode_batch`, reading back each segment text's ids from `where` when
    a frontend has encoded it before, and keeping those it encodes there."""
    failed = []  # a write that failed is logged once a process

    def call(*args, **kwargs):
        texts = args[0] if len(args) == 1 else None
        if (
            kwargs != {"add_special_tokens": False}
            or not isinstance(texts, list)
            or backend.truncation is not None
            or backend.padding is not None
            or not all(isinstance(t, str) for t in texts)
        ):
            return encode_batch(*args, **kwargs)
        out, missing = [None] * len(texts), []
        for i, text in enumerate(texts):
            ids = _read(where, text) if len(text) >= MIN_CHARS else None
            if ids is None:
                missing.append(i)
            else:
                out[i] = _Ids(ids)
        if missing:
            encoded = encode_batch([texts[i] for i in missing], **kwargs)
            for i, encoding in zip(missing, encoded):
                out[i] = encoding
                if len(texts[i]) >= MIN_CHARS:
                    _write(where, texts[i], encoding.ids, failed)
        return out

    return call


def _path(where: Path, text: str) -> Path:
    key = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
    return where / key[:2] / key


def _read(where: Path, text: str) -> np.ndarray | None:
    try:
        return np.fromfile(_path(where, text), dtype=_DTYPE)
    except FileNotFoundError:
        return None


def _write(where: Path, text: str, ids, failed: list) -> None:
    path = _path(where, text)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(ids, dtype=_DTYPE).tofile(temporary)
        os.replace(temporary, path)
    except OSError as error:
        if not failed:
            failed.append(error)
            logger.warning(
                "token ids: cannot keep %s (%s); encoding as before", path, error
            )
        temporary.unlink(missing_ok=True)

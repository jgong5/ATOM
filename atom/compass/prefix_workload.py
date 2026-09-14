"""Deterministic prompt-prefix identity for the pinned cc-traces corpus.

Opaque local IDs name prefix nodes, not recovered original token blocks.
The distinguishing code fits inside the first native 16-token block.
"""

from array import array
import hashlib
import json
from pathlib import Path
import re
import sys


SCHEMA = "compass.cc_prefix_encoding/1"
CODEC = "cc_traces_local_prefix_v1"
CORPUS_SHA256 = "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef"
WORDS = "the of and to in a is that it for on with as was at by an be this from".split()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def token_digest(tokens):
    values = array("i", tokens)
    if values.itemsize != 4:
        raise ValueError("prefix token digests require 32-bit integers")
    if sys.byteorder != "little":
        values.byteswap()
    return _sha(values.tobytes())


def tokenizer_identity(tokenizer, model):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("prefix encoding requires a pinned fast tokenizer")
    return {
        "model": model,
        "revision": tokenizer.init_kwargs.get("_commit_hash"),
        "backend_sha256": _sha(_canonical(json.loads(tokenizer.backend_tokenizer.to_str()))),
        "special_ids_sha256": _sha(_canonical(sorted(tokenizer.all_special_ids))),
        "vocab_size": len(tokenizer),
    }


def make_encoding(tokenizer, model):
    """Build an artifact from verified ordinary single-token vocabulary entries."""
    ids = []
    for word in WORDS:
        encoded = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] in tokenizer.all_special_ids:
            raise ValueError(f"{word!r} is not one ordinary tokenizer token")
        ids.append(encoded[0])
    spec = {
        "schema": SCHEMA, "codec": CODEC,
        "codec_sha256": _sha(Path(__file__).read_bytes()),
        "corpus_sha256": CORPUS_SHA256,
        "source_block_tokens": 64, "native_block_tokens": 16,
        "root_ordinal_bits": 12, "local_id_bits": 20, "warmup_index_bits": 16,
        "sharing": "declared_prompt_prefix_only",
        "tokenizer": tokenizer_identity(tokenizer, model),
        "ordinary_words": WORDS, "ordinary_token_ids": ids,
    }
    PrefixEncoding(spec).verify_tokenizer(tokenizer, model)
    return spec


class PrefixEncoding:
    def __init__(self, spec, *, path=None, sha256=None):
        if not isinstance(spec, dict):
            raise ValueError("prefix encoding must be an object")
        expected = {
            "schema": SCHEMA, "codec": CODEC, "corpus_sha256": CORPUS_SHA256,
            "source_block_tokens": 64, "native_block_tokens": 16,
            "root_ordinal_bits": 12, "local_id_bits": 20, "warmup_index_bits": 16,
            "sharing": "declared_prompt_prefix_only", "ordinary_words": WORDS,
            "codec_sha256": _sha(Path(__file__).read_bytes()),
        }
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("unsupported or changed corpus prefix codec")
        ids = spec.get("ordinary_token_ids")
        if (not isinstance(ids, list) or len(ids) != 20
                or any(type(i) is not int or not 0 <= i < 2**31 for i in ids)
                or len(set(ids)) != 20):
            raise ValueError("prefix codec needs 20 distinct ordinary token IDs")
        tokenizer = spec.get("tokenizer") or {}
        if (not tokenizer.get("model") or type(tokenizer.get("vocab_size")) is not int
                or max(ids) >= tokenizer["vocab_size"]
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(tokenizer.get(key, "")))
                       for key in ("backend_sha256", "special_ids_sha256"))):
            raise ValueError("prefix codec has no valid tokenizer identity")
        self.spec, self.path, self.sha256 = spec, path, sha256
        self.ids = ids

    @classmethod
    def load(cls, path, sha256):
        data = Path(path).read_bytes()
        if not isinstance(sha256, str) or _sha(data) != sha256:
            raise ValueError("prompt-encoding artifact differs from its explicit SHA-256")
        return cls(json.loads(data), path=str(Path(path).resolve()), sha256=sha256)

    def verify_tokenizer(self, tokenizer, model):
        if tokenizer_identity(tokenizer, model) != self.spec["tokenizer"]:
            raise ValueError("runtime tokenizer differs from pinned prefix encoding")
        for word, token in zip(WORDS, self.ids):
            if (tokenizer.encode(" " + word, add_special_tokens=False) != [token]
                    or token in tokenizer.all_special_ids):
                raise ValueError("prefix encoding contains an unverified/special token")

    def tokens(self, row, *, phase="measured", warmup_index=None):
        ordinal = row.get("corpus_line_1based")
        hashes = row.get("hash_ids")
        if (row.get("hash_id_scope") != "local"
                or not isinstance(row.get("session"), str) or not row["session"].strip()
                or type(ordinal) is not int or not 1 <= ordinal < 2**12
                or not isinstance(hashes, list) or not hashes
                or any(type(h) is not int or not 0 <= h < 2**20 for h in hashes)
                or type(row.get("input_tokens")) is not int
                or row["input_tokens"] != 64 * len(hashes)):
            raise ValueError("invalid scoped corpus hash blocks or prompt length")
        if phase == "measured" and warmup_index is None:
            marker, nonce = self.ids[17], 0
        elif phase == "warmup" and type(warmup_index) is int and 0 <= warmup_index < 65535:
            marker, nonce = self.ids[18], warmup_index + 1
        else:
            raise ValueError("invalid prefix phase or warmup namespace index")
        result = []
        nonce_tokens = [self.ids[(nonce >> shift) & 15] for shift in (12, 8, 4, 0)]
        for block_id in hashes:
            value = (ordinal << 20) | block_id
            code = [self.ids[(value >> shift) & 15] for shift in range(28, -1, -4)]
            # 2 markers + 8 injective ID digits + 4 warmup digits < 16.
            result.extend([self.ids[16], marker] + code + nonce_tokens + [self.ids[19]] * 50)
        return result

    def validate_rows(self, rows):
        """Check scoped parent consistency and the pinned measurement token bytes."""
        roots, ordinals, parents, digests = {}, {}, {}, []
        for row in rows:
            tokens = self.tokens(row)
            root, ordinal = row["session"], row["corpus_line_1based"]
            if (root in roots and roots[root] != ordinal
                    or ordinal in ordinals and ordinals[ordinal] != root):
                raise ValueError("root identities and corpus ordinals must form a bijection")
            roots[root], ordinals[ordinal] = ordinal, root
            previous = None
            for position, block_id in enumerate(row["hash_ids"]):
                key, value = (root, block_id), (position, previous)
                if key in parents and parents[key] != value:
                    raise ValueError("a scoped hash ID has conflicting position/parent")
                parents[key] = value
                previous = block_id
            digest = token_digest(tokens)
            if row.get("prompt_token_sha256") != digest:
                raise ValueError("row prompt-token digest differs from its scoped hash blocks")
            digests.append(digest)
        return digests

    def rows_digest(self, rows):
        self.validate_rows(rows)
        fields = ("arrival_s", "input_tokens", "output_tokens", "session",
                  "corpus_line_1based", "hash_id_scope", "hash_ids", "prompt_token_sha256",
                  "client_index", "json_path", "actor", "agent_id", "subagent_type")
        normalized = [{key: row.get(key) for key in fields} for row in rows]
        for row in normalized:
            row["arrival_s"] = float(row["arrival_s"])
        return _sha(_canonical(normalized))

    def evidence(self, digests, phase):
        return {"path": self.path, "sha256": self.sha256, "codec": CODEC,
                "phase": phase, "row_token_sha256": digests}

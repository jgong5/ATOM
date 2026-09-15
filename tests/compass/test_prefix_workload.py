"""Prefix identities are exact at native block boundaries, not probabilistic seeds."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atom.compass.prefix_workload import PrefixEncoding, WORDS, make_encoding, token_digest


MODEL = "Qwen/Qwen3.8-27B"


class Tokenizer:
    is_fast = True
    init_kwargs = {}
    all_special_ids = [999]

    def __init__(self):
        self.vocab = {" " + word: 100 + i for i, word in enumerate(WORDS)}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(self.vocab))

    def __len__(self):
        return 1000

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]]


def encoding(tmp_path):
    tokenizer = Tokenizer()
    path = tmp_path / "encoding.json"
    path.write_text(json.dumps(make_encoding(tokenizer, MODEL)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PrefixEncoding.load(path, digest), tokenizer


def row(codec, hashes, *, ordinal=1, root="root-a", arrival=0.0, output=16):
    result = {"arrival_s": arrival, "input_tokens": len(hashes) * 64,
              "output_tokens": output, "hash_ids": hashes, "hash_id_scope": "local",
              "corpus_line_1based": ordinal, "session": root,
              "client_index": 0, "json_path": "/requests/0", "actor": "root",
              "agent_id": None, "subagent_type": None}
    result["prompt_token_sha256"] = token_digest(codec.tokens(result))
    return result


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        "prefix_test_" + name, Path(__file__).resolve().parents[2] / "scripts/compass" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = load_script("replay")


def test_shared_and_different_prefixes_match_only_declared_native_blocks(tmp_path):
    codec, _ = encoding(tmp_path)
    a, b = row(codec, [10, 11, 12]), row(codec, [10, 11, 13])
    codec.validate_rows([a, b])
    left, right = codec.tokens(a), codec.tokens(b)
    assert len(left) == len(right) == 192
    assert left[:128] == right[:128]
    assert left[128:144] != right[128:144]
    other = codec.tokens(row(codec, [10, 11, 12], ordinal=2, root="root-b"))
    assert left[:16] != other[:16]  # local IDs cannot create cross-root hits
    renamed = dict(a, agent_id="subagent-other", source_model="another-label", client_index=7)
    assert codec.tokens(renamed) == left  # these labels do not split a root's identity


def test_injective_boundary_values_and_private_warmup_namespaces(tmp_path):
    codec, _ = encoding(tmp_path)
    prefixes = set()
    for ordinal in [1, 393, 4095]:
        for block_id in [0, 15, 16, 238450, 2**20 - 1]:
            tokens = codec.tokens(row(codec, [block_id], ordinal=ordinal))
            assert tuple(tokens[:16]) not in prefixes
            prefixes.add(tuple(tokens[:16]))
    source = row(codec, [10, 11])
    measured = codec.tokens(source)
    a = codec.tokens(source, phase="warmup", warmup_index=0)
    b = codec.tokens(source, phase="warmup", warmup_index=1)
    assert len(a) == len(b) == len(measured) == 128
    assert len({tuple(x[:16]) for x in [measured, a, b]}) == 3


@pytest.mark.parametrize("field,value", [("corpus_line_1based", 4096),
                                         ("hash_ids", [2**20]), ("hash_ids", [True]),
                                         ("input_tokens", 63), ("hash_id_scope", "global")])
def test_no_overflow_modulo_or_undeclared_identity(tmp_path, field, value):
    codec, _ = encoding(tmp_path)
    source = row(codec, [10])
    source[field] = value
    with pytest.raises(ValueError):
        codec.tokens(source)


def test_source_graph_and_token_digest_cannot_silently_change(tmp_path):
    codec, _ = encoding(tmp_path)
    a = row(codec, [10, 11])
    b = row(codec, [12, 11])
    with pytest.raises(ValueError, match="position/parent"):
        codec.validate_rows([a, b])
    b = row(codec, [10, 11], ordinal=2)
    with pytest.raises(ValueError, match="bijection"):
        codec.validate_rows([a, b])
    a["prompt_token_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        codec.validate_rows([a])


def test_tokenizer_artifact_and_ordinary_token_validation(tmp_path):
    codec, tokenizer = encoding(tmp_path)
    codec.verify_tokenizer(tokenizer, MODEL)
    tokenizer.vocab[" the"] = 998
    with pytest.raises(ValueError, match="tokenizer"):
        codec.verify_tokenizer(tokenizer, MODEL)
    tokenizer = Tokenizer()
    tokenizer.all_special_ids = [tokenizer.vocab[" the"]]
    with pytest.raises(ValueError, match="ordinary"):
        make_encoding(tokenizer, MODEL)
    with pytest.raises(ValueError, match="SHA-256"):
        PrefixEncoding.load(codec.path, "0" * 64)


def test_direct_payloads_keep_source_prefix_and_output_lengths(tmp_path):
    codec, tokenizer = encoding(tmp_path)
    rows = [row(codec, [10, 11], output=0), row(codec, [10, 12], arrival=2.0, output=32)]
    payloads, info = replay._encode_requests(
        rows, MODEL, tokenizer, declared=True, byte_budget=100000, prefix_encoding=codec)
    bodies = [json.loads(x) for x in payloads]
    assert bodies[0]["prompt"][:64] == bodies[1]["prompt"][:64]
    assert bodies[0]["prompt"][64:80] != bodies[1]["prompt"][64:80]
    assert [x["max_tokens"] for x in bodies] == [0, 32]
    assert [x["compass_arrival"] for x in bodies] == [0.0, 2.0]
    assert info["corpus_encoding"]["row_token_sha256"] == [x["prompt_token_sha256"] for x in rows]
    warm, warm_info = replay._encode_requests(
        rows, MODEL, tokenizer, declared=False, byte_budget=100000,
        prefix_encoding=codec, phase="warmup")
    assert warm_info["corpus_encoding"]["phase"] == "warmup"
    assert json.loads(warm[0])["prompt"][:16] != bodies[0]["prompt"][:16]


def test_workload_preserves_prefix_metadata_and_requires_explicit_mode(tmp_path):
    codec, _ = encoding(tmp_path)
    rows = [row(codec, [10, 11]), row(codec, [10, 12], arrival=2.0)]
    trace = tmp_path / "rows.jsonl"
    trace.write_text("".join(json.dumps(x) + "\n" for x in rows))
    args = SimpleNamespace(trace=str(trace), num_requests=0, time_scale=1,
                           input_tokens=1, output_tokens=1, prompt_encoding=codec.path)
    assert replay._workload(args) == rows
    args.prompt_encoding = None
    with pytest.raises(ValueError, match="explicit"):
        replay._workload(args)

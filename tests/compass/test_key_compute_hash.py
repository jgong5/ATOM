# SPDX-License-Identifier: MIT
"""`keyed` against the inputs ATOM's `Config.compute_hash` reads: ATOM keeps the
hash naming each config field that changes its computation graph. Each input of
a resolved server config, and each ATOM environment variable, changed alone:
every one that moves `compute_hash` moves `keyed`, is stated by the served
config.json, under its digest, or is left out below with the reason. An input
ATOM adds to the hash fails here until it is placed.

No device: aiter's arch probe answers a declared arch.
"""

import copy
import dataclasses
import json

import pytest
from test_key_index_cache_dtype import V41, V41_SERVER, served

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.run.step_grid import keyed
from atom.quant_spec import LayerQuantConfig
from atom.utils import envs

BOUND = (
    "a bound, checked as one: a grid refuses a value above the one it was planned "
    "for, and `keyed` names it only where it cuts the ladder"
)
RANK = "which rank this process is: every rank of the group runs the same kernels"
RENDEZVOUS = "where the DP group meets: picks no kernel"
#: Each input that moves `compute_hash` and neither `keyed` nor the config.json
#: digest covers, and why.
LEFT_OUT = {
    "max_model_len": BOUND,
    "max_num_seqs": BOUND,
    "prefill_context_parallel_size": (
        "a run priced by a step law or grid refuses a width other than 1, the one "
        "both are measured at"
    ),
    "compilation_config.level": "picks torch.compile's kernels: #3052",
    "compilation_config.use_cudagraph": (
        "picks no kernel: ATOM reads it only to refuse a compiled buffer write"
    ),
    "compilation_config.local_cache_dir": "where compiled graphs are cached",
    "compilation_config.cuda_graph_sizes": (
        "the ladder `keyed` names follows it only where no capture sizes are "
        "declared, and this server declares them"
    ),
    "parallel_config.data_parallel_size_local": (
        "picks MoRI's inter-node all-to-all: #3053"
    ),
    "parallel_config.data_parallel_rank": RANK,
    "parallel_config.data_parallel_rank_local": RANK,
    "parallel_config.data_parallel_master_ip": RENDEZVOUS,
    "parallel_config.data_parallel_master_port": RENDEZVOUS,
    "dcp_config.indexer_dcp_only": (
        "picks MiniMax-M3's context-parallel indexer: #3054"
    ),
    "hf_config.use_index_cache": (
        "set by `--hf-overrides`, which neither `keyed` nor the digest reads: #3055"
    ),
    "ATOM_REPLICATE_VOCAB_EMBED": "picks GLM-5.2's embedding path: #3056",
    "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4": (
        "fuses MiniMax-M3.1's shared expert into its MoE: #3056"
    ),
}
#: A value no config holds.
OTHER = object()
#: Another value for each layer list of the quantization config, which `keyed`
#: reads through ATOM's remap of its layer names: a pattern is a (name, spec)
#: pair and an exclude a name, and `OTHER` is neither.
NAMED = {
    "quant_config.layer_pattern_specs": [("model.layers.0.mlp", LayerQuantConfig())],
    "quant_config.exclude_layers": ["model.layers.0.mlp"],
}


def other(value):
    """A value of `value`'s kind that differs from it."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, list):
        return value[:-1] or [OTHER]
    return OTHER


def holding(owner, name, value):
    """A shallow copy of `owner` that holds `value` as `name`, past any setter."""
    owner = copy.copy(owner)
    vars(owner)[name] = value
    return owner


def names(owner):
    """The attributes `owner` holds, its dataclass fields among them."""
    fields = dataclasses.fields(owner) if dataclasses.is_dataclass(owner) else ()
    return {*vars(owner), *(f.name for f in fields)}


def changes(config, monkeypatch):
    """Each input of `config` changed alone: its name and the config as changed.
    An ATOM config the config holds, and the model's, change field by field; an
    `--hf-overrides` field changes where the server states it too."""
    for name in names(config):
        value = getattr(config, name)
        if type(value).__module__ != "atom.config" and name != "hf_config":
            yield name, holding(config, name, other(value))
            continue
        for field in names(value):
            held = NAMED.get(f"{name}.{field}", other(getattr(value, field)))
            owner = holding(value, field, held)
            changed = holding(config, name, owner)
            if name == "hf_config" and field in config.hf_overrides:
                stated = config.hf_overrides | {field: getattr(owner, field)}
                changed = holding(changed, "hf_overrides", stated)
            yield f"{name}.{field}", changed
    for name in envs.environment_variables:
        for flag in ("0", "1"):
            monkeypatch.setenv(name, flag)
            yield name, config
        monkeypatch.delenv(name)


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A V4.1-Flash MI455 server's resolved config, sharing its index cache:
    `index_topk_freq` stated in its config.json, `use_index_cache` set by
    `--hf-overrides`; no ATOM or performance variable set."""
    for name in (*PERFORMANCE_KEYS, *envs.environment_variables):
        monkeypatch.delenv(name, raising=False)
    stated = json.loads(V41.read_text())
    stated["text_config"]["index_topk_freq"] = 4
    model = tmp_path / "v41.json"
    model.write_text(json.dumps(stated))
    sharing = ["--hf-overrides", json.dumps({"use_index_cache": True})]
    return served(
        tmp_path, monkeypatch, "gfx1250", model, *V41_SERVER.split(), *sharing
    )


def test_each_input_that_moves_the_hash_is_keyed_digested_or_left_out(
    config, monkeypatch
):
    hashed, engine = config.compute_hash(), keyed(config)

    left_out = set()
    for name, changed in changes(config, monkeypatch):
        field = name.removeprefix("hf_config.")
        if field != name and field not in config.hf_overrides:
            continue  # the model's own config, keyed by its config.json's digest
        if changed.compute_hash() != hashed and keyed(changed) == engine:
            left_out.add(name)

    assert left_out == set(LEFT_OUT)

# SPDX-License-Identifier: MIT
"""`keyed` against the inputs ATOM's `Config.compute_hash` reads: ATOM keeps the
hash naming each config field that changes its computation graph. Each input of
a resolved server config, and each ATOM environment variable, changed alone:
every one that moves `compute_hash` moves `keyed`, is stated by the served
config.json, under its digest, or is left out below with the reason. An input
ATOM adds to the hash fails here until it is placed.

A run whose data-parallel group spans nodes, or that runs MiniMax-M3's
indexer-only context parallelism, `--hf-overrides`, or an ATOM switch `keyed`
names off its default, refuses a step linear law fitted without it, naming it.

No device: aiter's arch probe answers a declared arch.
"""

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest
from test_drive_runfile import build, install, step_linear
from test_key_index_cache_dtype import V41, V41_SERVER, served
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.run.step_grid import keyed
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
        "a run priced by a step law or grid refuses a data-parallel group across "
        "nodes, whose memory readings are taken on one node"
    ),
    "parallel_config.data_parallel_rank": RANK,
    "parallel_config.data_parallel_rank_local": RANK,
    "parallel_config.data_parallel_master_ip": RENDEZVOUS,
    "parallel_config.data_parallel_master_port": RENDEZVOUS,
}
#: ATOM's switches `keyed` names only off their default: each name, its default
#: and a value off it.
SWITCHES = [
    ("ATOM_REPLICATE_VOCAB_EMBED", "1", "0"),
    ("ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4", "0", "1"),
]
#: A value no config holds.
OTHER = object()


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
            owner = holding(value, field, other(getattr(value, field)))
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


def fitted_on(tmp_path, r) -> dict:
    """The run file of a step linear law fitted on the runner `r`."""
    return build(tmp_path, "--step-linear", step_linear(tmp_path, r).key.token())


def refusal(tmp_path, monkeypatch, run, r) -> str:
    """Why the runner `r` refuses the run file `run`'s step linear law."""
    with pytest.raises(ValueError, match="the step linear law") as refused:
        install(tmp_path, monkeypatch, run, r)
    print(f"\n{refused.value}")
    return str(refused.value)


def test_a_dp_group_across_nodes_refuses_a_law_fitted_on_one_naming_it(
    tmp_path, monkeypatch
):
    one_node = SimpleNamespace(
        data_parallel_size=2, data_parallel_rank=0, pipeline_parallel_rank=0
    )
    across = SimpleNamespace(**vars(one_node), data_parallel_size_local=1)
    run = fitted_on(tmp_path, deployment(tmp_path, parallel_config=one_node))

    r = deployment(tmp_path, parallel_config=across)
    named = refusal(tmp_path, monkeypatch, run, r)
    assert "data_parallel_size_local 1 of data_parallel_size 2" in named


@pytest.mark.parametrize(
    "here, named",
    [
        (
            {"dcp_config": SimpleNamespace(indexer_dcp_only=True)},
            "indexer_dcp_only None in the law and True here",
        ),
        (
            {"hf_overrides": {"use_index_cache": True}},
            "hf_overrides None in the law and {'use_index_cache': True} here",
        ),
    ],
    ids=["indexer-cp", "hf-overrides"],
)
def test_a_run_under_a_model_option_refuses_a_law_fitted_without_it_naming_it(
    tmp_path, monkeypatch, here, named
):
    run = fitted_on(tmp_path, deployment(tmp_path))

    assert named in refusal(tmp_path, monkeypatch, run, deployment(tmp_path, **here))


def test_no_override_and_the_indexer_mode_off_key_as_before(tmp_path):
    plain = keyed(deployment(tmp_path).config)
    off = {"dcp_config": SimpleNamespace(indexer_dcp_only=False), "hf_overrides": {}}

    assert keyed(deployment(tmp_path, **off).config) == plain


@pytest.mark.parametrize(
    "name, default, off", SWITCHES, ids=[name for name, *_ in SWITCHES]
)
def test_an_atom_switch_off_its_default_refuses_a_law_fitted_at_it_naming_it(
    tmp_path, monkeypatch, name, default, off
):
    for switch, *_ in SWITCHES:
        monkeypatch.delenv(switch, raising=False)
    r = deployment(tmp_path)
    run, unset = fitted_on(tmp_path, r), keyed(r.config)
    monkeypatch.setenv(name, default)
    assert keyed(r.config) == unset

    monkeypatch.setenv(name, off)
    named = refusal(tmp_path, monkeypatch, run, r)
    assert f"env {name} None in the law and {off!r} here" in named

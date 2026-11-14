# SPDX-License-Identifier: MIT
"""Two drafts of one target key apart. `keyed` states the digest of a draft's
config.json where it is not the target's, so a step linear law, host laws,
memory readings and a step grid bundle taken with one draft hold for a run
with that draft and refuse a run with the other at `compass run --check`,
naming the draft; a drafter read from the target's checkpoint keys as before;
and `compass fit steps --ops` refuses points taken with another draft.

The drafts are MiniMax-M3's two EAGLE3 heads, whose config.json files differ
only in their KV heads, 64 and 4 (GQA). The `--check` runs are
`test_run_lacking`'s, on `test_run_step_grid`'s deployment.
"""

import json
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_drive_runfile import host_laws, step_linear
from test_fit_steps import planted_roots  # noqa: F401  autouse
from test_key_index_cache_dtype import QWEN
from test_key_index_cache_dtype import served as built
from test_run_lacking import checked, lacked, served
from test_run_step_grid import deployment, publish
from test_step_labels_floor import _fit_traced, _ops

from atom.compass.bench import ops
from atom.compass.run.step_grid import keyed
from atom.compass.validate import bound

#: The engine-config field naming a draft by its config.json, as refusals state it.
DRAFT = "draft_model"
#: MiniMax-M3-EAGLE3's config.json; the GQA head's differs in its KV heads.
EAGLE3 = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "attention_bias": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "intermediate_size": 18432,
    "max_position_embeddings": 1048576,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 1,
    "num_key_value_heads": 64,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 200064,
    "draft_vocab_size": 200064,
}
HEADS = {"eagle3": EAGLE3, "eagle3-gqa": EAGLE3 | {"num_key_value_heads": 4}}


@pytest.fixture
def drafts(tmp_path):
    """Each draft's checkpoint directory by name, holding its config.json."""
    made = {}
    for name, config in HEADS.items():
        made[name] = tmp_path / "drafts" / name
        made[name].mkdir(parents=True)
        (made[name] / "config.json").write_text(json.dumps(config))
    return made


def drafter(path):
    """The fields `keyed` reads of ATOM's speculative config for an EAGLE3
    drafter of three tokens whose checkpoint is at `path`."""
    return SimpleNamespace(method="eagle3", num_speculative_tokens=3, model=str(path))


def drafting(config, path):
    """`config` with its drafter's checkpoint at `path` instead."""
    return SimpleNamespace(**{**vars(config), "speculative_config": drafter(path)})


def eagle3(tmp_path, monkeypatch, draft):
    """ATOM's config of a Qwen3 server drafting with EAGLE3 from `draft`, as
    `compass run` builds it."""
    args = ["--method", "eagle3", "--num-speculative-tokens", "3"]
    return built(tmp_path, monkeypatch, "gfx950", QWEN, *args, "--draft-model", draft)


def test_two_drafts_of_one_target_key_apart_by_their_config_json(
    tmp_path, monkeypatch, drafts
):
    mha, gqa = (keyed(eagle3(tmp_path, monkeypatch, str(d))) for d in drafts.values())

    assert {k for k in mha | gqa if mha.get(k) != gqa.get(k)} == {DRAFT}
    assert mha["speculative"] == gqa["speculative"] == ["eagle3", 3]
    assert mha[DRAFT] != gqa[DRAFT] and mha[DRAFT].startswith("sha256:")


def test_a_drafter_read_from_the_target_s_checkpoint_keys_as_before(
    tmp_path, monkeypatch
):
    config = eagle3(tmp_path, monkeypatch, str(tmp_path / QWEN.stem))

    assert config.speculative_config.model == config.model
    assert keyed(config)["speculative"] == ["eagle3", 3]
    assert DRAFT not in keyed(config)


def _entries(tmp_path, taken):
    """The step linear law and host laws taken with `taken`'s draft, beside the
    memory readings `deployment` published, as `--check` flags."""
    law, host = step_linear(tmp_path, taken), host_laws(tmp_path, taken)
    return ["--step-linear", law.key.token(), "--host-laws", host.key.token()]


def test_entries_taken_with_one_draft_hold_with_it_and_refuse_the_other(
    tmp_path, monkeypatch, capsys, drafts
):
    taken = deployment(tmp_path, speculative_config=drafter(drafts["eagle3"]))
    flags = _entries(tmp_path, taken)
    served(monkeypatch, taken.config)
    assert checked(tmp_path, *flags) == 0
    capsys.readouterr()
    here = drafting(taken.config, drafts["eagle3-gqa"])
    served(monkeypatch, here)

    assert checked(tmp_path, *flags) == 1

    out = capsys.readouterr().out
    assert lacked(out) == [
        "step linear law: stale",
        "host laws: stale",
        "memory readings: missing",
    ]
    theirs, ours = keyed(taken.config)[DRAFT], keyed(here)[DRAFT]
    assert f"{DRAFT} {theirs!r} in the law and {ours!r} here" in out
    assert f"fitted under another ['engine.{DRAFT}']" in out
    nearest = out.split("the nearest entry differs in engine_config")[1]
    assert f'"{DRAFT}": "{theirs}"' in nearest.split(" there)")[0]


def test_a_step_grid_bundle_taken_with_one_draft_refuses_the_other(
    tmp_path, monkeypatch, capsys, drafts
):
    monkeypatch.setattr(bound, "walked", lambda atom_args, arch: [])
    monkeypatch.setattr(ops, "_child", lambda fn, *args: fn(*args))
    taken = deployment(tmp_path, speculative_config=drafter(drafts["eagle3"]))
    token = publish(tmp_path, taken).key.token()
    served(monkeypatch, taken.config)
    assert checked(tmp_path, "--step-grid", token) == 0
    capsys.readouterr()
    here = drafting(taken.config, drafts["eagle3-gqa"])
    served(monkeypatch, here)

    assert checked(tmp_path, "--step-grid", token) == 1

    out = capsys.readouterr().out
    assert lacked(out) == ["step grid bundle: stale", "memory readings: missing"]
    theirs, ours = keyed(taken.config)[DRAFT], keyed(here)[DRAFT]
    assert f"{DRAFT} {theirs!r} in the bundle and {ours!r} here" in out


def test_fit_steps_ops_refuses_points_taken_with_another_draft(tmp_path, monkeypatch):
    draft = "sha256:" + "4" * 64
    ops_dir = _ops(tmp_path, monkeypatch, **{DRAFT: draft})

    with pytest.raises(SystemExit, match=f"records: {DRAFT} '{draft}', not None"):
        _fit_traced(tmp_path, ops_dir)

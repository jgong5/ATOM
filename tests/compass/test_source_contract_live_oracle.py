"""A source check must not register a second model, including across repeats."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atom.compass.core.loaded_input import load_json
from atom.compass.runtime import cache_region_oracle


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"seconds": 1.0}\n')
    _, loaded = load_json(str(path), role="oracle.price")
    oracle = SimpleNamespace(compass_loaded_inputs=(loaded,))
    module_path = Path(__file__).parents[2] / "scripts/compass/cc_traces_opening.py"
    spec = importlib.util.spec_from_file_location("live_source_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, oracle, [loaded.as_dict()], path


def test_initialized_oracle_is_reused_for_two_startup_checks(source, monkeypatch):
    opening, oracle, observed, _ = source

    def duplicate_model(**options):
        raise AssertionError("a second model would duplicate global attention registration")

    monkeypatch.setattr(cache_region_oracle, "source_cost_oracle", duplicate_model)
    for _ in range(2):
        assert opening._source_contract_oracle({"model": "Qwen"}, observed, oracle) is oracle


def test_offline_repeats_construct_once_but_reopen_source_bytes(source, monkeypatch):
    opening, oracle, observed, path = source
    builds = []

    def construct(**options):
        builds.append(options)
        if len(builds) > 1:
            raise AssertionError("second model initialization")
        return oracle

    monkeypatch.setattr(cache_region_oracle, "source_cost_oracle", construct)
    for _ in range(3):
        assert opening._source_contract_oracle({"model": "Qwen"}, observed) is oracle
    assert len(builds) == 1
    # Same path and size; a cached object must not hide changed source bytes.
    path.write_text('{"seconds": 2.0}\n')
    with pytest.raises(ValueError, match="input changed"):
        opening._source_contract_oracle({"model": "Qwen"}, observed)
    assert len(builds) == 1


@pytest.mark.parametrize("damage", ["missing", "changed", "duplicate"])
def test_live_oracle_must_match_each_repeat_manifest(source, damage):
    opening, oracle, observed, _ = source
    changed = copy.deepcopy(observed)
    if damage == "missing":
        changed.clear()
    elif damage == "changed":
        changed[0]["sha256"] = "0" * 64
    else:
        changed.append(changed[0])
    with pytest.raises(ValueError, match="loaded input identity"):
        opening._source_contract_oracle({"model": "Qwen"}, changed, oracle)


def test_changed_offline_options_do_not_reuse_or_rebuild_the_model(source, monkeypatch):
    opening, oracle, observed, _ = source
    monkeypatch.setattr(cache_region_oracle, "source_cost_oracle", lambda **kwargs: oracle)
    opening._source_contract_oracle({"model": "Qwen", "price": "one"}, observed)
    with pytest.raises(ValueError, match="options changed"):
        opening._source_contract_oracle({"model": "Qwen", "price": "two"}, observed)

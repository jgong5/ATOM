"""Native allocation capacity is conditioning; per-block layout remains scope."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from atom.compass.core.cost.families import attention as A, attention_scope
from .test_attention_family import _cold_designs, _library, _resolved_record, _unified


def test_normal_source_factory_import_does_not_depend_on_family_import_order():
    result = subprocess.run(
        [sys.executable, "-c", "from atom.compass.core.cost.library import PriceLibrary; "
         "from atom.compass.runtime.source_oracle import build_source_group"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cached", [False, True])
def test_generic_prefill_accepts_capacity_only_change_without_refitting(tmp_path, cached):
    scope = attention_scope.read_resolved(_resolved_record(), where="source").for_family("unified")
    if cached:
        regime = A.REGIMES["unified.prefill.cached"]
        designs = []
        for q, h in ((64, 64), (128, 256), (512, 1024), (256, 2048), (768, 512), (1024, 4096)):
            op = _unified([q], [q + h], is_prefill=True, has_cached=True)
            features = A.features_for(regime, A.structure_of(op))
            designs.append((op, sum(a * b for a, b in zip(features, (1e-6, 2e-10, 1e-8, 3e-9)))))
        asked = _unified([256], [768], is_prefill=True, has_cached=True)
    else:
        designs = _cold_designs()
        asked = _unified([641], [641], is_prefill=True, has_cached=False)
    library = _library(tmp_path, designs, scope=scope)
    library.request_attention_scope = dict(scope)
    original, why = library.lookup(asked)
    assert original is not None, why
    fits_before = {key: (fit.coefficients, copy.deepcopy(fit.scope))
                   for key, fit in library.attention_model().fits.items()}
    target = json.loads(json.dumps(scope))
    for _name, fields in target["kv_cache_layout"]:
        for field, value in fields:
            if field == "shape":
                value[0] = 266768
    library.request_attention_scope = attention_scope.declaration_of(target, where="derived target")
    assert library._treatment_for(asked, library._declared_scope(asked)) is not None
    record, why = library.lookup(asked)
    assert record is not None, why
    assert record["seconds"] == original["seconds"]
    assert {key: (fit.coefficients, fit.scope) for key, fit in library.attention_model().fits.items()} == fits_before
    retained = record["interpolation"]["source_conditioning"]["kv_cache_layout"]
    assert dict(dict(retained)["k"])["shape"][0] == 131072
    target["kv_cache_layout"][0][1][1][1][0] += 1  # change the K view's stride
    library.request_attention_scope = attention_scope.declaration_of(target, where="different layout")
    assert library.lookup(asked)[0] is None

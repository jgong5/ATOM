"""An observed autotune configuration identifies its own acquisition treatment."""
import json

from atom.compass.core.cost.families.adapter import ParametricPriceLibrary, _measurement_identity
from .test_attention_family import SCOPE, _cold_designs, _gdn, _gdn_designs, _library, _unified


def test_different_kernel_configurations_do_not_pool_or_select_together(tmp_path):
    designs = _gdn_designs()
    library = ParametricPriceLibrary()
    for label, digest in (("config_a", "a" * 64), ("config_b", "b" * 64)):
        _library(tmp_path, designs, name=label)
        for index in range(len(designs)):
            price = tmp_path / f"{label}{index}.json"
            blob = json.loads(price.read_text())
            for record in blob["prices"].values():
                record["kernel_config_digest"] = digest
            price.write_text(json.dumps(blob))
            library.add(str(price), str(tmp_path / f"g{label}{index}.json"))
    assert len(library.attention_design_points()) == 2 * len(designs)
    op = _gdn([777], initial=[False])
    assert library._treatment_for(op, dict(SCOPE)) is None
    for digest in ("a" * 64, "b" * 64):
        library.select_attention_treatments({"gdn.prefill": {"kernel_config_digest": digest}})
        selected = library._treatment_for(op, dict(SCOPE))
        assert selected is not None
        assert dict(selected[1:])["kernel_config_digest"] == digest


def test_absent_configuration_keeps_legacy_identity_and_pricing(tmp_path):
    legacy = {"cache": "graph", "kernels": {"kernel": 1.0}}
    identity = _measurement_identity(legacy)
    assert "kernel_config_digest" not in dict(identity[1:])
    assert identity != _measurement_identity(dict(legacy, kernel_config_digest="a" * 64))
    library = _library(tmp_path, _cold_designs())
    library.request_attention_scope = dict(SCOPE)
    record, _ = library.lookup(_unified([641], [641], is_prefill=True, has_cached=False))
    assert record is not None and record["interpolated"]

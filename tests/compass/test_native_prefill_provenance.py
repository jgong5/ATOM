"""New native A/P sources retain the existing source/provenance contract."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.core.loaded_input import manifest
from atom.compass.runtime import cache_region_oracle as runtime
from .test_native_prefill_sources import bundle
from .test_opening_harness import wrapper_evidence, opening, validate


def grouped_inputs(rank):
    path = Path(__file__).resolve().parents[2] / "atom/entrypoints/openai/api_server.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name == "_loaded_option_files" or
             isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "_ROLE_OPTIONS" for target in node.targets)]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_loaded_option_files"]([rank])


@pytest.mark.parametrize("damage", [
    None, "missing_read", "unregistered", "aggregate", "unconfigured", "changed_scope", "changed_source",
])
def test_native_region_factory_api_and_pair_contract_agree(tmp_path, damage):
    compass, _ = wrapper_evidence.__wrapped__(tmp_path)
    options = compass["oracle_options"]
    options.update(include_failed_outputless=False, include_failed_final=False, diagnostic_only=False)
    source_dir = tmp_path / "native"
    source_dir.mkdir()
    path, _ = bundle(source_dir)
    handoff = json.loads(path.read_text())
    scope = Path(options["attention_scope"])
    handoff.update(source_qualified=True, deployment_scope={
        "path": str(scope), "sha256": hashlib.sha256(scope.read_bytes()).hexdigest()})
    path.write_text(json.dumps(handoff))
    options.update(native_prefill_handoff=str(path),
                   native_prefill_handoff_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    oracle = runtime.source_cost_oracle(**options)
    rank = manifest(oracle.compass_loaded_inputs)
    rank["regions"] = oracle.compass_region_snapshot
    compass["loaded_inputs"]["ranks"] = [rank]
    grouped = grouped_inputs(rank)
    assert len(grouped["native_prefill_handoff"]) == 11
    digests = {key: next(iter(value.values())) if len(value) == 1 else validate._rolled_digest(value)
               for key, value in grouped.items()}
    compass.update(oracle_option_files=grouped, oracle_option_sha256=digests)
    provenance = dict(kind="source_calibration", measured_at_tp=1, from_target_engine=False,
        sources=[dict(path="isolated-source", sha256="8" * 64)], code={"collector.py": "9" * 64})
    artifacts = [dict(provenance, sha256=row["sha256"],
                      contents={Path(row["path"]).name: row["sha256"]}) for row in rank["inputs"]]
    artifacts += [dict(provenance, sha256=digests[key], contents=value) for key, value in grouped.items()]
    artifacts.append(dict(provenance, kind="region_model", sha256=rank["regions"]["sha256"]))
    role = "oracle.native_prefill_regions.verdict"
    if damage == "missing_read":
        rank["inputs"] = [row for row in rank["inputs"] if row["role"] != role]
    elif damage == "unregistered":
        digest = next(row["sha256"] for row in rank["inputs"] if row["role"] == role)
        artifacts = [row for row in artifacts if row["sha256"] != digest]
    elif damage == "aggregate":
        grouped["native_prefill_handoff"].pop("verdict.json")
    elif damage == "unconfigured":
        options.pop("native_prefill_handoff")
        options.pop("native_prefill_handoff_sha256")
    elif damage == "changed_scope":
        next(row for row in rank["inputs"] if row["role"] == "oracle.attention_scope")["sha256"] = "f" * 64
    elif damage == "changed_source":
        (source_dir / "source.json").write_text("{}")
    modelled = SimpleNamespace(manifest={"server": {"compass": compass, "tensor_parallel_size": 1}})
    bad, notes = opening.check_source_contract(modelled, {"artifacts": artifacts}, "7" * 64, {}, "native fixture")
    if damage is None:
        assert bad == [], bad
        assert notes == []
    else:
        assert bad, damage

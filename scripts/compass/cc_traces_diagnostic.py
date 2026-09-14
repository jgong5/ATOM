"""Pinned corpus case identity for diagnostic execution, without selecting data.

The emitter owns corpus membership/episode completeness validation. This module
checks a pinned manifest and its replay rows; it is not an independent census.
"""

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path


SCHEMA = "compass.explicit_corpus_diagnostic_workload/1"
CASE_SCHEMA = "compass.corpus_diagnostic_case/1"
RESERVED_CLASSES = {"short", "long", "clients_short", "clients_large"}
# Declared target behavior, not a reconstruction of source cancellation/errors.
# See atom/compass/ZERO_OUTPUT_CONTRACT.md.
ZERO_OUTPUT_SEMANTICS = "full_prompt_zero_retained_output_v1"


def registered_cell_name(name):
    return bool(re.fullmatch(r"tp\d+_(?:(?:clients_short|clients_large)_c\d+|short|long)", name))


def _corpus():
    path = Path(__file__).with_name("cc_traces_workload.py")
    spec = importlib.util.spec_from_file_location("diagnostic_corpus_source", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CORPUS


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be an explicit SHA-256")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def replay_rows(rows):
    """The driver-visible fields, preserving order and every arrival interval."""
    normalized = []
    previous = 0.0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("case rows must be JSON objects")
        at = row.get("arrival_s")
        if (type(at) not in (int, float) or not math.isfinite(at)
                or at < previous or (i == 0 and at != 0)):
            raise ValueError("case arrivals must start at zero and be finite/nondecreasing")
        normalized.append({
            "arrival_s": float(at),
            "input_tokens": _integer(row.get("input_tokens"), "input_tokens", 1),
            "output_tokens": _integer(row.get("output_tokens"), "output_tokens"),
        })
        previous = at
    return normalized


def rows_digest(rows):
    return _sha(json.dumps(replay_rows(rows), sort_keys=True).encode())


def load_case(workload, manifest, manifest_sha256, case_id, *, target_model):
    """Validate supplied pins and declared source/replay consistency.

    Source validation remains an emitter claim identified by the manifest hash;
    this does not re-open the complete corpus or prove descendant completeness.
    """
    if (not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case_id)
            or case_id in RESERVED_CLASSES):
        raise ValueError("case-id must be a distinct diagnostic name, not a registered class")
    workload, manifest = Path(workload).resolve(), Path(manifest).resolve()
    manifest_bytes = manifest.read_bytes()
    if _sha(manifest_bytes) != _digest(manifest_sha256, "manifest digest"):
        raise ValueError("diagnostic manifest digest differs from its explicit pin")
    stated = json.loads(manifest_bytes)
    if not isinstance(stated, dict):
        raise ValueError("diagnostic manifest must be a JSON object")
    if (stated.get("schema") != SCHEMA or stated.get("class") != case_id
            or stated.get("purpose") != "diagnostic"
            or stated.get("registered_acceptance_cell") is not False):
        raise ValueError("manifest must identify this diagnostic case and deny registered acceptance")
    corpus = _corpus()
    if stated.get("corpus") != corpus or stated.get("corpus_sha256_observed") != corpus["sha256"]:
        raise ValueError("manifest does not identify the pinned cc-traces corpus")
    if stated.get("target_model") != target_model:
        raise ValueError("diagnostic manifest names a different target model")
    for field in ("gaps_clipped", "outputs_altered", "subagents_pruned", "requests_serialised"):
        if type(stated.get(field)) is not int or stated[field] != 0:
            raise ValueError(f"diagnostic manifest must preserve source requests: {field}=0")
    emitter = stated.get("generator") or {}
    if not isinstance(emitter, dict):
        raise ValueError("diagnostic emitter provenance must be an object")
    _digest(emitter.get("sha256"), "emitter digest")
    if (not emitter.get("path") or not isinstance(stated.get("validation"), dict)
            or not stated["validation"] or not stated.get("selection_scope")):
        raise ValueError("diagnostic manifest must retain emitter, source validation and selection scope")
    data = workload.read_bytes()
    workload_sha = _digest(stated.get("sha256"), "workload digest")
    if _sha(data) != workload_sha:
        raise ValueError("diagnostic workload digest differs from the pinned manifest")
    rows = replay_rows([json.loads(line) for line in data.splitlines() if line.strip()])
    has_zero_output = any(row["output_tokens"] == 0 for row in rows)
    if has_zero_output and stated.get("zero_output_semantics") != ZERO_OUTPUT_SEMANTICS:
        raise ValueError(
            "zero-output diagnostics must explicitly declare zero_output_semantics="
            f"{ZERO_OUTPUT_SEMANTICS!r}"
        )
    count = _integer(stated.get("requests"), "requests", 1)
    clients = _integer(stated.get("clients"), "clients", 1)
    provenance = stated.get("provenance")
    if len(rows) != count or not isinstance(provenance, list) or len(provenance) != count:
        raise ValueError("case must keep every workload row and one provenance entry per row")
    identities, roots, origins = set(), set(), {}
    actors = {"root": 0, "subagent": 0}
    for row, source in zip(rows, provenance):
        if not isinstance(source, dict):
            raise ValueError("case provenance entries must be JSON objects")
        if (_integer(source.get("input_tokens"), "source input_tokens", 1) != row["input_tokens"]
                or _integer(source.get("output_tokens"), "source output_tokens") != row["output_tokens"]):
            raise ValueError("case row token lengths disagree with source provenance")
        if source.get("arrival_s") != row["arrival_s"]:
            raise ValueError("case row arrival disagrees with source provenance")
        root, path = source.get("session"), source.get("json_path")
        actor = source.get("actor")
        if (not isinstance(root, str) or not root.strip()
                or not isinstance(path, str) or not re.fullmatch(r"/requests/\d+(?:/requests/\d+)?", path)
                or actor not in actors or (actor == "root") != (path.count("/requests/") == 1)):
            raise ValueError("case has invalid native request identity/ancestry")
        identity = (root, path)
        if identity in identities:
            raise ValueError("case repeats a native request identity")
        identities.add(identity)
        roots.add(root)
        actors[actor] += 1
        if row["input_tokens"] != _integer(source.get("input_blocks"), "input_blocks", 1) * 64:
            raise ValueError("case source token/hash-block lengths disagree")
        if row["input_tokens"] + row["output_tokens"] > 256000:
            raise ValueError("case exceeds the pinned corpus's decimal 256000-token cap")
        at, origin = source.get("source_t_s"), source.get("origin_shift_s")
        if (type(at) not in (int, float) or type(origin) not in (int, float)
                or not math.isfinite(at) or not math.isfinite(origin)
                or not math.isclose(at - origin, row["arrival_s"], rel_tol=0, abs_tol=1e-9)):
            raise ValueError("case changed a source arrival interval")
        if root in origins and origins[root] != origin:
            raise ValueError("case must use one common origin shift per root")
        origins[root] = origin
    used = stated.get("roots_used")
    if not isinstance(used, list) or not all(isinstance(r, dict) for r in used):
        raise ValueError("case must declare its root identities")
    if (len(roots) != clients or stated.get("root_requests") != actors["root"]
            or stated.get("descendant_requests") != actors["subagent"]
            or {r.get("id") for r in used} != roots):
        raise ValueError("case root/client/request accounting disagrees with provenance")
    for field, key in (("input_token_total", "input_tokens"), ("output_token_total", "output_tokens")):
        if stated.get(field) != sum(r[key] for r in rows):
            raise ValueError(f"case {field} disagrees with workload rows")
    case = {
        "schema": CASE_SCHEMA, "case_id": case_id,
        "manifest": str(manifest), "manifest_sha256": manifest_sha256,
        "workload": str(workload), "workload_sha256": workload_sha,
        "rows_sha256": rows_digest(rows), "requests": count, "clients": clients,
        "target_model": target_model, "corpus_sha256": corpus["sha256"],
        "selection_scope": stated["selection_scope"], "emitter": emitter,
        "source_validation": "Emitter validation in the pinned manifest; not an independent corpus census",
    }
    if has_zero_output:
        case["zero_output_semantics"] = ZERO_OUTPUT_SEMANTICS
    return case


def recheck(case):
    current = load_case(case["workload"], case["manifest"], case["manifest_sha256"],
                        case["case_id"], target_model=case["target_model"])
    if current != case:
        raise ValueError("diagnostic case identity changed after planning")


def identity(case):
    """Portable pins: the two machines need not mount inputs at identical paths."""
    keys = ("schema", "case_id", "manifest_sha256", "workload_sha256", "rows_sha256",
            "requests", "clients", "target_model", "corpus_sha256")
    if not isinstance(case, dict) or any(key not in case for key in keys):
        raise ValueError("diagnostic case identity is incomplete")
    result = {key: case[key] for key in keys}
    if "zero_output_semantics" in case:
        result["zero_output_semantics"] = case["zero_output_semantics"]
    return result


def check_result(blob, case):
    """A full-file hash alone does not detect replay's --num-requests slicing."""
    if (blob.get("run") or {}).get("requests") != case["requests"]:
        raise ValueError("diagnostic result request count differs from the pinned case")
    if len(blob.get("results") or []) != case["requests"]:
        raise ValueError("diagnostic result count differs from the pinned case")
    if rows_digest(blob.get("workload") or []) != case["rows_sha256"]:
        raise ValueError("diagnostic result rows differ from the pinned case")

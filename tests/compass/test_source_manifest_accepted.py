"""The canonical DSL, end to end, into the real calibration validator.

Every piece of this chain has its own tests. What none of them establishes is
that the pieces fit: that what the *actual* source factory records, reading the
*actual* comma-separated `price:graph:regime` option over per-rank files on
disk, is a thing the *actual* calibration validator will accept -- and that it
refuses the same run when the declaration stops matching.

Four seams, none of them faked:

  `source_cost_oracle`  reads the DSL and the per-rank files
  `CompassPredictMixin` freezes what it read into a manifest
  `/compass/provenance` maps roles onto the option keys a report is written in
  `check_calibration`   reads that against a registry declared independently

Nothing here writes a loaded-input row. The rows come from the loader, and the
registry is built from the bytes on disk rather than from the manifest, so the
two sides are only ever compared, never copied from one another.
"""

import asyncio
import hashlib
import importlib
import json
import os
import types

import pytest

from atom.compass.config import CompassConfig
from atom.compass.runtime.predict import CompassPredictMixin
from atom.compass.runtime.source_oracle import source_cost_oracle

from . import test_cc_traces_validate as base

validate = base.validate
compare = base.validate.compare

try:
    api_server = importlib.import_module("atom.entrypoints.openai.api_server")
except Exception:  # noqa: BLE001 - environment-dependent
    api_server = None

pytestmark = pytest.mark.skipif(api_server is None,
                                reason="api_server import unavailable")

WIDTH = 2
QUERIES = [1, 1]
CONTEXTS = [128, 130]


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _ops():
    return [{"name": "mm", "input_shapes": [[2, 64], [64, 64]],
             "dtypes": ["bfloat16"]}]


def _price_file(tmp_path, name, seconds):
    from atom.compass.runtime.microbench import signature_of

    path = tmp_path / name
    path.write_text(json.dumps({
        "provenance": {"topology": {"tp": WIDTH}},
        "prices": {signature_of(op): {"name": op["name"], "seconds": seconds,
                                      "occurrences": 1,
                                      "kernels": {"k0": seconds}}
                   for op in _ops()},
        "unpriced": {},
    }), encoding="utf-8")
    return str(path)


def _graph_file(tmp_path, name, rank):
    path = tmp_path / name
    path.write_text(json.dumps({
        "ops": _ops(),
        "key": {"topology": [["tp", WIDTH]], "rank_coords": [["tp", rank]]},
        "provenance": {
            "batch_spec": {"kind": "decode", "query_lens": QUERIES,
                           "context_lens": CONTEXTS},
            "execution": {"capture_bucket": None},
        },
    }), encoding="utf-8")
    return str(path)


def _tree(tmp_path):
    """Per-rank price lists, their graphs, and body and head templates."""
    for rank in range(WIDTH):
        _price_file(tmp_path, f"prices.tp{rank}.json", 0.0013 + rank / 1000)
        _graph_file(tmp_path, f"pricegraph.tp{rank}.json", rank)
        _graph_file(tmp_path, f"body.tp{rank}.json", rank)
        _graph_file(tmp_path, f"head.tp{rank}.json", rank)
    return {
        # The canonical DSL: a triple, with the collective registration regime
        # in the third field, over a stem each rank expands for itself.
        "price": (f"{tmp_path / 'prices.json'}:"
                  f"{tmp_path / 'pricegraph.json'}:unregistered"),
        "template": str(tmp_path / "body.json"),
        "head_template": str(tmp_path / "head.json"),
        "head": "true", "tp": WIDTH, "derive": 0,
        "require_complete": "true", "regions": "none",
    }


def _served(tmp_path, options):
    """The real chain: factory -> mixin -> endpoint. No fabricated rows."""
    oracle = source_cost_oracle(**{k: v for k, v in options.items()},
                               rank_coords={"tp": 0})

    stub = CompassPredictMixin.__new__(CompassPredictMixin)
    stub.__dict__["_compass_config_cache"] = CompassConfig(enabled=True)
    stub.config = types.SimpleNamespace()
    stub._build_oracle = lambda config: oracle
    stub._topology = lambda: {"tp": WIDTH}
    stub._rank_coords = lambda: {"tp": 0}
    stub._warn_if_compiled = lambda: None
    stub._init_compass_state()
    manifest = stub.compass_input_manifest()

    engine = types.SimpleNamespace(
        config=types.SimpleNamespace(
            compass_config=CompassConfig(
                enabled=True,
                oracle_qualname=validate.SOURCE_FACTORY,
                oracle_options={k: str(v) for k, v in options.items()}),
            model_config=None, parallel_config=None),
        get_compass_inputs=lambda timeout=10.0: {"ranks": [manifest]})
    return engine, manifest


def _provenance(monkeypatch, engine):
    monkeypatch.setattr(api_server, "engine", engine)
    return asyncio.run(api_server.compass_provenance())


def _run(blob):
    """The saved modelled run, as `replay.py` writes the parts read here."""
    return compare.Run(
        path="modelled.json", label="modelled",
        manifest={"server": blob, "server_revision": "abc",
                  "model_revision": "rev1"})


def _registry_from_disk(blob):
    """Declare each option's files from the bytes on disk, independently.

    Built by reading the files again here, not by copying the manifest. The
    point of the check is that two accounts agree, and an account derived from
    the other cannot disagree.
    """
    artifacts = []
    for key, found in (blob["compass"]["oracle_option_files"] or {}).items():
        contents = {name: _sha(_locate(blob, key, name))
                    for name in found}
        sha = blob["compass"]["oracle_option_sha256"][key]
        artifacts.append({
            "sha256": sha,
            "kind": "source_calibration",
            "measured_at_tp": 1,
            "produced_by": "microbench.py",
            "workload_sha256": None,
            "contents": contents,
            "sources": [{"path": "/m/sweep.json", "sha256": base.SWEEP_SHA}],
            "code": {"scripts/compass/microbench.py": base.CODE_SHA},
        })
    return {"artifacts": artifacts}


def _locate(blob, key, name):
    """Where on disk a reported basename came from, per the rank's own record."""
    for rank in blob["compass"]["loaded_inputs"]["ranks"]:
        for row in rank["inputs"]:
            if os.path.basename(row["path"]) == name:
                return row["path"]
    raise AssertionError(f"{key}: {name} is not in any rank's record")


class TestTheCanonicalDSLReachesTheValidator:

    def test_the_option_members_are_the_files_the_ranks_opened(
            self, monkeypatch, tmp_path):
        """Before the validator: the triple and both templates arrive as the
        per-rank files that were actually read, not as the stems.

        Every rank of the group, not only the executor's own. `tp=2` builds
        two compositions and each resolves its own files, so one option key
        stands for four files here -- two ranks times the two members of the
        triple -- and that is what the option has to be shown to stand for.
        """
        engine, _manifest = _served(tmp_path, _tree(tmp_path))
        blob = _provenance(monkeypatch, engine)

        files = blob["compass"]["oracle_option_files"]
        assert set(files) == {"price", "template", "head_template"}
        assert set(files["price"]) == {
            "prices.tp0.json", "pricegraph.tp0.json",
            "prices.tp1.json", "pricegraph.tp1.json"}
        assert set(files["template"]) == {"body.tp0.json", "body.tp1.json"}
        assert set(files["head_template"]) == {"head.tp0.json", "head.tp1.json"}
        assert all(
            source == "loaded"
            for source in blob["compass"]["oracle_option_digest_source"].values()
        )

    def test_each_rank_appears_under_its_own_coordinates(
            self, monkeypatch, tmp_path):
        """The summary folds the ranks together; the per-rank record does not,
        and that is where "which rank read which file" survives."""
        _engine, manifest = _served(tmp_path, _tree(tmp_path))

        by_rank = {}
        for row in manifest["inputs"]:
            if row["role"] == "oracle.price":
                by_rank[row["rank_coords"]["tp"]] = row["path"]

        assert by_rank[0].endswith("prices.tp0.json")
        assert by_rank[1].endswith("prices.tp1.json")
        assert all(row["rank_own"] for row in manifest["inputs"])

    def test_the_declared_run_is_accepted(self, monkeypatch, tmp_path):
        """The whole point: what the canonical factory recorded is a thing the
        real calibration validator accepts, given an honest declaration."""
        engine, _ = _served(tmp_path, _tree(tmp_path))
        blob = _provenance(monkeypatch, engine)

        bad = validate.check_calibration(
            _run(blob), _registry_from_disk(blob), WIDTH, "d" * 64, {})

        assert bad == []

    def test_the_rolled_digest_the_server_reports_is_the_one_recomputed(
            self, monkeypatch, tmp_path):
        """The price option stands for two files, so the server rolls them
        into one digest and the validator recomputes it. Two implementations,
        in two programs, that have to agree byte for byte."""
        engine, _ = _served(tmp_path, _tree(tmp_path))
        blob = _provenance(monkeypatch, engine)

        contents = {name: _sha(_locate(blob, "price", name))
                    for name in blob["compass"]["oracle_option_files"]["price"]}
        assert (validate._rolled_digest(contents)
                == blob["compass"]["oracle_option_sha256"]["price"])


class TestADeclarationThatStopsMatchingIsRefused:

    def _blob(self, monkeypatch, tmp_path):
        engine, _ = _served(tmp_path, _tree(tmp_path))
        return _provenance(monkeypatch, engine)

    def test_a_declared_member_with_the_wrong_digest_is_refused(
            self, monkeypatch, tmp_path):
        blob = self._blob(monkeypatch, tmp_path)
        registry = _registry_from_disk(blob)
        for entry in registry["artifacts"]:
            if "pricegraph.tp0.json" in entry["contents"]:
                entry["contents"]["pricegraph.tp0.json"] = "9" * 64

        bad = validate.check_calibration(
            _run(blob), registry, WIDTH, "d" * 64, {})

        assert any("differs between what the server read" in b for b in bad)

    def test_a_missing_declared_member_is_refused(self, monkeypatch, tmp_path):
        """The graph half of the triple dropped. A price matched by signature
        alone is a price without a layout, so losing it from the declaration
        loses what the price was a price of."""
        blob = self._blob(monkeypatch, tmp_path)
        registry = _registry_from_disk(blob)
        for entry in registry["artifacts"]:
            entry["contents"].pop("pricegraph.tp0.json", None)

        bad = validate.check_calibration(
            _run(blob), registry, WIDTH, "d" * 64, {})

        assert any("which the registry does not declare" in b for b in bad)

    def test_an_undeclared_option_is_refused(self, monkeypatch, tmp_path):
        blob = self._blob(monkeypatch, tmp_path)
        registry = _registry_from_disk(blob)
        registry["artifacts"] = registry["artifacts"][:1]

        bad = validate.check_calibration(
            _run(blob), registry, WIDTH, "d" * 64, {})

        assert any("does not declare" in b or "cannot be attributed" in b
                   for b in bad)

    def test_a_calibration_measured_at_the_predicted_width_is_refused(
            self, monkeypatch, tmp_path):
        blob = self._blob(monkeypatch, tmp_path)
        registry = _registry_from_disk(blob)
        for entry in registry["artifacts"]:
            entry["measured_at_tp"] = WIDTH

        bad = validate.check_calibration(
            _run(blob), registry, WIDTH, "d" * 64, {})

        assert any("width being predicted" in b for b in bad)

    def test_replacing_a_file_after_the_read_refuses_rather_than_follows(
            self, monkeypatch, tmp_path):
        """The manifest keeps describing the bytes that were loaded, so the
        registry rebuilt from disk now disagrees with it -- which is the
        refusal one wants. The alternative, a digest that followed the file,
        would have quietly agreed."""
        blob = self._blob(monkeypatch, tmp_path)
        _price_file(tmp_path, "prices.tp0.json", 0.9)

        bad = validate.check_calibration(
            _run(blob), _registry_from_disk(blob), WIDTH, "d" * 64, {})

        assert any("differs between what the server read" in b for b in bad)

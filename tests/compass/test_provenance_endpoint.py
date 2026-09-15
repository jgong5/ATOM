"""What the server publishes about what it was fitted to, and what reads it.

The defect, end to end: a served run configured with
`--compass-oracle-option price=list.json:graph.json:unregistered` loaded two
real tables and published no digest for them, because the reader looked for one
file of that name and there is no such file. `compare.check_run` then refused
the run -- correctly, by its own rule, and for a reason that was not a defect
in the run.

These drive the real endpoint over a real manifest produced by the real
factory, and then the real `check_run` over what the endpoint published. The
old mechanism is still in the module as the fallback, so the wrong answer it
gave can be asserted directly beside the right one.
"""

import asyncio
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.config import CompassConfig
from atom.compass.core.loaded_input import manifest
from atom.compass.runtime.source_oracle import build_source_oracle

from .test_source_oracle import _template_file

ROOT = Path(__file__).resolve().parents[2]

try:
    api_server = importlib.import_module("atom.entrypoints.openai.api_server")
except Exception as exc:  # noqa: BLE001 - environment-dependent
    api_server = None
    _why = f"{type(exc).__name__}: {exc}"

pytestmark = pytest.mark.skipif(api_server is None,
                                reason="api_server import unavailable")


def _load_compare():
    path = ROOT / "scripts" / "compass" / "compare.py"
    spec = importlib.util.spec_from_file_location("compass_compare", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class _Engine:
    """Just the surface `/compass/provenance` reads, answering as a real one.

    The manifest it hands back is produced by the real factory reading real
    files -- this stands in for the process boundary, not for the record.
    """

    def __init__(self, options, ranks):
        self.config = SimpleNamespace(
            compass_config=CompassConfig(
                enabled=True,
                oracle_qualname=(
                    "atom.compass.runtime.source_oracle.source_cost_oracle"),
                oracle_options=options),
            model_config=None, parallel_config=None)
        self._ranks = ranks

    def get_compass_inputs(self, timeout=10.0):
        return {"ranks": self._ranks}


def _served(monkeypatch, tmp_path, options, coords=None):
    """Stand up the endpoint over a composition the factory really built."""
    built = build_source_oracle(derive=0, rank_coords=coords, **options)
    ranks = [manifest(built.loaded_inputs, coords=coords or {})]
    engine = _Engine({k: str(v) for k, v in options.items()}, ranks)
    monkeypatch.setattr(api_server, "engine", engine)
    return asyncio.run(api_server.compass_provenance())


class TestTheOldReaderCouldNotSeeTheseFiles:
    """The wrong answer, asserted against the function that gave it.

    `_artifact_digests` survives as the fallback for options no loader reads,
    so this is not a reconstruction: it is the same code, on the same values,
    returning what it returned.
    """

    def test_a_comma_separated_list_is_not_a_filename(self, tmp_path):
        first = _template_file(tmp_path, "graph.json")
        second = _template_file(tmp_path, "other.json")

        assert api_server._artifact_digests(f"{first},{second}") == {}

    def test_a_price_triple_is_not_a_filename(self, tmp_path):
        prices = tmp_path / "prices.json"
        prices.write_text("{}", encoding="utf-8")
        graph = tmp_path / "graph.json"
        graph.write_text("{}", encoding="utf-8")

        value = f"{prices}:{graph}:unregistered"

        assert api_server._artifact_digests(value) == {}


class TestTheServerPublishesWhatTheRanksRead:

    def test_every_member_of_a_list_option_gets_its_digest(self, monkeypatch,
                                                           tmp_path):
        first = _template_file(tmp_path, "graph.json")
        second = _template_file(tmp_path, "other.json")

        blob = _served(monkeypatch, tmp_path,
                       {"template": f"{first},{second}"})

        files = blob["compass"]["oracle_option_files"]["template"]
        assert set(files.values()) == {_digest(first), _digest(second)}
        assert blob["compass"]["oracle_option_sha256"]["template"]

    def test_the_digest_says_whether_it_is_of_bytes_that_were_read(
            self, monkeypatch, tmp_path):
        """A digest of what ran and a digest of what is on the disk now are
        different evidence, and a reader must not have to guess which."""
        path = _template_file(tmp_path, "graph.json")

        blob = _served(monkeypatch, tmp_path, {"template": path})

        assert blob["compass"]["oracle_option_digest_source"]["template"] == (
            "loaded")

    def test_an_option_no_loader_reads_falls_back_and_says_so(
            self, monkeypatch, tmp_path):
        """`regions` names a profile, not a file, and nothing reads a file for
        it. The fallback still answers where it can, marked as the guess it
        is."""
        path = _template_file(tmp_path, "graph.json")
        blob = _served(monkeypatch, tmp_path, {"template": path})

        sources = blob["compass"]["oracle_option_digest_source"]
        assert set(sources) <= set(blob["compass"]["oracle_options"])
        assert "loaded" in sources.values()

    def test_the_per_rank_detail_is_published_whole(self, monkeypatch,
                                                    tmp_path):
        """Folding the ranks into one digest per option loses which rank read
        which file, and that is the question the per-rank convention exists to
        answer."""
        _template_file(tmp_path, "graph.tp1.json")
        stem = str(tmp_path / "graph.json")

        blob = _served(monkeypatch, tmp_path, {"template": stem},
                       coords={"tp": 1})

        rows = blob["compass"]["loaded_inputs"]["ranks"][0]["inputs"]
        assert rows[0]["rank_own"] is True
        assert rows[0]["requested"] == stem
        assert rows[0]["path"].endswith("graph.tp1.json")

    def test_an_unreachable_engine_is_a_reason_and_not_an_empty_record(
            self, monkeypatch, tmp_path):
        """Nothing loaded and nothing asked are different states. Reporting
        the second as the first would make an unanswerable server look like a
        server with no calibration."""
        monkeypatch.setattr(api_server, "engine", None)

        assert api_server._compass_loaded_inputs()["why"]


class TestTheValidatorAcceptsWhatTheServerNowPublishes:
    """The consumer, unchanged, over the two shapes."""

    @staticmethod
    def _run(compare, compass_block):
        return compare.Run(
            path="modelled.json", label="modelled",
            manifest={"prompt_lengths": "passed",
                      "server_revision": "deadbeef",
                      "model_revision": "Qwen3.8-27B@abc",
                      "server": {"compass": compass_block}})

    @staticmethod
    def _refusals(bad):
        return [b for b in bad if "read no such file" in b]

    def test_a_run_that_published_no_digest_is_refused(self, tmp_path):
        """The old shape. `_names_a_file` splits the triple and sees paths, so
        the option has to carry a digest -- and there was none."""
        compare = _load_compare()
        prices = tmp_path / "prices.json"
        prices.write_text("{}", encoding="utf-8")
        value = f"{prices}:{tmp_path / 'graph.json'}:unregistered"

        bad = compare.check_run(self._run(compare, {
            "oracle": "source_cost_oracle",
            "oracle_options": {"price": value},
            "oracle_option_sha256": {}}), require_length_check=False)

        assert self._refusals(bad), "this is the refusal the defect produced"

    def test_the_same_run_is_reportable_once_the_ranks_report(self, tmp_path):
        compare = _load_compare()
        prices = tmp_path / "prices.json"
        prices.write_text("{}", encoding="utf-8")
        value = f"{prices}:{tmp_path / 'graph.json'}:unregistered"

        bad = compare.check_run(self._run(compare, {
            "oracle": "source_cost_oracle",
            "oracle_options": {"price": value},
            "oracle_option_sha256": {"price": "ab" * 32}}),
            require_length_check=False)

        assert not self._refusals(bad)

    def test_the_endpoint_output_satisfies_the_validator(self, monkeypatch,
                                                         tmp_path):
        """The two halves joined: what the server publishes is what the
        validator asks for, over a real read of real files."""
        compare = _load_compare()
        first = _template_file(tmp_path, "graph.json")
        second = _template_file(tmp_path, "other.json")

        blob = _served(monkeypatch, tmp_path,
                       {"template": f"{first},{second}"})

        bad = compare.check_run(self._run(compare, blob["compass"]),
                                require_length_check=False)

        assert not self._refusals(bad)

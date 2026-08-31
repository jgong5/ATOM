"""A simulated run's latency has to leave the engine, or it does not exist.

`benchmark_serving` times an HTTP stream and divides by its own wall clock, so
against a Compass server it measures the simulator. The engine's own readings
are the simulated latency, and before this they were computed only in
`postprocess`, which the serving path never calls.
"""

import ast
from pathlib import Path

import pytest

from atom.model_engine.request import RequestOutput

ROOT = Path(__file__).resolve().parents[2]


class TestRequestOutputCarriesTheEngineClock:
    def test_the_fields_exist_and_default_to_unstamped(self):
        out = RequestOutput(request_id=1, output_tokens=[], finished=False)
        assert out.arrive_time == 0.0
        assert out.first_token_time == 0.0
        assert out.finish_time == 0.0

    def test_they_survive_the_process_boundary(self):
        """Pickled: the streaming process is not the one that owns the seq."""
        import pickle

        out = RequestOutput(request_id=1, output_tokens=[7], finished=True,
                            arrive_time=100.0, first_token_time=100.5,
                            finish_time=102.0)
        back = pickle.loads(pickle.dumps(out))
        assert (back.arrive_time, back.first_token_time, back.finish_time) == (
            100.0, 100.5, 102.0)


class TestEveryStampGoesThroughTheClock:
    """One of the three sites used `time.time()` and produced a mixed-clock TTFT.

    A wall-clock instant minus a virtual one is not a duration. Guarded by
    inspection rather than by a run, because the site only fires on the
    speculative-decode path.
    """

    def test_no_first_token_time_is_stamped_from_wall_time(self):
        source = (ROOT / "atom/model_engine/scheduler.py").read_text()
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t for t in node.targets
                       if isinstance(t, ast.Attribute)
                       and t.attr in ("first_token_time", "finish_time",
                                      "arrive_time")]
            if not targets:
                continue
            call = node.value
            if isinstance(call, ast.Call):
                func = call.func
                # time.time() / time.monotonic() rather than get_clock().time()
                if isinstance(func, ast.Attribute) and isinstance(
                        func.value, ast.Name) and func.value.id == "time":
                    offenders.append((targets[0].attr, node.lineno))
        assert not offenders, (
            f"stamped off the wall clock, bypassing get_clock(): {offenders}")


class TestTheRecorder:
    @pytest.fixture(autouse=True)
    def _server(self):
        import atom.entrypoints.openai.api_server as server
        server._compass_records.clear()
        yield server
        server._compass_records.clear()

    def _out(self, **kw):
        base = dict(request_id=1, output_tokens=[], finished=True,
                    arrive_time=10.0, first_token_time=10.25, finish_time=11.0)
        base.update(kw)
        return RequestOutput(**base)

    def test_it_records_what_the_engine_measured(self, _server):
        _server._record_engine_timings("req-a", self._out())
        row = _server._compass_records["req-a"]
        assert row["ttft"] == pytest.approx(0.25)
        assert row["latency"] == pytest.approx(1.0)

    def test_a_request_with_no_token_has_no_ttft(self, _server):
        """0.0 is not a time-to-first-token; averaging it in would be a lie."""
        _server._record_engine_timings("req-b", self._out(first_token_time=0.0))
        assert _server._compass_records["req-b"]["ttft"] is None
        assert _server._compass_records["req-b"]["latency"] == pytest.approx(1.0)

    def test_an_unstamped_request_is_not_recorded(self, _server):
        _server._record_engine_timings("req-c", self._out(finish_time=0.0))
        _server._record_engine_timings("req-d", self._out(arrive_time=0.0))
        assert _server._compass_records == {}

    def test_it_is_bounded_and_drops_the_oldest(self, _server, monkeypatch):
        """A long-lived server must not grow a row per request forever."""
        monkeypatch.setattr(_server, "COMPASS_MAX_RECORDS", 3)
        for i in range(5):
            _server._record_engine_timings(f"r{i}", self._out())
        assert list(_server._compass_records) == ["r2", "r3", "r4"]

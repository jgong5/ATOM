"""What the acceptance plan must say, so a leased hour is not spent finding out.

The plan runs nothing, so there is no result to check. What there is to check is
that the sequence it prints is the protocol's: the real side paced and prepared
on the leased node, the modelled side declared and unprepared in a container
with no device, a fresh server for every repeat on both sides, the device-free
probe taken where the prediction happened, and the validator last.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = _load("cc_traces_plan")
validate = _load("cc_traces_validate")


@pytest.fixture
def plan():
    return json.loads(_run(["--root", "/r"]))


def _run(argv, capsys=None):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert plan_mod.main(argv) == 0
    return buffer.getvalue()


def _by_id(cell):
    return {s["id"]: s for s in cell["steps"]}


def _role(cell, role, side=None):
    return [
        s
        for s in cell["steps"]
        if s["role"] == role and (side is None or s.get("side") == side)
    ]


class TestTheMatrixItCovers:
    def test_every_width_and_class_appears_once(self, plan):
        seen = [(c["tp"], c["class"]) for c in plan["cells"]]
        assert sorted(seen) == sorted(
            (tp, klass) for tp in plan_mod.TPS for klass in plan_mod.CLASSES
        )
        assert len(seen) == len(set(seen))

    def test_each_cell_has_its_own_directory(self, plan):
        dirs = [c["cell"] for c in plan["cells"]]
        assert len(set(dirs)) == len(dirs)
        assert all(d.startswith("/r/") for d in dirs)

    def test_the_matrix_step_names_every_cell(self, plan):
        for cell in plan["cells"]:
            assert cell["cell"] in plan["matrix"]


class TestARepeatIsAProcess:
    """Section 3 registers three repeats a side, each from a fresh process.

    On the modelled side this is not bookkeeping: a predicting server's virtual
    epoch is fixed when it starts, so a second replay against the same process
    is stamped from an origin the first one already moved.
    """

    def test_each_repeat_starts_and_stops_its_own_server(self, plan):
        for cell in plan["cells"]:
            for side in ("real", "modelled"):
                serves = _role(cell, "serve", side)
                stops = [s for s in _role(cell, "stop", side) if "serve" in s["stops"]]
                replays = _role(cell, "replay", side)
                assert len(serves) == len(replays) == len(stops) == plan["repeats"]
                assert [s["repeat"] for s in serves] == [1, 2, 3]

    def test_a_repeat_s_replay_sits_between_its_own_serve_and_stop(self, plan):
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            for side in ("real", "modelled"):
                for n in range(1, plan["repeats"] + 1):
                    serve = ids.index(f"serve-{side}-{n}")
                    replay = ids.index(f"replay-{side}-{n}")
                    stop = ids.index(f"stop-{side}-{n}")
                    assert serve < replay < stop

    def test_no_server_outlives_the_repeat_that_started_it(self, plan):
        """The previous repeat's process is stopped before the next starts."""
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            for side in ("real", "modelled"):
                for n in range(1, plan["repeats"]):
                    assert ids.index(f"stop-{side}-{n}") < ids.index(
                        f"serve-{side}-{n + 1}"
                    )

    def test_every_stop_names_the_step_that_started_the_process(self, plan):
        for cell in plan["cells"]:
            ids = {s["id"] for s in cell["steps"]}
            for step in _role(cell, "stop"):
                assert step["stops"] in ids
                assert step["command"] is None
                assert (
                    "signalling the process it started" in step["provided_by"]
                    or "signalling the sampler it started" in step["provided_by"]
                )

    def test_each_repeat_writes_its_own_step_table(self, plan):
        """The engine opens measure_out with "w": one shared name would leave
        one table and two runs that overwrote each other."""
        for cell in plan["cells"]:
            tables = [
                s["command"][s["command"].index("--compass-measure-out") + 1]
                for s in _role(cell, "serve", "real")
            ]
            assert len(set(tables)) == len(tables) == plan["repeats"]


class TestTheTwoSidesAreNotRunTheSameWay:
    def test_the_real_side_is_paced_and_prepared(self, plan):
        for cell in plan["cells"]:
            for step in _role(cell, "replay", "real"):
                assert "--pace" in step["command"]
                assert "--prepare" in step["command"]
                assert step["where"] == "gpu"

    def test_the_modelled_side_is_neither(self, plan):
        """A paced modelled side answers a different arrival process, and a
        prepared one has been warmed -- both are refusals in the validator."""
        for cell in plan["cells"]:
            for step in _role(cell, "replay", "modelled"):
                assert "--pace" not in step["command"]
                assert "--prepare" not in step["command"]
                assert step["where"] == "device_free"

    def test_the_modelled_side_predicts_and_the_real_side_measures(self, plan):
        steps = _by_id(plan["cells"][0])
        real = steps["serve-real-1"]["command"]
        modelled = steps["serve-modelled-1"]["command"]
        assert real[real.index("--compass-mode") + 1] == "measure"
        assert modelled[modelled.index("--compass-mode") + 1] == "predict"
        assert "--compass-measure-out" in real
        assert steps["serve-real-1"]["where"] == "gpu"
        assert steps["serve-modelled-1"]["where"] == "device_free"

    def test_the_modelled_side_names_the_rank_aggregation(self, plan):
        """Left at the default the one executor prices the rank it calls
        itself, and the TP4 rank-1 outlier disappears with nothing to show
        that it did. The choice is in the plan, not in a parser default."""
        for cell in plan["cells"]:
            for step in _role(cell, "serve", "modelled"):
                cmd = step["command"]
                assert cmd[cmd.index("--compass-rank-aggregation") + 1] == "slowest"
            for step in _role(cell, "serve", "real"):
                # The real side runs one process per rank and has no such
                # question: each reports its own step.
                assert "--compass-rank-aggregation" not in step["command"]

    def test_the_modelled_side_serves_through_the_device_free_entry_point(self, plan):
        """The module entry point asks the driver for the chip at import, which
        on a machine with no device fails before a flag could be parsed."""
        for cell in plan["cells"]:
            for step in _role(cell, "serve", "modelled"):
                assert "scripts/compass/replay_server.py" in step["command"]
                assert "--compass-replay-target" in step["command"]
            for step in _role(cell, "serve", "real"):
                assert "atom.entrypoints.openai.api_server" in step["command"]
                assert "--compass-replay-target" not in step["command"]

    def test_each_width_is_served_its_own_target(self):
        got = json.loads(_run(["--root", "/r", "--replay-target", "1=/w/t1.json",
                               "--replay-target", "2=/w/t2.json",
                               "--replay-target", "4=/w/t4.json"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve", "modelled"):
                command = step["command"]
                assert command[command.index("--compass-replay-target") + 1] == (
                    f"/w/t{cell['tp']}.json"
                )

    def test_one_target_cannot_be_handed_to_the_whole_matrix(self):
        # The replay runner refuses a target whose width is not the one being
        # replayed, so a bare path is two cells that cannot run -- and it read
        # like a plan that was configured.
        with pytest.raises(SystemExit) as raised:
            _run(["--root", "/r", "--replay-target", "/w/t.json"])
        assert "TP=PATH" in str(raised.value)

    def test_the_registry_resolves_a_target_of_the_right_width(self):
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a"]))
        seen = {}
        for cell in got["cells"]:
            for step in _role(cell, "serve", "modelled"):
                command = step["command"]
                seen[cell["tp"]] = command[
                    command.index("--compass-replay-target") + 1
                ]
        assert seen == {
            tp: plan_mod.registry.replay_target(tp, "/a") for tp in plan_mod.TPS
        }
        assert len(set(seen.values())) == 3

    def test_every_width_is_sized_from_the_analytical_profile(self):
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve", "modelled"):
                command = step["command"]
                assert command[command.index("--compass-memory-model") + 1] == (
                    f"/a/memval/capture_replay/profile/profile.tp{cell['tp']}.json"
                )

    def test_the_real_side_is_never_handed_either(self):
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve", "real"):
                assert "--compass-memory-model" not in step["command"]
                assert "--compass-replay-target" not in step["command"]

    def test_both_sides_carry_the_registered_engine_configuration(self, plan):
        for cell in plan["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                for arg in plan_mod.ENGINE_ARGS:
                    assert arg in command
                assert "--no-enable_prefix_caching" in command
                assert command[command.index("-tp") + 1] == str(cell["tp"])

    def test_each_side_replays_the_registered_workload_for_its_class(self, plan):
        for cell in plan["cells"]:
            expected = f"atom/compass/cc_traces_{cell['class']}.jsonl"
            replays = _role(cell, "replay")
            assert replays
            for step in replays:
                assert expected in step["command"]
                assert (ROOT / expected).exists()

    def test_the_repeat_count_is_the_protocol_s_on_both_sides(self, plan):
        for cell in plan["cells"]:
            for side in ("real", "modelled"):
                assert len(_role(cell, "replay", side)) == plan["repeats"]
                assert plan["repeats"] == plan_mod.REPEATS

    def test_an_oracle_option_reaches_only_the_modelled_side(self):
        got = json.loads(
            _run(["--root", "/r", "--oracle", "O", "--oracle-option", "library=/w/l"])
        )
        steps = _by_id(got["cells"][0])
        assert "--compass-oracle-option" in steps["serve-modelled-1"]["command"]
        assert "library=/w/l" in steps["serve-modelled-1"]["command"]
        assert "--compass-oracle-option" not in steps["serve-real-1"]["command"]


class TestTheNodeIsWatchedForTheWholeWindow:
    def test_a_baseline_sample_is_taken_before_any_server(self, plan):
        """The only sample that can show a card was already somebody else's."""
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            assert ids.index("sample-baseline") < ids.index("serve-real-1")
            baseline = _by_id(cell)["sample-baseline"]["command"]
            assert baseline[baseline.index("--phase") + 1] == "baseline"
            assert "--once" in baseline

    def test_the_sampler_runs_across_every_real_repeat(self, plan):
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            assert ids.index("sample") < ids.index("serve-real-1")
            assert ids.index(f"stop-real-{plan['repeats']}") < ids.index("stop-sample")
            assert _by_id(cell)["sample"]["background"] is True

    def test_the_sampler_writes_what_the_audit_reads(self, plan):
        for cell in plan["cells"]:
            for name in ("sample", "sample-baseline"):
                command = _by_id(cell)[name]["command"]
                assert "scripts/compass/gpu_sampler.py" in command
                assert f"{cell['cell']}/gpu.jsonl" in command
            audit = _by_id(cell)["isolation"]["command"]
            assert f"{cell['cell']}/gpu.jsonl" in audit


class TestTheOrderIsTheProtocolS:
    def test_the_device_free_probe_happens_before_the_verdict(self, plan):
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            assert ids.index("gpu-free") < ids.index("validate")

    def test_the_probe_runs_where_the_prediction_ran(self, plan):
        """Taken in the container that served the modelled side; taken on the
        leased node it would say nothing about that container."""
        for cell in plan["cells"]:
            steps = _by_id(cell)
            assert steps["gpu-free"]["where"] == "device_free"
            assert steps["gpu-free"]["where"] == steps["serve-modelled-1"]["where"]

    def test_the_isolation_audit_follows_the_measured_window(self, plan):
        ids = [s["id"] for s in plan["cells"][0]["steps"]]
        assert ids.index("sample") < ids.index("replay-real-1")
        assert ids.index("replay-real-1") < ids.index("isolation")

    def test_costs_are_merged_before_the_verdict_reads_them(self, plan):
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            assert ids.index("costs") < ids.index("validate")
            assert ids.index(f"replay-modelled-{plan['repeats']}") < ids.index("costs")

    def test_the_verdict_is_the_last_thing_in_a_cell(self, plan):
        for cell in plan["cells"]:
            assert cell["steps"][-1]["id"] == "validate"
            assert cell["steps"][-1]["where"] == "cpu"

    def test_nothing_that_reads_artifacts_asks_for_a_device(self, plan):
        for cell in plan["cells"]:
            for step in cell["steps"]:
                if step["id"] in ("validate", "isolation", "verify-workload", "stamp"):
                    assert step["where"] == "cpu"


class TestItSaysWhatItDoesNotProvide:
    def test_a_step_with_no_command_says_who_provides_it(self, plan):
        for cell in plan["cells"]:
            for step in cell["steps"]:
                if not step.get("command"):
                    assert step["provided_by"]

    def test_the_gaps_are_carried_in_the_plan_itself(self, plan):
        assert plan["gaps"]
        assert any("target record of its own width" in gap for gap in plan["gaps"])
        assert any("memory profile" in gap for gap in plan["gaps"])
        assert any("oracle" in gap for gap in plan["gaps"])
        assert any("capture, calibration, derivation" in gap for gap in plan["gaps"])

    def test_the_evidence_list_matches_what_the_validator_requires(self, plan):
        """A plan that forgets an artifact is a cell refused after the lease."""
        assert validate.GPU_FREE_EVIDENCE in plan["evidence"]
        assert "costs.json" in plan["evidence"]
        assert "isolation.json" in plan["evidence"]
        assert "cc_traces_protocol.json" in plan["evidence"]
        assert "registry.json" in plan["evidence"]
        assert "gpu.jsonl" in plan["evidence"]

    def test_it_does_not_claim_to_have_run_anything(self, plan):
        assert "no result is claimed here" in plan["means"]


class TestWhatItPrints:
    def test_the_shell_rendering_is_the_same_plan(self, tmp_path):
        text = _run(["--root", "/r", "--shell"])
        assert "cc_traces_validate.py cell /r/tp4_long" in text
        assert "[device_free]" in text and "[gpu]" in text
        assert "what this plan does not provide" in text

    def test_the_json_can_be_written_out(self, tmp_path):
        out = tmp_path / "plan.json"
        assert plan_mod.main(["--root", "/r", "--out", str(out)]) == 0
        assert json.loads(out.read_text())["cells"]


class TestTheServerListensWhereTheHarnessLooks:
    """A server that binds a different port than the plan polls is a hang.

    The api_server parser carries two ports: `--server-port` is the HTTP
    listener, and `--port` is the engine's internal port. Naming the wrong one
    costs a whole lease before anyone sees why.
    """

    def test_the_listener_port_is_the_one_the_health_check_polls(self, plan):
        for cell in plan["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                assert "--server-port" in command, command
                port = command[command.index("--server-port") + 1]
                assert step["health"] == f"http://127.0.0.1:{port}/health"

    def test_the_engine_internal_port_is_not_given_the_listener_port(self, plan):
        """`--port` is the engine's own rendezvous port; setting it to the
        listener's is what left the listener on its default. It is named, at
        its own value, rather than left to the engine to pick."""
        for cell in plan["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                listener = command[command.index("--server-port") + 1]
                rendezvous = command[command.index("--port") + 1]
                assert rendezvous != listener
                assert rendezvous == str(plan_mod.ENGINE_PORT)

    def test_the_replay_client_still_dials_with_port(self, plan):
        """The client's `--port` is the address it connects to, not a server
        flag, and it stays."""
        for cell in plan["cells"]:
            for step in _role(cell, "replay"):
                command = step["command"]
                port = command[command.index("--port") + 1]
                serve = _role(cell, "serve", step["side"])[0]["command"]
                assert port == serve[serve.index("--server-port") + 1]

    def test_the_chosen_port_reaches_both_sides(self):
        got = json.loads(_run(["--root", "/r", "--port", "53013"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                assert command[command.index("--server-port") + 1] == "53013"

    def test_the_flag_is_the_one_the_entry_point_binds(self):
        """Cross-check against the server itself, so a rename there is not a
        silent hang here."""
        source = (
            ROOT / "atom" / "entrypoints" / "openai" / "api_server.py"
        ).read_text()
        assert "port=args.server_port," in source
        assert '"--server-port",' in source


class TestTheTwoPortsAreChosenSeparately:
    """The listener and the engine's rendezvous port are different sockets.

    Only the listener was ever named. The rendezvous port was left at the
    engine's own default, so every server on a host asked for the same one:
    two cells alive at once collide on it, and at TP>1 the collision is
    between rendezvous groups rather than between binds.
    """

    def test_each_scope_is_named_with_the_flag_that_carries_it(self, plan):
        for cell in plan["cells"]:
            for step in _role(cell, "serve"):
                ports = step["ports"]
                command = step["command"]
                assert set(ports) == {"http_listener", "engine_rendezvous"}
                for scope in ports.values():
                    flag = scope["flag"]
                    assert command[command.index(flag) + 1] == str(scope["port"])
                assert (
                    ports["http_listener"]["port"] != ports["engine_rendezvous"]["port"]
                )

    def test_the_rendezvous_port_can_be_chosen_per_run(self):
        got = json.loads(_run(["--root", "/r", "--engine-port", "53117"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                assert command[command.index("--port") + 1] == "53117"
                assert step["ports"]["engine_rendezvous"]["port"] == 53117

    def test_one_socket_cannot_serve_both_scopes(self):
        with pytest.raises(SystemExit) as raised:
            plan_mod.cell_steps(
                2,
                "long",
                root="/r",
                oracle=None,
                options=(),
                port=8006,
                engine_port=8006,
                repeats=1,
            )
        assert "8006" in str(raised.value)

    def test_the_engine_flag_is_the_one_the_engine_parses(self):
        """Cross-checked against the engine, so a rename there is caught here
        rather than by two runs quietly sharing a rendezvous group."""
        source = (ROOT / "atom" / "model_engine" / "arg_utils.py").read_text()
        assert '"--port",' in source
        assert f"default={plan_mod.ENGINE_PORT}," in source
        runner = (ROOT / "atom" / "model_engine" / "model_runner.py").read_text()
        assert 'os.environ["MASTER_PORT"] = str(self.config.port)' in runner


class TestTheRegistryIsWhereTheConfigurationComesFrom:
    """`--artifact-root` in place of a dozen hand-typed options.

    The modelled side's oracle configuration was free text on the command
    line: one qualname and twelve `KEY=VALUE` strings, none of them reachable
    by a test. An artifact resolved from the wrong width prices a different
    deployment and the run says nothing about it, so the set is data now and
    the plan reads it.
    """

    def test_naming_a_root_configures_every_modelled_server(self):
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a"]))
        for cell in got["cells"]:
            tp = cell["tp"]
            for step in _role(cell, "serve"):
                command = step["command"]
                if "--compass-oracle" not in command:
                    continue
                assert (command[command.index("--compass-oracle") + 1]
                        == plan_mod.registry.ORACLE)
                supplied = [command[i + 1] for i, token in enumerate(command)
                            if token == "--compass-oracle-option"]
                assert supplied == plan_mod.registry.options(tp, "/a")

    def test_the_real_side_is_given_no_oracle_at_all(self):
        # It is the thing being predicted. A price list on that side would be
        # an input from the engine under test.
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve"):
                if "--compass-mode" not in step["command"]:
                    continue
                command = step["command"]
                mode = command[command.index("--compass-mode") + 1]
                if mode == "measure":
                    assert "--compass-oracle" not in command
                    assert "--compass-rank-aggregation" not in command

    def test_a_typed_option_still_wins_over_the_registry(self):
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a",
                               "--oracle-option", "derive=0"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                if "--compass-oracle-option" not in command:
                    continue
                supplied = [command[i + 1] for i, token in enumerate(command)
                            if token == "--compass-oracle-option"]
                assert supplied == ["derive=0"]

    def test_without_a_root_nothing_is_invented(self):
        # The plan has always been allowed to print a cell whose oracle the
        # operator supplies later. Naming no root keeps that, rather than
        # silently resolving paths under a default that does not exist.
        got = json.loads(_run(["--root", "/r"]))
        for cell in got["cells"]:
            for step in _role(cell, "serve"):
                assert "--compass-oracle" not in step["command"]

    def test_the_aggregation_is_the_registry_s_and_not_a_second_copy(self):
        assert plan_mod.registry.RANK_AGGREGATION == "slowest"
        got = json.loads(_run(["--root", "/r", "--artifact-root", "/a"]))
        seen = 0
        for cell in got["cells"]:
            for step in _role(cell, "serve"):
                command = step["command"]
                if "--compass-rank-aggregation" not in command:
                    continue
                seen += 1
                assert (command[command.index("--compass-rank-aggregation") + 1]
                        == plan_mod.registry.RANK_AGGREGATION)
        assert seen == 6 * got["repeats"]

    def test_native_allocation_is_selected_at_every_width(self):
        # `carry_allocation=1` reuses the template's blocks and declares them
        # unmeasured. It is inadmissible for acceptance, and the way it would
        # arrive is by being typed once and never noticed again.
        for tp in plan_mod.TPS:
            options = plan_mod.registry.options(tp, "/a")
            assert "allocation=native" in options
            assert not any(o.startswith("carry_allocation=") for o in options)

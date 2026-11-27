# SPDX-License-Identifier: MIT
"""A pipelined prefill that offloads to a Mooncake Store hands in its Store
counters from its head stage alone.

A later stage's `StoreShell` is inert (the head's keeps the ranks) and states
no counters, so its engine hands in no ``offload``, and the run summary reads
the head's. Before, each later stage handed in its prefix cache's counters
alone, and the summary's read of their connector counters ended the authority
before it wrote ``summary.json``.
"""

from types import SimpleNamespace

from test_run_speculative_summary import _run

from atom.compass import run as compass_run
from atom.model_engine.engine_stats import EngineStats
from atom.utils import clock
from atom.utils.clock import LPRuntime

#: The head's Store shell's counters, as `StoreShell.offload_statistics` gives them.
HEAD = {
    "connector": {"loaded_tokens": 256, "saved_tokens": 512},
    "copies": {"load": {"copies": 1}, "save": {"copies": 2}},
    "store": {"lookups": 3},
}


def _core(stated: dict):
    """An engine whose scheduler holds a Store shell stating `stated`."""
    shell = SimpleNamespace(offload_statistics=lambda: stated)
    stats = EngineStats()
    scheduler = SimpleNamespace(engine_stats=stats, kv_connector=shell)
    return SimpleNamespace(runner_mgr=None, scheduler=scheduler)


def test_a_later_stage_s_inert_store_shell_hands_in_no_offload(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path)
    a = compass_run._RecordingAuthority(run)
    monkeypatch.setattr(compass_run, "_workers_say", lambda core: {"refusals": []})
    said = {}
    for name, stated in (("engine-P-pp0", HEAD), ("engine-P-pp1", {})):
        rt = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
        rt.me = SimpleNamespace(name=name)
        monkeypatch.setattr(clock, "_installed", rt)
        monkeypatch.setattr(
            compass_run,
            "_hand_in",
            lambda run, lp, reasons, client_run=None, **sections: said.update(
                {lp: {"refusals": reasons, **sections}}
            ),
        )
        compass_run._engine_hand_in(run, _core(stated), {})
    assert "offload" not in said["engine-P-pp1"]
    assert said["engine-P-pp0"]["offload"]["connector"] == HEAD["connector"]

    out = compass_run._summary(run, a, a, 0.0, said, {})

    assert set(out["offload"]) == {"engine-P-pp0"}
    assert out["offload"]["engine-P-pp0"]["connector"]["loaded_tokens"] == 256

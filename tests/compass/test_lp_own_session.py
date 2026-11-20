# SPDX-License-Identifier: MIT
"""LP `lmcache` starts in a session of its own: `compass run` stops the server
that started it with a SIGTERM to that server's process group, and a stop that
reached the LP before its ``+inf`` grant would leave the run with no hand-in of
it, so its authority would write no summary (#3809's 3P4D C1228 run lost LP
mooncake_store's, started the same way, at its stop)."""

import os
import time

from test_lmcache_lp import CHUNK, _config, _run

from atom.compass import run as compass_run


def test_lp_lmcache_runs_in_a_session_the_server_s_stop_does_not_reach(
    tmp_path, monkeypatch
):
    sid = tmp_path / "sid"
    # The LP's process as `_launch_lmcache` starts it, writing its session.
    main = f"import os; open({str(sid)!r}, 'w').write(str(os.getsid(0)))"
    monkeypatch.setattr(compass_run, "LMCACHE_MAIN", main)
    compass_run._launch_lmcache(
        _config(kv_cache_block_size=CHUNK // 2), _run(tmp_path), None
    )
    deadline = time.monotonic() + 60
    while not (sid.exists() and sid.read_text()):
        assert time.monotonic() < deadline, "the LP's process wrote no session"
        time.sleep(0.05)
    assert int(sid.read_text()) != os.getsid(0)

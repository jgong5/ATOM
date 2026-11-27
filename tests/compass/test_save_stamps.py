# SPDX-License-Identifier: MIT
"""COMPASS_SAVE_STAMPS=1 (#3720): each KV offload copy stamped at its submit, start and end."""

import json
import time
import types
from concurrent.futures import ThreadPoolExecutor

from atom.compass.parity import save_stamps


class Mixin:
    """`OffloadWorkerMixin` as the stamps reach it: executors made by `_init_worker_common`, jobs via `_guard`."""

    def _init_worker_common(self):
        self._save_executor = ThreadPoolExecutor(max_workers=1)
        self._load_executor = ThreadPoolExecutor(max_workers=1)
        self.rank = 2

    def _guard(self, kind, fn, req, **kwargs):
        return fn(req, **kwargs)


def test_off_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(save_stamps.ENV, raising=False)
    init = Mixin._init_worker_common
    assert not save_stamps.install(tmp_path, Mixin)
    assert Mixin._init_worker_common is init


def test_each_copy_is_stamped_at_submit_start_and_end(monkeypatch, tmp_path):
    monkeypatch.setenv(save_stamps.ENV, "1")
    init = Mixin._init_worker_common
    try:
        assert save_stamps.install(tmp_path, Mixin) and save_stamps.install(tmp_path, Mixin)
        c = Mixin()
        c._init_worker_common()
        req = types.SimpleNamespace(req_id=7)
        slow = lambda r, producer_event=None: time.sleep(0.02) or ("saved", r.req_id)  # noqa: E731
        f1 = c._save_executor.submit(c._guard, "save", slow, req, producer_event="ev")
        f2 = c._save_executor.submit(c._guard, "save", slow, types.SimpleNamespace(req_id=8))
        assert f1.result() == ("saved", 7) and f2.result() == ("saved", 8)
        c._save_executor.shutdown(wait=True)
        lines = [json.loads(x) for x in next((tmp_path / "save_stamps").glob("*.jsonl")).read_text().splitlines()]
        a, b = sorted(lines, key=lambda d: d["req"])
        assert (a["kind"], a["req"], a["rank"]) == ("save", "7", 2)
        assert a["t_submit_ns"] <= a["t_start_ns"] < a["t_end_ns"]
        # The second waited behind the first on the one save worker: started after it ended.
        assert b["t_start_ns"] >= a["t_end_ns"] and b["t_submit_ns"] < a["t_end_ns"]
    finally:
        Mixin._init_worker_common = init

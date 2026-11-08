# SPDX-License-Identifier: MIT
"""`compass measure ops` on a DeepSeek-V4-Pro DPA-TP8 rank's decode, off the
card: its graph traced on a declared MI355X, each leaf taken through
`ops.measure` and `Stub.point` on ATOM's own runner, KV allocation and
metadata builder up to its launch (`v41_ops_rehearsal.py`), at one row of
1024 tokens.

V4-Pro's KV allocation binds its planes on the runner, and its builder caches
a view of each plane's rows for each state slot; a leaf bound to such a view
indexes past the plane from the view's base.
"""

import json
import shutil

import pytest
from test_bench_ops_v41_rehearsal import HARNESS, _run
from test_capture_v4_pro import DECODE, DPA_TP8, LAYERS, QKN, trace
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture

#: The decode attention and the leaf writing its rows: the slots each takes
#: of the NoPE plane and of the RoPE plane.
PLANES = {
    DECODE: ("kv_buffer", "kvrope"),
    QKN: ("swa_nope_scale_buff", "swa_rope_buff"),
}


@pytest.fixture(scope="module")
def rehearsed(model_dir, tmp_path_factory):  # noqa: F811
    """The rehearsal's record of each leaf of a decode of width 1."""
    root = tmp_path_factory.mktemp("v4pro_kv")
    done = trace(root, "--width 1" + DPA_TP8)
    assert done.returncode == 0, done.stderr[-4000:]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, root / "model" / name)
    point = ["--arch", "gfx950", "--width", "1", "--context", "1024"]
    point += ["--llc-mib", "256", "--store", root / "store"]
    atom = ["--model", root / "model", "-tp", "8", *DPA_TP8.split()[1:]]
    atom += ["--enforce-eager", "--level", "0"]
    _run(HARNESS, root / "got.json", *point, "--", *atom)
    return json.loads((root / "got.json").read_text())


def test_the_decode_attention_reads_the_planes_the_runner_bound(rehearsed):
    """Each such leaf takes its last layer's view of each plane, which the
    runner holds, not a state slot's view the builder cut from it."""
    leaves = [r for r in rehearsed if r["op"] in PLANES]
    assert {r["op"] for r in leaves} == set(PLANES)
    planes = f"v4_unified_kv.{LAYERS - 1}", f"v4_unified_kv_rope.{LAYERS - 1}"
    for r in leaves:
        assert tuple(r["bound"][slot] for slot in PLANES[r["op"]]) == planes

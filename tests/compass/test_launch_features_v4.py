# SPDX-License-Identifier: MIT
"""DeepSeek-V4's launch feature: a step split into two micro-batches.

A recording runner of a model ATOM wraps in its `UBatchWrapper` writes on each
line whether the wrapper ran the step's micro-batches, and `fit_host` holds
`predict` to it: the split `ForwardMode.decide` settles, less a decode that
replays a graph recorded split. On planted middle chunks a split launch costs
2.5 times a plain one: without the provider the JIT cut names every split,
with it none, and the fit adopts the column, which a run prices.
"""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_parity_device_window import _recorded
from test_parity_record import _config, _drive
from torch import nn

from atom.compass.backends.launch_features import GRAPHS, REGISTRY, stated
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import Sample, label
from atom.compass.backends.step_linear_fit import fit_host
from atom.compass.parity import ENV, read
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.runner.projection import forward_mode
from atom.utils.tbo import ubatch_wrapper
from atom.utils.tbo.ubatch_splitting import maybe_create_ubatch_slices
from atom.utils.tbo.ubatching import local_tbo_precompute

ARCH = "DeepseekV4ForCausalLM"
V4 = REGISTRY[ARCH]


class _Model(nn.Module):
    def forward(self, input_ids, positions):
        return "one batch"


@pytest.mark.parametrize("decode_graphs", [False, True])
def test_a_recording_runner_writes_each_split_atom_ran_and_the_fit_agrees(
    tmp_path, monkeypatch, decode_graphs
):
    """Three prompts of 200 tokens under a budget of 64, a prefill split from
    40 tokens; a decode of three requests splits, and replays a graph recorded
    split under `decode_graphs`."""
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "40")
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1, 2]})[0][0]
    recorded, device = _recorded(batches[-1])
    context = SimpleNamespace(ubatch_slices=None)
    monkeypatch.setattr(ubatch_wrapper, "get_forward_context", lambda: context)

    def run_model(self, input_ids, batch):
        """`prepare_model`'s split and `run_model`'s dispatch, as ATOM's."""
        mode, prefill = forward_mode(batch, self), batch.total_seqs_num_prefill
        if not mode.use_cudagraph:
            context.ubatch_slices = None
            if mode.tbo_collective_active:
                context.ubatch_slices = maybe_create_ubatch_slices(
                    num_reqs=prefill or batch.total_seqs_num,
                    scheduled_tokens=batch.total_tokens_num,
                    is_prefill=bool(prefill),
                    num_scheduled_tokens=(
                        batch.num_scheduled_tokens if prefill else None
                    ),
                    force=True,
                )
            self.model(None, None)
        return "logits"

    def eligibility(self, batch):  # `_local_tbo_eligibility` off prefill CP
        prefill = batch.total_tokens_num_prefill > 0
        return local_tbo_precompute(
            self.config, batch, prefill, batch.num_scheduled_tokens
        )

    base = recorded.__mro__[-2]
    monkeypatch.setattr(base, "run_model", run_model)
    monkeypatch.setattr(base, "_local_tbo_eligibility", eligibility, raising=False)
    wrapper = ubatch_wrapper.UBatchWrapper(_Model())
    wrapper._run_ubatches = lambda *args: "micro-batches"  # its streams need a device
    monkeypatch.setattr(recorded, "model", wrapper, raising=False)
    config = _config(
        hf_config=SimpleNamespace(architectures=[ARCH]),
        enable_tbo=True,
        enable_tbo_decode=True,
    )
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, config)
    runner.tbo_decode_graphs = decode_graphs
    runner.host_key = {"engine": {"ladder": [1, 2, 4, 8]}}
    for batch in batches:
        runner.forward(batch)
    device.tick = 10**9
    runner.exit()

    lines = read(tmp_path / "real")[0]
    for line in lines:
        rows = line["rows"]
        if all(decode for *_, decode in rows):
            split = len(rows) > 2 and not decode_graphs
        else:
            split = sum(tokens for tokens, *_ in rows) >= 40
        assert line["launch"] == {"split": float(split)}, line
    assert {line["launch"]["split"] for line in lines} == {0.0, 1.0}
    replayed = [line for line in lines if "tbo_tokens" in line and line["rung"]]
    assert bool(replayed) == decode_graphs
    key = json.loads((tmp_path / "real" / "dp0.key.json").read_text())
    assert key["engine"][GRAPHS]["architecture"] == ARCH
    fit_host(label(str(tmp_path / "real" / "dp0.jsonl"))[0])  # each as it launched


def _middle(tokens, split):
    halves = ((tokens // 2,), (tokens - tokens // 2,))
    return BatchView(
        (RequestShape(tokens, 2 * tokens, False),),
        produces_output=False,
        tbo_tokens=halves if split else None,
    )


def test_split_launches_leave_the_jit_cut_and_their_column_is_priced():
    """Middle chunks whose launch is 120 ms and 1 us a token, 210 ms more when
    split, so 2.5 times a plain one at 16k tokens; a third split."""
    rng, samples = np.random.default_rng(3114), []
    key = {"engine": {GRAPHS: {"architecture": ARCH}}}
    for run in ("a", "b", "c"):
        for i in range(90):
            tokens = int(rng.integers(1024, 16384))
            view = _middle(tokens, i % 3 == 0)
            launch = V4.predict(view, {})
            w = 0.12 + 1e-6 * tokens + 0.21 * launch["split"]
            host = {"E": 1e-4, "W_pre": 1e-4, "W_launch": w * rng.uniform(0.99, 1.01)}
            provenance = {"source": "record", "run": run, "requests": [i]}
            samples.append(
                Sample(
                    i,
                    0,
                    view,
                    0.01,
                    None,
                    host,
                    False,
                    provenance | {"config": key},
                    launch=launch,
                )
            )
    bare = [
        dataclasses.replace(
            s, launch={}, provenance=s.provenance | {"config": {"engine": {}}}
        )
        for s in samples
    ]
    assert sum(map(len, fit_host(bare)[1]["jit_steps"].values())) == 90

    host, report = fit_host(samples)
    assert report["jit_steps"] == {}
    assert report["middle.W_launch"]["launch_columns"]["adopted"] == ["split"]
    assert host["columns"]["middle"]["W_launch"] == pytest.approx(
        {"split": 0.21}, rel=0.05
    )
    rec = HostDeviceRecurrence(
        host["laws"], (), lambda: 0.0, columns=host["columns"], launch=stated(key)
    )
    batch = SimpleNamespace(
        total_seqs_num_prefill=1,
        total_tokens_num=16384,
        total_seqs_num=1,
        produces_output=lambda: False,
        compass_launch=rec.predicted(_middle(16384, True)),
    )
    rec.charge(batch, 0.01, 0.0)
    assert rec.segments["W_launch"] == pytest.approx(0.12 + 16384e-6 + 0.21, rel=0.02)

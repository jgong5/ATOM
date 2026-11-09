# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's launch features: an eager body and an eager decoder tail.

`predict` is held to ATOM's own decisions: `ForwardMode.decide` picks the
prefill token-graph bucket and `running_bs`, and the decoder's `DecoderGraph`
replays its request graph or runs eagerly. A recording runner of a model shaped
as V4.1's writes what replayed on each line and the decoder's recorded buckets
in its key, and its labelled record passes `fit_host`'s check; on planted
samples the fit adopts both columns and a run prices them.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("triton")  # the decoder's tail gather is a Triton kernel

import test_parity_device_window
from test_parity_device_window import _recorded
from test_parity_record import _config, _drive
from test_parity_recurrence import _labelled, _law, record

from atom.compass.backends.launch_features import GRAPHS, REGISTRY, stated
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import Sample, label
from atom.compass.backends.step_linear_fit import fit_host
from atom.compass.parity import ENV, read, recorded_graphs, recurrence
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.runner.projection import batch_view, forward_mode
from atom.models.deepseek_v41 import decoder_graph
from atom.models.deepseek_v41.decoder_graph import DecoderGraph
from atom.utils.forward_context import ForwardMode
from atom.utils.region_graph import ReqGraph, TokenGraph

ARCH = "DeepseekV41ForCausalLM"
V41 = REGISTRY[ARCH]
#: #2218's recipe: its capture ladder, the token buckets to 3072, the
#: decoder's default request buckets.
LADDER = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 48, 64, 128], dtype=np.int32)
BUCKETS = (256, 512, 1024, 2048, 3072)
DECODER = (1, 2, 3, 4, 8, 16)


class _ReqGraph:
    """The decoder's `ReqGraph`, holding the buckets `recorded` names."""

    def __init__(self, recorded):
        self.recorded, self.inputs_ = sorted(recorded), (torch.zeros(1),)

    def has(self, key, phase=None):
        return key in self.recorded

    def keys(self):
        return list(self.recorded)

    def inputs(self, key, phase=None):
        return self.inputs_

    def replay(self, args, kwargs=None, key=None, phase=None):
        return "replayed"


def _decoder(monkeypatch, recorded):
    """ATOM's `DecoderGraph`, its tail gather and eager decode stood in for."""
    monkeypatch.setattr(
        decoder_graph, "gather_rows", lambda rows, *t, dim, out=None: out or t
    )
    graph = DecoderGraph(SimpleNamespace(decoder_specs=(), parallel_config=None))
    graph.graph = _ReqGraph({(n, True) for n in recorded})
    graph._decode = lambda *args: "decoded"
    return graph


def _run_decoder(decoder, mode):
    """`DecoderGraph.run` on a replayed prefill's tail, which `lay_out_tail`
    lays out for the step's `running_bs` requests."""
    forward = SimpleNamespace(
        tr_graph="replay" if mode.token_graph else None,
        context=SimpleNamespace(forward_mode=mode),
        dp_metadata=None,
        attn_metadata=SimpleNamespace(
            step=SimpleNamespace(indptrs={}, tiles={}, selected={}, candidates={})
        ),
    )
    plan = SimpleNamespace(
        tail=SimpleNamespace(width=1, requests=(object(),)),
        tail_rows=None,
        last=torch.zeros(mode.running_bs),
    )
    return decoder.run(forward, plan, torch.zeros(1, 1), [])


def _prefill(lengths, output=True):
    lengths = np.asarray(lengths)
    return SimpleNamespace(
        total_seqs_num=len(lengths),
        total_seqs_num_prefill=len(lengths),
        total_tokens_num=int(lengths.sum()),
        total_tokens_num_prefill=int(lengths.sum()),
        num_scheduled_tokens=lengths,
        context_lens=lengths,
        num_spec_step=0,
        is_dummy_run=False,
        final_prefill_requests=lambda: len(lengths) if output else 0,
        produces_output=lambda: output,
    )


@pytest.fixture
def buckets(monkeypatch):
    """`decide` reads the recorded prefill buckets off `TokenGraph`."""
    sizes = {"prefill": BUCKETS}
    monkeypatch.setattr(
        TokenGraph, "captured_sizes", staticmethod(lambda p=None: sizes.get(p, ()))
    )


def _graphs(decoder=DECODER, buckets=BUCKETS):
    return {
        "architecture": ARCH,
        "token": {"prefill": list(buckets), "decode": []},
        "request": [],
        "decoder": None if decoder is None else [[n, True] for n in decoder],
    }


@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("seqs", [1, 2, 4, 5, 7, 8, 9, 16, 17])
@pytest.mark.parametrize("each", [16, 192, 384, 1500])
def test_predict_is_what_atom_decides_for_a_prefill(
    monkeypatch, buckets, output, seqs, each
):
    batch = _prefill([each] * seqs, output)
    runner = SimpleNamespace(
        config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    )
    mode = ForwardMode.decide(
        batch=batch,
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        step_needs_eager=False,
        capture_sizes=LADDER,
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=1,
        running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
    )
    tail = output and _run_decoder(_decoder(monkeypatch, DECODER), mode)
    atom = {
        "eager_body": float(not mode.prefill_token_graph),
        "eager_tail": float(tail == "decoded"),
    }
    assert V41.predict(batch_view(batch, mode, runner), _graphs()) == atom


def test_a_decode_and_a_model_that_does_not_replay_its_decoder_run_no_tail():
    decode = BatchView((RequestShape(6, 900, True),) * 5, produces_output=True)
    prefill = BatchView((RequestShape(100, 100, False),) * 5, produces_output=True)
    assert set(V41.predict(decode, _graphs()).values()) == {0.0}
    assert V41.predict(prefill, _graphs(decoder=None))["eager_tail"] == 0.0
    assert V41.predict(prefill, _graphs(decoder=()))["eager_tail"] == 1.0


class _TokenGraph:
    """The body's `TokenGraph`, replaying any bucket."""

    def replay(self, args=(), kwargs=None, key=None, phase=None):
        return "replayed"


class _Runtime(nn.Module):
    """V4.1's runtime model as the provider finds it: its body's token graph,
    made once ATOM records it, and its decoder."""

    def __init__(self, decoder):
        super().__init__()
        self.replay, self.decoder, self.early = True, decoder, nn.Module()

    def decoder_replay_active(self):
        return self.replay


def test_a_recording_runner_writes_what_replayed_and_the_decoder_buckets(
    tmp_path, monkeypatch
):
    """Under a budget of 64 tokens: a middle chunk of 64, a one-request output
    prefill of 26 and a two-request one of 50, against a 32-token bucket and a
    decoder that recorded a `running_bs` of 1 alone."""
    prompts = {0: 90, 1: 40, 2: 10}
    batches = _drive(
        tmp_path / "simulated",
        monkeypatch,
        {0: list(prompts)},
        prompt=lambda i: list(range(5 + 1000 * i, 5 + 1000 * i + prompts[i])),
    )[0][0]
    monkeypatch.setattr(
        TokenGraph,
        "captured_sizes",
        staticmethod(lambda p=None: (32,) * (p != "decode")),
    )
    runtime = _Runtime(_decoder(monkeypatch, (1,)))
    builder = SimpleNamespace(
        step_needs_eager=lambda batch: False, prepare_prefill=lambda *a: None
    )
    monkeypatch.setattr(test_parity_device_window, "GRAPHABLE", builder)
    recorded, device = _recorded(batches[-1])

    def run_model(self, input_ids, batch):
        """What ATOM runs: a prefill's metadata, its body's bucket, its tail."""
        if batch.total_seqs_num_prefill:
            self.attn_metadata_builder.prepare_prefill(batch)
            mode = forward_mode(batch, self)
            if mode.prefill_token_graph:
                vars(runtime.early)["_token_graph"].replay((), {}, phase="prefill")
            if batch.produces_output():
                _run_decoder(runtime.decoder, mode)
        return "logits"

    monkeypatch.setattr(recorded.__mro__[-2], "run_model", run_model)
    monkeypatch.setattr(recorded, "model", nn.ModuleList([runtime]), raising=False)
    config = _config(hf_config=SimpleNamespace(architectures=[ARCH]))
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, config)
    # ATOM's capture, after `install`: the decoder runs, and the body's token
    # graph is made.
    capture = SimpleNamespace(token_graph=False, req_graphs=False, running_bs=1)
    assert _run_decoder(runtime.decoder, capture) == "decoded"
    vars(runtime.early)["_token_graph"] = _TokenGraph()
    runner.host_key = {"engine": {"ladder": [1, 2, 4, 8]}}
    for batch in batches:
        runner.forward(batch)
    device.tick = 10**9
    runner.exit()

    lines = read(tmp_path / "real")[0]
    prefills = [s for s in lines if not all(r[2] for r in s["rows"])]
    assert [tuple(s["launch"].values()) for s in prefills] == [(1, 0), (0, 0), (1, 1)]
    key = json.loads((tmp_path / "real" / "dp0.key.json").read_text())
    assert key["engine"][GRAPHS]["decoder"] == [[1, True]]
    samples = label(str(tmp_path / "real" / "dp0.jsonl"))[0]
    fit_host(samples)  # every step predicted as it launched
    key["engine"][GRAPHS]["decoder"] = [[1, True], [2, True]]
    for s in samples:
        s.provenance["config"] = key
    with pytest.raises(ValueError, match="its launch provider predicts"):
        fit_host(samples)


def test_a_model_without_decoder_replay_states_no_decoder(monkeypatch):
    runtime = _Runtime(_decoder(monkeypatch, (1,)))
    runtime.replay = False
    runner = SimpleNamespace(model=nn.ModuleList([runtime]))
    assert V41.recorded(runner) == {"decoder": None}
    with pytest.raises(ValueError, match="holds no DeepSeek-V4.1 runtime model"):
        V41.install(SimpleNamespace(model=nn.Linear(1, 1)))


def test_the_recorded_request_graphs_hold_int_and_tuple_keys():
    """A DSpark draft keys its request graph by `running_bs`, a decode by
    `(requests, drafts)`."""
    graphs = ReqGraph(), ReqGraph()
    graphs[0]._buckets[(None, 4)] = graphs[1]._buckets[(None, (4, 5))] = None
    config = _config(hf_config=SimpleNamespace(architectures=[ARCH]))
    request = recorded_graphs(config)["request"]
    assert [4] in request and [(4, 5)] in request


def test_both_columns_are_adopted_and_priced(monkeypatch):
    """Output prefills whose launch is 12 ms and 1 us a token, 30 ms more on an
    eager body and 40 ms more on an eager tail."""
    rng, samples, graphs = np.random.default_rng(2958), [], _graphs()
    key = {"engine": {GRAPHS: graphs}}
    for run in ("a", "b", "c"):
        for i in range(90):
            seqs, each = int(rng.integers(1, 12)), int(rng.integers(64, 900))
            rung = int(LADDER[np.searchsorted(LADDER, seqs)])
            view = BatchView(
                (RequestShape(each, each, False),) * seqs,
                produces_output=True,
                target_bs=rung if rung > seqs else None,
            )
            launch = V41.predict(view, graphs)
            w = 12e-3 + 1e-6 * each * seqs
            w += 30e-3 * launch["eager_body"] + 40e-3 * launch["eager_tail"]
            host = {"E": 1e-4, "W_pre": 1e-4, "W_post": 1e-4}
            host["W_launch"] = w * rng.uniform(0.99, 1.01)
            provenance = {
                "source": "record",
                "run": run,
                "requests": [i],
                "config": key,
            }
            samples.append(
                Sample(i, 0, view, 0.01, None, host, False, provenance, launch=launch)
            )
    host, report = fit_host(samples)
    columns = host["columns"]["prefill"]["W_launch"]
    assert columns == pytest.approx(
        {"eager_body": 30e-3, "eager_tail": 40e-3}, rel=0.05
    )
    assert report["prefill.W_launch"]["launch_columns"]["adopted"] == list(columns)

    rec = HostDeviceRecurrence(
        host["laws"], (), lambda: 0.0, columns=host["columns"], launch=stated(key)
    )
    view = BatchView((RequestShape(700, 700, False),) * 5, produces_output=True)
    batch = SimpleNamespace(
        total_seqs_num_prefill=5,
        total_tokens_num=3500,
        total_seqs_num=5,
        produces_output=lambda: True,
        compass_launch=rec.predicted(view),
    )
    rec.charge(batch, 0.01, 0.0)
    law = np.dot(host["laws"]["prefill"]["W_launch"], (1, 3500, 5))
    assert rec.segments["W_launch"] == pytest.approx(law + 70e-3, rel=0.05)


def test_an_e_column_prices_a_busy_forward_and_leaves_an_idle_one_measured(tmp_path):
    """`record()`'s middle chunks of 50 tokens run past a 32-token bucket: an
    `E` column on an eager body moves the busy one's period 2 ms, and the first,
    idle, keeps its measured wait."""
    steps = record()
    samples, law = _labelled(tmp_path, steps), _law()
    bare = recurrence.priced(steps, samples, law)
    law["host"]["key"] = {"engine": {GRAPHS: _graphs(decoder=None, buckets=(32,))}}
    law["host"]["columns"] = {"middle": {"E": {"eager_body": 2e-3}}}
    priced = recurrence.priced(steps, samples, law)

    def residual(report):
        return {r[0]: r[3] for r in report["failures"]}

    assert residual(priced)[1] == pytest.approx(residual(bare).get(1, 0) + 2.0)
    assert 0 not in residual(priced)


def test_the_admission_law_is_fitted_past_the_e_column():
    """Output prefills whose `E` is 1 ms, 5 ms more on an eager body, and 2 ms
    and 100 ns an uncached token more where they admit a request."""
    rng, samples, graphs = np.random.default_rng(29580), [], _graphs()
    key = {"engine": {GRAPHS: graphs}}
    for run in ("a", "b", "c"):
        for i in range(120):
            each = int(rng.integers(64, 3000))
            view = BatchView(
                (RequestShape(each, each, False),) * 2, produces_output=True
            )
            launch = V41.predict(view, graphs)
            uncached = int(rng.integers(0, 8000)) * (i % 3 == 0)
            admitted = (
                (1, uncached, int(rng.integers(0, 4000))) if uncached else (0, 0, 0)
            )
            e = (
                1e-3
                + 5e-3 * launch["eager_body"]
                + (2e-3 + 1e-7 * uncached) * (uncached > 0)
            )
            host = {"E": e * rng.uniform(0.995, 1.005), "W_pre": 1e-4, "W_post": 1e-4}
            host["W_launch"] = 1e-2
            provenance = {
                "source": "record",
                "run": run,
                "requests": [i],
                "config": key,
            }
            samples.append(
                Sample(
                    i,
                    0,
                    view,
                    0.01,
                    None,
                    host,
                    False,
                    provenance,
                    admitted=admitted,
                    launch=launch,
                )
            )
    host, _ = fit_host(samples)
    assert host["columns"]["prefill"]["E"]["eager_body"] == pytest.approx(
        5e-3, rel=0.05
    )
    assert host["admit"][:2] == pytest.approx([2e-3, 1e-7], rel=0.1)

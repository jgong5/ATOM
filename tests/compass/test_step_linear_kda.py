# SPDX-License-Identifier: MIT
"""A KDA model's step linear law: the prefill vector's KDA state columns.

Kimi-K3's KDA layers carry a recurrent state per request, read and written on
every forward, saved at midstep checkpoints, forked off a prefix-cache hit and
moved between slots, and their prefill recurrence runs each row in chunks of
`KDA_CHUNK` tokens. `runner.projection.kda_counts` reads that work off the
scheduled batch for a real record and a simulated run alike, as the view's
`kda`, and `compass fit steps --model` adds the prefill vector's `KDA` columns
over it for a model with KDA layers.

The records are written from a known law whose prefill pays a cost per chunk,
`Sum ceil(N_Q / c)`: the fit gives back `tokens` with the chunks' share,
`c_chunk / c` a token, and `kda_pad_blocks` at `c_chunk`, the padding of each
row's last chunk being all `tokens` does not carry. A column of the chunks
themselves is that much collinear with `tokens`, which is why the law counts
the padding alone (the VIF test below). With the switch off the fit is the one
it was, byte for byte.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MockConfig, _MockHFConfig
from test_backend_step_linear_fit import DECODE, _engine_file, _record, _view
from test_parity_record import _drive

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_labels import digest
from atom.compass.backends.step_linear import KDA, KDA_CHUNK, kda_layers
from atom.compass.backends.step_linear_fit import _features, fit, label_records
from atom.compass.parity import read
from atom.compass.runner.projection import kda_counts

PREFILL = {
    "step": 0.02,
    "tokens": 6e-5,
    "query_square": 2e-10,
    "query_cached": 3e-10,
    "history": 2e-7,
}
#: Seconds per state row, midstep save, fork and move.
STATE = {"kda_seqs": 4e-4, "kda_saves": 2e-4, "kda_forks": 3e-4, "kda_moves": 1e-4}
#: Seconds per chunk of the prefill recurrence, `Sum ceil(N_Q / KDA_CHUNK)`.
CHUNK_S = 5e-5
#: Midstep checkpoints every this many tokens, as the recipe's prefix cache.
INTERVAL = 8192
#: Kimi-K3's config.json, as `test_pd_kda_slot.py` builds it.
KIMI_JSON = Path(__file__).with_name("kimi_k3_config.json")
#: Kimi-K3's model config: its text config's KDA layers.
KIMI = _MockHFConfig()
KIMI.text_config = _MockHFConfig()
KIMI.text_config.linear_attn_config = {"kda_layers": [1, 2, 3], "head_dim": 128}


def _chunks(rows) -> int:
    return sum(-(-q // KDA_CHUNK) for q, _, decode in rows if not decode)


def _steps(seed, n=240):
    """Alternating prefill and decode steps of a random shape, each with its
    `kda` counts and its price under `PREFILL`, `STATE` and `CHUNK_S`, a
    decode's under `DECODE`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if i % 2:
            rung = (None, 2, 4)[i % 3]
            width = 1 if rung is None else int(rng.integers(1, rung + 1))
            rows = [(1, int(rng.integers(1, 60000)), True) for _ in range(width)]
            c = DECODE["eager" if rung is None else str(rung)]
            view = _view(rows, rung)
            price = (
                c["step"]
                + c["context"] * sum(r[1] for r in rows)
                + c["graph_padding"] * view.graph_padding
            )
            out.append((rows, rung, price, (width, 0, 0, 0)))
            continue
        rows, saves, forks = [], 0, 0
        for _ in range(int(rng.integers(1, 4))):
            q = int(rng.integers(16, 8192))
            cached = int(rng.integers(0, 60000)) if rng.random() < 0.7 else 0
            rows.append((q, q + cached, False))
            saves += (q + cached) // INTERVAL - cached // INTERVAL
            forks += int(cached > 0 and rng.random() < 0.4)
        moves = int(rng.integers(0, 3)) if rng.random() < 0.3 else 0
        counts = (len(rows), saves, forks, moves)
        q = np.array([r[0] for r in rows])
        n_kv = np.array([r[1] for r in rows])
        price = (
            PREFILL["step"]
            + PREFILL["tokens"] * q.sum()
            + PREFILL["query_square"] * (q * q).sum()
            + PREFILL["query_cached"] * (q * (n_kv - q)).sum()
            + PREFILL["history"] * (n_kv - q).sum()
            + sum(c * n for c, n in zip(STATE.values(), counts))
            + CHUNK_S * _chunks(rows)
        )
        out.append((rows, None, price, counts))
    return out


def _kda_record(path, steps, kda=True):
    """`steps` as a real record, each line stating its `kda` counts with `kda`."""
    _record(path, [s[:3] for s in steps])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    if kda:
        for line, step in zip(lines, steps):
            line["kda"] = list(step[3])
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    _engine_file(path)
    return path


def _law(records, **options):
    samples = label_records(records)[0]
    return fit(samples, {"form": "history", **options})[0]


def test_the_fit_recovers_the_kda_state_work_and_the_chunks_padding(tmp_path):
    steps = _steps(0)
    law = _law([_kda_record(tmp_path / "dp0.jsonl", steps)], kda=True)

    prefill = law["prefill"]
    assert list(prefill["coefficients"]) == [*PREFILL, *KDA]
    assert prefill["kda_chunk"] == KDA_CHUNK
    expected = PREFILL | STATE | {"kda_pad_blocks": CHUNK_S}
    # The chunks' share of each token goes to `tokens`.
    expected["tokens"] += CHUNK_S / KDA_CHUNK
    # Each label is whole nanoseconds.
    assert prefill["coefficients"] == pytest.approx(expected, rel=1e-5)

    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, rung, seconds, counts in steps[:-1]:
        view = BatchView(
            tuple(RequestShape(*r) for r in rows), capture_rung=rung, kda=counts
        )
        assert backend.price(view).seconds == pytest.approx(seconds, rel=1e-6)
    priced = backend.price(
        BatchView((RequestShape(100, 100, False),), kda=(1, 0, 0, 0))
    )
    assert priced.terms[-1].name == "prefill.kda_pad_blocks"
    assert priced.terms[-1].seconds == pytest.approx(CHUNK_S * 28 / KDA_CHUNK, rel=1e-5)


def _vif(x: np.ndarray) -> np.ndarray:
    """Each column's variance inflation factor over the others and an intercept."""
    out = []
    for j in range(x.shape[1]):
        others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        fitted = others @ np.linalg.lstsq(others, x[:, j], rcond=None)[0]
        residual = ((x[:, j] - fitted) ** 2).sum()
        out.append(((x[:, j] - x[:, j].mean()) ** 2).sum() / residual)
    return np.array(out)


def test_the_padding_column_is_identifiable_where_the_chunks_are_not(tmp_path):
    record = _kda_record(tmp_path / "dp0.jsonl", _steps(1))
    rows = _features(label_records([record])[0], "history", 1, kda=True)[0]
    x = np.array([r[1][1:] for r in rows if r[0] == "prefill"], dtype=np.float64)
    names = [*list(PREFILL)[1:], *KDA]
    base = len(PREFILL) - 1

    # The collinearity gate: each column added at most 10, and none that was at
    # most 10 without them above it.
    without = dict(zip(names, _vif(x[:, :base])))
    padded = dict(zip(names, _vif(x)))
    print(f"\nVIF, base: {without}\nVIF, padded chunks: {padded}")
    assert max(padded[f] for f in KDA) <= 10.0
    assert all(padded[f] <= 10.0 for f, v in without.items() if v <= 10.0)
    # The chunks themselves, `tokens / c` plus that padding.
    pad = names.index("kda_pad_blocks")
    x[:, pad] += x[:, names.index("tokens")] / KDA_CHUNK
    chunks = dict(zip(names, _vif(x)))
    print(f"VIF, whole chunks: {chunks}")
    assert chunks["kda_pad_blocks"] > 1e3 and chunks["tokens"] > 1e3


def test_with_the_switch_off_the_law_is_the_one_without_counts(tmp_path):
    steps = _steps(2)
    plain = _kda_record(tmp_path / "plain.jsonl", steps, kda=False)
    counted = _kda_record(tmp_path / "counted.jsonl", steps)

    def law(record, **options):
        # The run is named by its file and bytes, which the counts move.
        text = json.dumps(_law([record], **options), sort_keys=True)
        return text.replace(digest(record), "<run>")

    off = law(plain)
    assert "kda" not in off
    assert law(counted) == off
    assert law(counted, kda=False) == off


def test_a_kda_law_refuses_a_step_and_a_record_stating_no_counts(tmp_path):
    steps = _steps(3)
    law = _law([_kda_record(tmp_path / "dp0.jsonl", steps)], kda=True)
    with pytest.raises(CostRefused, match="no KDA state counts on this step"):
        StepLinearBackend(StepLinearLaw(law)).price(_view([(100, 100, False)]))
    plain = _kda_record(tmp_path / "plain.jsonl", steps, kda=False)
    with pytest.raises(ValueError, match="states no kda counts"):
        _law([plain], kda=True)


def test_a_law_over_another_chunk_refuses(tmp_path):
    law = _law([_kda_record(tmp_path / "dp0.jsonl", _steps(4))], kda=True)
    law["prefill"]["kda_chunk"] = 64
    with pytest.raises(ValueError, match="chunk of 64 tokens"):
        StepLinearLaw(law)


def test_kda_layers_reads_the_text_configs_linear_attention_config():
    assert kda_layers(KIMI) == 3
    assert kda_layers(KIMI.text_config) == 3
    assert kda_layers(_MockHFConfig()) == 0
    assert kda_layers(None) == 0


def test_kda_counts_reads_the_scheduled_batchs_state_work():
    from atom.model_engine.state_runtime import StateMaintenanceOps

    runner = MockConfig(hf_config=KIMI)
    batch = MockConfig(
        # Rows 0 and 2 prefill and hold a state; row 3 decodes.
        num_scheduled_tokens=np.array([100, 64, 33, 1]),
        total_seqs_num_prefill=3,
        state_rows=[0, 2, 3],
        state_save_all=[[(5, 8192)], [], []],
        state_fork_srcs=[7, -1, -1],
        state_maintenance_ops=StateMaintenanceOps(
            relocations=((1, 2),), checkpoint_stores=("store",)
        ),
    )
    counted = kda_counts(batch, MockConfig(config=runner))
    assert counted == (3, 1, 1, 2)
    assert kda_counts(batch, MockConfig(config=MockConfig())) is None


@pytest.mark.usefixtures("unpriced")
@pytest.mark.parametrize("kimi", [True, False], ids=["kda", "none"])
def test_a_record_states_each_steps_kda_counts_for_a_kda_model(
    tmp_path, monkeypatch, kimi
):
    hf = {"hf_config": KIMI} if kimi else {}
    built = _drive(
        tmp_path,
        monkeypatch,
        {0: [0, 1]},
        per_req_cache=True,
        pool_entries={"state": 8},
        **hf,
    )
    lines = read(tmp_path)[0]
    assert len(lines) == len(built[0][0])
    for line in lines:
        if not kimi:
            assert "kda" not in line
            continue
        assert line["kda"] == [len(line["rows"]), 0, 0, 0]


def test_fit_steps_model_turns_the_columns_on_for_a_model_with_kda_layers(tmp_path):
    from test_memory_readings import CONFIG_JSON

    from atom.compass.fit.steps import _model_kda

    kimi, other = tmp_path / "kimi", tmp_path / "other"
    for directory, config in ((kimi, KIMI_JSON), (other, CONFIG_JSON)):
        directory.mkdir()
        (directory / "config.json").write_text(config.read_text())
    assert _model_kda(str(kimi)) and not _model_kda(str(other))


def test_a_data_parallel_dummy_runs_no_state_work_beside_a_kda_peer():
    from atom.compass.backends.step_linear import dummy_view, kda_columns

    peer = BatchView((RequestShape(300, 300, False),), kda=(1, 1, 0, 2))
    assert dummy_view(peer).kda == (0, 0, 0, 0)
    assert kda_columns(dummy_view(peer)) == (0, 0, 0, 0, 0.0)
    assert dummy_view(BatchView((RequestShape(300, 300, False),))).kda is None

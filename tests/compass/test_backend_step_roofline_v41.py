# SPDX-License-Identifier: MIT
"""The step roofline over DeepSeek-V4.1's modules and its DSpark drafter's:
sparse attention by layer mode, the index keys a compressing layer writes,
the fused Q and KV norm, a prefill that replays its decoder on each finishing
request's last rows, and the drafter's block, sampler and KV pass priced as
the step's draft terms.

Each is checked against a count by hand, on a machine with no kernel floor,
once memory-bound and once compute-bound. The published V4.1-Flash config at
TP4 under DSpark5, expert parallelism off and at EP4, builds on the meta
device with its drafter, every module classified, and prices on the MI455
spec; the build owns its process because the driverless declarations are
process-global.
"""

import collections
import json
import os
import pathlib
import subprocess
import sys

import pytest
from test_backend_step_linear import backend
from test_backend_step_linear import view as linear_view
from test_backend_step_roofline_v4 import BOUND, _module, machine, view

from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_roofline import (
    CONTEXT,
    Module,
    StepFallbackBackend,
    StepRooflineBackend,
)
from atom.compass.backends.work import Work, allgather, elementwise, gemm

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = TREE / "tests" / "models" / "deepseek_v41" / "fixtures" / "config.json"
MI455 = TREE / "scripts" / "compass" / "machines" / "mi455.json"


def _cost(modules, step, spec):
    cost = StepRooflineBackend(spec, modules).price(step)
    return {t.name: t.seconds for t in (*cost.terms, *cost.draft_terms)}


def prefill(*rows, output=True):
    shapes = tuple(RequestShape(q, c, False) for q, c in rows)
    return BatchView(shapes, produces_output=output)


@BOUND
def test_a_layer_reads_the_top_k_it_is_given_and_scores_at_most_its_candidates(
    spec,
):
    # Window 4, every 3 tokens compressed, the top 2 read: a layer reusing
    # another's choice scores nothing, one choosing again scores 3 heads of
    # 4 over at most 1 candidate entry, in fp8.
    reuse = Module("attn", "sparse_attention", (2, 8, 4, 3, 2, 0, 0), "bf16")
    reindex = Module("attn", "sparse_attention", (2, 8, 4, 3, 2, 3, 4, 1), "bf16")
    rows = [(3, 10, False), (1, 7, True)]
    keys = scored = 0
    for q, kv, _ in rows:
        for p in range(kv - q + 1, kv + 1):
            keys += min(p, 4) + min(p // 3, 2)
            scored += min(p // 3, 1)
    # Q read and O written in bf16; each row's window and its top 2 once.
    read = (4 + 2) + (4 + 2)
    attention = Work({"bf16": 4 * 2 * 8 * keys}, 4 * 2 * 8 * 4 + read * 8 * 2)
    # Each row's one candidate index key, at a byte.
    index = Work({"fp8": 2 * 3 * 4 * scored}, 2 * 4)

    assert _cost([reuse], view(*rows), spec)["roofline.sparse_attention"] == (
        pytest.approx(_module(attention, spec))
    )
    assert _cost([reindex], view(*rows), spec)["roofline.sparse_attention"] == (
        pytest.approx(_module(attention + index, spec))
    )


@BOUND
def test_index_keys_and_the_fused_norm_by_hand(spec):
    keys = Module("attn.indexer", "index_keys", (4, 8, 3), "fp8")
    norm = Module("attn.qk_norm", "norm", (12, 8))
    rows = [(3, 10, False), (1, 7, True)]
    # 10 // 3 - 7 // 3 entries completed by the first row, none by the second:
    # each projected from 8 wide to 4 in bf16, written at a byte.
    made = gemm(1, 4, 8) + elementwise(4, reads=0, fmt="fp8")
    # 12 elements a token read in bf16, 4 written in bf16 and 8 in fp8.
    fused = Work({}, 4 * (12 * 2 + 4 * 2 + 8))
    by_name = _cost([keys, norm], view(*rows), spec)

    assert by_name["roofline.index_keys"] == pytest.approx(_module(made, spec))
    assert by_name["roofline.norm"] == pytest.approx(_module(fused, spec))


#: A drafter of block width 3: its attention of 2 heads of 8 over a window of
#: 4, its LM head's vocabulary shard of 16, its sampler at rank 4 over two
#: ranks, and its KV pass's projection of the target's rows.
DRAFTER = [
    Module("draft.attn", "block_attention", (2, 8, 4), "bf16", draft=3),
    Module("draft.head", "gemm", (16, 8), draft=3),
    Module("draft.markov", "markov", (16, 4, 3), ranks=2, draft=3),
    Module("draft.kv", "gemm", (32, 8), "fp8", act="fp8", draft=CONTEXT),
]
TARGET = Module("proj", "gemm", (8, 8))


def _block(spec, contexts):
    """The drafter's block terms by hand, one block of 3 for each context."""
    keys = sum(3 * (min(c, 4) + 3) for c in contexts)
    q, held = 3 * len(contexts), sum(min(c, 4) for c in contexts)
    # Q, O and the block's keys in bf16, and the window before the block.
    attention = Work({"bf16": 4 * 2 * 8 * keys}, q * 5 * 8 * 2 + held * 8 * 2)
    # Each position: the shard's table and logits and the rows' Markov rows,
    # then each rank's (max, id) pairs gathered.
    rows = len(contexts)
    position = Work({"bf16": 2 * rows * 16 * 4}, (16 * 4 + rows * 16 + rows * 4) * 2)
    position += allgather(2, rows * 8 * 2)
    return {
        "draft roofline.block_attention": _module(attention, spec),
        "draft roofline.markov": _module(position + position + position, spec),
        "draft roofline.gemm": _module(gemm(q, 16, 8), spec),
    }


def _kv(tokens, spec):
    return _module(gemm(tokens, 32, 8, "fp8", "fp8"), spec)


@BOUND
def test_a_drafter_prices_its_block_and_its_kv_pass_apart_from_the_target(spec):
    # Two decode rows of 2, at contexts 3 and 9: the target and the KV pass
    # on their 4 tokens, a block of 3 for each row.
    step = view((2, 3, True), (2, 9, True))
    by_name = _cost([TARGET, *DRAFTER], step, spec)
    drafted = _block(spec, [3, 9])
    drafted["draft roofline.gemm"] += _kv(4, spec)

    assert by_name.pop("roofline.gemm") == pytest.approx(_module(gemm(4, 8, 8), spec))
    assert by_name == pytest.approx(drafted)


@BOUND
def test_a_prefill_drafts_one_block_and_a_middle_chunk_drafts_none(spec):
    # Which rows finish their prompts a view does not say: one block, at the
    # shortest context. The KV pass runs on every row of either.
    drafted = _block(spec, [9])
    drafted["draft roofline.gemm"] += _kv(8, spec)
    idle = _cost(DRAFTER, prefill((5, 20), (3, 9), output=False), spec)

    assert _cost(DRAFTER, prefill((5, 20), (3, 9)), spec) == pytest.approx(drafted)
    assert idle == pytest.approx(
        {
            "draft roofline.block_attention": 0.0,
            "draft roofline.markov": 0.0,
            "draft roofline.gemm": _kv(8, spec),
        }
    )


#: Layer 0 before the decoder, layer 1 replayed on its last 4 rows, layer 2
#: on the last row; attention of 2 heads of 8 over a window of 4.
REPLAYED = [
    Module("layers.0.proj", "gemm", (8, 8), layer=0),
    Module("layers.1.proj", "gemm", (8, 16), layer=1, replay=4),
    Module("layers.2.proj", "gemm", (8, 32), layer=2, replay=1),
    Module(
        "layers.1.attn", "sparse_attention", (2, 8, 4, 0, 0, 0, 0), layer=1, replay=4
    ),
]


@BOUND
def test_a_prefill_replays_its_decoder_on_the_last_rows_of_a_finishing_row(spec):
    def gemms(rows, tail, last):
        return (
            _module(gemm(rows, 8, 8), spec)
            + _module(gemm(tail, 8, 16), spec)
            + _module(gemm(last, 8, 32), spec)
        )

    # The tail's 4 rows attend only each other: 1 + 2 + 3 + 4 keys, Q and O
    # in bf16, its 4 window rows read once.
    tail = Work({"bf16": 4 * 2 * 8 * 10}, 4 * 2 * 8 * 4 + 4 * 8 * 2)

    assert _cost(REPLAYED, prefill((10, 10)), spec) == pytest.approx(
        {
            "roofline.gemm": gemms(10, 4, 1),
            "roofline.sparse_attention": _module(tail, spec),
        }
    )
    # Of two rows, the fewer tokens: 3 rows, whichever of them finishes.
    two = _cost(REPLAYED, prefill((10, 10), (3, 30)), spec)
    assert two["roofline.gemm"] == pytest.approx(gemms(13, 3, 1))
    # A chunk that finishes no prompt runs no decoder; a decode step all of it.
    assert _cost(REPLAYED, prefill((10, 10), output=False), spec) == {
        "roofline.gemm": _module(gemm(10, 8, 8), spec)
    }
    decode = _cost(REPLAYED, view((2, 30, True)), spec)
    assert decode["roofline.gemm"] == pytest.approx(gemms(2, 2, 2))


def test_a_step_the_law_refuses_takes_the_drafters_terms_from_the_roofline():
    batch = linear_view((1, 10, True), rung=4)
    with pytest.raises(CostRefused) as refused:
        backend().price(batch)
    roofline = StepRooflineBackend(machine(1e18), [TARGET, *DRAFTER])
    cost = StepFallbackBackend(backend(), roofline).price(batch)
    alone = roofline.price(batch)

    assert [t.name for t in cost.draft_terms] == [t.name for t in alone.draft_terms]
    assert cost.draft_seconds == alone.draft_seconds > 0
    for term in cost.draft_terms:
        assert term.provenance.refusals == refused.value.declined


BUILD = """
import json, sys
ep, root = sys.argv[1] == "1", sys.argv[2]
from atom.compass.validate.bound import _build
args = ["--kv_cache_dtype", "bf16", "--index-cache-dtype", "fp8", "--method",
        "dspark", "--num-speculative-tokens", "5"]
args += ["--enable-expert-parallel"] * ep
config, tree = _build(root + "/model", 4, args, "gfx1250", root)
from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.run.modules import drafter, walk
from atom.compass.runner.overrides import FactoriesOnMeta
from atom.compass.spec import MachineSpec
held = set(config.compilation_config.static_forward_context)
draft = drafter(config)
kept = set(config.compilation_config.static_forward_context) == held
with FactoriesOnMeta():  # ATOM builds its own drafter next, of the same names
    type(draft)(config)
modules = walk(tree, draft)
spec = MachineSpec.from_mapping(json.loads(open(sys.argv[3]).read()))
roofline = StepRooflineBackend(spec, modules)
verify = RequestShape(6, 4096, True)
steps = {
    "verify-1": BatchView((verify,), produces_output=True),
    "verify-16": BatchView((verify,) * 16, produces_output=True),
    "prefill": BatchView((RequestShape(4096, 4096, False),), produces_output=True),
    "middle": BatchView((RequestShape(4096, 4096, False),), produces_output=False),
}
priced = {}
for name, step in steps.items():
    cost = roofline.price(step)
    priced[name] = [{t.name: t.seconds for t in terms}
                    for terms in (cost.terms, cost.draft_terms)]
found = [[m.draft, m.replay, m.kind, m.dims, m.fmt, m.act] for m in modules]
print(json.dumps({"kept": kept, "modules": found, "priced": priced}))
"""


@pytest.fixture(scope="module", params=[False, True], ids=["ep-off", "ep4"])
def built(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("v41")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    done = subprocess.run(
        [sys.executable, "-c", BUILD, str(int(request.param)), str(root), str(MI455)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return request.param, json.loads(done.stdout.splitlines()[-1])


def test_the_v41_flash_tp4_dspark_build_classifies_every_module(built):
    ep, done = built
    found = collections.Counter(
        (draft, replay, kind, tuple(dims), fmt, act)
        for draft, replay, kind, dims, fmt, act in done["modules"]
    )
    # Each module kind's count, the experts' by their first four dims.
    counted = collections.Counter()
    for (draft, replay, kind, dims, *_), n in found.items():
        counted[draft, replay, kind, dims[:4] if kind == "moe" else dims] += n
    attention = {k[1:]: n for k, n in counted.items() if k[2] == "sparse_attention"}
    owner = (512, 32, 128)
    moe = (384, 96, 6, 2304) if ep else (384, 384, 6, 640)
    draft_moe = (128, 32, 3, 2304) if ep else (128, 128, 3, 640)
    latents = ("gemm", (1792, 5120), "fp8", "fp8")

    assert done["kept"]
    assert not [m for m in found if m[2] == "unclassified"]
    # Two mixes a block: the first decoder layer's and the decoder's on the
    # window, the last layer's on its last row.
    assert counted[0, 0, "mhc", (4, 5120)] == 2 * 20
    assert counted[0, 128, "mhc", (4, 5120)] == 2 * 19
    assert counted[0, 1, "mhc", (4, 5120)] == 2
    # Window-only layers 0 and 1; the ratio-2 encoder, layers 2, 8 and 14
    # choosing the top 512 for the rest; layer 20 choosing for the decoder,
    # which chooses again at 24, 28, 32 and 36 among 2048 blocks of 8.
    assert attention == {
        (0, "sparse_attention", (16, 512, 128, 0, 0, 0, 0)): 2,
        (0, "sparse_attention", (16, 512, 128, 2, 512, 0, 0)): 15,
        (0, "sparse_attention", (16, 512, 128, 2, *owner)): 3,
        (128, "sparse_attention", (16, 512, 128, 1, *owner)): 1,
        (128, "sparse_attention", (16, 512, 128, 1, *owner, 16384)): 4,
        (128, "sparse_attention", (16, 512, 128, 1, 512, 0, 0)): 14,
        (1, "sparse_attention", (16, 512, 128, 1, 512, 0, 0)): 1,
    }
    # The compressing layers write index keys from every row, at fp8.
    assert found[0, 0, "index_keys", (128, 512, 2), "fp8", "bf16"] == 3
    assert found[0, 0, "index_keys", (128, 512, 1), "fp8", "bf16"] == 1
    # The target's, whatever rows they run on.
    target = collections.Counter()
    for (draft, _, kind, dims), n in counted.items():
        if not draft:
            target[kind, dims] += n
    # The native group-32 linears read fp8: the Q and KV latents' GEMM.
    assert sum(n for k, n in found.items() if k[0] == 0 and k[2:] == latents) == 40
    assert target["moe", moe] == 40
    # Each layer's Q and KV norms in one launch, and the final norm: the
    # block's run in its mixes, the compressor's and the indexer's in theirs.
    assert {k[1]: n for k, n in target.items() if k[0] == "norm"} == {
        (1792, 1280): 40,
        (5120,): 1,
    }
    assert not target["gemm", (128, 512)]
    # The drafter: three stages over blocks of 5 attending the window, the
    # LM head and the sampler on a quarter of the vocabulary, and its KV pass
    # on the target's rows, on a replayed prefill's window as the decoder.
    assert counted[5, 0, "mhc", (4, 5120)] == 6
    assert counted[5, 0, "moe", draft_moe] == 3
    assert found[5, 0, "block_attention", (16, 512, 128), "bf16", "bf16"] == 3
    assert found[5, 0, "markov", (32320, 256, 5), "bf16", "bf16"] == 1
    assert found[5, 0, "gemm", (32320, 5120), "bf16", "bf16"] == 1
    assert found[CONTEXT, 128, "gemm", (1536, 5120), "fp8", "fp8"] == 1
    assert found[CONTEXT, 128, "gemm", (5120, 15360), "fp8", "fp8"] == 1
    assert found[CONTEXT, 128, "norm", (5120,), "bf16", "bf16"] == 1


def test_the_v41_flash_tp4_dspark_steps_price_target_and_draft_apart(built):
    _, done = built
    priced = done["priced"]
    target = {name: sum(terms[0].values()) for name, terms in priced.items()}
    draft = {name: sum(terms[1].values()) for name, terms in priced.items()}

    for terms, drafts in priced.values():
        assert all(n.startswith("roofline.") for n in terms)
        assert drafts and all(n.startswith("draft roofline.") for n in drafts)
    assert 0 < target["verify-1"] < target["verify-16"]
    assert 0 < draft["verify-1"] < draft["verify-16"]
    # A middle chunk stops at the first decoder layer's compression: no
    # decoder, no LM head and no draft.
    assert target["middle"] < target["prefill"]
    assert "roofline.lm_head" not in priced["middle"][0]
    assert draft["middle"] == 0 < draft["prefill"]

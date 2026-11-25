# SPDX-License-Identifier: MIT
"""The forms the scenario tests cover, and how a case prices its steps (`08` D120).

A scenario case carries ``capability(name, level)``: the form `name` it runs,
at level 0 under the step stub or at level 1 under the step roofline alone,
the backends `cost_overrides` puts in its run file. A refused form has no case;
its entry names the test that asserts its refusal. A supported form with no case
yet, or a refused one with no refusal test yet, names the open issue that adds
it. `test_capability_matrix.py` holds the marks to this vocabulary, and
`scripts/compass/capability_table.py` prints the table from them. Target levels
live on the tracker (#2674), not here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

#: The mark's name; `tests/conftest.py` registers it.
MARK = "capability"
#: The levels a case may claim: 0 under the step stub, 1 under the step roofline.
LEVELS = (0, 1)
#: The backend of a level-1 case: the step roofline alone (D120 point 3).
ROOFLINE = "step_roofline"
#: The machine a level-1 case is priced on: a pit2 MI355X's spec document, the
#: GPU gate's host (D120 point 3), which states every field the roofline reads.
ROOFLINE_MACHINE = json.loads(
    (Path(__file__).parent / "machines" / "mi355x-gpu40.json").read_text()
)


@dataclass(frozen=True)
class Form:
    what: str
    refused: bool = False
    #: A refused form's refusal test, ``path::function``, without parameters.
    test: str = ""
    #: The open issue that adds a supported form's first case, or a refused
    #: form's refusal test; the change that adds it drops this.
    issue: str = ""


def refused(what: str, test: str = "", issue: str = "") -> Form:
    return Form(what, refused=True, test=test, issue=issue)


HERE = "tests/compass/"
FORMS = {
    "tp1": Form("one engine at TP 1"),
    "tp_above_1": Form("one engine above TP 1, its ranks one engine LP"),
    "pcp": Form("prefill context parallelism"),
    "dcp": Form("decode context parallelism"),
    "dp_attention": Form("DP-attention, each DP rank a member of one engine LP"),
    "multinode_dp": Form("multi-node DP, two process trees on one host"),
    "pd_1p1d": Form("one prefill and one decode deployment through atomesh"),
    "pd_dp": Form("1P1D at DP-attention width 2, by ATOM's dispatch or the router LP"),
    "pd_widths": Form("1P1D with each side at its own DP width"),
    "pd_xpyd": Form("several prefill and decode deployments"),
    "pd_prefill_pp": Form("a prefill side of two pipeline stages"),
    "pp": Form("pipeline parallelism"),
    "mtp": Form("MTP speculative decoding"),
    "dspark": Form("DSpark speculative decoding"),
    "eagle3": Form("MiniMax-M3 with its EAGLE3 drafter (`14` D87.1)"),
    "lmcache_mp": Form("KV offload to an lmcache MP server"),
    "mooncake_store": Form("KV offload to a Mooncake Store"),
    "rapidserve": Form("RapidServe's prefill and decode engines (#2598)"),
    "router_lp": Form("atomesh as the router LP: dp_sticky, random, round_robin"),
    "ep": Form("EP on a MoE model, no DP", issue="#2668"),
    "ep_dp": Form("DP-attention with EP on a MoE model", issue="#2668"),
    "tbo": Form("two-batch overlap", issue="#2668"),
    "pure_dp": Form("DP without DP-attention", issue="#2668"),
    "cuda_graph": Form("CUDA graph mode", issue="#2669"),
    "hbm_prefix_hit": Form("a prefix hit in HBM", issue="#2669"),
    "deepseek_v4": Form("DeepSeek-V4 sparse attention", issue="#2671"),
    "mla": Form("MLA (DeepSeek-R1)", issue="#2671"),
    "router_cache_aware": Form(
        "the router LP's cache_aware, with and without timed eviction"
    ),
    "router_prefix_hash": Form("the router LP's prefix_hash"),
    "images": refused(
        "a step with image tokens: no real backend prices a vision encoder",
        HERE + "test_image_preprocess.py::"
        "test_a_steps_image_tokens_are_refused_by_every_fitted_backend",
    ),
    "router_power_of_two": refused(
        "the router LP's power_of_two",
        HERE + "test_router_lp_acceptance.py::test_power_of_two_is_refused_by_name",
    ),
    "plugin": refused(
        "ATOM as a vLLM, SGLang or rtpllm plugin",
        HERE + "test_refused_bootstraps.py::test_a_plugin_is_refused",
    ),
    "diffusion": refused(
        "the diffusion engine",
        HERE + "test_refused_bootstraps.py::test_the_diffusion_engine_is_refused",
    ),
    "fake_eplb_dp": refused(
        "--fake-eplb above DP width 1",
        HERE + "test_dp_slice.py::test_fake_eplb_is_refused_at_dp_width_above_one",
    ),
    "kimi_k3": refused("Kimi-K3", issue="#2672"),
    "windowed_attention": refused("a model with windowed attention", issue="#2672"),
}


def capability(name: str, level: int = 0):
    """The mark of a scenario case of form `name` at `level`."""
    return getattr(pytest.mark, MARK)(name, level)


def cost_overrides(backend, machine) -> dict:
    """The run-file keys that price a scenario's steps by `backend` on
    `machine`: a step stub's coefficients, for a level-0 case, or `ROOFLINE`,
    for a level-1 case, the step roofline alone, which `run.runner` installs
    when the run file names no other backend. Its ``step_stub`` is None, which
    `test_vertical_slice._run_file` leaves out."""
    if backend == ROOFLINE:
        return {"step_stub": None, ROOFLINE: True, "machine": machine}
    return {"step_stub": backend, "machine": machine}

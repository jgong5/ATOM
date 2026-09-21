# SPDX-License-Identifier: MIT
"""Where a recorded graph applies, decided against a step.

Two things are being defended here and they pull in opposite directions.

**A step outside the record has to be refused, by name.** A trace records one
path. Priced against a step that went down another one it returns a number with
no warning attached, which is worse than returning nothing: the prior flat
operator list answered for exactly the batch it was traced at while appearing to
answer for all of them. So the tests below check the refusals as closely as the
admissions, and check that a refusal names the condition and what the step
offered -- a bare False is a detection nobody can act on.

**Deciding must not change the record.** The guards are read off a live tracing
run, and `int()` on a symbolic size, or a Python comparison against one, asks
that run a question and it answers by installing a guard. A predicate written
with either idiom would narrow the graph it was reading every time it was read.
That is not asserted here in a comment; the wrong idiom is executed and the
guard it installs is counted, beside the right idiom that installs none.
"""

import pytest
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

from atom.compass.ir import (
    Applicability,
    Graph,
    GuardedApplicability,
    NodeKind,
    Op,
    Verdict,
)

DECODE_KEY = {
    "mode": "decode",
    "replayed": True,
    "has_cached": True,
    "produces_output": True,
    "spec_width": 0,
    "tbo": False,
    "attention_backend": "aiter",
    "dummy_run": False,
}
PREFILL_KEY = {**DECODE_KEY, "mode": "prefill", "replayed": False}


def _op():
    return Op(
        name="aiter::gemm_a16w16",
        kind=NodeKind.CAPTURED,
        in_shapes=((4, 8),),
        out_shapes=((4, 16),),
    )


def _traced(hint):
    """One tracing run that branches on its token count, as the model does.

    The branch is the whole point: `tokens > 128` is what separates the prefill
    path from the decode path, and taking it is what leaves a guard behind for
    the record to be built out of.
    """
    shape_env = ShapeEnv()
    source = ConstantSource("tokens")
    symbol = shape_env.create_symbol(
        hint, source=source, dynamic_dim=DimDynamic.DYNAMIC
    )
    tokens = shape_env.create_symintnode(symbol, hint=hint, source=source)
    key = PREFILL_KEY if tokens > 128 else DECODE_KEY
    return shape_env, str(symbol), tokens, key


def _recorded(hint):
    shape_env, name, tokens, key = _traced(hint)
    return shape_env, name, tokens, GuardedApplicability.from_shape_env(shape_env, key)


# --- the record is produced, not authored ------------------------------------


def test_the_two_paths_record_different_domains():
    decode_env, decode_name, _, decode = _recorded(17)
    prefill_env, prefill_name, _, prefill = _recorded(512)

    assert [str(guard) for guard in decode.guards] == [f"{decode_name} <= 128"]
    assert [str(guard) for guard in prefill.guards] == [f"{prefill_name} > 128"]
    assert dict(decode.ranges)[decode_name] == (2, 128)
    assert dict(prefill.ranges)[prefill_name][0] == 129
    assert dict(decode.key)["mode"] == "decode"
    assert dict(prefill.key)["mode"] == "prefill"
    assert len(decode_env.guards) == len(prefill_env.guards) == 1


def test_a_step_inside_the_domain_is_admitted():
    _, name, _, decode = _recorded(17)
    verdict = decode.decide(key=DECODE_KEY, sizes={name: 17})
    assert verdict == Verdict(True)
    assert verdict.applies and not verdict.reason


def test_a_step_outside_the_domain_is_refused_naming_guard_and_binding():
    _, name, _, prefill = _recorded(512)
    verdict = prefill.decide(key=PREFILL_KEY, sizes={name: 17})
    assert not verdict.applies
    assert f"{name} > 128" in verdict.reason
    assert f"{name}=17" in verdict.reason


def test_the_two_records_split_one_step_between_them():
    _, decode_name, _, decode = _recorded(17)
    _, prefill_name, _, prefill = _recorded(512)
    admitted = decode.decide(key=DECODE_KEY, sizes={decode_name: 17})
    refused = prefill.decide(key=PREFILL_KEY, sizes={prefill_name: 17})
    assert admitted.applies
    assert not refused.applies


# --- deciding installs no guard ----------------------------------------------


def test_converting_a_symbolic_size_installs_a_guard():
    shape_env, _, tokens, _ = _traced(17)
    before = len(shape_env.guards)
    assert int(tokens) == 17
    assert len(shape_env.guards) == before + 1
    assert any("Eq(" in str(guard.expr) for guard in shape_env.guards)


def test_comparing_a_symbolic_size_installs_a_guard():
    shape_env, name, tokens, _ = _traced(17)
    before = len(shape_env.guards)
    assert not bool(tokens > 64)
    assert len(shape_env.guards) == before + 1
    assert f"{name} <= 64" in [str(guard.expr) for guard in shape_env.guards]


def test_asking_the_tracing_run_for_a_hint_installs_no_guard():
    shape_env, _, tokens, _ = _traced(17)
    before = len(shape_env.guards)
    assert shape_env.size_hint(tokens.node.expr) == 17
    assert len(shape_env.guards) == before


def test_recording_and_deciding_leave_the_tracing_run_untouched():
    """The named result: the same run, decided against, guard count unmoved."""
    shape_env, name, _, key = _traced(17)
    before = len(shape_env.guards)
    recorded = GuardedApplicability.from_shape_env(shape_env, key)
    during = len(shape_env.guards)
    assert recorded.decide(key=DECODE_KEY, sizes={name: 17}).applies
    assert not recorded.decide(key=DECODE_KEY, sizes={name: 129}).applies
    assert before == during == len(shape_env.guards) == 1


def test_a_size_that_is_still_symbolic_is_refused():
    shape_env, name, tokens, recorded = _recorded(17)
    before = len(shape_env.guards)
    with pytest.raises(TypeError, match="installs a guard"):
        recorded.decide(key=DECODE_KEY, sizes={name: tokens})
    assert len(shape_env.guards) == before


def test_the_refusal_of_a_symbolic_size_names_how_to_resolve_it():
    _, name, tokens, recorded = _recorded(17)
    with pytest.raises(TypeError, match="Ask the tracing run for the hint"):
        recorded.decide(key=DECODE_KEY, sizes={name: tokens})


# --- the recorded range, which no branch asked about -------------------------


def test_a_size_below_the_recorded_range_is_refused_naming_both():
    _, name, _, decode = _recorded(17)
    verdict = decode.decide(key=DECODE_KEY, sizes={name: 1})
    assert not verdict.applies
    assert f"{name} in [2, 128]" in verdict.reason
    assert f"{name}=1" in verdict.reason


def test_an_unbounded_range_admits_what_no_guard_excludes():
    _, name, _, prefill = _recorded(512)
    assert prefill.decide(key=PREFILL_KEY, sizes={name: 100_000}).applies


def test_a_step_that_binds_nothing_the_record_names_is_refused():
    _, name, _, decode = _recorded(17)
    verdict = decode.decide(key=DECODE_KEY, sizes={"unrelated": 17})
    assert not verdict.applies
    assert name in verdict.reason
    assert "unrelated" in verdict.reason


def test_a_guard_over_an_unbound_symbol_is_refused_rather_than_skipped():
    _, name, _, decode = _recorded(17)
    only_guards = GuardedApplicability(key=DECODE_KEY, guards=decode.guards)
    verdict = only_guards.decide(key=DECODE_KEY, sizes={"other": 4})
    assert not verdict.applies
    assert f"{name} <= 128" in verdict.reason
    assert "does not bind" in verdict.reason


# --- the discrete key, which the guards cannot cover -------------------------


@pytest.mark.parametrize(
    "condition,value",
    [
        ("mode", "prefill"),
        ("replayed", False),
        ("has_cached", False),
        ("produces_output", False),
        ("spec_width", 3),
        ("tbo", True),
        ("attention_backend", "triton"),
        ("dummy_run", True),
    ],
)
def test_a_branch_that_left_no_guard_is_caught_by_the_key(condition, value):
    _, name, _, decode = _recorded(17)
    verdict = decode.decide(key={**DECODE_KEY, condition: value}, sizes={name: 17})
    assert not verdict.applies
    assert condition in verdict.reason
    assert repr(value) in verdict.reason
    assert repr(DECODE_KEY[condition]) in verdict.reason


def test_a_condition_neither_side_states_is_named_rather_than_ignored():
    _, name, _, decode = _recorded(17)
    thinner = {key: value for key, value in DECODE_KEY.items() if key != "tbo"}
    verdict = decode.decide(key=thinner, sizes={name: 17})
    assert not verdict.applies
    assert "tbo" in verdict.reason

    wider = {**DECODE_KEY, "pipeline_stage": 2}
    verdict = decode.decide(key=wider, sizes={name: 17})
    assert not verdict.applies
    assert "pipeline_stage" in verdict.reason


def test_a_key_is_the_same_key_whatever_order_it_was_written_in():
    reversed_key = dict(reversed(list(DECODE_KEY.items())))
    assert GuardedApplicability(key=reversed_key) == GuardedApplicability(
        key=DECODE_KEY
    )


def test_a_key_that_states_no_condition_is_refused():
    with pytest.raises(ValueError, match="admits every step"):
        GuardedApplicability(key={})


def test_a_key_value_that_cannot_be_held_in_a_value_is_refused():
    with pytest.raises(TypeError, match="hashable"):
        GuardedApplicability(key={"ctx": [1, 2, 3]})


# --- what the pieces refuse to be built as -----------------------------------


def test_a_guard_that_cannot_be_substituted_into_is_refused():
    with pytest.raises(TypeError, match="free symbols"):
        GuardedApplicability(key=DECODE_KEY, guards=("tokens > 128",))


def test_a_refusal_without_a_reason_is_itself_refused():
    with pytest.raises(ValueError, match="states which condition"):
        Verdict(False)
    with pytest.raises(ValueError, match="nothing to explain"):
        Verdict(True, "admitted, but")


def test_a_recorded_predicate_is_what_a_graph_is_built_with():
    _, _, _, decode = _recorded(17)
    assert isinstance(decode, Applicability)
    assert Graph(applicability=decode, region=_op()).applicability is decode

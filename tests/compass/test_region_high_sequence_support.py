"""High per-sequence support preserves every checked old TP1 component and band."""
from dataclasses import fields

import pytest

from atom.compass.core.cost.base import StepShape
from atom.compass.core.cost.regions import (
    SOURCE_27B_TP1_HISTORY_2M as OLD,
    SOURCE_27B_TP1_HISTORY_256K as NEW,
)


COUNTS = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 21, 31, 32)


def decode(histories, tp=1):
    return StepShape((1,) * len(histories), tuple(histories), topology={"tp": tp},
                     capture_bucket=OLD.bucket_for(len(histories)))


def balanced(total, n):
    return tuple(total // n + (i < total % n) for i in range(n))


def old_layouts(n):
    cell = (OLD.bucket_for(n), OLD.bucket_for(n) != n)
    lo, hi, total_limit = dict(OLD.decode_context_cells)[cell]
    maximum = min(n * hi, total_limit)
    totals = {n * lo, min(n * lo + 1, maximum), maximum - 1, maximum}
    for boundary in [n * 1152, 65536, 1572864, 2097152]:
        totals.update(total for total in [boundary - 1, boundary, boundary + 1]
                      if n * lo <= total <= maximum)
    result = set()
    for total in totals:
        result.add(balanced(total, n))
        if n > 1:
            first = min(hi, total - (n - 1) * lo)
            rest = balanced(total - first, n - 1)
            result.add((first,) + rest)
            result.add(tuple(reversed((first,) + rest)))
    return sorted(result)


@pytest.mark.parametrize("n", COUNTS)
def test_old_domain_complete_breakdown_and_band_are_exact(n):
    for histories in old_layouts(n):
        query = decode(histories)
        assert OLD.refusal(query) is None
        assert NEW.refusal(query) is None
        assert NEW.breakdown(query) == OLD.breakdown(query), histories
        assert NEW.band(query) == OLD.band(query), histories


def test_existing_measurements_and_16_32_continuations_are_same_objects():
    changed = {"decode_context_cells", "prepare_decode_history_extensions",
               "topologies", "version", "provenance"}
    for field in fields(OLD):
        if field.name not in changed:
            assert getattr(NEW, field.name) is getattr(OLD, field.name)
    previous = OLD.prepare_decode_history_extensions
    assert len(NEW.prepare_decode_history_extensions) == len(previous) + 6
    assert all(before is after for before, after in zip(
        previous, NEW.prepare_decode_history_extensions))


@pytest.mark.parametrize("n,total,output", [
    (1, 640, True), (1, 9216, True), (1, 16384, False),
    (2, 2048, True), (2, 12000, True),
    (3, 1536, True), (8, 8192, True), (32, 16384, True),
])
def test_prefill_components_and_bands_do_not_change(n, total, output):
    scheduled = balanced(total, n)
    query = StepShape(scheduled, scheduled, num_prefill_tokens=total,
                      topology={"tp": 1}, produces_output=output)
    assert OLD.refusal(query) is None
    assert NEW.breakdown(query) == OLD.breakdown(query)
    assert NEW.band(query) == OLD.band(query)


@pytest.mark.parametrize("history", [196609, 250049, 262143])
def test_new_singleton_support_reaches_validated_guard(history):
    query = decode([history])
    assert OLD.refusal(query) is not None
    assert NEW.refusal(query) is None
    low, high = NEW.band(query)
    assert low <= NEW.seconds(query) <= high


def test_new_per_sequence_support_does_not_remove_existing_sum_bound():
    histories = (262143,) + balanced(2097152 - 262143, 31)
    assert NEW.refusal(decode(histories)) is None
    assert NEW.refusal(decode((262144,))) is not None
    assert NEW.refusal(decode((histories[0] + 1,) + histories[1:])) is not None
    # Exceed the sum while every row stays inside the per-sequence bound.
    over_sum = histories[:1] + (histories[1] + 1,) + histories[2:]
    assert max(over_sum) <= 262143
    assert NEW.refusal(decode(over_sum)) is not None


def test_new_candidate_is_tp1_only_and_does_not_expand_low_prefill():
    assert NEW.topologies == (1,)
    for tp in [2, 4]:
        query = decode([1152], tp=tp)
        assert OLD.refusal(query) is None
        assert NEW.refusal(query) is not None
    for n, total in [(1, 128), (2, 512), (3, 512)]:
        scheduled = balanced(total, n)
        query = StepShape(scheduled, scheduled, num_prefill_tokens=total,
                          topology={"tp": 1}, produces_output=True)
        assert NEW.refusal(query) == OLD.refusal(query)

# SPDX-License-Identifier: MIT
"""`step_linear.moe_column`'s `moe_experts` for a data-parallel group whose idle ranks forward empty batches: those
dummies add no token to the group's tokens (#3682), so subtracting `dummies - 1` from them must not go below zero.
"""

import itertools

from atom.compass.backends.step_linear import activated_experts, moe_column

ROUTING = (384, 6)


def test_three_idle_ranks_beside_one_token_do_not_price_negative_experts():
    got = moe_column("moe_experts", 1, ROUTING, 3)
    assert got >= activated_experts(*ROUTING, 1) > 0


def test_no_group_shape_prices_a_negative_column():
    for tokens, dummies in itertools.product(range(9), range(6)):
        assert moe_column("moe_experts", tokens, ROUTING, dummies) >= 0


def test_a_step_that_was_already_positive_is_unchanged():
    # 12 tokens counting 3 dummies' rows: reached = 12 - 2 = 10 tokens, as before
    assert moe_column("moe_experts", 12, ROUTING, 3) == activated_experts(*ROUTING, 10)
    assert moe_column("moe_experts", 6, ROUTING, 0) == activated_experts(*ROUTING, 6)

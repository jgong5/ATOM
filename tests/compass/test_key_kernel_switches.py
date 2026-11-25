# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's amd-kernels decode stack (`KERNEL_SWITCHES`) and aiter's
tuned-table variables (`TABLES`) change the kernels a step runs, so a law, host
law or memory reading taken under them is keyed apart: each switch by its value
where ATOM reads it off its default, each table by its file's content. A server
that sets none, or sets a switch at its default, keys as before.

No device: `keyed_env` reads only this process's environment.
"""

import pytest

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.run.step_grid import (
    KERNEL_SWITCHES,
    READ_OFF_DEFAULT,
    TABLES,
    keyed_env,
)

#: The 1P2D AgentX recipe's 06d decode (e1561f3d5 env/d_env_akmoe_lmhead.sh):
#: every kernel switch it sets, as it sets them.
D06 = {
    "ATOM_M3_AK_MOE": "1",
    "ATOM_P1_MOE_TILES": "1",
    "ATOM_P1_MOE_IDROWS": "1",
    "ATOM_P1_MBF_ASM": "1",
    "ATOM_P1_MBF_BACK": "1",
    "ATOM_DG_LMHEAD_AK": "1",
    "ATOM_P1_LMHEAD_RING": "1",
    "ATOM_M3_AK_DECODE": "1",
    "ATOM_P1_INDEXER_SPLIT4": "1",
    "ATOM_P1_INDEXER_SPLIT4_WIDE": "1",
    "ATOM_P1_INDEXER_SPLIT4_MIN8": "6",
    "ATOM_P1_K2_SCORER": "rot2_wr2_h1024",
    "ATOM_M3_AK_INSERT": "1",
    "ATOM_M3_AK_DECODE_QOUT": "1",
    "ATOM_P1_AK_WIDE": "1",
    "ATOM_P1_L2_D2": "1",
    "ATOM_M3_FUSED_DECODE": "1",
    "ATOM_AK_MXFP8_GEMM": "1",
    "ATOM_P1_FRONT_QKV_CHAIN": "1",
    "ATOM_P1_AR_QKV_CODES": "1",
    "ATOM_P1_L4_PULLQ": "1",
    "ATOM_MINIMAX_AK_AR_NORM": "1",
    "ATOM_M3_DRAFT_AK_AR_NORM": "1",
    "ATOM_M3_DRAFT_FUSED_CTX_ROPE": "1",
    "ATOM_M3_DRAFT_MARKOV_WIDE": "1",
    "ATOM_M3_DRAFT_POS_MAJOR": "1",
    "ATOM_P1_DRAFT_MARKOV_PERSIST": "1",
    "ATOM_P1_DRAFT_FC_AK": "1",
    "ATOM_P1_FAST_CONCAT": "1",
    "ATOM_P1_DSPARK_AUX_NEXT": "64",
    "ATOM_M3_VERIFY_HEAD_CHUNK_TUNED": "1",
    "ATOM_P1_PREFIX_DEMOTE_DECODE": "1",
}
#: The AK MoE's row range and w13 sharing, which the recipe leaves at ATOM's defaults: keyed only off them.
AK_MOE_RANGE = {
    "ATOM_M3_AK_MOE_MIN": "1",
    "ATOM_M3_AK_MOE_MAX": "87381",
    "ATOM_M3_AK_MOE_SHARE_W13": "1",
}
#: The same recipe's settings that are no kernel switch: never keyed here.
NOT_KEYED = {
    "ATOM_M3_INDEXER_CP": "0",
    "ATOM_STEP_WATCHDOG_S": "5",
    "ATOM_STEP_KICK_S": "5",
    "HSA_ENABLE_IPC_MODE_LEGACY": "1",
    "ATOM_M3_SOMETHING_ELSE": "1",
}
BF16 = "AITER_CONFIG_GEMM_BF16"
MX8 = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """No performance variable, switch or table from the test's own process."""
    import os

    for name in (*PERFORMANCE_KEYS, *READ_OFF_DEFAULT, *KERNEL_SWITCHES, *NOT_KEYED):
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(TABLES):
            monkeypatch.delenv(name)


def test_the_recipes_decode_switches_are_each_keyed_by_value(monkeypatch):
    for name, value in (D06 | NOT_KEYED).items():
        monkeypatch.setenv(name, value)

    assert keyed_env() == D06


def test_every_switch_is_in_the_recipes_decode_or_its_ak_moe_range():
    assert set(KERNEL_SWITCHES) == set(D06) | set(AK_MOE_RANGE)


def test_the_ak_moe_range_at_atoms_defaults_keys_as_unset(monkeypatch):
    for name, value in (D06 | AK_MOE_RANGE).items():
        monkeypatch.setenv(name, value)

    assert keyed_env() == D06


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ATOM_M3_AK_MOE_MAX", "2048"),
        ("ATOM_M3_AK_MOE_MIN", "17"),
        ("ATOM_M3_AK_MOE_SHARE_W13", "0"),
    ],
)
def test_an_ak_moe_range_off_default_is_keyed(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert keyed_env() == {name: value}


@pytest.mark.parametrize("name", sorted(KERNEL_SWITCHES))
def test_a_switch_at_its_default_keys_as_unset(monkeypatch, name):
    default, _ = KERNEL_SWITCHES[name]
    monkeypatch.setenv("AITER_LOG_LEVEL", "WARNING")
    before = keyed_env()
    monkeypatch.setenv(name, default)

    assert keyed_env() == before == {"AITER_LOG_LEVEL": "WARNING"}


@pytest.mark.parametrize(
    ("name", "value", "keyed"),
    [
        ("ATOM_M3_AK_MOE", "0", False),
        ("ATOM_M3_AK_MOE", "true", False),  # ATOM reads only "1" as on
        ("ATOM_M3_AK_MOE", "1", True),
        ("ATOM_P1_INDEXER_SPLIT4_MIN8", "08", False),  # int("08") == 8
        ("ATOM_P1_INDEXER_SPLIT4_MIN8", "6", True),
        ("ATOM_P1_INDEXER_SPLIT4_MIN8", "six", True),  # refused by ATOM too
        ("ATOM_P1_DSPARK_AUX_NEXT", "", False),  # `int(value or 0)`
        ("ATOM_P1_DSPARK_AUX_NEXT", "64", True),
        ("ATOM_P1_K2_SCORER", "", False),
        ("ATOM_P1_K2_SCORER", "rot2_wr2_h1024", True),
    ],
)
def test_a_switch_is_keyed_where_its_parser_reads_it_off_default(
    monkeypatch, name, value, keyed
):
    monkeypatch.setenv(name, value)

    assert keyed_env() == ({name: value} if keyed else {})


def test_a_table_is_keyed_by_its_files_content_not_its_path(monkeypatch, tmp_path):
    one, same, other = (tmp_path / n for n in ("a.csv", "b.csv", "c.csv"))
    one.write_text("M,N,K,kernel\n1,2,3,x\n")
    same.write_text("M,N,K,kernel\n1,2,3,x\n")
    other.write_text("M,N,K,kernel\n1,2,3,y\n")

    monkeypatch.setenv(BF16, str(one))
    at_one = keyed_env()
    monkeypatch.setenv(BF16, str(same))
    at_same = keyed_env()
    monkeypatch.setenv(BF16, str(other))
    at_other = keyed_env()

    assert at_one == at_same
    assert at_one[BF16].startswith("sha256:") and len(at_one[BF16]) == 7 + 16
    assert at_other[BF16] != at_one[BF16]


def test_a_table_no_file_answers_is_keyed_by_its_path(monkeypatch, tmp_path):
    monkeypatch.setenv(MX8, str(tmp_path / "absent.csv"))

    assert keyed_env() == {MX8: str(tmp_path / "absent.csv")}


def test_an_empty_table_variable_is_not_keyed(monkeypatch):
    monkeypatch.setenv(BF16, "")

    assert keyed_env() == {}

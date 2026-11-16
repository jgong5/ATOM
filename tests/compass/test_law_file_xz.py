# SPDX-License-Identifier: MIT
"""A law file stored xz-compressed, as `scripts/compass/laws/` stores each one:
`step_linear_fit pack` stores a JSON law so, in the same bytes for the same
document, and the fit and its `figures` write one when its name ends in `.xz`.
`compass run`, the reuse check and `figures` read it as they read its JSON, so
a run, a verdict and a fold come out as they do from the JSON.
"""

import json
import lzma
from pathlib import Path

from test_advice_reuse import advise, law, verdicts
from test_advice_reuse import at_head, searched  # noqa: F401  fixtures
from test_advice_reuse import law_file as checked_in
from test_backend_step_linear_fit import SPEC
from test_drive_runfile import build
from test_drive_runfile_law_file import law_file
from test_run_step_grid import deployment
from test_step_linear_figures import _fits, _plain

from atom.compass.backends.json_file import read_json, write_json
from atom.compass.backends.step_linear_fit import main

#: The directory of the checked-in law files.
LAWS = Path(__file__).resolve().parents[2] / "scripts/compass/laws"


def test_every_checked_in_law_is_stored_xz_compressed():
    names = sorted(p.name for p in LAWS.iterdir())
    assert names and [n for n in names if not n.endswith(".json.xz")] == []


def test_pack_stores_each_law_in_a_directory_in_the_bytes_of_its_document(tmp_path):
    document = {"form": "base", "decode": {"1": {"coefficients": [1e-3, 0.1]}}}
    (tmp_path / "laid-out.json").write_text(json.dumps(document, indent=2))
    (tmp_path / "written.json").write_text(json.dumps(document) + "\n")

    main(["pack", str(tmp_path)])
    packed = sorted(tmp_path.iterdir())
    main(["pack", str(tmp_path)])

    assert [p.name for p in packed] == ["laid-out.json.xz", "written.json.xz"]
    assert sorted(tmp_path.iterdir()) == packed
    first, second = (p.read_bytes() for p in packed)
    assert first == second
    assert lzma.decompress(first) == (json.dumps(document) + "\n").encode()
    assert read_json(packed[0]) == document


def test_a_run_file_built_from_a_compressed_law_is_the_one_its_json_builds(tmp_path):
    path, _ = law_file(tmp_path, deployment(tmp_path))
    plain = build(tmp_path, "--step-linear", path, "--host-laws", path)

    main(["pack", path])

    packed = f"{path}.xz"
    assert build(tmp_path, "--step-linear", packed, "--host-laws", packed) == plain


def test_the_reuse_check_reuses_a_compressed_law_file_as_its_json(
    tmp_path, searched  # noqa: F811
):
    where = checked_in(tmp_path, law(searched))
    main(["pack", where])
    packed = f"{where}.xz"

    judged = advise(tmp_path)

    reused = {"verdict": "reuse", "requirement": None, "reason": packed, "law": packed}
    assert verdicts(judged, "law") == [reused | {"run": [packed, packed]}] * 2


def test_the_fit_and_its_figures_write_a_law_compressed_when_its_name_ends_in_xz(
    tmp_path,
):
    fitted, records = _fits(tmp_path, _plain)
    packed = tmp_path / "law.json.xz"

    main([*map(str, records), "--out", str(packed), *SPEC])
    assert read_json(packed) == fitted

    write_json({**fitted, "figures": {}}, packed)
    main(["figures", *map(str, records), "--law", str(packed)])
    assert read_json(packed) == fitted

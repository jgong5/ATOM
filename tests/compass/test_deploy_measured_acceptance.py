# SPDX-License-Identifier: MIT
"""A simulated prefill-decode deployment's acceptance, read from a real decode
record: `deploy.sh run --acceptance-record` builds the run file through
`runfile.prefill_decode`, which names the record as `compass run` does, and
each instance's `compass run --join` hands ATOM the rates its histograms give,
the decode instance always and the prefill instance when it drafts, so each
engine's hand-in names its input ``measured``.

Each record's scheduler is ATOM's, as `test_run_measured_acceptance` records it;
`deploy.sh` runs on `test_deploy_sh`'s stubs.
"""

import argparse
import json
import signal
import subprocess
from types import SimpleNamespace

import pytest
from test_deploy_sh import ROUTER_ARGS, SCRIPT, _until, planted  # noqa: F401
from test_drive_run import stub
from test_memory_readings import DOCUMENT
from test_run_measured_acceptance import mtp, recorded
from test_run_speculative_summary import RATES, _stated

from atom.compass import run as compass_run
from atom.compass.drive import run as drive_run
from atom.compass.drive import runfile, serve
from atom.model_engine.engine_stats import EngineStats

ROLES = {"P": "kv_producer", "D": "kv_consumer"}
DPA2 = ["-tp", "2", "--enable-dp-attention"]


def built(tmp_path, record, prefill, decode) -> dict:
    """The run file `prefill_decode` builds from `deploy.sh run`'s flags with
    `record` as its acceptance record, for a prefill and a decode instance
    started with ATOM's arguments `prefill` and `decode`."""
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    parser.add_argument("--acceptance-record")
    machine = ["--machine", str(tmp_path / "machine.json")]
    store = ["--store", str(tmp_path / "store")]
    flags = ["--step-stub", stub(tmp_path), *machine, *store]
    args = parser.parse_args([*flags, "--acceptance-record", str(record)])
    model = ["--model", str(tmp_path)]
    return runfile.prefill_decode(
        args, [*model, *prefill], [*model, *decode], "intra_node"
    )


def joined(monkeypatch, tmp_path, run, side, atom) -> list[str]:
    """The ATOM arguments the `side` instance's `compass run --join` of the run
    file `run` serves with, started as `deploy.sh` starts it with `atom`."""
    path = tmp_path / "run_file.json"
    path.write_text(json.dumps(run | {"out_dir": "o", "clock_endpoint": "tcp://x:1"}))
    seen = {}
    monkeypatch.setattr(
        serve, "serve", lambda atom_args, out, run, joined: seen.update(atom=atom_args)
    )
    monkeypatch.setattr(drive_run, "write_grading", lambda out, report, side: None)
    kv = json.dumps({"kv_role": ROLES[side], "kv_connector": "compass"})
    given = ["--model", str(tmp_path), *atom, "--kv-transfer-config", kv]
    drive_run.main(["--join", str(path), "--out", str(tmp_path / side), "--", *given])
    return seen["atom"]


def said(monkeypatch, atom_args, run) -> dict:
    """The acceptance input an engine started with `atom_args` hands in under
    the run file `run`."""
    rates = runfile._engine_args(atom_args).spec_decode_acceptance_rates
    spec = _stated(monkeypatch, len(rates), synthetic_acceptance_rates=rates)
    engine = SimpleNamespace(
        config=SimpleNamespace(speculative_config=spec),
        engine_stats=EngineStats(use_spec=True, mtp_k=len(rates)),
    )
    return compass_run._acceptance(engine, run)["input"]


@pytest.mark.parametrize("prefill_drafts", [True, False])
def test_a_1p1d_run_file_with_a_decode_record_gives_the_decode_its_measured_rates(
    monkeypatch, tmp_path, prefill_drafts
):
    record, first, _ = recorded(monkeypatch, tmp_path, [0.8], rank=0)
    _, second, _ = recorded(monkeypatch, tmp_path, [0.4], rank=1)
    counts = [
        sum(s.engine_stats.distribution[a] for s in (first, second)) for a in (0, 1)
    ]
    prefill = mtp(1) if prefill_drafts else []

    run = built(tmp_path, record, prefill, [*mtp(1), *DPA2])
    decode = joined(monkeypatch, tmp_path, run, "D", [*mtp(1), *DPA2])
    prefilled = joined(monkeypatch, tmp_path, run, "P", prefill)

    assert run[compass_run.ACCEPTANCE_RECORD] == str(record.resolve())
    rates = runfile._engine_args(decode).spec_decode_acceptance_rates
    assert rates == pytest.approx([counts[1] / sum(counts)])
    input_ = said(monkeypatch, decode, run)
    print("decode input:", input_)
    record_ = {"record": str(record.resolve())}
    assert input_ == {"flag": RATES, "value": rates, "method": "measured"} | record_
    stated = runfile._engine_args(prefilled).spec_decode_acceptance_rates
    assert stated == (rates if prefill_drafts else None)


@pytest.mark.parametrize(
    "prefill, decode, refusal",
    [
        (
            [*mtp(1), "--spec-decode-acceptance-length", "1.5"],
            mtp(1),
            (
                "^the prefill instance: --acceptance-record states the acceptance, "
                "and ATOM's arguments state --spec-decode-acceptance-length too$"
            ),
        ),
        (mtp(1), mtp(2), "^the decode instance: --acceptance-record: .* drafted 1 "),
        ([], [], "^the decode instance: .*and ATOM's arguments draft 0$"),
        ([], [*mtp(1), *DPA2], "^the decode instance: .* holds no dp1.acceptance"),
    ],
)
def test_a_record_either_instance_would_refuse_refuses_the_run_file_naming_it(
    monkeypatch, tmp_path, prefill, decode, refusal
):
    record, *_ = recorded(monkeypatch, tmp_path, [0.7], rank=0)

    with pytest.raises(ValueError, match=refusal):
        built(tmp_path, record, prefill, decode)


def deployed(tmp_path, flags, record, decode):
    """`deploy.sh run`'s words for a deployment whose decode instance runs
    `decode`, with `record` as its acceptance record."""
    model = ["--model", str(tmp_path)]
    return [
        "bash", str(SCRIPT), str(tmp_path / "d"), "run", *flags,
        "--acceptance-record", str(record),
        "--prefill", "--", *model,
        "--decode", "--", *model, *decode,
        "--router", "--", *ROUTER_ARGS,
    ]  # fmt: skip


def test_deploy_names_the_record_in_the_run_file_its_instances_join(
    monkeypatch, tmp_path, planted  # noqa: F811
):
    env, flags = planted
    record, *_ = recorded(monkeypatch, tmp_path, [0.7], rank=0)
    d = tmp_path / "d"
    script = subprocess.Popen(
        deployed(tmp_path, flags, record, mtp(1)),
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
    )
    _until(d / "server.json", script)

    run = json.loads((d / "run_file.json").read_text())
    assert run[compass_run.ACCEPTANCE_RECORD] == str(record.resolve())
    seen = json.loads((d / "decode.argv.json").read_text())
    assert seen[:3] == ["run", "--join", str(d / "run_file.json")]

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)
    assert script.returncode == 0, out


def test_deploy_refuses_a_record_its_decode_instance_would_refuse_before_it_starts(
    monkeypatch, tmp_path, planted  # noqa: F811
):
    env, flags = planted
    record, *_ = recorded(monkeypatch, tmp_path, [0.7], rank=0)

    done = subprocess.run(
        deployed(tmp_path, flags, record, mtp(2)),
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 1, done.stderr
    assert "deploy: the decode instance: --acceptance-record: " in done.stderr
    assert "drafted 1 tokens a step, and ATOM's arguments draft 2" in done.stderr
    assert not (tmp_path / "d" / "run_file.json").exists()
    assert not (tmp_path / "d" / "authority.argv.json").exists()

---
name: compass-calibration
description: Design, rehearse, record, check or adjust a Compass step-law calibration (the #739 v1.1 procedure). Use when an agent writes or edits a calibration config, generates its traces, dry-runs it on a law, records it on GPUs, checks a calibration record against its plan, or adapts a calibration to a changed deployment (capture sizes, max-num-seqs, TP/DP/EP, DCP, batched tokens, KV capacity, block size, speculative decoding, MoE routing, a new layer kind).
version: 1.0.0
scope: jgong5/ATOM, atom/compass/calib/, scripts/compass/calib/, scripts/compass/{gen,plan,check,dry,cal}_v1.*
last_updated: 2026-11-24
---

# compass-calibration

Read `scripts/compass/calib/README.md` first. It says which deployment variables re-key the
laws and the memory readings, what each variable moves in a design, the steps with their
commands, what each dry-run failure asks for, and what never changes.

## The pieces

- Configs: `scripts/compass/calib/*.json`, read by `atom/compass/calib/deploy.py`
  (`load`, `from_record`; a config may `extend` another one).
- Generate the traces: `scripts/compass/gen_v1.py CONFIG DESIGN [--record RECORD]`.
- Plan and rehearse on a law: `scripts/compass/plan_v1.py DESIGN --law LAW --host HOST [--target RECORD]`.
- Both, plus the verdict: `[RECORD=...] [TARGET=...] scripts/compass/dry_v1.sh CONFIG LAW_DIR OUT`.
- Record on one server: `scripts/compass/cal_v1.sh DESIGN OUT record ...`. Hold a gpool
  lease for it.
- Check a record against its plan: `scripts/compass/check_v1.py DESIGN REC --law LAW --host HOST`.

## Rules

- A deployment change starts from a smoke record under the new deployment. Pass it with
  `--record` / `RECORD=` so that no field is edited by hand.
- Segment sizes stay relative (`design.count`, `design.tokens`). Write a plain number only
  where the README says sizes do not scale.
- Never loosen `deploy.ACCEPTANCE`. Never pick a column variant by its dry-run result:
  the frozen form-selection rule chooses.
- Fix a failing vector in the segments, following the README's section 4, and dry-run
  again before any GPU time.
- Tests: `tests/compass/test_calib_v1.py`.

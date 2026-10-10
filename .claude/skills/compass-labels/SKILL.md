---
name: compass-labels
description: Apply the `module: compass-*` and `dev-process` GitHub labels to an ATOM PR or issue, and keep the label set in sync with the tree. Use when opening or updating a PR or issue in the Compass area, or when asked what labels a PR should carry.
version: 1.0.0
scope: jgong5/ATOM, atom/compass/, tests/compass/, scripts/compass/, tools/compass/
last_updated: 2026-10-10
---

# compass-labels

The rules behind the `module: compass-*` and `dev-process` labels on
`jgong5/ATOM`, and `compass_labels.py`, which applies the part of them a
changed-file set and a title decide on their own.

## Labels

- `module: compass-<dir>` per module directory under `atom/compass/`
  (underscores become hyphens), colour `0E8A16`, description
  `atom/compass/<dir>/ and its tests`.
- `module: compass-script` for `scripts/compass/`, `module: compass-tools`
  for `tools/compass/`, same colour.
- `dev-process`, colour `BFD4F2`, for the development workflow itself.

A module is a subdirectory of `atom/compass/` carrying a file other than
`__init__.py` -- never a hard-coded list, so a directory with only an
`__init__.py` (nothing shipped there yet) carries no label, and a new module
is picked up with no edit here. `compass_labels.py discover_modules` reads
this from the tree at run time.

## When a PR gets a module label

A PR gets `module: compass-<dir>` when it changes any of:

- `atom/compass/<dir>/**`
- `tests/compass/<dir>/**` (the file's own directory, not its name)
- `tests/compass/test_<dir>.py` or `tests/compass/test_<dir>_*.py` -- a flat
  file directly under `tests/compass/`. When a filename's prefix matches more
  than one module (`test_clock_transport_wire.py` matches both `clock` and
  `clock_transport`), the longer module name wins.
- `scripts/compass/**` (`module: compass-script`) or `tools/compass/**`
  (`module: compass-tools`)

A PR may carry several module labels.

## When a PR gets `dev-process`

A PR gets `dev-process` when it changes `AI_DEV_RULES.md`, anything under
`.claude/`, or any file named `CLAUDE.md`, or when its title's scope is
`compass(rules)` or `compass(process)`, or its title starts with `Process:`
or `agent-team`. A gate-script change (`scripts/compass/**`) only ever reaches
`module: compass-script`, never `dev-process`, even though it is part of the
development workflow: the path rule above does not test for it.

## Judgement calls (read and apply by hand, not run by the script)

These stay prose because they need a reader, not a rule:

- **Purpose label.** A compass PR that also changes ATOM code outside
  `atom/compass/`, `tests/compass/`, `scripts/compass/` and `tools/compass/`
  also gets the label of the module that change enables -- for example
  `module: compass-clock` for an engine clock read added to wire the Clock
  Authority in. Apply it only when the purpose is clear; when it is not,
  apply nothing rather than guess.
- **Issues.** An issue inherits the labels of every PR whose body says
  `Closes #N`. Otherwise, label it from its text only when the text names a
  path, file or symbol that resolves to a module; when unsure, apply no
  label.

Labels are only ever added. Nothing here ever removes one.

## Using `compass_labels.py`

```bash
# print the labels PR 680's files and title give, by the rules above
python3 .claude/skills/compass-labels/compass_labels.py pr 680

# the same, then add whichever of those labels the PR does not already carry
python3 .claude/skills/compass-labels/compass_labels.py pr 680 --apply

# create any module/script/tools/dev-process label the current tree implies
# and the repo is missing -- a no-op once the repo already carries them all
python3 .claude/skills/compass-labels/compass_labels.py sync
```

`pr <N>` only ever adds: it never removes a label the PR already carries that
the rules above would not themselves add, so a purpose or issue-inheritance
label applied by hand survives a later `--apply`.

## Wiring into the workflow

`.claude/agent-team.md` names this skill: an issue or PR gets its labels from
`compass_labels.py pr <N> --apply` when it is created, and again whenever a PR
round adds files the opening round did not touch.

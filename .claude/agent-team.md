---
repo: jgong5/ATOM
integration_branch: feature/atomcompass_new
gate_task: scripts/compass/gate_cpu.sh
gate_wave: scripts/compass/gate_gpu.sh
design_entry: atom/compass/design/README.md
never_touch: ROCm/ATOM
new_tests_dir: tests/compass
import_check: python3 -c "import atom; print(atom.__file__)"
---
**This environment.** Each owner runs from their own environment and keeps a
git-ignored `.claude/agent-team.local.md` in the main worktree
(`$(git rev-parse --path-format=absolute --git-common-dir)/../.claude/`, not
the task worktree) saying how it meets the requirements below: where git,
`gh` and `python3` run, path mappings, file ownership, the CPU gate container
and how to stage into it, GPU node access. Read it before anything else. If
it is missing, stop and ask the owner to write it from this file. Scratch and
working logs go in `agent_scratch/`, beside the repository.

## Several owners

Several owners drive agents on this repository at once, each through their
own `gh` login. Ownership is the assignee:

- **Scope.** A `run` acts only on issues assigned to its own `gh` login and on
  the PRs that deliver them, with their stacks: claim, review, develop and
  land all stay inside that set. Every other open issue and PR belongs to
  someone else; `status` may report them, nothing else touches them.
- **Assignee on creation.** `gh issue create` always passes
  `--assignee @me`. `gh pr create` passes the assignee of the issue it
  delivers.
- **Reassignment** takes effect at the next pass: the old owner's run stops
  touching the issue, the new owner's run takes it over (recreating the
  worktree from the PR branch). A push rejected during the overlap is
  fetched and merged, never forced.
- **`need human`** is removed by the issue's assignee, except an escalation
  about a design decision (a D-numbered decision, a design document, a new
  hook point in ATOM): only @jgong5 removes that one.

## Environment requirements

- Git, `gh`, `python3`, `gate_wave` and the `agent-team` scripts run where
  the local file says, with the project's toolchain; a stray host Python
  that happens to work tells you nothing.
- A `gh` login with write access to `repo`, recent enough that `gh pr view`
  accepts `closingIssuesReferences` (`pr_state.py` asks for it on every PR;
  2.45.0 rejects it), and the `gh stack` extension.
- Files a git command writes are owned by the user who edits them; if git
  runs as another user (root in a container), chown the worktree after every
  git command that writes files (worktree add, fetch into a branch, merge,
  checkout) and check nothing is left.
- **`gate_task` runs in a CPU environment with no GPU driver**, on a
  `scripts/compass/snapshot.sh` tarball of the head
  (`scripts/compass/README.md`), and is judged against the same run on the
  base. Not where a driver is present: kernel tests in the CPU tier run on
  the device there and fail at the base too. New tests in `new_tests_dir`
  are CPU-only: they pass there with no driver.
- **Python resolution.** Run `python3` and `pytest` from the tree's root, or
  with `PYTHONPATH` set to it, and run `import_check` before trusting a
  result: an environment with another ATOM checkout resolves `atom` there
  from anywhere else. A container that runs a tree mounts the worktree
  root's parent, not the tree alone; mounting one tree makes `pytest` fail
  with a path error.
- **The local integration branch.** The Compass scripts diff against the
  local `feature/atomcompass_new` when it exists, before
  `fork/feature/atomcompass_new`, so a stale local branch gives them a wrong
  base unless `COMPASS_INTEGRATION_REF` names the ref. The fast-forward
  after every landing keeps it current.

## Project rules

- **Few ATOM hook points.** Hook Compass into ATOM at the fewest points (a
  factory, a constructor, or a bootstrap patch applied only when Compass is
  on), never with per-use-site wrappers in ATOM's engine or frontend files.
  Compass-off behaviour stays byte-identical. A PR touching ATOM engine code
  compares the hook options in its body before building.
- **Inline reviews.** Owners may review with inline comments
  (`pulls/<n>/comments`, `pulls/<n>/reviews`), which `pr_state.py` and the
  issue-comments endpoint do not show. Read both endpoints before acting on
  a PR, and treat an owner's inline comment as a required finding. A search
  over a PR that found nothing counts only if it covered both. On REST,
  merge state is `.merged` / `.merged_at`, not `.mergedAt`.
- **`gh` and the fork's parent.** Commands that look up the fork's parent
  (`pr create`, `issue create`, `pr ready`, `stack`) can fail with a ROCm
  SAML error. Never follow the authorization link it prints: use REST on
  `repo`, and prefix `gh stack` with `GH_REPO=<repo>`. Pass file bodies as
  `-F body=@<file>`; `-f` posts the literal path and still succeeds.
  `land.py` cannot land a draft PR: open PRs ready for review, or clear
  draft with the GraphQL `markPullRequestReadyForReview` mutation.
- **Re-basing a stack.** A PR in a `gh stack` refuses a base PATCH with 422.
  Run `gh stack unstack`, PATCH each base, then `gh stack link --base
  <integration_branch>` the whole chain again.
- **Gating a stacked PR.** Set `COMPASS_INTEGRATION_REF` to the parent PR's
  head, so `.compass-changed` lists only this PR's diff. Otherwise a
  GPU-trigger path changed lower in the stack makes `gate_cpu.sh` exit 98
  for every PR above it; only the PR that changes that path needs
  `gate_gpu.sh`.
- **Timing noise.** `tests/entrypoints/test_stream_marker_properties.py`
  holds wall-clock timing classes that can fail a correct tree under load.
  A failure there is noise until a re-run reproduces it; never edit or
  exclude the file to get a gate green.
- **Bound every gate.** Run gates under `timeout`. A gate that failed prints
  `GATE_CPU_RC=`; one that hung prints a pytest summary and nothing after
  it. Never pipe a gate or a lint into `tail`: capture each exit status
  separately. The MANUAL entries in `cpu_gate_exclude.txt` are
  load-bearing: each one hangs the tier.

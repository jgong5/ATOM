## How to talk with owner
- **If there are no blocking issues, say so explicitly** in every message.
- **Quote the context; stop only for critical decisions.**
- **Always communicate PR status.** The owner does not care about local worktree state.
- **Answer with the conclusion first.** When the owner asks what a task
  established, the first line is the **finding**, not the method or the process;
  for that question this overrides the next-action lead below. A reader who stops
  after one line should have the answer; measurement follows, then caveats and
  cost. The shape: *a real model traces at both widths, and its shapes are
  entirely concrete -- 0 symbolic of 13,107* -- then the evidence for each half.
- Output shaping (`/i-have-adhd`): lead with the next action, number multi-step
  tasks, end with one concrete next action, restate state every turn, specific time
  estimates, matter-of-fact error tone, cap lists at 5, no preamble or closing
  pleasantries.

## Execution rules
- Don't modify the main worktree. Develop with linked worktrees, one per in-flight
  task, under `compass-worktrees/<task-id>`, beside the repo.
- **On every landing, the landing agent fast-forwards the main worktree** to
  `feature/atomcompass_new` (the integration branch). That updates the main
  worktree; it does not develop there, so the rule above stands. It is easy to
  skip because nothing visibly breaks, but every linked worktree shares the main
  worktree's local `feature/atomcompass_new`, and `compass_resolve_ref` tries that
  bare name first, so a stale local branch wins for any command that omits
  `COMPASS_INTEGRATION_REF`. Measured: 14 commits behind after one session of
  landings.

  Container git runs as uid 0 and the repo belongs to the host user, 13797, so
  container git works only because `/root/.gitconfig` carries
  `safe.directory = *`. `/root` does not survive `teardown.sh`: after a rebuild,
  run `git config --global --add safe.directory '*'` before any git command, then
  `gh auth setup-git` — the same file holds the credential helper `push` needs.
  Measured 2026-09-23: with that entry removed, container git refuses both the
  main worktree and a linked one with `dubious ownership`. A pull as root leaves
  new files root-owned, which fails host-side edits silently, so the chown
  follows every pull:

  ```
  cd <main worktree> && git fetch fork --quiet
  git merge --ff-only fork/feature/atomcompass_new
  chown -R 13797:13797 .      # whole tree, .git included -- the repo's standing state
  ```

  Verify afterwards: `stat -c "%u %n" . .git` prints `13797` on both lines, and
  `git -C <each worktree> rev-parse HEAD` succeeds. Do not assume it worked.
- **Four setup rules.** Each failure behind them came from a shared mutable
  non-git source tree that things silently resolved against, not from worktrees.
  1. No shared mutable source root exists. Every tree is a worktree or a
     `git archive` snapshot.
  2. Containers mount the worktree parent, so every task's tree is reachable at a
     stable path; mounting one tree makes `pytest` fail with a path error.
  3. Every command sets `PYTHONPATH` to its own worktree and verifies it with
     `python -c "import atom; print(atom.__file__)"` before trusting a result. A
     run once resolved `atom` to another branch's snapshot and failed silently.
  4. Snapshots use `git archive`, never `rsync`, so a snapshot names one commit.
     An rsync'd tree mixing two generations gave a `TypeError` that read as a
     code bug.
- **No design-doc references in code.** No `D18`, `P0.4`, `T5`, `W2.5`, backticked
  doc numbers, "principle N", or numbered labels like "Gate 1". No quoting design
  principles as justification. Say what the code does, its functions, how it works.
  Design docs may cite each other freely; code may not cite them at all. This
  extends to **runtime data** — a `(BEYOND-D18)` suffix on an emitted stub name was
  a citation in the output record.
- **The design-doc rule is checked at the head, over the PR's whole file set** —
  never over the added lines of a delta, which cannot see a reference that
  arrived before the range. A design document a test opens **by path** is a
  functional dependency, not a citation, and stays. Measured: one PR carried
  **41** through cycles that each reported clean; a board-wide census found nine
  reaching runtime output.
- Merge conflicts are the agent's call, not the owner's ("don't bother me on merge
  conflict, it's on you"). Tasks are cut so each touches one module plus its tests,
  which makes most of them disjoint, but they are **not guaranteed disjoint** and no
  allocation-time file locking is imposed — a merge conflict is cheaper than the
  machinery to prevent it. **Frequent conflicts mean the task decomposition is
  wrong, not that coordination failed.** The response is to re-cut the tasks, not
  to add a scheduler.
- Working logs and scratch go in `agent_scratch/`. Nothing durable lives in the
  tree: the task record is the GitHub issue and its PR.
- **Tasks are a pool, not a track assignment.** Work is a DAG of GitHub issues; a
  task becomes claimable when its dependencies land, and any free agent assigns
  itself the issue — there is no permanent per-track ownership. A task carries
  four sections, the first written before it is claimable and the rest written as
  it runs:

  | Section | Written by | Lives in | Contains |
  |---|---|---|---|
  | Brief | the planner | the issue body | what to build; the interfaces it implements and consumes; its file set; its exit criteria; its effort estimate |
  | Dev record | the developer | the PR body | what was found, what was decided that the design did not cover, what surprised it, what was left undone |
  | Review record | the reviewer | the PR review comment | what was checked, what was accepted with reservation, what the next task in this area should watch |
  | Handoff | both | a closing comment on the issue | what a successor needs to know that is not in the code |

  Claiming a task is assigning yourself its issue — the issue exists before the
  branch does, which is why the brief lives there. Every task's brief links to its
  predecessors' issues, so continuity survives agent turnover. **A brief that
  cannot name its file set is not claimable.**
- Task management is GitHub: the PR names its issue, and the issue is closed
  deliberately, with the handoff comment. Agents open, assign, comment on and
  close issues, including issues they did not open.
- **Check delivery before claiming or briefing an issue.** Read its
  **comments**, not only its body, and check whether any open PR names it in its
  title or with a delivering verb (closes / fixes / resolves / addresses /
  implements). **Do not write "checked" unless the check you ran is the one that
  answers the claim.** Measured: two briefs in one day were written for issues an
  open PR already delivered.
- **A finding that outlives its PR needs an issue, not a PR body.** PR bodies are
  squashed away on landing, so a finding recorded only there is lost to the next
  reader. If a finding is not fixed in the PR that found it, open an issue and
  point the PR body at it. Measured: two reviews re-derived findings that had been
  written down repeatedly with no issue to point at.
- **A PR's state is its last thread entry; its verdict is the last comment that
  carries one**, and a verdict covers only the head it names. Read both from the
  thread, never from a carried-forward summary: developer rounds and review
  verdicts alternate in one stream and both open with a bold heading, so a
  remembered approval may belong to an earlier round. Measured: one PR sat
  recorded as approved through six consecutive checks while its last entry was a
  developer record and no reviewer had seen its head.
- **When something does not work as expected, stop and diagnose it. Do not work
  around it.** This covers: a design document that contradicts the code; a test
  that fails for a reason the task did not predict; a measurement outside its
  stated range; an interface that cannot be implemented as specified. Each is
  investigated first — the workaround is forbidden, the diagnosis is not. If the
  cause is a bug in the task's own change, it is a finding: fixed, and recorded
  in the PR body. Only if settling it needs an owner ruling is it an
  **escalation** (defined below), labelled and discussed with the owner. A
  workaround improvised under build pressure is exactly the class of decision
  that never gets written down.
- Concurrency: 5 tasks in flight, up to 10 agents (5 developer + 5 reviewer). The
  cap is review throughput, not the task DAG.
- Both developer and reviewer agents must be told to read `atom/compass/design/README.md`'s
  eight principles first.
- **A developer agent owns development and PR updates; the main agent orchestrates
  and does not write the change itself.** After each push a reviewer agent
  reviews, the developer amends, and that repeats until the verdict is APPROVE.
  The owner is asked only for an escalation, never for a finding.
- **An escalation is anything that needs an owner ruling before work can
  continue; it is labelled `need human` when it is declared. Anything the
  developer can fix without a ruling is a finding, and is not labelled.** A halt
  declared in prose does not stop automation; the label does. When the ruling
  lives on a separate issue, **label each PR it holds anyway** and name the issue
  on the PR. Measured: four PRs declared effort halts pointing at a ruling on
  #89, none was labelled, and an agent dispatched work at #91 because it looked
  unlabelled and approved.
- **Automation is on by default.** An agent acts on any issue or PR that does
  not carry the `need human` label — no opt-in, no waiting to be told.
- **`need human` stops all agent action on that issue or PR** — no agent
  commits to it, reviews it, amends it, or merges it, not even a labelled PR
  whose review already passed. An agent applies the label the moment it
  escalates, so it can stop itself; only the owner removes it, and removal is
  what restarts the work.
- **The review loop has its own stop.** If the same finding survives two cycles,
  or the loop passes three cycles, it halts, goes to the owner and applies
  `need human` to the PR: a task that cannot converge is mis-cut, not
  under-worked.
- Reviewer agents must post their review to the PR; **the verdict goes in the
  comment body text**, since GitHub refuses APPROVE/REQUEST_CHANGES on
  self-authored PRs. That limitation is about the verdict only — it says
  nothing about where findings go.
- **Prefer inline comments.** A finding that points at specific lines is posted as
  an inline review comment on those lines, via
  `gh api repos/<owner>/<repo>/pulls/<n>/comments` with `body`, `commit_id`,
  `path` and `line` — `commit_id` is required alongside the other three;
  omitting it fails the call rather than silently defaulting. A reviewer does not
  fold a line-anchorable finding into one standalone comment.
- **The standalone comment carries the verdict and the summary**, plus any
  finding that genuinely has no line to sit on — a missing file, a count wrong
  across a whole document, a claim in the PR body rather than the diff.
- **Four gates land a task, all required:**
  1. ATOM's test suite passes unmodified — in two tiers, not as one GPU-free
     suite. P0.2 measured it: of the 187 files, 30 are plugin and 29 of the
     remaining 157 reach the GPU driver, so what is GPU-free is a tier, run per
     task, and the rest is a GPU superset run per wave as a delta. `16`'s
     measured test and lint baselines carry the derivation and the counts.
     Needing to edit an ATOM test means the change altered ATOM's behaviour and
     must be justified on its own terms, not absorbed.
  2. New CPU-only tests for what the task added, in `tests/compass/`, in ATOM's
     style. **The tests must exercise something the PR did not itself add.** A
     new module plus tests for that module, imported by nothing else, is
     self-confirming: it passes every gate and demonstrates nothing.
  3. One named result, stated in the issue body before the task is claimed and
     not chosen afterwards. An umbrella brief whose children carry the work
     states one anyway, or names the child that carries it; a developer choosing
     one afterwards is the case this rule forbids.
  4. Review by the task's reviewer agent, looping to APPROVE as above.
     **A check counts only once someone has seen it fire** — a test, a pin, or
     any instrument in this file. A reviewer credits a test with holding a
     defect only after reinstating the defect — the pre-fix code via
     `git show`, nothing else changed — re-running, and recording both counts
     plus the failing node id and assertion. A developer reverts their own fix
     before claiming it; if nothing reddens, they add the pin. A claim that a
     fix is unobservable is checked the same way: if the reviewer can make the
     reinstated defect fail a test, the claim is wrong. **An inert pin on a
     required finding is itself a required finding: the reviewer does not
     approve over it.** Mutations
     preserve line count, because a test that asserts a source line number or a
     file's length fails on any edit and would look as if it caught the
     mutation. Measured: #163's cycle-2 reviewer reinstated the defect, recorded
     the pin as inert (`39 passed`), and approved anyway.

  Baselines are recorded first (the suite's and ruff's pass/fail state, before the
  first Compass commit) — the lint baseline on this repository is already known
  to be dirty, and a pre-existing failure attributed to Compass costs a day.
- **Effort is estimated in lines of code, not time.** Wall-clock appears only for
  machine time with a measured basis. **A task that overruns its estimate by more
  than ~2x is a halt-and-discuss event, not a reason to keep going** — an
  escalation, labelled as above; the usual cause is that the task was mis-cut.
- **PRs land squashed onto `feature/atomcompass_new`, the integration branch**,
  one commit per task. GitHub enforces this structurally
  (`allow_merge_commit=false`, `allow_rebase_merge=false`). Base branch is always
  `feature/atomcompass_new` — never `main`, never `master`, never a branch on
  upstream `ROCm/ATOM`.
- **Landing is the agents' job; no owner approval is needed or sought.** An agent
  lands any PR whose verdict is APPROVE covering its current head, with no
  `need human` on it **or anywhere below it in its stack**. The APPROVE is the
  reviewer's statement that gates 1-3 hold for that head; landing does not wait
  for the per-wave GPU superset. **The hold is the label**: a PR whose body
  declares an escalation but carries no label gets the label, and is then held
  by it. The only other holds are rules in this file, such as an approval that
  does not cover the head, and **an agent that holds a PR names the rule**. An
  agent that sees a rule violation in an approved PR lands it anyway and files
  the violation as an issue; it does not post a verdict.
  The owner stated this after ~42 approved, unlabelled PRs sat for a day because
  a handoff note called landing "the owner's call".
  - **A handoff note is a predecessor's judgement, not a rule.** Where a note
    contradicts this file, this file wins.
  - **Before landing on a moved tip, compute the tree that will land.** An
    independent PR: `git merge-tree --write-tree <current tip> <reviewed head>`.
    A PR stacked on another adds `--merge-base <parent's reviewed head>`: its
    head still carries the parent's original commits while the tip carries the
    parent's squash, so the plain form reports conflicts that do not exist. If
    the result is `<reviewed head>^{tree}`, the gate stands. Otherwise apply the
    whole batch this way, bottom-first per chain, and gate the combined tree once
    before landing — two green PRs can merge red, and package-wide globs are the
    known mechanism. Measured: 19 PRs landed as one batch; every computed tree
    differed from its reviewed head, the plain form falsely conflicted on all 5
    stacked children, the combined tree `23b288eee` was gated once (4829 passed,
    against 4601 on the old tip), and the landed tip `05880556e` carries exactly
    that tree.
  - **After landing:** fast-forward the main worktree (above); close, with a
    handoff comment, a tracker issue whose tasks have all landed; a follow-up
    filed as "claimable once X lands" is now claimable.
- **An approval covers a tree, not a PR.** When a head moves past the comment
  that approved it, the new commits get a delta review pinned to
  `<approved sha>..<head>` before the PR lands. A content-preserving restack needs
  a verification, not a full review — by tree hash or a chunk-by-chunk comparison
  of the result, never a diff of diffs. Measured: **13** heads had moved past
  their approvals, one with an unreviewed commit under two other approved PRs.
- **Recommended, not required: stack a dependent task's PR on its unlanded
  parent** with `gh stack` (`github/gh-stack` v0.1.1, `gh` 2.45.0) rather than
  waiting for it to land. Independent tasks do not stack. It installs under
  `/root`, which `teardown.sh` discards; reinstall with
  `./shell.sh /workspace/gpu_docker/install-gh-stack.sh` (idempotent).
- **Land a stack with `gh stack merge <pr-number> --squash --yes`**: it squashes
  up to and including that PR, leaves the rest open and retargets the PR above —
  no rebase, force-push or base patch. On a stacked PR `PUT /pulls/<n>/merge`
  returns 403 and `PATCH /pulls/<n> -f base=` returns 422 (measured on a probe).
- **`gh stack` gotchas.** `link` needs `--base <branch>`. Only open, non-draft
  PRs merge. There is **no `--message` flag**: with one commit the body survives
  as the squash message, with many GitHub's default applies — the one real cost
  of stacking. `unstack` **can refuse outright** and `DELETE /stacks/<n>` 404s,
  so **link a chain only when you mean it.**
- **Never force-push a branch under review. A restack after its parent has
  landed is permitted.**
- **A chain is linked as a whole or not at all**: `gh stack merge` retargets
  members inside the stack and not those outside it. When a PR joins a chain,
  re-link the whole chain in the same step —
  `gh stack link --base feature/atomcompass_new <bottom> ... <top>`, safe to
  repeat. **Do not link a chain with `need human` anywhere below it.** **Drift
  check:** for each open PR whose base is another open PR's branch, both sit in
  one stack (`gh api "repos/<o>/<r>/stacks?pull_request=<n>"`); held chains are
  counted, not failed. Measured: two chains linked at two PRs each grew to five
  and four.
- **An unlinked chain pays a restack of everything above on each parent
  landing** — `git rebase --onto <new> <old> <branch>` (a plain rebase
  conflicts) plus a REST base patch, per child. A linked `gh stack` does not.
- Except for the main branch, free updates to `jgong5/ATOM` — branches, PRs and
  issues alike, untouched until the project agrees to upstream the milestone.
  Never touch `ROCm/ATOM`.

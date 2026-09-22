## How to talk with owner
- **If there are no blocking issues, say so explicitly** in every message.
- **Quote the context; stop only for critical decisions.**
- **Always communicate PR status.** The owner does not care about local worktree state.
- **Answer with the conclusion first.** When the owner asks what a task
  established, the first line is the **finding**, not the method, the process, or
  what happens next. A reader who stops after one line should have the answer.
  Supporting measurement follows; caveats and cost follow that. The shape, from a
  capture task: *a real model traces at both widths, and its shapes are entirely
  concrete -- 0 symbolic of 13,107* -- then the evidence for each half. What this
  replaces is a status report that recites what was done and leaves the owner to
  infer what it means.
- Output shaping (`/i-have-adhd`): lead with the next action, number multi-step
  tasks, end with one concrete next action, restate state every turn, specific time
  estimates, matter-of-fact error tone, cap lists at 5, no preamble or closing
  pleasantries.

## Execution rules
- Don't modify the main worktree. Develop with linked worktrees, one per in-flight
  task, under `compass-worktrees/<task-id>`, beside the repo.
- **On every landing, the main agent fast-forwards the main worktree** to
  `feature/atomcompass_new` (the integration branch) promptly. This doesn't
  contradict the rule above: that rule
  forbids developing there, not updating it. The pull runs through the container
  as root and leaves the tree root-owned, which fails host-side edits silently;
  chowning it to the host user fixes that but then makes container git refuse
  the same tree with `dubious ownership` until a `safe.directory` entry for that
  path exists in the container's git config. That entry lives under `/root`,
  which does not survive a full `teardown.sh`, so expect to re-add it after a
  container rebuild — script it under `/workspace` rather than doing it by hand
  each time. This happened on #11's landing and needed a manual repair.
- **The fast-forward is easy to skip because nothing visibly breaks**, and it was
  measured **14 commits behind** after a full session of landings. It matters
  beyond tidiness: the main worktree holds the local `feature/atomcompass_new`,
  **every linked worktree shares that ref**, and `compass_resolve_ref` tries the
  bare name first -- so a stale local branch beats `fork/...` for any command that
  omits `COMPASS_INTEGRATION_REF`. That is the defect the resolver was changed to
  announce rather than resolve silently.
- **The chown back is two steps, and the second is not optional.** The container
  runs as uid 0 and the host user is 13797, so root writes with umask 0022 and the
  tree stays root-owned without it. But `chown -R` over the main worktree also
  hits `.git/worktrees/`, **which every linked worktree shares**, and container git
  then refuses all of them with `dubious ownership`. Measured: it broke four
  worktrees at once, mid-session, with agents running.

  ```
  cd <main worktree> && git fetch fork --quiet
  git merge --ff-only fork/feature/atomcompass_new
  chown -R 13797:13797 .      # working tree back to the host user
  chown -R 0:0 .git           # MUST follow, or every linked worktree breaks
  ```

  Working tree 13797-owned, `.git` root-owned -- the state the repo was already
  in, which is why no `safe.directory` entry is needed. Verify with
  `stat -c "%u %n" . .git` and by running `git -C <each worktree> rev-parse HEAD`
  afterwards. Do not assume it worked.
- **Four setup rules, from failures already recorded on this hardware.** None was
  caused by worktrees; all were caused by a shared mutable non-git source tree
  that things silently resolved against.
  1. No shared mutable source root exists. Every tree is a worktree or a
     `git archive` snapshot.
  2. Containers mount the worktree parent, so every task's tree is reachable at a
     stable path. Mounting only one tree makes `pytest` on worktree code fail
     with a path error rather than a test failure.
  3. Every command sets `PYTHONPATH` to its own worktree and verifies it —
     `python -c "import atom; print(atom.__file__)"` before trusting a result. A
     prior run resolved `atom` to a non-git snapshot of a different branch, 72
     files divergent, and failed silently wherever both trees had the symbol.
  4. Snapshots use `git archive`, never `rsync` — so a snapshot names a commit and
     cannot be a mixture of generations. One shared tree held two files from two
     different generations, producing a `TypeError` that named the callee and
     read as a code bug.
- **No design-doc references in code.** No `D18`, `P0.4`, `T5`, `W2.5`, backticked
  doc numbers, "principle N", or numbered labels like "Gate 1". No quoting design
  principles as justification. Say what the code does, its functions, how it works.
  Design docs may cite each other freely; code may not cite them at all. This
  extends to **runtime data** — a `(BEYOND-D18)` suffix on an emitted stub name was
  a citation in the output record.
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
- **A finding that outlives its PR needs an issue, not a PR body.** PR bodies are
  squashed away on landing, so a finding recorded only there is lost to the next
  reader. Two reviews have now re-derived findings that had been written down
  repeatedly with no issue to point at. If a finding is not fixed in the PR that
  found it, open an issue and point the PR body at it.
- **A PR's state is read from the last entry in its comment thread**, never from
  a carried-forward summary. Developer rounds and review verdicts alternate in one
  stream and both open with a bold heading, so a remembered approval may belong to
  an earlier round. One PR sat recorded as approved through six consecutive checks
  while its last entry was a developer record and no reviewer had seen its head.
- Design and implement solutions while keeping the solution as simple as possible.
- **When something does not work as expected, stop and discuss. Do not work
  around it.** This covers: a design document that contradicts the code; a test
  that fails for a reason the task did not predict; a measurement outside its
  stated range; an interface that cannot be implemented as specified. A
  workaround improvised under build pressure is exactly the class of decision
  that never gets written down.
- Concurrency: 5 tasks in flight, up to 10 agents (5 developer + 5 reviewer). The
  cap is review throughput, not the task DAG.
- Both developer and reviewer agents must be told to read `atom/compass/design/README.md`'s
  eight principles first.
- **A developer agent owns development and PR updates; the main agent orchestrates
  and does not write the change itself.** After each push a reviewer agent
  reviews, the developer amends, and that repeats until the verdict is APPROVE.
  The owner is asked only for a critical blocking issue or a scope call — an
  actionable review finding is not an escalation, and is not labelled.
- **Automation is on by default.** An agent acts on any issue or PR that does
  not carry the `need human` label — no opt-in, no waiting to be told.
- **`need human` stops all agent action on that issue or PR** — no agent
  commits to it, reviews it, amends it, or merges it, not even a labelled PR
  whose review already passed. An agent applies the label the moment it
  escalates, so it can stop itself; only the owner removes it, and removal is
  what restarts the work.
- **The review loop has its own stop.** If the same finding survives two cycles,
  or the loop passes three cycles, it halts and goes to the owner and applies
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
     style.
     **The tests must exercise something the PR did not itself add.** A new
     module plus tests for that module, imported by nothing else, is
     self-confirming: it passes every gate and demonstrates nothing. Where a
     task is verification rather than implementation, the brief says so and the
     deliverable is evidence, not a package.
  3. One named result, stated in the issue body before the task is claimed and
     not chosen afterwards.
     An umbrella brief whose children carry the work states one anyway, or
     names the child that carries it. A developer choosing one afterwards is
     the case this rule forbids, and it has happened.
  4. Review by the task's reviewer agent, looping to APPROVE as above.

  Baselines are recorded first (the suite's and ruff's pass/fail state, before the
  first Compass commit) — the lint baseline on this repository is already known
  to be dirty, and a pre-existing failure attributed to Compass costs a day.
- **Effort is estimated in lines of code, not time.** Wall-clock appears only for
  machine time with a measured basis. **A task that overruns its estimate by more
  than ~2x is a halt-and-discuss event, not a reason to keep going** — the usual
  cause is that the task was mis-cut.
- **PRs land squashed onto `feature/atomcompass_new`, the integration branch**,
  one commit per task. GitHub enforces this structurally
  (`allow_merge_commit=false`, `allow_rebase_merge=false`). Base branch is always
  `feature/atomcompass_new` — never `main`, never `master`, never a branch on
  upstream `ROCm/ATOM`.
- **Recommended, not required: stack a dependent task's PR on its unlanded
  parent** with `gh stack` rather than waiting for it to land. Independent
  tasks do not stack. `gh stack` is GitHub's own extension
  (`github/gh-stack`), already installed at v0.1.1 against `gh` 2.45.0. It
  installs under `/root`, which `teardown.sh` discards, so it does not
  survive a container rebuild; reinstall with
  `./shell.sh /workspace/gpu_docker/install-gh-stack.sh`, idempotent. Its
  stack metadata lives in `.git/gh-stack` and is not committed.
- **Land a stack with `gh stack merge <pr-number> --squash --yes`**, which merges
  up to and including that PR and leaves the rest open. Measured on this fork:
  landing the bottom of a two-PR stack produced **one squashed commit carrying
  that PR's message body**, left the upper PR open, **retargeted its base to the
  merged branch automatically**, and left it at `mergeable_state=clean` — no
  rebase, no force-push, no base patch. The plain endpoints are what fail on a
  stacked PR: `PUT /pulls/<n>/merge` returns 403 naming the stack merge path, and
  `PATCH /pulls/<n> -f base=` returns 422. Those are the wrong tools, not a
  limitation.
- **Four things about `gh stack` that cost time if rediscovered.** `gh stack link`
  fails with `unable to determine default branch` unless `--base <branch>` is
  given. `gh stack unstack <n>` refuses while any member is queued for merge, takes
  no `--yes`, and can leave an orphan stack object; `DELETE /stacks/<n>` 404s. Only
  open, non-draft PRs merge. And there is **no `--message` flag**, so a
  hand-written squash message cannot be supplied at merge time — with one commit
  the body survives, with many GitHub's default applies. Since the squash message
  is where a task's measured result is recorded, that is the one real cost of
  stacking.
- **Never force-push a branch under review. A restack after its parent has
  landed is permitted.**
- **Landing the bottom of a hand-managed base chain forces one restack of
  everything above it** — `git rebase --onto <new> <old> <branch>` plus a REST base
  patch, per child, per parent move. A plain `git rebase` conflicts where `--onto`
  does not. **A linked `gh stack` does not pay this**, which is the argument for
  linking a chain rather than hand-managing it.
- Except for the main branch, free updates to `jgong5/ATOM` — branches, PRs and
  issues alike, untouched until the project agrees to upstream the milestone.
  Never touch `ROCm/ATOM`.

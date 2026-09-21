## How to talk with owner
- **If there are no blocking issues, say so explicitly** in every message.
- **Quote the context; stop only for critical decisions.**
- **Always communicate PR status.** The owner does not care about local worktree state.
- Output shaping (`/i-have-adhd`): lead with the next action, number multi-step
  tasks, end with one concrete next action, restate state every turn, specific time
  estimates, matter-of-fact error tone, cap lists at 5, no preamble or closing
  pleasantries.

## Execution rules
- Don't modify the main worktree. Develop with linked worktrees (D97).
- **On every landing, the main agent fast-forwards the main worktree** to the
  integration branch promptly (D97). This doesn't contradict the rule above: that rule
  forbids developing there, not updating it. The pull runs through the container
  as root and leaves the tree root-owned, which fails host-side edits silently;
  chowning it to the host user fixes that but then makes container git refuse
  the same tree with `dubious ownership` until a `safe.directory` entry for that
  path exists in the container's git config. That entry lives under `/root`,
  which does not survive a full `teardown.sh`, so expect to re-add it after a
  container rebuild — script it under `/workspace` rather than doing it by hand
  each time. This happened on #11's landing and needed a manual repair.
- **No design-doc references in code.** No `D18`, `P0.4`, `T5`, `W2.5`, backticked
  doc numbers, "principle N", or numbered labels like "Gate 1". No quoting design
  principles as justification. Say what the code does, its functions, how it works.
  Design docs may cite each other freely; code may not cite them at all. This
  extends to **runtime data** — a `(BEYOND-D18)` suffix on an emitted stub name was
  a citation in the output record.
- Merge conflicts are the agent's call, not the owner's ("don't bother me on merge
  conflict, it's on you").
- Working logs and scratch go in `agent_scratch/`. Nothing durable lives in the
  tree: the task record is the GitHub issue and its PR.
- Task management is GitHub (D96, D97): the PR names its issue, and the issue is
  closed deliberately, with the handoff comment. Agents open, assign, comment on
  and close issues, including issues they did not open.
- Design and implement solutions while keeping the solution as simple as possible.
- Concurrency: 5 tasks in flight, up to 10 agents (`16_execution_plan.md`, D95).
- Both developer and reviewer agents must be told to read `atom/compass/design/README.md`'s
  eight principles first.
- **A developer agent owns development and PR updates; the main agent orchestrates
  and does not write the change itself** (D95). After each push a reviewer agent
  reviews, the developer amends, and that repeats until the verdict is APPROVE.
  The owner is asked only for a critical blocking issue or a scope call — an
  actionable review finding is not an escalation, and is not labelled.
- **Automation is on by default.** An agent acts on any issue or PR that does
  not carry the `need human` label — no opt-in, no waiting to be told.
- **`need human` stops all agent action on that issue or PR** (D95) — no agent
  commits to it, reviews it, amends it, or merges it, not even a labelled PR
  whose review already passed. An agent applies the label the moment it
  escalates, so it can stop itself; only the owner removes it, and removal is
  what restarts the work.
- **The review loop has its own stop.** If the same finding survives two cycles,
  or the loop passes three cycles, it halts and goes to the owner and applies
  `need human` to the PR (`16_execution_plan.md`, D95): a task that cannot
  converge is mis-cut, not under-worked.
- Reviewer agents must post their review to the PR; **the verdict goes in the
  comment body text**, since GitHub refuses APPROVE/REQUEST_CHANGES on
  self-authored PRs (D98). That limitation is about the verdict only — it says
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
- **PRs land squashed onto the integration branch**, one commit per task.
  GitHub enforces this structurally (`allow_merge_commit=false`,
  `allow_rebase_merge=false`) (D97).
- **Recommended, not required: stack a dependent task's PR on its unlanded
  parent** with `gh stack` rather than waiting for it to land (D97.1).
  Independent tasks do not stack. `gh stack` is GitHub's own extension
  (`github/gh-stack`), already installed at v0.1.1 against `gh` 2.45.0. It
  installs under `/root`, which `teardown.sh` discards, so it does not
  survive a container rebuild; reinstall with
  `./shell.sh /workspace/gpu_docker/install-gh-stack.sh`, idempotent. Its
  stack metadata lives in `.git/gh-stack` and is not committed.
- **Never force-push a branch under review. A restack after its parent has
  landed is permitted** (D97.1).
- **Landing the bottom of a stack forces one restack of everything stacked
  above it** (D97.1).
- Except for the main branch, free updates to `jgong5/ATOM` — branches, PRs and
  issues alike. Never touch `ROCm/ATOM`.

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
  self-authored PRs (D98).
- **PRs land squashed onto the integration branch**, one commit per task —
  enforced structurally by the repo's merge settings (D97).
- Except for the main branch, free updates to `jgong5/ATOM` — branches, PRs and
  issues alike. Never touch `ROCm/ATOM`.

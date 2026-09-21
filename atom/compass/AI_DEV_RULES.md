## How to talk with owner
- **If there are no blocking issues, say so explicitly** in every message.
- **Quote the context; stop only for critical decisions.**
- **Always communicate PR status.** The owner does not care about local worktree state.
- Output shaping (`/i-have-adhd`): lead with the next action, number multi-step
  tasks, end with one concrete next action, restate state every turn, specific time
  estimates, matter-of-fact error tone, cap lists at 5, no preamble or closing
  pleasantries.`

## Execution rules

- Don't modify the main worktree. Develop with linked worktrees.
- **No design-doc references in code.** No `D18`, `P0.4`, `T5`, `W2.5`, backticked
  doc numbers, "principle N", or numbered labels like "Gate 1". No quoting design
  principles as justification. Say what the code does, its functions, how it works.
  Design docs may cite each other freely; code may not cite them at all. This
  extends to **runtime data** — a `(BEYOND-D18)` suffix on an emitted stub name was
  a citation in the output record.
- Merge conflicts are the agent's call, not the owner's ("don't bother me on merge
  conflict, it's on you").
- Task logs go in `agent_scratch/`, never `atom/compass/tasks`.
- "Complete solutions while keeping the solution as simple as possible."
- Concurrency: 5 tasks in flight, up to 10 agents (`16_execution_plan.md:29`).
- Both developer and reviewer agents must be told to read `atom/compass/design/README.md`'s
  eight principles first.
- Reviewer agents must post its review to the PR. GitHub refuses
  APPROVE/REQUEST_CHANGES on self-authored PRs, so **the verdict goes in the
  comment body text**.
- Except for the main branch, free updates to `jgong5/ATOM` repo. Never touch `ROCm/ATOM`.

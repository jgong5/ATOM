## How to talk with owner
- **If there are no blocking issues, say so explicitly** in every message.
- **Quote the context; stop only for critical decisions.**
- **Always communicate PR status.** The owner does not care about local worktree state.
- Output shaping (`/i-have-adhd`): lead with the next action — or, when asked what
  a task established, with the finding — number multi-step tasks, end with one
  concrete next action, matter-of-fact error tone, cap lists at 5, no preamble or
  closing pleasantries.

## Execution rules
- Don't modify the main worktree. Develop with linked worktrees, one per in-flight
  task, under `compass-worktrees/<task-id>`, beside the repo.
- **On every landing, the landing agent fast-forwards the main worktree's local
  `feature/atomcompass_new`** (the integration branch). Every linked worktree shares
  its local `feature/atomcompass_new`, and the Compass scripts
  (`scripts/compass/`) diff against that local branch when it exists, before
  falling back to `fork/feature/atomcompass_new` — so a stale local branch gives
  a wrong base unless `COMPASS_INTEGRATION_REF` names the ref explicitly. `fork`
  is the remote for `jgong5/ATOM` (`origin` is `ROCm/ATOM`); in a clone that
  names it differently, substitute that name. After a container rebuild
  (`/root` does not survive `teardown.sh`), run
  `git config --global --add safe.directory '*'` and `gh auth setup-git` before
  any git command. `setup-git` needs a `gh` login, `/root/.config/gh/hosts.yml`,
  which a full teardown also discards, so `gh auth login` or the token file
  `gpu_docker/CLAUDE.md` names may be needed first (unverified: untested after a
  real teardown). The main worktree may have another branch checked out, so a
  refspec fetch fast-forwards the integration branch without a checkout. If
  `git worktree list` shows `[feature/atomcompass_new]` in some worktree, git
  refuses that fetch; run `git fetch fork && git merge --ff-only
  fork/feature/atomcompass_new` in that worktree instead. Git as root leaves
  files root-owned, which fails host-side edits silently, so chown whichever
  worktree git ran in, and the shared `.git`, after every update:

  ```
  cd <main worktree>
  git fetch fork feature/atomcompass_new:feature/atomcompass_new
  chown -R 13797:13797 . "$(git rev-parse --git-common-dir)"
  find . "$(git rev-parse --git-common-dir)" ! -uid 13797 | head -1   # prints nothing
  git -C <each worktree> rev-parse HEAD
  ```
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
- **No design-doc references in code or runtime output.** No `D18`, `P0.4`, `T5`,
  `W2.5`, backticked doc numbers, "principle N", or labels like "Gate 1"; no
  quoting principles as justification. Say what the code does. Design docs may
  cite each other. Check at the head over the PR's whole file set, never over a
  delta's added lines. A design document a test opens by path is a dependency,
  not a citation.
- Merge conflicts are the agent's call, not the owner's. Tasks are cut to one
  module plus its tests but are not guaranteed disjoint. **Frequent conflicts mean
  the decomposition is wrong**: re-cut the tasks, don't add a scheduler.
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

  Every brief links its predecessors' issues. **A brief that cannot name its file
  set is not claimable.** **Decompose a complex task into sub-tasks** before it is
  claimed (roughly: an estimate over ~1000 lines including tests, or three or
  more deliverables).
  Each sub-task is its own issue, with its own brief and its dependencies stated.
- Task management is GitHub: the PR names its issue, and the issue is closed
  deliberately, with the handoff comment. Agents open, assign, comment on and
  close issues, including issues they did not open. A finding not fixed in the PR
  that found it gets an issue: PR bodies are squashed away on landing.
- **Check delivery before claiming or briefing an issue.** Read its
  **comments**, not only its body, and check whether any open PR names it in its
  title or with a delivering verb (closes / fixes / resolves / addresses /
  implements). **Do not write "checked" unless the check you ran is the one that
  answers the claim.**
- **A PR's state is its last thread entry; its verdict is the last comment that
  carries one.** Read both from the thread, never from a carried-forward summary:
  developer rounds and review verdicts alternate and both open with a bold heading.
- **When something does not work as expected, stop and diagnose it. Do not work
  around it** — the workaround is forbidden, the diagnosis is not. This covers: a
  design document that contradicts the code; a test that fails for a reason the
  task did not predict; a measurement outside its stated range; an interface that
  cannot be implemented as specified. The outcome is a finding or an escalation
  (below).
- Concurrency: 5 tasks in flight, up to 10 agents (5 developer + 5 reviewer). The
  cap is review throughput, not the task DAG.
- Both developer and reviewer agents must be told to read `atom/compass/design/README.md`'s
  eight principles first.
- **A developer agent owns development and PR updates; the main agent orchestrates
  and does not write the change itself.** After each push a reviewer agent
  reviews, the developer pushes fixes, and that repeats until the verdict is APPROVE.
  The developer works under the
  [`ponytail`](https://github.com/DietrichGebert/ponytail/blob/main/skills/ponytail/SKILL.md)
  skill at level `full` (`/ponytail full`).
- **Escalations and `need human`.** An escalation is anything that needs an owner
  ruling before work can continue; anything the developer can fix without one is
  a finding, and the owner is never asked about findings. An agent applies
  `need human` the moment it escalates — a halt declared in prose stops nothing;
  when the ruling lives on another issue, label each PR it holds and name that
  issue. The label stops all agent action on that issue or PR (no commit, review,
  amend or merge, even after a passed review), with two exceptions. One is `gh
  stack link` by PR number, which lands and pushes nothing, though it retargets
  the linked PRs' bases (then and when a PR below lands). **The other is a base
  update: an agent may merge as the branch-update rule (below) calls for,
  patching the PR's base via REST just before the push if it is an unlinked child
  whose parent landed, changing nothing else.** The merge keeps every change from
  both sides; where it cannot, the agent commits nothing and names the conflict
  hunk in a PR comment. A PR comment lists each resolved file, and the label
  allows one delta review of the resolutions. Only the owner removes the label.
  **Without it, automation is on by default**: agents act with no opt-in.
- **The review loop has its own stop.** If the same finding survives two cycles,
  or the loop passes three cycles, it halts, goes to the owner and applies
  `need human` to the PR: a task that cannot converge is mis-cut, not
  under-worked.
- **Reviews go on the PR.** The verdict and summary go in one standalone comment
  body (GitHub refuses APPROVE/REQUEST_CHANGES on self-authored PRs). A finding
  that points at lines is an inline comment via
  `gh api repos/<owner>/<repo>/pulls/<n>/comments` with `body`, `commit_id`,
  `path` and `line`; only findings with no line go in the standalone comment.
- **Four gates land a task, all required:**
  1. ATOM's test suite passes unmodified, in two tiers: the GPU-free tier per
     task, the GPU superset per wave as a delta. Needing to edit an ATOM test
     means the change altered ATOM's behaviour and must be justified on its own
     terms, not absorbed.
  2. New CPU-only tests for what the task added, in `tests/compass/`, in ATOM's
     style. **The tests must exercise something the PR did not itself add.** A
     new module plus tests for that module, imported by nothing else, is
     self-confirming: it passes every gate and demonstrates nothing.
  3. One named result, stated in the issue body before the task is claimed and
     not chosen afterwards. An umbrella brief whose children carry the work
     states one anyway, or names the child that carries it; a developer choosing
     one afterwards is the case this rule forbids.
  4. Review by the task's reviewer agent, looping to APPROVE as above. The
     reviewer also runs the
     [`ponytail-review`](https://github.com/DietrichGebert/ponytail/blob/main/skills/ponytail-review/SKILL.md)
     skill over the diff to catch over-engineering; its findings are posted like
     any others. **A check
     counts only once someone has seen it fire.** A reviewer credits a test with
     holding a defect only after reinstating it (the pre-fix code via `git show`,
     line count preserved, nothing else changed) and recording the red: both
     counts, the failing node id and assertion. A developer reverts their own
     fix the same way before claiming it. **An inert pin on a required finding
     blocks APPROVE.**

  Record the suite's and ruff's baselines before the first Compass commit; the
  ruff baseline is already dirty.
- **Effort is estimated in lines of code, not time.** Wall-clock appears only for
  machine time with a measured basis. **A task that overruns its estimate by more
  than ~2x is an escalation**, not a reason to keep going.
- **PRs land squashed onto `feature/atomcompass_new`**, one commit per task with a
  written message — never `main` or `master`. Land with `PUT
  repos/<o>/<r>/pulls/<n>/merge-async` (stacked or not), `merge_method=squash`,
  `merge_action=direct_merge`, `sha=<approved head>`, a `commit_title` ending in
  ` (#<n>)` (GitHub does not add it) and `-F commit_message=@<file>`. It is
  asynchronous: poll `GET repos/<o>/<r>/pulls/<n>/merge-async/<uuid>`, with the
  `uuid` from the response's `details`, until `status` is `merged` (`failed` is a
  finding to diagnose), before the fast-forward below or the next PR up a stack,
  which lands bottom-first.
- **Landing is the agents' job; no owner approval is needed or sought.** An agent
  lands any PR whose APPROVE covers its current head (below) and with no
  `need human` on it or anywhere below it in its stack; it does not wait for the
  per-wave GPU superset. Landing a stacked PR lands every unlanded PR below it,
  so each of those needs the same: its own APPROVE covering its head, and no
  label. A PR whose body declares an escalation without the label
  gets the label. **Holds are landing preconditions, and there are three:**
  `need human` on the PR or below it (a label on an issue a PR delivers counts as
  on that PR; every escalation rule in this file holds through this label), an
  APPROVE covering each head (the reviewer checks gates 1-3 as they apply per task
  before approving, and the approval is gate 4), and the tree check below. A
  hold names the one that is unmet. **A violation of any other rule seen in an
  approved PR is landed and filed as an issue, not held.** Where a
  handoff note contradicts this file, this file wins.
  - **Before landing on a moved tip, compute the tree that will land:**
    `git merge-tree --write-tree <current tip> <reviewed head>`, adding
    `--merge-base <parent's reviewed head>` for a stacked PR (the plain form
    reports false conflicts). If the result is `<reviewed head>^{tree}`, the gate
    stands; otherwise apply the whole batch bottom-first per chain and gate the
    combined tree once before landing — two green PRs can merge red.
  - **After landing:** fast-forward the main worktree (above); close, with a
    handoff comment, a tracker issue whose tasks have all landed; a follow-up
    filed as "claimable once X lands" is now claimable.
- **An approval covers a tree, not a PR.** When a head moves past the comment
  that approved it, the new commits get a delta review pinned to
  `<approved sha>..<head>` before the PR lands, read with `git log -p
  --first-parent --diff-merges=remerge <approved sha>..<head>`: each commit's
  own diff, and only the conflict resolutions of each merge. What a merge brings
  in is reviewed in its own PR, landed or still under review.
- **Recommended, not required: stack a dependent task's PR on its unlanded
  parent** with `gh stack` (`github/gh-stack` v0.1.1); independent tasks do not
  stack. Reinstall after a container rebuild with
  `./shell.sh /workspace/gpu_docker/install-gh-stack.sh`.
  - Link a chain whole or not at all, and only when you mean it (`unstack` can
    refuse): `gh stack link --base feature/atomcompass_new <bottom-pr#> ... <top-pr#>`,
    re-run whenever a PR joins, held members included (linking lands nothing).
    Only linked members are retargeted when a parent lands. A fork (two or more
    open PRs based on one open PR's branch) links at most one arm. Drift check:
    every open PR based on another open PR's branch sits in one stack, fork arms
    excepted
    (`gh api "repos/<o>/<r>/stacks?pull_request=<n>"`).
  - Land a stack one PR at a time (above), never with `gh stack merge`: it takes
    no message, and it force-pushes the child after landing. When a parent lands,
    GitHub retargets a linked child itself. An unlinked child (a fork arm, or a
    chain never linked) gets its base patched via REST and the new integration
    tip merged into it, so its diff shows only its own changes.
- **PR branches only gain commits: no force-push, no rebase, no amend.** Answer
  review findings with new commits. Update a branch only when it conflicts, needs
  code landed since, or it is an unlinked child whose parent landed (above) — a
  moved tip alone needs no update, the tree check covers it — and then by merging
  freshly fetched `fork/feature/atomcompass_new`, or the parent's head, into it.
  Never run `gh stack rebase`, `sync`, `push` or `submit`: they rebase or
  force-push. A secret or large binary pushed by mistake
  is the one case that needs a force-push, and it is an escalation. The branch
  lands squashed, so its merge commits never reach the integration branch, and
  GitHub's incremental review and the delta review stay intact.
- Except for the main branch, free updates to `jgong5/ATOM` — branches, PRs and
  issues alike, untouched until the project agrees to upstream the milestone.
  Never touch `ROCm/ATOM`.

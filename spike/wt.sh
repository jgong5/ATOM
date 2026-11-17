#!/bin/bash
# usage: wt.sh -- as root in a gate container: the task worktree ATOM-worktrees/3636 on task-3636 from the fetched
# origin/feature/atomcompass_new, owned back to xingjche; the import check there. Writes logs/wt.log, ending WT_RC.
exec >> /shared_nfs/xingjche/agent_scratch/3636/logs/wt.log 2>&1
A=/shared_nfs/xingjche/ATOM W=/shared_nfs/xingjche/ATOM-worktrees/3636
git config --global --add safe.directory '*'
[[ -e $W ]] && { echo "$W exists: WT_RC=2"; exit 2; }
git -C $A worktree add $W -b task-3636 origin/feature/atomcompass_new || { echo WT_RC=1; exit 1; }
chown -R 50109788:1000 $W $A/.git/worktrees/3636 $A/.git/refs/heads/task-3636 2>/dev/null
chown 50109788:1000 $A/.git/logs/refs/heads/task-3636 $A/.git/index 2>/dev/null
find $A/.git -user 0 2>/dev/null | head -5
git -C $W log --oneline -1
(cd / && PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 python3 -c 'import atom; print(atom.__file__)' 2>&1 | tail -1)
(cd / && PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 python3 -c 'from atom.compass.run import atom_digest; print("atom_digest", atom_digest())' 2>&1 | tail -1)
echo WT_RC=0

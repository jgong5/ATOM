#!/bin/bash
# usage: mesh.sh -- as root in a gate container: atomesh at 3911f83ba (#1498's port worktree, clean) into the
# container's /tmp, copied to 3636/mesh/atomesh. Writes logs/mesh.log, ending MESH_RC.
W=/shared_nfs/xingjche/ATOM-worktrees/1498-port S=/shared_nfs/xingjche/agent_scratch/3636
exec >> $S/logs/mesh.log 2>&1
git config --global --add safe.directory '*'
ulimit -n "$(ulimit -Hn)"
[[ $(git -C $W rev-parse HEAD) == 3911f83bab97ec0a6951cfd5202de3011a84da1e && -z $(git -C $W status --porcelain) ]] || { echo "$W not clean at 3911f83ba: $(git -C $W status --porcelain | head -3)"; echo MESH_RC=2; exit 2; }
rm -rf /tmp/3636mesh
bash /shared_nfs/xingjche/agent_scratch/975/run/mesh_b.sh $W /tmp/3636mesh
mkdir -p $S/mesh && cp /tmp/3636mesh/release/atomesh $S/mesh/atomesh && rm -rf /tmp/3636mesh
$S/mesh/atomesh --version-verbose | grep -E 'Git (Commit|Status)'
[[ -z $(git -C $W status --porcelain) ]] || { echo "the build left $W unclean"; echo MESH_RC=3; exit 3; }
chown -R 50109788:1000 $S/mesh $S/logs
echo MESH_RC=0

#!/bin/bash
# Run a command under the shared per-host agent lock (a directory under $LOCKROOT): wait until mkdir succeeds, record the
# task, run, release. usage: scripts/with_host_lock.sh TASK_NAME command [args...]
LOCKROOT=${LOCKROOT:-/exp/mu2e/data/users/ram2aq/agent_locks}
L="$LOCKROOT/$(hostname -s)"
TASK=$1; shift
mkdir -p "$LOCKROOT"
until mkdir "$L" 2>/dev/null; do sleep 20; done
# release only a lock this process still owns (another holder may have replaced it)
trap 'grep -q "(pid $$," "$L/task" 2>/dev/null && rm -rf "$L"' EXIT
echo "$TASK (pid $$, started $(date -u +%FT%TZ))" > "$L/task"
echo "# lock taken $(date -u +%FT%TZ)"
"$@" &
child=$!
# a signal to the wrapper stops the command first; the lock is released only after it has ended
trap 'kill -TERM $child 2>/dev/null' TERM INT HUP
wait $child
rc=$?
while kill -0 $child 2>/dev/null; do wait $child; rc=$?; done
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc

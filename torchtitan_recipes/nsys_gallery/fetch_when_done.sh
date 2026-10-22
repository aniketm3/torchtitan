#!/usr/bin/env bash
# Laptop side: wait for the pod's DONE file, then copy results back.
# Exits 2 (and still copies what exists) if progress.log is unchanged for 10 min.
# Usage: bash fetch_when_done.sh <host> <port> <local_dir>
set -u
host=$1; port=$2; dest=$3
remote=/root/nsys_gallery_out
mkdir -p "$dest"
ssh_cmd=(ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -p "$port" "root@$host")
last_size=-1; stale=0
while true; do
    state=$("${ssh_cmd[@]}" "test -f $remote/DONE && echo DONE; stat -c %s $remote/progress.log 2>/dev/null || echo 0")
    echo "$(date -u +%T) $(echo $state | tr '\n' ' ')"
    if echo "$state" | grep -q DONE; then code=0; break; fi
    size=$(echo "$state" | tail -1)
    if [[ "$size" == "$last_size" ]]; then stale=$((stale + 1)); else stale=0; fi
    last_size=$size
    if [[ $stale -ge 10 ]]; then echo "STALL: progress.log unchanged for 10 min"; code=2; break; fi
    sleep 60
done
rsync -az -e "ssh -p $port" "root@$host:$remote/" "$dest/" && echo "COPIED to $dest"
exit $code

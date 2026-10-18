#!/usr/bin/bash
# Run on the laptop: wait for the pod's sweep to finish (DONE marker), copy
# sweep_logs/ and outputs/ back, then tell the pod it may stop (COPIED marker).
#   bash fetch_and_stop.sh <host> <port> <remote_log_dir relative to repo>
set -u
host=$1
port=$2
log_dir=$3
ssh_cmd=(ssh -p "$port" -o BatchMode=yes -o ConnectTimeout=20 "root@$host")
repo=/root/torchtitan

for _ in $(seq 240); do
    if "${ssh_cmd[@]}" "test -f $repo/$log_dir/DONE"; then
        rsync -az -e "ssh -p $port -o BatchMode=yes" "root@$host:$repo/sweep_logs/" ./sweep_logs/
        rsync -az -e "ssh -p $port -o BatchMode=yes" "root@$host:$repo/outputs/" ./outputs/
        "${ssh_cmd[@]}" "touch $repo/$log_dir/COPIED"
        echo "Copied $log_dir and outputs/ at $(date)"
        exit 0
    fi
    sleep 60
done
echo "Timed out waiting for $log_dir/DONE"
exit 1

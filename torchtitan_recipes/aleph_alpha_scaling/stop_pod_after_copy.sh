#!/usr/bin/bash
# Run on the pod at the end of a sweep: mark it DONE, wait up to 30 min for the
# laptop to copy results off (fetch_and_stop.sh touches COPIED), then stop the
# pod. Stopping wipes the container disk, hence the wait.
#   stop_pod_after_copy.sh <log_dir>
set -u
log_dir=$1
touch "$log_dir/DONE"
# The pod ID is only exported to interactive shells; read it from RunPod's env file.
pod_id=$(sed -n 's/^.*RUNPOD_POD_ID="\{0,1\}\([a-z0-9]*\).*$/\1/p' /etc/rp_environment)
echo "Waiting for $log_dir/COPIED before stopping pod $pod_id"
for _ in $(seq 180); do
    [[ -f "$log_dir/COPIED" ]] && break
    sleep 10
done
echo "Stopping pod $pod_id"
# The pod's built-in RUNPOD_API_KEY cannot stop pods; use the account key
# installed with `runpodctl config --apiKey=...`.
env -u RUNPOD_API_KEY runpodctl stop pod "$pod_id"

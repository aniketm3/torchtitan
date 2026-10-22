#!/usr/bin/env bash
# Make `nsys` (Nsight Systems CLI) available on a RunPod pod. Run as root.
set -euo pipefail

if command -v nsys >/dev/null; then
    nsys --version
    exit 0
fi
# CUDA images sometimes ship it outside PATH.
for candidate in /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems*/*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys; do
    if [[ -x "$candidate" ]]; then
        ln -sf "$candidate" /usr/local/bin/nsys
        nsys --version
        exit 0
    fi
done

# NVIDIA devtools apt repo -> latest nsight-systems-cli.
. /etc/os-release
repo="ubuntu${VERSION_ID//./}/$(dpkg --print-architecture)"
apt-get update -qq
apt-get install -y -qq --no-install-recommends gnupg wget ca-certificates >/dev/null
wget -qO- "https://developer.download.nvidia.com/devtools/repos/${repo}/nvidia.pub" |
    gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/${repo}/ /" \
    >/etc/apt/sources.list.d/nvidia-devtools.list
apt-get update -qq
apt-get install -y -qq --no-install-recommends nsight-systems-cli >/dev/null
nsys --version

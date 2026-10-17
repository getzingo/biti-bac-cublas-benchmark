#!/usr/bin/env bash

set -euo pipefail
cd -- "$(dirname -- "$0")"

NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
DCGM_BINDINGS="${DCGM_BINDINGS:-/usr/local/dcgm/bindings/python3}"

if [[ $EUID -eq 0 ]]; then
  SUDO=()
elif sudo -n true 2>/dev/null; then
  SUDO=(sudo)
else
  echo 'no sudo' >&2
  exit 1
fi

apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "already installed: $*"
    return
  fi
  echo "installing: ${missing[*]}"
  "${SUDO[@]}" apt-get update -qq
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y "${missing[@]}"
}

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi missing: no usable GPU/driver here' >&2; exit 1; }

dcgm_policy=$(apt-cache policy datacenter-gpu-manager 2>/dev/null || true)
if ! grep -q 'Candidate: [0-9]' <<<"$dcgm_policy"; then
  echo 'adding the NVIDIA cuda-keyring repository for DCGM'
  keyring=/tmp/cuda-keyring_1.1-1_all.deb
  curl -fsSL -o "$keyring" \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
  "${SUDO[@]}" dpkg -i "$keyring"
  rm -f "$keyring"
fi

apt_install python3-venv datacenter-gpu-manager

"${SUDO[@]}" systemctl enable --now nvidia-dcgm >/dev/null 2>&1 || true
for _ in $(seq 30); do
  pgrep -x nv-hostengine >/dev/null && break
  sleep 1
done
if ! pgrep -x nv-hostengine >/dev/null; then
  "${SUDO[@]}" nv-hostengine || true
  sleep 2
fi
pgrep -x nv-hostengine >/dev/null || {
  echo 'nv-hostengine is not running; telemetry cannot record without it' >&2
  exit 1
}

command -v "$NVCC" >/dev/null || { echo 'CUDA toolkit required, set NVCC=/path/to/nvcc' >&2; exit 1; }
command -v nvidia-cuda-mps-control >/dev/null || \
  echo 'warning: no nvidia-cuda-mps-control on PATH, MPS cells will fail'

python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

"$NVCC" -std=c++14 -O3 -arch="$CUDA_ARCH" run_worker.cu -lcublas -o run_worker
if ldd run_worker | grep -q 'not found'; then
  echo 'run_worker cannot resolve its shared libraries:' >&2
  ldd run_worker | grep 'not found' >&2
  exit 1
fi

DCGM_BINDINGS="$DCGM_BINDINGS" .venv/bin/python - <<'PY'
import os, sys

sys.path.insert(0, os.environ["DCGM_BINDINGS"])
import pydcgm  # noqa: F401
import dcgm_agent, dcgm_fields, dcgm_structs, dcgm_field_helpers  # noqa: F401
from DcgmHandle import DcgmHandle
from DcgmFieldGroup import DcgmFieldGroup  # noqa: F401

handle = DcgmHandle(ipAddress="127.0.0.1")
print("DCGM host engine reachable")
handle.Shutdown()
PY
dcgm_version=$(dcgmi --version | awk 'NF{print $NF; exit}')
echo "DCGM host engine running (dcgmi version ${dcgm_version:-unknown})"

mig_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader)
mig_uuid=$(nvidia-smi -L | grep -o 'MIG-[0-9a-f-]*' | head -1 || true)
if [[ "$mig_mode" == "Enabled" && -z "$mig_uuid" ]]; then
  echo 'worker smoke test skipped MIG is enabled with no GPU instances yet'
else
  printf 'GO\n' | CUDA_VISIBLE_DEVICES="${mig_uuid:-0}" ./run_worker 0 8192 8192 8192 fp32 0 1 >/dev/null
  echo "worker smoke test passed on ${mig_uuid:-the whole GPU}"
fi

echo "setup complete: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

#!/usr/bin/env bash
# Prepare a benchmark host: OS packages (python venv, DCGM), the DCGM host
# engine, the python environment, and the CUDA worker binary.
#
# The host is disposable: the VM is deleted between campaigns, so this script
# is what carries the knowledge of a bare box. A fresh Hyperstack "Ubuntu
# Server 22.04 LTS R535 CUDA 12.2" image comes with nvidia-smi, /usr/local/cuda
# (nvcc NOT on PATH), cuBLAS on the loader path, /usr/bin/nvidia-cuda-mps-control,
# python3.10 with pip, passwordless sudo - and nothing else this benchmark
# needs: no DCGM (no host engine, no python bindings) and no python3-venv, so a
# bare `python3 -m venv` fails with "ensurepip is not available". It also boots
# with MIG mode already ENABLED and zero GPU instances, and in that state no
# CUDA process can initialise at all.
#
# Everything here is checked before it is done, so a fresh clone, a rebooted
# box and an already prepared box all end up in the same state and the script
# is safe to re-run. It needs passwordless sudo, because the package install
# and the DCGM service belong to root.
#
# Running a campaign afterwards - this part the script cannot do for you:
#   sudo .venv/bin/python entrypoint.py            # as root: as a normal user
#                                                  # every clock/MIG nvidia-smi
#                                                  # call is refused
#   ...reboot when it prints "Reboot required"...   # a MIG on/off switch only
#   sudo .venv/bin/python entrypoint.py --resume   # lands on the next boot
#   sudo .venv/bin/python post_processing.py <run dir>
# The run dir belongs to root, so post_processing needs sudo too. The clock
# lock outlives the process: only a reboot or `nvidia-smi -rgc` clears it.
# entrypoint line-buffers its output, so a redirected log is readable live.

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
  echo 'setup needs sudo (passwordless) to install packages' >&2
  exit 1
fi

# apt, but only for what is actually absent. The package database knows that
# better than a marker file we would have to keep in sync.
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

# DCGM lives in NVIDIA's CUDA repo. The CUDA images ship that repo already, a
# plain ubuntu image does not, so only reach for the keyring when apt cannot
# see the package at all. The output is captured first: grep -q walks away
# early, and under pipefail the SIGPIPE that gives apt-cache would otherwise
# read as "no candidate".
dcgm_policy=$(apt-cache policy datacenter-gpu-manager 2>/dev/null || true)
if ! grep -q 'Candidate: [0-9]' <<<"$dcgm_policy"; then
  echo 'adding the NVIDIA cuda-keyring repository for DCGM'
  keyring=/tmp/cuda-keyring_1.1-1_all.deb
  curl -fsSL -o "$keyring" \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
  "${SUDO[@]}" dpkg -i "$keyring"
  rm -f "$keyring"
fi

# python3-venv is its own package on ubuntu: without it "python3 -m venv" dies
# with "ensurepip is not available". DCGM carries the python bindings and the
# host engine that telemetry.py talks to.
apt_install python3-venv datacenter-gpu-manager

# telemetry.py connects to a host engine on 127.0.0.1 and deliberately does not
# start one, so the daemon has to be running before the first cell. The deb
# calls the unit nvidia-dcgm.service (dcgm.service is the deprecated twin that
# conflicts with it) and leaves it disabled, so enable it here and then wait
# for the process rather than for the unit, which also covers a manual start.
"${SUDO[@]}" systemctl enable --now nvidia-dcgm >/dev/null 2>&1 || true
for _ in $(seq 30); do
  pgrep -x nv-hostengine >/dev/null && break
  sleep 1
done
if ! pgrep -x nv-hostengine >/dev/null; then
  # No systemd, or the unit refused to start: it is a daemon, so run it.
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

# Everything the benchmark touches is exercised here instead of in the middle
# of a 126-step run: the CLI imports, the DCGM bindings against the live host
# engine, and one real GEMM through the freshly built worker.
.venv/bin/python entrypoint.py --help >/dev/null

DCGM_BINDINGS="$DCGM_BINDINGS" .venv/bin/python - <<'PY'
import os, sys
# Same import order as telemetry.py: the binding modules import each other, so
# pydcgm has to come first or DcgmHandle.py hits a half-built module.
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

# The worker needs a device CUDA can actually use, and picks it the way
# workers.py does: a MIG instance's UUID when MIG is on, device 0 when it is
# off. A stock A100 boots MIG-enabled with no instances, and then the CUDA
# runtime cannot initialise at all, so there is nothing to test until the
# benchmark creates a layout; say so instead of failing.
mig_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader)
mig_uuid=$(nvidia-smi -L | grep -o 'MIG-[0-9a-f-]*' | head -1 || true)
if [[ "$mig_mode" == "Enabled" && -z "$mig_uuid" ]]; then
  echo 'worker smoke test skipped: MIG is enabled with no GPU instances yet'
else
  printf 'GO\n' | CUDA_VISIBLE_DEVICES="${mig_uuid:-0}" ./run_worker 0 8192 8192 8192 fp32 0 1 >/dev/null
  echo "worker smoke test passed on ${mig_uuid:-the whole GPU}"
fi

echo "setup complete: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

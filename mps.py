from config import Config
from utils import run_cmd
import os

CONTROL = "nvidia-cuda-mps-control"


class Mps:
  def __init__(self, config: Config):
    self.config = config
    self._started = False

  def is_running(self) -> bool:
    """check if mps daeomon is even running"""
    try:
      return run_cmd([CONTROL], input="get_server_list\n")["rc"] == 0
    except FileNotFoundError:
      return False  # MPS isn't installed, so it can't be running

  def server_pids(self) -> set[int]:
    """find PIDs of all running mps servers"""
    if self.config.dry_run or not self._started:
      return set()
    result = run_cmd([CONTROL], input="get_server_list\n")
    return {int(w) for w in result["stdout"].split() if w.isdigit()}

  def start(self):
    if self.config.dry_run:
      print("[dry-run] would start the MPS control daemon")
      return
    # fix mps daemon resource usage
    caps = [k for k in os.environ
            if k in ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                     "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT")]
    if caps:
      raise RuntimeError(f"Unset {', '.join(caps)} before running MPS cells")
    if self.is_running():
      raise RuntimeError(
        "a MPS daemon is already running stop it:"
        f"'echo quit | {CONTROL}'")
    try:
      result = run_cmd([CONTROL, "-d"])
    except FileNotFoundError:
      raise RuntimeError(f"{CONTROL} not found, MPS installed?")
    if result["rc"] != 0:
      raise RuntimeError(f"MPS daemon failed to start: {result['stderr']}")
    self._started = True
    if not self.is_running():
      raise RuntimeError("MPS daemon hangs again, forgot sudo again?")

  def stop(self):
    if self.config.dry_run or not self._started:
      return
    result = run_cmd([CONTROL], input="quit\n")
    self._started = False
    if result["rc"] != 0:
      print(f"WARNING: MPS died ungracefully: {result['stderr']}")

import time

import pynvml as nvml

from config import Config
from mig import MigLayout, PROFILES
from utils import run_cmd


class RebootRequired(RuntimeError):
  pass


class A100:
  # workaround slow MIG layout changes
  LAYOUT_TIMEOUT_S = 120.0

  def __init__(self, config: Config):
    self.config = config
    if config.dry_run:
      return
    nvml.nvmlInit()
    self._handle = nvml.nvmlDeviceGetHandleByIndex(0)
    name = nvml.nvmlDeviceGetName(self._handle)
    self.name = name.decode() if isinstance(name, bytes) else name
    self.uuid = nvml.nvmlDeviceGetUUID(self._handle)
    if isinstance(self.uuid, bytes):
      self.uuid = self.uuid.decode()
    self.driver_version = nvml.nvmlSystemGetDriverVersion()
    if isinstance(self.driver_version, bytes):
      self.driver_version = self.driver_version.decode()

  def shutdown(self):
    if not self.config.dry_run:
      nvml.nvmlShutdown()

  def apply_controls(self):
    """lock SM clock and power limit"""
    target_clock = self.config.clock_mhz
    if self.config.dry_run:
      print(f"[dry-run] would lock SM clock to {target_clock} MHz")
      return
    result = run_cmd(["nvidia-smi", "-i", "0", "-lgc", f"{target_clock},{target_clock}"])
    if result.get("rc") != 0:
      raise RuntimeError(f"Clock locking failed: {result['stderr']}")

  def get_mig_layout(self) -> MigLayout:
    return MigLayout(self._handle)

  def set_mig_layout(self, profile_name: str):
    profile = PROFILES.get(profile_name)

    if self.config.dry_run:
      print(f"[dry-run] would repartition GPU into "
            f"{profile.instance_count}x {profile.name}")
      return

    current_mig_layout = self.get_mig_layout()
    if current_mig_layout.detect_profile() == profile_name:
      return

    if self.confounding_gpu_processes():
      raise RuntimeError("processes still running on GPU, not changing profiles")

    if not current_mig_layout.enabled:
      self._set_mig_mode(1)
      current_mig_layout = self.get_mig_layout()

    # workaroung lingering MIG slices
    if current_mig_layout.instances:
      self._destroy_mig_slices()

    self._smi_checked(
      ["mig", "-cgi", ",".join([profile.name] * profile.instance_count), "-C"])

    self._await_layout(profile_name)

  def _await_layout(self, profile_name: str):
    deadline = time.monotonic() + self.LAYOUT_TIMEOUT_S
    while True:
      layout = self.get_mig_layout()
      if layout.detect_profile() == profile_name:
        return
      if time.monotonic() >= deadline:
        layout.validate(profile_name)  # raises with actual mismatch
        return
      time.sleep(0.5)

  def disable_mig(self):
    if self.config.dry_run:
      print("[dry-run] would make sure MIG is disabled")
      return
    current = self.get_mig_layout()
    if not current.enabled:
      return
    if self.confounding_gpu_processes():
      raise RuntimeError("processes still running on GPU, not changing profiles")
    if current.instances:
      self._destroy_mig_slices()
    self._set_mig_mode(0)

  def _set_mig_mode(self, mode):
    """on == 1, off == 0"""
    self._smi_checked(["-i", "0", "-mig", str(mode)])
    current, pending = nvml.nvmlDeviceGetMigMode(self._handle)
    if current != pending:
      state = "enable" if mode else "disable"
      raise RebootRequired(f"MIG {state} pending until bench reboots")

  def _destroy_mig_slices(self):
    self._smi_checked(["mig", "-dci"])
    # double-kill needed
    self._smi_checked(["mig", "-dgi"])

  @staticmethod
  def _smi_checked(args):
    """nvidia-smi plus args"""
    result = run_cmd(["nvidia-smi", *args])
    if result["rc"] != 0:
      raise RuntimeError(
        f"nvidia-smi {' '.join(args)} failed: {result['stderr']}")


  def confounding_gpu_processes(self):
    """makes sure no process like the lingering DCGM uses MIG slice"""
    if self.config.dry_run:
      return set()
    handles = [self._handle]
    try:
      for slot in range(nvml.nvmlDeviceGetMaxMigDeviceCount(self._handle)):
        try:
          handles.append(
            nvml.nvmlDeviceGetMigDeviceHandleByIndex(self._handle, slot))
        except nvml.NVMLError_NotFound:
          # empty slot
          continue
    except nvml.NVMLError:
      pass
    pids = set()
    for handle in handles:
      for query in (nvml.nvmlDeviceGetComputeRunningProcesses,
                    nvml.nvmlDeviceGetMPSComputeRunningProcesses):
        try:
          pids.update(p.pid for p in query(handle))
        except nvml.NVMLError:
          continue
    return pids

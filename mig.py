"""MIG layout inspection and validation for the A100."""

from dataclasses import dataclass
import pynvml as nvml


@dataclass(frozen=True)
class MigProfile:
  """
  instance_count: how many slices the profile can create from gpu
  memory_gib: gpu memory the slice holds
  sms: SM count the slice holds
  """
  name: str
  instance_count: int
  memory_gib: int
  sms: int

PROFILES = {
  "1g.10gb": MigProfile("1g.10gb", instance_count=7, memory_gib=10, sms=14),
  "1g.20gb": MigProfile("1g.20gb", instance_count=4, memory_gib=20, sms=14),
  "3g.40gb": MigProfile("3g.40gb", instance_count=2, memory_gib=40, sms=42),
}

# NVML inaccuracy compensation
MEMORY_TOLERANCE_GIB = 2


@dataclass
class MigInstance:
  """single MIG slice from nvml"""
  uuid: str
  gpu_instance_id: int
  memory_gib: float


class MigLayout:
  """current MIG config"""

  def __init__(self, handle, gpu_index=0):
    self._handle = handle
    current, pending = nvml.nvmlDeviceGetMigMode(handle)
    if current != pending:
      # pending means the gpu is mid-transition
      raise RuntimeError(
        "MIG mode change is pending; finish or revert it before measuring")
    self.enabled = bool(current)
    self.instances = self._enumerate() if self.enabled else []

  def _enumerate(self):
    """dont recreate existing MIG slices"""
    found = []
    for slot in range(nvml.nvmlDeviceGetMaxMigDeviceCount(self._handle)):
      try:
        child = nvml.nvmlDeviceGetMigDeviceHandleByIndex(self._handle, slot)
      except nvml.NVMLError_NotFound:
        continue  # empty slot in a partially populated layout
      found.append(MigInstance(
        uuid=nvml.nvmlDeviceGetUUID(child),
        gpu_instance_id=nvml.nvmlDeviceGetGpuInstanceId(child),
        memory_gib=nvml.nvmlDeviceGetMemoryInfo(child).total / (1024 ** 3),
      ))
    return found

  def detect_profile(self):
    """helper to match MIG slice to profile name"""
    for profile in PROFILES.values():
      if self._matches(profile):
        return profile.name
    return None

  def _matches(self, profile):
    if len(self.instances) != profile.instance_count:
      return False
    return all(
      abs(inst.memory_gib - profile.memory_gib) <= MEMORY_TOLERANCE_GIB
      for inst in self.instances)

  def validate(self, profile_name):
    """MIG slice idempotency"""
    if profile_name not in PROFILES:
      raise ValueError(f"Unknown MIG profile: {profile_name}")
    if not self.enabled:
      raise RuntimeError(
        f"Profile {profile_name} requested but MIG is disabled on this GPU")
    profile = PROFILES[profile_name]
    if not self._matches(profile):
      actual = self.detect_profile() or f"{len(self.instances)} unrecognized instances"
      raise RuntimeError(
        f"Existing MIG layout ({actual}) does not match requested "
        f"profile {profile_name}")
    return profile

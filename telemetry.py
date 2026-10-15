"""async DCGM recording for one benchmark cell

power, energy, clocks, temperature for each MIG slice if enabled,
otherwise whole GPU

python modules come from DCGM
"""

import contextlib
import csv
import os
import sys
import threading
import time
from pathlib import Path

from config import Config
from mig import MigLayout

DEFAULT_BINDINGS = "/usr/local/dcgm/bindings/python3"


class Telemetry:
  PROF_FIELDS = {
    "sm_active": 1002,
    "sm_occupancy": 1003,
    "tensor_pipe_active": 1004,
    "dram_active": 1005,
    "fp32_pipe_active": 1007,
    "tensor_hmma_active": 1014,
  }
  DEVICE_FIELDS = {
    "sm_clock_mhz": 100,
    "mem_clock_mhz": 101,
    "temp_c": 150,
    "power_w": 155,
    "energy_mj": 156,
  }

  # dcgm timings
  READ_INTERVAL_S = 1.0
  KEEP_AGE_S = 30.0

  # delay for slow mig profile changes
  HIERARCHY_TIMEOUT_S = 5.0
  HIERARCHY_POLL_S = 0.25

  def __init__(self, config: Config, layout: MigLayout, out_dir):
    self.config = config
    self.layout = layout
    self.out_dir = Path(out_dir)

    # as time_s, entity, field_id, value
    # time_s is CLOCK_MONOTONIC_RAW
    # entity is "GPU0" or MIG slice uuid
    self.samples = []
    # (entity, field_id) tuples from DCGM
    self.watched = set()

    self.error = None

    self._handle = None  # ip/locahost
    self._groups = []    # group, field_group pairs deleted on stop
    self._readers = []   # (collection, field_group, {entity key: label})
    self._file = None
    self._writer = None
    self._thread = None
    self._stop = threading.Event()


  def start(self) -> bool:
    if self.config.dry_run:
      print(f"[dry-run] would record DCGM into {self.out_dir / 'dcgm.csv'}")
      return True
    try:
      self._watch()
    except Exception as e:
      self.error = f"DCGM failed starting: {e}"
      self._cleanup()
      return False

    # wall-clock microseconds vs workers monotonic clock sync
    self._offset = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - time.time()
    self._file = (self.out_dir / "dcgm.csv").open("w", newline="")
    self._writer = csv.writer(self._file)
    self._writer.writerow(["time_s", "entity", "field_id", "value"])
    self._thread = threading.Thread(target=self._record, daemon=True)
    self._thread.start()
    return True

  def stop(self):
    """Fetch last samples, end thread and release"""
    if self.config.dry_run:
      return
    if self._thread is not None:
      self._stop.set()
      self._thread.join(timeout=10)
    self._cleanup()

  def _watch(self):
    bindings = os.environ.get("DCGM_BINDINGS", DEFAULT_BINDINGS)
    if bindings not in sys.path:
      sys.path.insert(0, bindings)
    import pydcgm
    import dcgm_agent
    import dcgm_fields
    import dcgm_structs
    import dcgm_field_helpers
    from DcgmHandle import DcgmHandle
    from DcgmFieldGroup import DcgmFieldGroup

    self._handle = DcgmHandle(ipAddress="127.0.0.1")
    system = self._handle.GetSystem()

    gpu = {(dcgm_fields.DCGM_FE_GPU, 0): "GPU0"}
    if self.layout is not None and self.layout.enabled:
      prof_entities = self._mig_entities(dcgm_agent, dcgm_fields)
    else:
      prof_entities = gpu

    for name, entities, fields in (("device", gpu, self.DEVICE_FIELDS),
                                   ("prof", prof_entities, self.PROF_FIELDS)):
      tag = f"bench_{name}_{os.getpid()}"
      pairs = [dcgm_structs.c_dcgmGroupEntityPair_t(group_id, entity_id)
               for group_id, entity_id in entities]
      group = system.GetGroupWithEntities(tag, pairs)
      field_group = DcgmFieldGroup(self._handle, name=tag,
                                   fieldIds=list(fields.values()))
      # Register for cleanup before watching
      # if failed -> delete
      self._groups.append((group, field_group))
      group.samples.WatchFields(field_group,
                                int(self.config.sample_interval_s * 1e6),
                                self.KEEP_AGE_S, 0)
      collection = dcgm_field_helpers.DcgmFieldValueEntityCollection(
        self._handle.handle, group._groupId)
      self._readers.append((collection, field_group, entities))
      self.watched.update((label, field_id)
                          for label in entities.values()
                          for field_id in fields.values())

    system.UpdateAllFields(True)

  def _mig_entities(self, dcgm_agent, dcgm_fields):
    deadline = time.monotonic() + self.HIERARCHY_TIMEOUT_S
    while True:
      hierarchy = dcgm_agent.dcgmGetGpuInstanceHierarchy(self._handle.handle)
      dcgm_ids = {}
      for item in hierarchy.entityList[:hierarchy.count]:
        if (item.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I
            and item.info.nvmlGpuIndex == 0):
          dcgm_ids[item.info.nvmlInstanceId] = item.entity.entityId
      missing = [inst for inst in self.layout.instances
                 if inst.gpu_instance_id not in dcgm_ids]
      if not missing:
        break
      if time.monotonic() >= deadline:
        raise RuntimeError(
          f"DCGM doesnt list MIG instance {missing[0].uuid} "
          f"(GPU instance {missing[0].gpu_instance_id}) within ")
      time.sleep(self.HIERARCHY_POLL_S)

    entities = {}
    for inst in self.layout.instances:
      key = (dcgm_fields.DCGM_FE_GPU_I, dcgm_ids[inst.gpu_instance_id])
      entities[key] = inst.uuid
    return entities

  def _record(self):
    try:
      while not self._stop.wait(self.READ_INTERVAL_S):
        self._drain()
      self._drain()
    except Exception as exc:
      self.error = f"DCGM recording failed: {exc}"

  def _drain(self):
    for collection, field_group, entities in self._readers:
      collection.GetAllSinceLastCall(field_group)
      for group_id, by_entity in collection.values.items():
        for entity_id, by_field in by_entity.items():
          label = entities[(group_id, entity_id)]
          for field_id, series in by_field.items():
            for value in series.values:
              if value.isBlank:
                continue
              row = (value.ts / 1e6 + self._offset, label, field_id, value.value)
              self.samples.append(row)
              self._writer.writerow(row)
      collection.EmptyValues()
    self._file.flush()

  def _cleanup(self):
    for group, field_group in self._groups:
      with contextlib.suppress(Exception):
        group.Delete()
      with contextlib.suppress(Exception):
        field_group.Delete()
    self._groups = []
    if self._handle is not None:
      with contextlib.suppress(Exception):
        self._handle.Shutdown()
      self._handle = None
    if self._file is not None:
      self._file.close()
      self._file = None

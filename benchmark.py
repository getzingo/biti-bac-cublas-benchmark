import json
from dataclasses import asdict
from pathlib import Path

import utils
from a100 import A100, RebootRequired
from config import Cell, Config, build_schedule
from mps import Mps
from telemetry import Telemetry
from workers import CellFailed, WorkerPool


class Benchmark:
  def __init__(self, config: Config, resume=False):
    self.config = config
    if resume:
      latest = utils.latest_run_dir()
      if latest is None:
        raise RuntimeError("ERROR: no previous run directory")
      self.work_dir = Path(latest)
      self.plan = json.loads((self.work_dir / "plan.json").read_text())
    else:
      self.work_dir = Path(utils.create_dir_structure())
      self.plan = [dict(asdict(cell), round=r, done=False)
                   for r, cell in build_schedule()]
      self._save_plan()
    self.gpu = A100(config)
    # layout last set up in this process, starts out unknown
    self._layout = None

  def _save_plan(self):
    # safety write as tmp file
    tmp = self.work_dir / "plan.json.tmp"
    tmp.write_text(json.dumps(self.plan, indent=2) + "\n")
    tmp.replace(self.work_dir / "plan.json")

  def run(self):
    """Work through every step that isn't done yet.

    Returns True when the whole plan is done or False when reboot needed
    """
    self.gpu.apply_controls()
    try:
      for step in self.plan:
        if step["done"]:
          continue
        cell = Cell(**{k: v for k, v in step.items() if k not in ("done", "round")})
        label = f"round {step['round']} {cell.key()}"
        try:
          self.prepare_gpu(cell)
        except RebootRequired as e:
          print(f"Reboot required before {label}: {e}")
          print("After reboot, continue with: "
                "python entrypoint.py --resume")
          return False
        cell_dir = (self.work_dir / f"round-{step['round']:02d}"
                    / cell.key().replace("/", "_"))
        tele = self.execute_cell(cell, cell_dir)
        if tele is None:
          # fail if dcgm didnt come up
          return False
        if not tele:
          continue
        missing = self.missing_metrics(tele)
        if missing:
          # leave undone if unsuccessful
          print(f"{label}: no data for {missing}, "
                f"will be repeated on --resume")
          continue
        step["done"] = True
        self._save_plan()
      return all(step["done"] for step in self.plan)
    finally:
      self.gpu.shutdown()

  def prepare_gpu(self, cell: Cell):
    """put gpu into the layout the cell needs"""
    if cell.profile == self._layout:
      return
    if cell.mechanism == "MIG":
      self.gpu.set_mig_layout(cell.profile)
    else:
      self.gpu.disable_mig()
    self._layout = cell.profile

  def execute_cell(self, cell, cell_dir):
    """returns Telemetry when metrics successfull,
    False when cell failed and has to be repeated
    or None when DCGM failed startup
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    foreign = self.gpu.confounding_gpu_processes()
    # wait till gpu is idle
    if foreign:
      print(f"{cell.key()}: GPU busy with PIDs {sorted(foreign)}; "
            f"will be repeated on --resume")
      return False

    layout = None
    if cell.mechanism == "MIG" and not self.config.dry_run:
      layout = self.gpu.get_mig_layout()
    devices = self.worker_devices(cell, layout)

    mps = Mps(self.config)
    if cell.mechanism == "MPS":
      mps.start()
    try:
      tele = Telemetry(self.config, layout, cell_dir)
      if not tele.start():
        print(f"Stopping: {tele.error}")
        return None
      pool = WorkerPool(self.config, cell, devices, cell_dir,
                        check=lambda pids: self.check_no_foreign(pids, mps, tele))
      try:
        pool.run()
      except CellFailed as e:
        print(f"{cell.key()}: {e}... will be repeated on --resume")
        return False
      finally:
        tele.stop()
    finally:
      mps.stop()

    missing = pool.missing_results()
    if missing:
      print(f"{cell.key()}: {' && '.join(missing)}; will be repeated on --resume")
      return False
    return tele

  def worker_devices(self, cell, layout):
    """One CUDA_VISIBLE_DEVICES value per worker"""
    if cell.mechanism != "MIG":
      return ["0"] * cell.n
    if self.config.dry_run:
      return [f"MIG-instance-{i}" for i in range(cell.n)]
    return [inst.uuid for inst in layout.instances[:cell.n]]

  def check_no_foreign(self, worker_pids, mps, tele):
    if tele.error:
      raise CellFailed(tele.error)
    foreign = self.gpu.confounding_gpu_processes() - worker_pids - mps.server_pids()
    if foreign:
      raise CellFailed(f"foreign GPU process(es) {sorted(foreign)} during the run")

  def missing_metrics(self, tele):
    """check if tele data exists"""
    if self.config.dry_run:
      return []
    if tele.error:
      return [tele.error]
    recorded = {(entity, field_id) for _, entity, field_id, _ in tele.samples}
    return sorted(tele.watched - recorded)

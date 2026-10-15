"""runs run_worker.cu with running cuBLAS

worker prints on stdout
  # META {...}                            device and build facts
  # READY                                 warmup done, waiting for GO
  wid,index,device_ms,launch_ns,end_ns    one line per timed GEMM

workers stderr goes to worker-<wid>.stderr.
"""

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from config import Cell, Config

WORKER_BINARY = Path(__file__).resolve().parent / "run_worker"


class CellFailed(Exception):
  pass


class WorkerPool:
  CHECK_INTERVAL_S = 0.5

  def __init__(self, config: Config, cell: Cell, devices, out_dir, check=None):
    self.config = config
    self.cell = cell
    self.devices = list(devices)
    self.out_dir = Path(out_dir)
    self.check = check
    self.meta = {}   # wid -> META dict
    self.rows = {}   # wid -> [(index, device_ms, launch_ns, end_ns)]
    self._ready = []
    self._errors = []
    self._procs = []
    self._readers = []

  def run(self):
    if self.config.dry_run:
      print(f"[dry-run] would run {len(self.devices)} workers for "
            f"{self.cell.key()} on {self.devices}")
      return
    if not WORKER_BINARY.is_file():
      raise RuntimeError(f"{WORKER_BINARY} not found, forgot setup.sh")
    deadline = time.monotonic() + self.config.timeout_s
    try:
      self._launch()
      self._wait(deadline, lambda: all(e.is_set() for e in self._ready))
      for wid, proc in enumerate(self._procs):
        try:
          proc.stdin.write("GO\n")
          proc.stdin.close()
        except OSError:
          raise CellFailed(f"worker {wid} died before GO, check worker-{wid}.stderr")
      self._wait(deadline, lambda: all(p.poll() is not None for p in self._procs))
      self._raise_on_failure()
    finally:
      self._kill_leftovers()
      self._write_results()

  def missing_results(self):
    if self.config.dry_run:
      return []
    problems = []
    for wid in range(len(self.devices)):
      if wid not in self.meta:
        problems.append(f"worker {wid}: no META line")
      got = len(self.rows.get(wid, []))
      if got != self.config.runs:
        problems.append(f"worker {wid}: {got}/{self.config.runs} timed runs")
    return problems

  def _launch(self):
    for wid, device in enumerate(self.devices):
      cmd = [str(WORKER_BINARY), str(wid), *map(str, self.config.shape),
             self.cell.precision, str(self.config.warmup), str(self.config.runs)]
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
      self.rows[wid] = []
      self._ready.append(threading.Event())
      with (self.out_dir / f"worker-{wid}.stderr").open("w") as err:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=err,
                                text=True, bufsize=1)
      self._procs.append(proc)
      reader = threading.Thread(target=self._read, args=(wid, proc), daemon=True)
      reader.start()
      self._readers.append(reader)

  def _read(self, wid, proc):
    """parse workers stdout when done"""
    try:
      for line in proc.stdout:
        line = line.strip()
        if line == "# READY":
          self._ready[wid].set()
        elif line.startswith("# META "):
          self.meta[wid] = json.loads(line[len("# META "):])
        elif line:
          w, index, ms, launch, end = line.split(",")
          if int(w) != wid:
            raise ValueError(f"row wrongly attributed to worker {w}")
          self.rows[wid].append((int(index), float(ms), int(launch), int(end)))
    except Exception as exc:
      self._errors.append(f"worker {wid} wrote unreadable output: {exc}")

  def _wait(self, deadline, done):
    next_check = 0.0
    while not done():
      self._raise_on_failure()
      if time.monotonic() > deadline:
        raise CellFailed(
          f"workers didn't finish within {self.config.timeout_s:.0f}s")
      if self.check and time.monotonic() >= next_check:
        self.check({p.pid for p in self._procs})
        next_check = time.monotonic() + self.CHECK_INTERVAL_S
      time.sleep(0.05)

  def _raise_on_failure(self):
    if self._errors:
      raise CellFailed("; ".join(self._errors))
    for wid, proc in enumerate(self._procs):
      code = proc.poll()
      if code is None:
        continue
      if code != 0:
        raise CellFailed(f"worker {wid} exited with code {code}; "
                         f"check worker-{wid}.stderr")
      if not self._ready[wid].is_set():
        raise CellFailed(f"worker {wid} exited before READY")

  def _kill_leftovers(self):
    for proc in self._procs:
      if proc.poll() is None:
        proc.terminate()
    for proc in self._procs:
      try:
        proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    for reader in self._readers:
      reader.join(timeout=5)

  def _write_results(self):
    with (self.out_dir / "timing.csv").open("w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["worker", "index", "device_ms", "launch_ns", "end_ns"])
      for wid, rows in sorted(self.rows.items()):
        for row in rows:
          writer.writerow([wid, *row])
    (self.out_dir / "workers.json").write_text(json.dumps(
      {str(wid): {"device": self.devices[wid], "meta": self.meta.get(wid)}
       for wid in range(len(self._procs))}, indent=2) + "\n")

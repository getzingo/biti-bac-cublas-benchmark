"""turns a finished run into tables for a statistical evaluation

writes <run dir>/dataset/ csvs
"""

import csv
import json
import math
import statistics
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from config import Config
from telemetry import Telemetry

# field id -> column name
PROF_NAMES = {v: k for k, v in Telemetry.PROF_FIELDS.items()}
DEVICE = Telemetry.DEVICE_FIELDS

# columns of the four tables, in file order (what they mean: data_dictionary.csv)
DESIGN = ["cell", "mechanism", "profile", "n_workers", "precision", "round"]
COLUMNS = {
  "invocations": DESIGN + [
    "worker", "index", "device_ms", "launch_s", "end_s", "window_class"],
  "workers": DESIGN + [
    "worker", "device", "timed_count", "retained_count", "crossing_count",
    "outside_count", "mean_ms"],
  "cell_rounds": DESIGN + [
    "window_start_s", "window_end_s", "window_s", "timed_count", "retained_count",
    "crossing_count", "outside_count", "timing_status", "tflops", "mean_ms",
    "energy_status", "energy_j", "mean_power_w", "gflop_per_j", "energy_start_gap_s",
    "energy_end_gap_s", "sm_clock_samples", "sm_clock_mean_mhz", "clock_locked"],
  "profiling_series": DESIGN + [
    "entity", "field", "assigned", "n_samples", "mean", "max_gap_s",
    "median_interval_s"]}


def read_timing(path) -> dict:
  """worker -> [(index, device_ms, launch_s, end_s)], sorted by host launch time"""
  workers = defaultdict(list)
  try:
    with open(path, newline="") as f:
      for row in csv.DictReader(f):
        try:
          workers[int(row["worker"])].append((
            int(row["index"]), float(row["device_ms"]),
            int(row["launch_ns"]) / 1e9, int(row["end_ns"]) / 1e9))
        except (KeyError, TypeError, ValueError):
          raise ValueError(f"malformed timing row {row!r} in {path}") from None
  except FileNotFoundError:
    raise ValueError(f"{path} is missing; a step marked done must have it") from None
  for rows in workers.values():
    rows.sort(key=lambda r: r[2])
  return workers


def read_dcgm(path) -> dict:
  """(entity, field_id) -> [(time_s, value)], sorted by time"""
  series = defaultdict(list)
  try:
    with open(path, newline="") as f:
      for row in csv.DictReader(f):
        try:
          series[(row["entity"], int(row["field_id"]))].append(
            (float(row["time_s"]), float(row["value"])))
        except (KeyError, TypeError, ValueError):
          raise ValueError(f"malformed telemetry row {row!r} in {path}") from None
  except FileNotFoundError:
    raise ValueError(f"{path} is missing; a step marked done must have it") from None
  for samples in series.values():
    samples.sort()
  return series


def read_devices(path) -> dict:
  """worker -> device from workers.json"""
  try:
    data = json.loads(path.read_text())
  except FileNotFoundError:
    raise ValueError(f"{path} is missing; a step marked done must have it") from None
  return {int(k): v.get("device") for k, v in data.items()}


def window_class(launch, done, start, end):
  """retained = fully inside the window, crossing = overlaps an edge, outside = no overlap"""
  if launch >= start and done <= end:
    return "retained"
  if launch < end and done > start:
    return "crossing"
  return "outside"


def counter_at(samples, times, t, max_gap_s):
  """(value, width, problem) of the cumulative counter at time t"""
  i = bisect_left(times, t)
  if i < len(times) and times[i] == t:
    return samples[i][1], 0.0, None
  if i == 0:
    return None, None, "no_sample_before"
  if i == len(times):
    return None, None, "no_sample_after"
  (t0, v0), (t1, v1) = samples[i - 1], samples[i]
  width = t1 - t0
  if width > max_gap_s:
    return None, width, "gap_over_max"
  return v0 + (v1 - v0) * (t - t0) / width, width, None


def energy_over(samples, start, end, max_gap_s):
  """(joules, start width, end width, status) over [start, end] from the
  cumulative energy counter (mJ).
  counter at end - counter at start"""
  if not samples:
    return None, None, None, "no_samples"
  times = [t for t, _ in samples]
  v0, width0, problem0 = counter_at(samples, times, start, max_gap_s)
  v1, width1, problem1 = counter_at(samples, times, end, max_gap_s)
  if v0 is None or v1 is None:
    return None, width0, width1, f"start_{problem0}" if problem0 else f"end_{problem1}"
  return (v1 - v0) / 1000, width0, width1, "ok"


def series_stats(samples, start, end):
  """sample count, mean, longest hole and median spacing of a series inside the window"""
  inside = [(t, v) for t, v in samples if start <= t <= end and math.isfinite(v)]
  if not inside:
    return {"n_samples": 0}
  times = [t for t, _ in inside]
  steps = [b - a for a, b in zip(times, times[1:])]
  return {
    "n_samples": len(inside),
    "mean": statistics.fmean(v for _, v in inside),
    "max_gap_s": max([times[0] - start, *steps, end - times[-1]]),
    "median_interval_s": statistics.median(steps) if steps else None}


def class_counts(counts, has_window):
  return {f"{name}_count": counts[name] if has_window else None
          for name in ("retained", "crossing", "outside")}


def reduce_cell(cell_dir, design, config):
  """rows of the four tables for one cell directory; design holds the columns
  that identify the cell and is copied onto every row"""
  cell_dir = Path(cell_dir)
  timing = read_timing(cell_dir / "timing.csv")
  dcgm = read_dcgm(cell_dir / "dcgm.csv")
  devices = read_devices(cell_dir / "workers.json")

  expected = list(range(design["n_workers"]))
  if sorted(timing) != expected:
    raise ValueError(f"{cell_dir}: expected workers {expected}, "
                     f"timing.csv has {sorted(timing)}")
  if not all(devices.get(w) for w in timing):
    raise ValueError(f"{cell_dir}: workers.json has no device for every worker")

  # the common window, from the latest first launch to the earliest last end
  start = max(rows[0][2] for rows in timing.values())
  end = min(rows[-1][3] for rows in timing.values())
  window_s = end - start
  has_window = window_s > 0

  invocations, workers, kept = [], [], []
  total = Counter()
  for wid in sorted(timing):
    counts, retained = Counter(), []
    for index, ms, launch, done in timing[wid]:
      cls = window_class(launch, done, start, end) if has_window else None
      counts[cls] += 1
      if cls == "retained":
        retained.append(ms)
      invocations.append({
        **design, "worker": wid, "index": index, "device_ms": ms,
        "launch_s": launch, "end_s": done, "window_class": cls})
    workers.append({
      **design, "worker": wid, "device": devices[wid],
      "timed_count": len(timing[wid]), **class_counts(counts, has_window),
      "mean_ms": statistics.fmean(retained) if retained else None})
    total.update(counts)
    kept += retained

  if not has_window:
    timing_status = "no_positive_overlap"
  elif not kept:
    timing_status = "no_fully_contained_calls"
  else:
    timing_status = "ok"

  cell_round = {
    **design, "window_start_s": start, "window_end_s": end, "window_s": window_s,
    "timed_count": sum(len(rows) for rows in timing.values()),
    **class_counts(total, has_window), "timing_status": timing_status}

  energy_status = "no_valid_window"
  if timing_status == "ok":
    flops = len(kept) * config.flops_per_call
    cell_round["tflops"] = flops / window_s / 1e12
    cell_round["mean_ms"] = statistics.fmean(kept)
    energy, width0, width1, energy_status = energy_over(
      dcgm.get(("GPU0", DEVICE["energy_mj"]), []), start, end, config.max_gap_s)
    cell_round.update(energy_j=energy, energy_start_gap_s=width0,
                      energy_end_gap_s=width1)
    if energy is not None:
      cell_round["mean_power_w"] = energy / window_s
      if energy > 0:  # 0 J is a valid energy, an efficiency of it is not
        cell_round["gflop_per_j"] = flops / energy / 1e9
  cell_round["energy_status"] = energy_status

  if has_window:
    clocks = [v for t, v in dcgm.get(("GPU0", DEVICE["sm_clock_mhz"]), [])
              if start <= t <= end and math.isfinite(v)]
    cell_round["sm_clock_samples"] = len(clocks)
    if clocks:
      cell_round["sm_clock_mean_mhz"] = statistics.fmean(clocks)
      cell_round["clock_locked"] = int(all(c == config.clock_mhz for c in clocks))

  # one row per profiled entity and field, including idle MIG instances
  assigned = {"GPU0" if devices[w] == "0" else devices[w] for w in timing}
  entities = sorted({e for e, f in dcgm if f in PROF_NAMES})
  profiling = [
    {**design, "entity": e, "field": name, "assigned": int(e in assigned),
     **series_stats(dcgm.get((e, fid), []), start, end)}
    for e in entities for fid, name in PROF_NAMES.items()]

  return {"invocations": invocations, "workers": workers,
          "cell_rounds": [cell_round], "profiling_series": profiling}


def process_run(run_dir, config=Config()):
  """tables of every step marked done, written to <run dir>/dataset"""
  run_dir = Path(run_dir)
  plan = json.loads((run_dir / "plan.json").read_text())
  tables = {name: [] for name in COLUMNS}
  for step in plan:
    if not step["done"]:
      continue
    cell = f"{step['mechanism']}/{step['profile']}/{step['n']}/{step['precision']}"
    design = {"cell": cell, "mechanism": step["mechanism"],
              "profile": step["profile"], "n_workers": step["n"],
              "precision": step["precision"], "round": step["round"]}
    cell_dir = run_dir / f"round-{step['round']:02d}" / cell.replace("/", "_")
    for name, rows in reduce_cell(cell_dir, design, config).items():
      tables[name].extend(rows)

  out = run_dir / "dataset"
  out.mkdir(exist_ok=True)
  for name, rows in tables.items():
    write_csv(out / f"{name}.csv", rows, COLUMNS[name])
  return tables


def write_csv(path, rows, names):
  """a missing key or None is written as a blank cell"""
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=names, restval="")
    writer.writeheader()
    writer.writerows(rows)


def main():
  if len(sys.argv) != 2:
    print("usage: python post_processing.py <run dir>")
    return 2
  run_dir = Path(sys.argv[1])
  if not (run_dir / "plan.json").is_file():
    print(f"error: {run_dir} has no plan.json")
    return 2
  try:
    tables = process_run(run_dir)
    print(f"{len(tables['cell_rounds'])} cell-rounds written to {run_dir / 'dataset'}")
  except ValueError as error:
    print(f"cannot reduce the run: {error}",
      "no tables were written. files in dataset/ are not as expected")
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())

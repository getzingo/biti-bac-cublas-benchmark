from dataclasses import dataclass
import random

from mig import PROFILES

PRECISIONS = ("fp16", "fp32", "fp64")
MECHANISMS = ("TS", "MPS", "MIG")


@dataclass(frozen=True)
class Cell:
  """Cell means one config to be run
  :param mechanism: enum TS, MPS or MIG
  :param profile: enum full or MIG profile
  :param n: number of workers running concurrently, 1 2 4 or 7
  :param precision: enum fp16, fp32 or fp64
  """
  mechanism: str
  profile: str     # 'full' for TS/MPS, otherwise a MIG profile name
  n: int
  precision: str

  def key(self):
    return f"{self.mechanism}/{self.profile}/{self.n}/{self.precision}"


def build_plan():
  cells = []
  for mechanism in ("TS", "MPS"):
    for n in (1, 2, 4, 7):  # concurrent worker counts
      for precision in PRECISIONS:
        cells.append(Cell(mechanism, "full", n, precision))
  for profile in PROFILES.values():
    # isolated baseline then concurrent
    for n in (1, profile.instance_count):
      for precision in PRECISIONS:
        cells.append(Cell("MIG", profile.name, n, precision))
  return cells

PLAN = tuple(build_plan())

ROUNDS = 3
# for later randomisation, pinned for reproducibility
SEED = 0


def build_schedule():
  """PLAN repeated ROUNDS times, as (round, cell) pairs

  cells with same GPU layout stay together as may need reboot
  order reverses every other round
  """
  layouts = {}
  for cell in PLAN:
    layouts.setdefault(cell.profile, []).append(cell)
  order = list(layouts)  # 'full' first, then the MIG profiles
  rng = random.Random(SEED)
  schedule = []
  for r in range(1, ROUNDS + 1):
    for profile in (order if r % 2 else reversed(order)):
      cells = list(layouts[profile])
      rng.shuffle(cells)
      schedule.extend((r, cell) for cell in cells)
  return schedule


@dataclass(frozen=True)
class Config:
  shape: tuple = (8192, 8192, 8192)

  # warmup calls untimed, runs timed
  warmup: int = 5
  runs: int = 500
  timeout_s: float = 1800.0

  clock_mhz: int = 1410

  dry_run: bool = False

  # dcgm interval = 10 Hz
  sample_interval_s: float = 0.1

  # widest gap between two energy readings that is still interpolated
  max_gap_s: float = 0.250

  @property
  def flops_per_call(self):
    return 2 * self.shape[0] * self.shape[1] * self.shape[2]

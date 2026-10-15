import os
import subprocess

from datetime import datetime


def create_dir_structure() -> str:
  root_dir = os.path.dirname(__file__)
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d%H%M%S")
  if not os.path.exists(os.path.join(root_dir, timestamp)):
    os.makedirs(os.path.join(root_dir, timestamp))
  working_dir = os.path.join(root_dir, timestamp)
  return working_dir

def latest_run_dir() -> str | None:
  """The newest run directory made by create_dir_structure, if there is one.
  The timestamp names sort in time order, so the newest is simply the last."""
  root_dir = os.path.dirname(__file__)
  runs = sorted(
    d for d in os.listdir(root_dir)
    if len(d) == 14 and d.isdigit() and os.path.isdir(os.path.join(root_dir, d)))
  return os.path.join(root_dir, runs[-1]) if runs else None

def run_cmd(cmd: list[str], input: str | None = None) -> dict:
  """Run a command and return its exit code and output.
  `input` is fed to the command's stdin, e.g. for nvidia-cuda-mps-control."""
  try:
    process = subprocess.run(
      cmd,
      input=input,
      capture_output=True,
      text=True
    )
    result = {
      "rc": process.returncode,
      "stdout": process.stdout.strip(),
      "stderr": process.stderr.strip(),
    }
    return result
  except subprocess.CalledProcessError as e:
    exit(e.returncode)
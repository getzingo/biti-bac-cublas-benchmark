#!/usr/bin/env python3
import argparse
import sys
from benchmark import Benchmark
from config import Config

sys.stdout.reconfigure(line_buffering=True)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--resume", action="store_true",
                help="continue the most recent run, e.g. after a reboot")
args = ap.parse_args()

benchmark = Benchmark(Config(dry_run=args.dry_run), resume=args.resume)
finished = benchmark.run()
print(f"{'Plan complete' if finished else 'Paused'}: {benchmark.work_dir}")

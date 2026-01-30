#!/usr/bin/env python3
"""
Run cpu_mini_validate.py multiple times with different seeds and save all output to a log file.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "cpu_mini_validate.py"
LOG_DIR = REPO / "scripts" / "logs"
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run minimal DCA validation multiple times and log")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds, e.g. 42,43,44. Default: 42-51")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: scripts/logs)")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = DEFAULT_SEEDS

    log_dir = Path(args.log_dir) if args.log_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / "cpu_mini_validate_{}.log".format(stamp)

    print("Running {} trials (seeds={}), logging to {}".format(len(seeds), seeds, log_path))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("DCA minimal validation – multi-run log\n")
        f.write("seeds={}, n_steps={}\n".format(seeds, args.n_steps))
        f.write("started at {}\n\n".format(datetime.now().isoformat()))

        for i, seed in enumerate(seeds):
            sep = "\n" + "=" * 60 + "\n"
            f.write(sep)
            f.write("Run {} / {}  seed={}\n".format(i + 1, len(seeds), seed))
            f.write("=" * 60 + "\n\n")
            f.flush()

            cmd = [sys.executable, str(SCRIPT), "--seed", str(seed), "--n_steps", str(args.n_steps)]
            ret = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")
            f.write(ret.stdout)
            if ret.stderr:
                f.write(ret.stderr)
            if ret.returncode != 0:
                f.write("(exit code {})\n".format(ret.returncode))
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("End of log  {}\n".format(datetime.now().isoformat()))

    print("Log saved to {}".format(log_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

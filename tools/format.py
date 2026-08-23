#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["kernels", "tests"]


def main():
    dirs = [d for d in TARGETS if (REPO_ROOT / d).is_dir()]
    result = subprocess.run(
        [sys.executable, "-m", "black", *dirs],
        cwd=REPO_ROOT,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

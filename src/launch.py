#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    train_script = root / "src" / "03_train_test" / "train_example.py"
    test_script = root / "src" / "03_train_test" / "test_example.py"

    subprocess.run([sys.executable, str(train_script)], check=True)
    subprocess.run([sys.executable, str(test_script)], check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULES = [
    "src.train",
    "src.eval",
    "src.infer",
    "src.utils",
    "src.utils.seed",
    "src.utils.checkpoint",
    "src.datasets",
    "src.metrics",
    "src.logger",
    "src.model",
]


def main() -> int:
    failed = False
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:
            failed = True
            print(f"[FAIL] {module_name}: {exc.__class__.__name__}: {exc}")

    if failed:
        print("Import smoke-check failed.")
        return 1

    print("Import smoke-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

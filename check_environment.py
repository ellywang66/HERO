#!/usr/bin/env python3
"""Check that the dependencies needed to train HERO are installed."""

import importlib
import sys

REQUIRED = ["torch", "transformers", "hydra", "omegaconf", "sklearn", "numpy", "tqdm", "loguru"]


def main():
    ok = sys.version_info >= (3, 10)
    print(f"python {sys.version.split()[0]} {'OK' if ok else '(need >= 3.10)'}")
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"{name} {getattr(module, '__version__', '')} OK")
        except ImportError:
            print(f"{name} MISSING")
            ok = False
    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        pass
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

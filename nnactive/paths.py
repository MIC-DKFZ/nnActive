import os
from pathlib import Path

key = "nnActive_results"
nnActive_results = Path(os.environ[key]) if key in os.environ.keys() else None


if nnActive_results is None:
    print("nnActive_results is not defined.")


def get_nnActive_results() -> Path:
    return nnActive_results

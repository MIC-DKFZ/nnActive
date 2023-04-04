import os
from pathlib import Path

KEY = "nnActive_results"
nnActive_results = Path(value) if (value := os.environ.get(KEY)) else None


def get_nnActive_results() -> Path | None:
    return nnActive_results


# if nnActive_results is None:
#     print("nnActive_results is not defined.")

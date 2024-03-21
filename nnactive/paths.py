import os
from pathlib import Path

nnActive_results = Path(value) if (value := os.environ.get("nnActive_results")) else None
nnActive_raw = Path(value) if (value := os.environ.get("nnActive_raw")) else None
nnActive_data = Path(value) if (value := os.environ.get("nnActive_data")) else None


def get_nnActive_results() -> Path | None:
    return nnActive_results


# if nnActive_results is None:
#     print("nnActive_results is not defined.")

from contextlib import contextmanager
import os
import nnunetv2.paths as paths
from pathlib import Path

nnActive_results = Path(value) if (value := os.environ.get("nnActive_results")) else None
nnActive_raw = Path(value) if (value := os.environ.get("nnActive_raw")) else None
nnActive_data = Path(value) if (value := os.environ.get("nnActive_data")) else None


def get_nnActive_results() -> Path | None:
    return nnActive_results


@contextmanager
def set_raw_paths():
    temp_raw = paths.nnUNet_raw
    temp_preprocessed = paths.nnUNet_raw
    paths.set_paths(nnUNet_raw=nnActive_raw / "nnUNet_raw", nnUNet_preprocessed=nnActive_raw / "nnUNet_preprocessed")
    yield
    paths.set_paths(nnUNet_raw=temp_raw, nnUNet_preprocessed=temp_preprocessed)


# if nnActive_results is None:
#     print("nnActive_results is not defined.")

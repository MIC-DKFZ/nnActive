import os
from contextlib import contextmanager
from pathlib import Path

import nnunetv2.paths as paths

__paths = {
    "nnActive_results": (
        Path(value) if (value := os.environ.get("nnActive_results")) else None
    ),
    "nnActive_raw": Path(value) if (value := os.environ.get("nnActive_raw")) else None,
    "nnActive_data": (
        Path(value) if (value := os.environ.get("nnActive_data")) else None
    ),
}

base_nnActive_results = (
    Path(value) if (value := os.environ.get("nnActive_results")) else None
)


def set_paths(
    nnActive_raw: str | Path | None = None,
    nnActive_results: str | Path | None = None,
    nnActive_data: str | Path | None = None,
):
    if nnActive_raw is not None:
        __paths["nnActive_raw"] = Path(nnActive_raw)
    if nnActive_results is not None:
        __paths["nnActive_results"] = Path(nnActive_results)
    if nnActive_data is not None:
        __paths["nnActive_data"] = Path(nnActive_data)


def __getattr__(item: str):
    if (value := __paths.get(item)) is not None:
        return value
    else:
        raise AttributeError(f"module {__name__} has no attribute {item}")


def get_nnActive_results() -> Path | None:
    return __paths["nnActive_results"]


def get_nnActive_data() -> Path | None:
    return __paths["nnActive_data"]


@contextmanager
def set_raw_paths():
    temp_raw = paths.nnUNet_raw
    temp_preprocessed = paths.nnUNet_raw
    temp_results = paths.nnUNet_results
    # we set nnUnet_results to nnActive_raw/nnUNet_raw
    # nnUNet_resuls is not needed for most use cases
    # nnUNet_results can lead to multiple identical names!
    paths.set_paths(
        nnUNet_raw=__paths["nnActive_raw"] / "nnUNet_raw",
        nnUNet_preprocessed=__paths["nnActive_raw"] / "nnUNet_preprocessed",
        nnUNet_results=__paths["nnActive_raw"]
        / "nnUNet_results",  # nnUNet_results is not used anyways in this context!
    )
    yield
    paths.set_paths(
        nnUNet_raw=temp_raw,
        nnUNet_preprocessed=temp_preprocessed,
        nnUNet_results=temp_results,
    )


# if nnActive_results is None:
#     print("nnActive_results is not defined.")

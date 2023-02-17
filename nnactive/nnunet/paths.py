from pathlib import Path

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results

NNUNET_RAW = Path(nnUNet_raw) if nnUNet_raw is not None else None
NNUNET_PREPROCESSED = (
    Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
)
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None

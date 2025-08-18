import argparse

from huggingface_hub import (  # Install huggingface_hub if not already installed
    snapshot_download,
)

DATASETS = [
    "Hippocampus",
    "ACDC",
    "KiTS2021",
    "AMOS2022_task1",
]


def hf_download(
    out_dir: str,
    dataset: str | None = None,
    experiment: str | None = None,
    analysis_only: bool = False,
):
    if dataset is not None:
        if dataset not in DATASETS:
            raise ValueError(
                f"Got invalid dataset {dataset}. Choose one of: {', '.join(DATASETS)}"
            )
        datasets = [dataset]
    else:
        datasets = DATASETS

    patterns = ["analysis/*"]
    if not analysis_only:
        if experiment is not None:
            patterns.append(f"nnActive_results/{experiment}")
            patterns.append(f"nnUNet_raw/{experiment}")
        else:
            patterns.append("nnActive_results/*")
            patterns.append("nnUNet_raw/*")

    for dset in datasets:
        print(f"Downloading results for {dset}...")
        download_path = snapshot_download(
            repo_id=f"nnActive/{dataset}",
            allow_patterns=patterns,
            local_dir=out_dir,
        )
        print(f"Saved files at {download_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Download nnActive results from Huggingface"
    )
    parser.add_argument("--out-dir", type=str, help="Path to local directory")
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Download results for a single dataset. Default: download all",
    )
    parser.add_argument(
        "--experiment", type=str, required=False, help="Download specific experiment"
    )
    parser.add_argument(
        "--analysis-only", action="store_true", help="Only download analysis files"
    )

    args = parser.parse_args()
    hf_download(
        args.out_dir,
        dataset=args.dataset if "dataset" in args else None,
        experiment=args.experiment if "experiment" in args else None,
        analysis_only=args.analysis_only,
    )


if __name__ == "__main__":
    main()

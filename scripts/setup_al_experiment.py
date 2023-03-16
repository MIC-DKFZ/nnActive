import os
from argparse import ArgumentParser

from nnactive.config import ActiveConfig
from nnactive.nnunet.utils import convert_id_to_dataset_name, get_patch_size
from nnactive.paths import get_nnActive_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer_20epochs")
    parser.add_argument("-p", "--patch-size", type=int, default=None, help="Patch Size")
    parser.add_argument(
        "--base_id",
        type=int,
        default=None,
        help="Dataset from which patch size is taken",
    )
    parser.add_argument("-qs", "--query-size", type=int, default = 10)
    parser.add_argument("--uncertainty", type=str, default="random")

    args = parser.parse_args()
    trainer = args.trainer
    query_size = args.query_size
    uncertainty = args.uncertainty

    if args.patch_size is None and args.base_id is None:
        raise ValueError("Either patch_size or base_id have to be set")
    patch_size = (
        [args.patch_size] * 3
        if args.patch_size is not None
        else get_patch_size(args.base_id)
    )
    dataset_id: int = args.dataset

    dataset_name: str = convert_id_to_dataset_name(dataset_id)

    results_path = get_nnActive_results()

    save_path = results_path / dataset_name

    config = ActiveConfig(trainer=trainer, patch_size=patch_size, )

    os.makedirs(save_path, exist_ok=True)

    config.save_id(dataset_id)

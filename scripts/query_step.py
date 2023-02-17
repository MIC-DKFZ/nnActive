from argparse import ArgumentParser

from nnactive.calculate_uncertainties import write_uncertainties_from_softmax_preds
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import get_raw_path, get_results_path, read_dataset_json
from nnactive.query_patches import query_most_uncertain_patches
from nnactive.uncertainty_aggregation.aggregate_uncertainties import (
    aggregate_uncertainties_per_image,
    read_images_to_numpy,
)


def main():
    # TODO: obtain Patch Size
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)
    parser.add_argument("-u", "--uncertainty_type", type=str, default="pred_entropy")
    parser.add_argument("-n", "--num_patches", type=int, default=10)
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()
    dataset_id = args.dataset_id
    uncertainty_type = args.uncertainty_type
    num_patches = args.num_patches
    loop = args.loop

    # patch_size: tuple[int] = nnUNetPlans.json["configurations"]["3d_fullres"]["patch_size"]
    patch_size = (10, 10, 10)

    raw_dataset_path = get_raw_path(dataset_id)
    dataset_json_path = raw_dataset_path / "dataset.json"
    base_softmax_path = get_results_path(dataset_id) / "predTr"
    uncertainty_path = base_softmax_path / "uncertainties"
    agg_uncertainty_path = base_softmax_path / "uncertainties_aggregated"

    if loop is None:
        loop = len(get_sorted_loop_files(raw_dataset_path))

    dataset_json = read_dataset_json(dataset_id)
    file_ending = dataset_json["file_ending"]
    ignore_label = dataset_json["labels"]["ignore"]

    write_uncertainties_from_softmax_preds(base_softmax_path, uncertainty_path)
    read_images_to_numpy(
        dataset_json_path,
        uncertainty_path,
        aggregate_uncertainties_per_image,
        target_folder=agg_uncertainty_path,
        patch_size=patch_size,
    )
    query_most_uncertain_patches(
        agg_uncertainty_path,
        uncertainty_type,
        num_patches,
        raw_dataset_path,
        loop,
        file_ending,
        ignore_label,
    )


if __name__ == "__main__":
    main()

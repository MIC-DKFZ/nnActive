import json
from argparse import Namespace

from nnactive.calculate_uncertainties import write_uncertainties_from_softmax_preds
from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.loops.loading import (
    get_patches_from_loop_files,
    get_sorted_loop_files,
    save_loop,
)
from nnactive.nnunet.utils import get_raw_path, get_results_path, read_dataset_json
from nnactive.query.random import (
    generate_random_patches,
    generate_random_patches_labels,
)
from nnactive.query_patches import query_most_uncertain_patches
from nnactive.results.state import State
from nnactive.uncertainty_aggregation.aggregate_uncertainties import (
    aggregate_uncertainties_per_image,
    read_images_to_numpy,
)


@register_subcommand(
    "query_step",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (
            ("-u", "--uncertainty_type"),
            {"type": str, "default": None},
        ),  # default="pred_entropy")
        (("-n", "--num_patches"), {"type": int, "default": None}),
        (("-s", "--patch_size"), {"type": int, "default": None}),
    ],
)
def main(args: Namespace):
    # TODO: obtain Patch Size
    # TODO: help
    dataset_id = args.dataset_id
    patch_size = args.patch_size

    # patch_size: tuple[int] = nnUNetPlans.json["configurations"]["3d_fullres"]["patch_size"

    config = ActiveConfig.get_from_id(dataset_id)

    uncertainty_type = args.uncertainty_type
    if uncertainty_type is None:
        uncertainty_type = config.uncertainty
    num_patches = args.num_patches
    if num_patches is None:
        num_patches = config.query_size
    if patch_size is not None:
        patch_size = list([patch_size] * 3)
    else:
        patch_size = config.patch_size
    seed = config.seed

    query_step(dataset_id, patch_size, uncertainty_type, num_patches, seed)


def query_step(
    dataset_id: int,
    patch_size: tuple[int],
    uncertainty_type: str,
    num_patches: int,
    seed: int = None,
):
    raw_dataset_path = get_raw_path(dataset_id)
    dataset_json_path = raw_dataset_path / "dataset.json"
    base_softmax_path = get_results_path(dataset_id) / "predTr"
    uncertainty_path = base_softmax_path / "uncertainties"
    agg_uncertainty_path = base_softmax_path / "uncertainties_aggregated"

    # if loop is None:
    loop = len(get_sorted_loop_files(raw_dataset_path))

    dataset_json = read_dataset_json(dataset_id)
    file_ending = dataset_json["file_ending"]
    ignore_label = dataset_json["labels"]["ignore"]
    state = State.get_id_state(dataset_id)

    if uncertainty_type.lower() == "random":
        labeled_patches = get_patches_from_loop_files(raw_dataset_path, loop - 1)
        # for i in range(loop):
        #     labeled_patches.extend(get_patches_from_loop_files(raw_dataset_path, i))
        patches = generate_random_patches(
            file_ending,
            raw_labels_path=raw_dataset_path / "labelsTr",
            patch_size=patch_size,
            n_patches=num_patches,
            labeled_patches=labeled_patches,
            seed=seed + loop,
            trials_per_img=600,
        )
        # bring into loop_XXX.json format and save!
        loop_json = {"patches": patches}

        save_loop(raw_dataset_path, loop_json, loop)
    elif uncertainty_type.lower() == "random-label":
        labeled_patches = get_patches_from_loop_files(raw_dataset_path, loop - 1)

        base_path = get_raw_path(dataset_json["annotated_id"])

        patches = generate_random_patches_labels(
            file_ending,
            seg_labels_path=base_path / "labelsTr",
            patch_size=patch_size,
            n_patches=num_patches,
            labeled_patches=labeled_patches,
            seed=seed + loop,
            trials_per_img=600,
        )
        # bring into loop_XXX.json format and save!
        loop_json = {"patches": patches}
        save_loop(raw_dataset_path, loop_json, loop)
    else:
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
    state.query = True
    state.save_state()

import shutil
from pathlib import Path

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results

from nnactive.cli.subcommands.convert_to_partannotated import (
    convert_dataset_to_partannotated,
)
from nnactive.loops.loading import get_loop_patches
from nnactive.nnunet.io import generate_custom_splits_file
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.utils.io import load_json

NNUNET_RAW = Path(nnUNet_raw) if nnUNet_raw is not None else None
NNUNET_PREPROCESSED = (
    Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
)
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None


def main():
    for factor in [
        1,
        2,
    ]:
        for dataset_id in [
            4,
            985,
            982,
            216,
            # 984, AMOS small does not have enough datapoints to support random-all-classes
            981,
        ]:
            try:
                base_id = dataset_id
                target_id = 217
                dataset_json = read_dataset_json(dataset_id=base_id)
                labels = [v for k, v in dataset_json["labels"].items() if k != "ignore"]

                num_patches = 2 * len(labels) * factor

                convert_dataset_to_partannotated(
                    base_id,
                    target_id,
                    full_images=0,
                    num_patches=num_patches,
                    patch_size=[1, 1, 1],
                    name_suffix="Convert",
                    strategy="random-label-all-classes",
                    force=True,
                )

                try:
                    generate_custom_splits_file(target_id, 0, 2 * factor, verify=True)

                    splits_file = load_json(
                        get_preprocessed_path(target_id) / "splits_final.json"
                    )
                    num_files = len(
                        set(
                            [
                                patch.file
                                for patch in get_loop_patches(get_raw_path(target_id))
                            ]
                        )
                    )
                    for split in splits_file:
                        assert (
                            len(split["train"] + split["val"]) == num_files
                        )  # some patches have gone lost or were added

                    shutil.rmtree(get_preprocessed_path(target_id))
                    shutil.rmtree(get_raw_path(target_id))

                    print(splits_file)
                except Exception as e:
                    shutil.rmtree(get_raw_path(target_id))
                    raise e

            except Exception as e:
                shutil.rmtree(get_raw_path(target_id))
                raise e


if __name__ == "__main__":
    main()

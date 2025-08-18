import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from loguru import logger

from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_nested_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.utils.io import load_json
from nnactive.utils.patches import create_patch_mask_for_image
from nnactive.utils.pyutils import rescale_pad_to_square, stitch_images


def visualize_query_trajectory(raw_folder: Path, output_folder: Path):
    """Create a folder structure with output_folder/loop_XXX containing binary masks with
    patches.

    Args:
        raw_folder (Path): experiment path in nnActive_data/.../nnUNet_raw/experiment
        output_folder (Path): folder to save masks for each loop
    """
    file_ending = load_json(raw_folder / "dataset.json")["file_ending"]
    save_folder = output_folder
    loop_patches = get_nested_patches_from_loop_files(raw_folder)

    if (raw_folder / "labelsTr").is_dir() is False:
        annotated_id = load_json(raw_folder / "dataset.json")["annotated_id"]
        logger.info(
            "Try Using dataset from annotated_id in nnUNet_raw default: {}".format(
                annotated_id
            )
        )

        raw_folder = get_raw_path(annotated_id)
    img_names = [
        f.name
        for f in (raw_folder / "labelsTr").iterdir()
        if f.name.endswith(file_ending)
    ]
    for i in range(len(loop_patches)):
        os.makedirs(save_folder / f"loop_{i:03d}", exist_ok=True)

    for img_name in img_names:
        all_img_patches = [x for xs in loop_patches for x in xs]
        if len([p for p in all_img_patches if p.file == img_name]) == 0:
            continue
        img = sitk.ReadImage(raw_folder / "labelsTr" / img_name)
        label_shape = sitk.GetArrayFromImage(img).shape
        for i, l_ps in enumerate(loop_patches):
            img_patches = [patch for patch in l_ps if patch.file == img_name]
            if len(img_patches) == 0:
                continue
            mask = create_patch_mask_for_image(
                img_name, l_ps, label_shape, identify_patch=False
            )
            mask = sitk.GetImageFromArray(mask)
            mask = copy_geometry_sitk(mask, img)
            sitk.WriteImage(
                mask,
                (save_folder / f"loop_{i:03d}" / img_name),
            )


def plot_query_trajectory(
    raw_folder: Path, img_folder: Path | None = None, save_folder: Path = None
):
    print(f"Saving results to folder: {save_folder}")

    file_ending = load_json(raw_folder / "dataset.json")["file_ending"]
    loop_patches = get_nested_patches_from_loop_files(raw_folder)
    if img_folder is not None:
        img_names = [
            "_".join(f.name.split("_")[:-1]) + file_ending
            for f in (img_folder).iterdir()
            if f.name.endswith(file_ending)
        ]
    else:
        img_names = [
            f.name
            for f in (raw_folder / "labelsTr").iterdir()
            if f.name.endswith(file_ending)
        ]

    for i in range(len(loop_patches)):
        os.makedirs(save_folder / f"loop_{i:03d}", exist_ok=True)

    for img_name in img_names:
        all_img_patches = [x for xs in loop_patches for x in xs]
        if len([p for p in all_img_patches if p.file == img_name]) == 0:
            continue
        img = sitk.ReadImage(
            img_folder / (img_name.replace(file_ending, "") + "_0000" + file_ending)
        )
        img: np.ndarray = sitk.GetArrayFromImage(img)
        img = (img - img.min()) / (img.max() - img.min())

        img_shape = img.shape
        for i, l_ps in enumerate(loop_patches):
            img_patches = [
                (p_id, patch)
                for p_id, patch in enumerate(l_ps)
                if patch.file == img_name
            ]
            for p_id, img_patch in img_patches:
                mask = create_patch_mask_for_image(
                    img_name, [img_patch], img_shape, identify_patch=False
                )
                center_axs = [0, 1, 2]
                views = []
                masks = []
                for center_ax in center_axs:
                    slices = []
                    for dim, shape in enumerate(img_shape):
                        if dim == center_ax:
                            center_coord = int(
                                img_patch.coords[center_ax]
                                + img_patch.size[center_ax] // 2
                            )
                            slices.append(slice(center_coord, center_coord + 1))
                        else:
                            slices.append(slice(0, int(shape)))

                    slices = tuple(slices)
                    viewplane = img[slices]
                    viewplane = viewplane.squeeze()
                    viewplane = rescale_pad_to_square(viewplane)
                    maskplane = mask[slices]
                    maskplane = maskplane.squeeze()
                    maskplane = rescale_pad_to_square(maskplane)
                    views.append(viewplane)
                    masks.append(maskplane)

                fig, axs = plt.subplots(1, len(center_axs))
                for c in range(len(center_axs)):
                    axs[c].imshow(views[c], cmap="gray", vmin=0, vmax=1)
                    axs[c].imshow(masks[c], cmap=plt.cm.Reds, alpha=0.3)
                    axs[c].set_xticks([])
                    axs[c].set_yticks([])
                file_id = img_name.replace(file_ending, "")
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)
                fig.suptitle(f"Patch {p_id} Loop {i} File {file_id}", y=0.72)
                filename = f"loop-{i:02d}__id-{p_id:02d}__img-{file_id}.png"
                plt.savefig(
                    save_folder / f"loop_{i:03d}" / filename, bbox_inches="tight"
                )
                plt.close("all")

    for i in range(len(loop_patches)):
        stitch_images(
            save_folder / f"loop_{i:03d}",
            save_folder / f"overview-loop_{i:03d}.png",
            columns=5,
            image_padding=0,
        )


def extract_al_method_from_path(path_str: str) -> str | None:
    method_dict = {
        "mutual_information": "BALD",
        "power_bald": "PowerBALD",
        "softrank_bald": "SoftrankBALD",
        "pred_entropy": "Predictive Entropy",
        "power_pe": "PowerPE",
        "random": "Random",
        "random-label2": "Random 33% FG",
        "random-label": "Random 66% FG",
    }
    path = Path(path_str)
    match = re.search(r"__unc-([a-zA-Z0-9_-]+)__", path.name)
    if not match:
        return None
    return method_dict.get(match.group(1), match.group(1))


def plot_region_predictions_across_loops(
    img_folder: Path,
    gt_folder: Path,
    image_name: str,
    save_folder: Path,
    raw_folder: Path | None = None,
    raw_folders_from_file: Path | None = None,
    results_folder: Path | None = None,
    slice_axis: int = 0,
    max_loops: int | None = 5,
):
    if (raw_folder is None) == (raw_folders_from_file is None):
        raise ValueError(
            "Must specify exactly one of: raw_folder, raw_folders_from_file"
        )

    if raw_folder is not None:
        raw_folders = [raw_folder]
    else:
        with open(raw_folders_from_file, "r") as f:
            raw_folders = [line.strip() for line in f]

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    subimages_folder = save_folder / f"{image_name}"
    subimages_folder.mkdir(
        exist_ok=True,
    )

    for method_idx, raw_folder in enumerate(raw_folders):
        raw_folder = Path(raw_folder)
        results_folder = raw_folder.parent.parent / "nnUNet_results" / raw_folder.name

        # Load dataset information
        dset_json = load_json(Path(raw_folder) / "dataset.json")
        num_classes = len(dset_json["labels"])
        file_ending = dset_json["file_ending"]
        img_id = image_name.replace(file_ending, "")
        image_name = img_id + file_ending

        # Get AL method name for plot label
        al_method = extract_al_method_from_path(results_folder)

        # Load background image
        img_path = Path(img_folder) / (img_id + "_0000" + file_ending)
        img = sitk.ReadImage(str(img_path))
        img_np = sitk.GetArrayFromImage(img)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Gather and sort loop prediction folders
        label_dirs = [gt_folder]
        label_dirs += sorted(
            results_folder.glob("loop_*__predVal"),
            key=lambda p: int(p.name.split("_")[1]),
        )

        # Add the final predVal folder
        final_pred_folder = results_folder / "predVal"
        if final_pred_folder.exists():
            label_dirs.append(final_pred_folder)

        if not label_dirs:
            print(f"No prediction folders found in {results_folder}")
            continue

        if max_loops is not None:
            label_dirs = label_dirs[: max_loops + 1]

        # Prepare the plot
        fig, axs = plt.subplots(
            1, len(label_dirs), figsize=(2.5 * len(label_dirs), 3), squeeze=False
        )
        axs = axs[0]
        for i, loop_folder in enumerate(label_dirs):
            pred_path = loop_folder / image_name
            if not pred_path.exists():
                print(f"Missing prediction: {pred_path}")
                axs[i].axis("off")
                continue

            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
            pred_shape = pred.shape
            slicer = [slice(None)] * 3
            slicer[slice_axis] = pred_shape[slice_axis] // 2
            pred = pred[tuple(slicer)]
            pred = np.array(pred, dtype=float)
            pred[pred == 0] = np.nan

            assert img_np.shape == pred_shape
            base_img = img_np[tuple(slicer)]

            axs[i].imshow(base_img, cmap="gray", vmin=0, vmax=1)
            axs[i].imshow(
                pred, cmap="gist_rainbow", alpha=0.6, vmin=0, vmax=num_classes - 1
            )
            axs[i].axis("off")

        axs[0].axis("on")
        axs[0].grid(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_ylabel(al_method)
        fig.tight_layout()
        fig.savefig(
            subimages_folder / f"{img_id}_{al_method}.png",
            bbox_inches="tight",
        )
        plt.close()

    stitch_images(
        subimages_folder,
        save_folder / f"overview_{image_name}.png",
        columns=1,
        image_padding=0,
    )


if __name__ == "__main__":
    raw_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347"
    )
    output_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/query__analysis"
    )
    # visualize_query_trajectory(raw_folder, output_folder)

    output_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnactive/results/visualization"
    )

    raw_folder = Path(
        "/home/c817h/network/cluster-data/Dataset135_KiTS2021/nnUNet_raw/Dataset010_KiTS2021__patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40__unc-mutual_information__seed-12345"
    )
    image_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_raw/nnUNet_raw/Dataset135_KiTS2021/imagesTr"
    )
    output_folder = output_folder / raw_folder.name

    # raw_folder = Path(
    #     "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347"
    # )
    # image_folder = None

    plot_query_trajectory(
        raw_folder=raw_folder, save_folder=output_folder, img_folder=image_folder
    )

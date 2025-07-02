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
    path = Path(path_str)
    match = re.search(r"__unc-([a-zA-Z0-9_]+)__", path.name)
    if match:
        return match.group(1)
    return None


def plot_region_predictions_across_loops(
    raw_folder: Path,
    results_folder: Path,
    image_name: str,
    save_folder: Path,
    img_folder: Path | None = None,
    gt_folder: Path | None = None,
):
    results_folder = Path(results_folder)
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)

    file_ending = load_json(Path(raw_folder) / "dataset.json")["file_ending"]
    img_id = image_name.replace(file_ending, "")
    image_name = img_id + file_ending

    al_method = extract_al_method_from_path(results_folder)

    # Load background image
    if img_folder:
        img_path = Path(img_folder) / (img_id + "_0000" + file_ending)
        img = sitk.ReadImage(str(img_path))
        img_np = sitk.GetArrayFromImage(img)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    else:
        img_np = None

    # Gather and sort loop prediction folders
    label_dirs = [gt_folder]
    label_dirs += sorted(
        results_folder.glob("loop_*__predVal"), key=lambda p: int(p.name.split("_")[1])
    )
    tags = ["GT"] + [f.name for f in label_dirs[1:]]

    # Add the final predVal folder
    final_pred_folder = results_folder / "predVal"
    if final_pred_folder.exists():
        label_dirs.append(final_pred_folder)
        tags += ["final"]

    if not label_dirs:
        print(f"No prediction folders found in {results_folder}")
        return

    # Prepare the plot
    fig, axs = plt.subplots(
        1, len(label_dirs), figsize=(3.5 * len(label_dirs), 3), squeeze=False
    )
    axs = axs[0]
    for i, (loop_folder, label) in enumerate(zip(label_dirs, tags)):
        pred_path = loop_folder / image_name
        if not pred_path.exists():
            print(f"Missing prediction: {pred_path}")
            axs[i].axis("off")
            axs[i].set_title(f"{label}\n(Missing)")
            continue

        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        pred_shape = pred.shape
        fixed_xcoord = pred_shape[0] // 2
        pred = pred[fixed_xcoord]
        pred = np.array(pred, dtype=float)
        pred[pred == 0] = np.nan

        if img_np is not None:
            assert img_np.shape == pred_shape
            base_img = img_np[fixed_xcoord]
        else:
            base_img = np.zeros_like(pred, dtype=np.float32)

        axs[i].imshow(base_img, cmap="gray", vmin=0, vmax=1)
        axs[i].imshow(pred, cmap="Set3", alpha=0.4)
        axs[i].set_title(label)
        axs[i].axis("off")

    axs[0].axis("on")
    axs[0].grid(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_ylabel(al_method)
    fig.tight_layout()
    fig.savefig(save_folder / f"{img_id}.pdf", bbox_inches="tight")
    plt.close()


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

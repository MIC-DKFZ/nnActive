import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from loguru import logger
from matplotlib import patches

from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_nested_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.utils.io import load_json
from nnactive.utils.patches import create_patch_mask_for_image
from nnactive.utils.pyutils import (
    get_bounding_box_from_mask,
    rescale_pad_to_square,
    stitch_images,
)


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
    raw_folder: Path,
    img_folder: Path | None = None,
    gt_folder: Path | None = None,
    save_folder: Path = None,
    center_axs: int | list | None = None,
    show_patches_only: bool = False,
):
    print(f"Saving results to folder: {save_folder}")
    if show_patches_only:
        prefix = "patch-only_"
    else:
        prefix = ""

    dset_json = load_json(raw_folder / "dataset.json")
    file_ending = dset_json["file_ending"]
    labels_dict = dset_json["labels"]
    num_classes = len(labels_dict)

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
        os.makedirs(save_folder / (prefix + f"loop_{i:03d}"), exist_ok=True)

    for img_name in img_names:
        all_img_patches = [x for xs in loop_patches for x in xs]
        if len([p for p in all_img_patches if p.file == img_name]) == 0:
            continue
        img = sitk.ReadImage(
            img_folder / (img_name.replace(file_ending, "") + "_0000" + file_ending)
        )
        img: np.ndarray = sitk.GetArrayFromImage(img)
        img = (img - img.min()) / (img.max() - img.min())

        gt = None
        if gt_folder is not None:
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_folder / img_name)))
            gt = np.array(gt, dtype=float)

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
                if center_axs is None:
                    center_axs = [0, 1, 2]
                elif isinstance(center_axs, int):
                    center_axs = [center_axs]
                views = []
                masks = []
                gts = []
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
                    maskplane = np.array(maskplane, dtype=float)
                    maskplane[maskplane == 0] = np.nan
                    if gt is not None:
                        gtplane = rescale_pad_to_square(gt[slices].squeeze())

                    if show_patches_only:
                        ys, xs = np.nonzero(~np.isnan(maskplane))
                        bbox_slice = (
                            slice(ys.min(), ys.max() + 1),
                            slice(xs.min(), xs.max() + 1),
                        )
                        maskplane = maskplane[bbox_slice]
                        viewplane = viewplane[bbox_slice]
                        gtplane = gtplane[bbox_slice]

                    views.append(viewplane)
                    masks.append(maskplane)
                    if gt is not None:
                        gts.append(gtplane)

                fig, axs = plt.subplots(1, len(center_axs), squeeze=False)
                axs = axs[0]
                for c in range(len(center_axs)):
                    axs[c].imshow(views[c], cmap="gray", vmin=0, vmax=1)
                    if gts:
                        _gt = gts[c]
                        _gt[
                            (_gt == labels_dict["background"])
                            | (_gt == labels_dict["ignore"])
                        ] = np.nan  # Set background and ignore label to NaN
                        axs[c].imshow(
                            _gt,
                            cmap="gist_rainbow",
                            alpha=0.3,
                            vmin=0,
                            vmax=num_classes - 1,
                        )

                    if not show_patches_only:
                        # axs[c].imshow(masks[c], cmap=plt.cm.Reds, alpha=0.2, vmin=0, vmax=1)
                        x_min, y_min, width, height = get_bounding_box_from_mask(
                            masks[c]
                        )
                        rect = patches.Rectangle(
                            (x_min, y_min),
                            width,
                            height,
                            linewidth=2,
                            edgecolor="red",
                            facecolor=(1.0, 1.0, 1.0, 0.2),
                        )
                        axs[c].add_patch(rect)

                    axs[c].set_xticks([])
                    axs[c].set_yticks([])
                file_id = img_name.replace(file_ending, "")
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)
                fig.suptitle(
                    f"Patch {p_id} Loop {i} File {file_id}",
                    y=0.72 if len(center_axs) == 3 else None,
                )
                filename = f"loop-{i:02d}__id-{p_id:02d}__img-{file_id}.png"
                plt.savefig(
                    save_folder / (prefix + f"loop_{i:03d}") / filename,
                    bbox_inches="tight",
                )
                plt.close("all")

    for i in range(len(loop_patches)):
        stitch_images(
            save_folder / (prefix + f"loop_{i:03d}"),
            save_folder / (prefix + f"overview-loop_{i:03d}.png"),
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
        "class_pe33": "Cla PE 33%",
        "class_pe66": "Cla PE 66%",
        "class_power_pe66_exp": "ClaSP PE",
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


def plot_class_stratification(
    img_path: Path,
    gt_path: Path,
    probs_path: Path,
    save_folder: Path,
    raw_folder: Path | None = None,
    slice_axis: int = 0,
):
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    raw_folder = Path(raw_folder)

    # Load dataset information
    dset_json = load_json(Path(raw_folder) / "dataset.json")
    labels_dict: dict = dset_json["labels"]
    bg_label = labels_dict.get("background")
    ignore_label = labels_dict.pop("ignore")
    num_classes = len(labels_dict)
    file_ending = dset_json["file_ending"]
    image_name = img_path.name
    img_id = image_name.replace(file_ending, "")

    # Load background image
    img = sitk.ReadImage(str(img_path))
    img = sitk.GetArrayFromImage(img)
    img = (img - img.min()) / (img.max() - img.min())

    # Load GT labels
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    gt_shape = gt.shape
    slicer = [slice(None)] * 3
    slicer[slice_axis] = gt_shape[slice_axis] // 2
    gt = gt[tuple(slicer)]
    gt = np.array(gt, dtype=float)
    gt[(gt == bg_label) | (gt == ignore_label)] = np.nan
    assert img.shape == gt_shape
    img = img[tuple(slicer)]

    # Load probabilities
    probs = np.load(probs_path)["probabilities"]
    preds = np.argmax(probs, axis=0)[tuple(slicer)]
    preds = np.array(preds, dtype=float)
    preds[preds == bg_label] = np.nan

    # Prepare the plot
    fig, axs = plt.subplots(
        2,
        num_classes + 1,
        figsize=(1.8 * num_classes, 3),
        squeeze=False,
        gridspec_kw={"hspace": 0.4},
    )

    # Plot image
    for ax in axs[0]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    axs[1, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    for ax in axs[1][2:]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    # Plot GT
    axs[0, 0].imshow(gt, cmap="gist_rainbow", alpha=0.6, vmin=0, vmax=num_classes)
    axs[0, 0].set_title("GT")

    # Plot Predictions
    axs[1, 0].imshow(preds, cmap="gist_rainbow", alpha=0.6, vmin=0, vmax=num_classes)
    axs[1, 0].set_title("Prediction")

    # Plot Uncertainty Maps
    pe = probs * np.log(probs)
    if np.isnan(pe).sum() > 0:
        print("Warning: some nan values encountered after log operation")
    pe = -np.sum(pe, axis=0)

    axs[0, 1].imshow(pe[tuple(slicer)], cmap="Reds", alpha=0.5)
    axs[0, 1].set_title("$H[x]$")

    for i in range(1, num_classes):
        axs[0, i + 1].imshow((pe * probs[i])[tuple(slicer)], cmap="Reds", alpha=0.5)
        axs[0, i + 1].set_title(rf"$H[x]\times p_{i}(x)$")

        axs[1, i + 1].imshow(probs[i][tuple(slicer)], cmap="Reds", alpha=0.5)
        axs[1, i + 1].set_title(
            rf"p: {[k for k, v in labels_dict.items() if v == i][0]}"
        )

    for ax in axs.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_folder / f"{img_id}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fnames = [
        "patient002_frame01",
        "patient002_frame12",
        "patient003_frame01",
        "patient003_frame15",
    ]
    for fname in fnames:
        plot_class_stratification(
            img_path=Path(
                f"/home/j211b/experiments/nnactive/nnactive_data/Dataset027_ACDC/nnUNet_raw/Dataset001_ACDC__patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30__unc-mutual_information__seed-12346/imagesTr/{fname}_0000.nii.gz"
            ),
            gt_path=Path(
                f"/home/j211b/experiments/nnactive/nnactive_raw/nnUNet_raw/Dataset027_ACDC/labelsTr/{fname}.nii.gz"
            ),
            probs_path=Path(
                f"/home/j211b/experiments/nnactive/nnactive_data/Dataset027_ACDC/nnUNet_results/Dataset001_ACDC__patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30__unc-mutual_information__seed-12346/tmp_predTr/{fname}.npz"
            ),
            raw_folder=Path(
                "/home/j211b/experiments/nnactive/nnactive_data/Dataset027_ACDC/nnUNet_raw/Dataset001_ACDC__patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30__unc-mutual_information__seed-12346/"
            ),
            save_folder=Path(
                "/home/j211b/experiments/nnactive/nnactive_data/Dataset027_ACDC/nnUNet_results/Dataset001_ACDC__patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30__unc-mutual_information__seed-12346/tmp_plot/"
            ),
        )

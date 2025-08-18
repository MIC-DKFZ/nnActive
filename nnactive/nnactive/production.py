from pathlib import Path

from loguru import logger

from nnactive.data.create_empty_masks import create_empty_mask


def produce_empty_masks(
    images_folder: Path,
    output_folder: Path,
    fill_value: int,
    file_ending: str,
    additional_label_folder: Path | None = None,
    modality_iden: str = "_0000",
):
    """Create empty labels for all images in images_folder.

    Args:
        images_folder (Path): Folder with images
        output_folder (Path): Folder for labels
        fill_value (int): Label for ignore regions
        file_ending (str): File ending
        additional_label_folder (Path, optional): Folder with additional labels. Defaults to None.
        modality_iden (str): _0000 modality string after img_identifier e.g. _0000. Defaults to _0000.
    """
    fns = [
        file.name[: -(len(modality_iden) + len(file_ending))]
        for file in images_folder.iterdir()
        if file.is_file() and file.name.endswith(file_ending)
    ]
    fns = list(set(fns))
    logger.info("Creating ignore label images for #{} images".format(len(fns)))

    if output_folder.exists():
        logger.warning(
            "Output folder already exists. Existing files will be overwritten."
        )
    else:
        output_folder.mkdir(parents=True, exist_ok=True)

    for fn in fns:
        image_file = images_folder / (fn + modality_iden + file_ending)
        save_file = output_folder / (fn + file_ending)
        additional_label_file = (
            additional_label_folder / (fn + file_ending)
            if additional_label_folder
            else None
        )
        create_empty_mask(
            image_file,
            fill_value,
            save_file,
            additional_label_file=additional_label_file,
            change_to_uint=True,
        )

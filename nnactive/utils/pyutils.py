import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Hashable, Iterable, List

import numpy as np
from PIL import Image
from pydantic import dataclasses


def rglob_follow_symlinks(root: Path, pattern: str):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        current_dir = Path(dirpath)
        # Check both files and dirs
        for name in filenames + dirnames:
            path = current_dir / name
            if path.match(pattern):
                yield path


def create_string_identifier(
    values: Iterable | None = None,
    ignore_ident: Iterable[int] | None = None,
    remove_list: Iterable[str] = (" ", "_", "-"),
) -> str:
    """Create a unique identifier from the given identifier values with commas separated.

    Args:
        identifier (Iterable | None, optional): _description_. Defaults to None.
        ignore_ident (Iterable[int] | None, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    ignore_ident = ignore_ident if ignore_ident is not None else []
    ident_name = (
        tuple([k for i, k in enumerate(values) if i not in ignore_ident])
        if values is not None
        else tuple()
    )
    ident_name = f"{ident_name}"
    for rm_char in remove_list:
        ident_name = ident_name.replace(rm_char, "")
    # remove the brackets from list
    ident_name = ident_name[1:-1]
    return ident_name


def stitch_images(
    image_dir: Path,
    output_file: Path | str,
    columns: int = 3,
    image_padding: int = 0,
    background_color: tuple[int, int, int] = (255, 255, 255),
):
    """
    Stitches multiple images from a directory into a single image arranged in a grid.

    Args:
        image_dir (Path): The directory containing the images to stitch.
        output_file (Path | str): The path to save the stitched image.
        columns (int, optional): The number of columns in the grid. Defaults to 3.
        image_padding (int, optional): The padding between images in pixels. Defaults to 0.
        background_color (tuple[int, int, int], optional): The background color of the stitched image. Defaults to (255, 255, 255).

    Returns:
        None
    """
    # Get all image files from the folder
    image_files = [
        f.name
        for f in image_dir.iterdir()
        if f.is_file() and f.name.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif"))
    ]
    image_files.sort()

    # Open all images and find the maximum width and height
    images = [Image.open(image_dir / f) for f in image_files]
    widths, heights = zip(*(img.size for img in images))

    max_width = max(widths)
    max_height = max(heights)

    # Calculate the number of rows needed
    rows = (len(images) + columns - 1) // columns

    # Create a new image with a background color
    grid_width = columns * max_width + (columns - 1) * image_padding
    grid_height = rows * max_height + (rows - 1) * image_padding
    grid_image = Image.new("RGB", (grid_width, grid_height), color=background_color)

    # Paste images into the grid
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x_offset = col * (max_width + image_padding)
        y_offset = row * (max_height + image_padding)
        grid_image.paste(image, (x_offset, y_offset))

    # Save the stitched image
    grid_image.save(output_file)
    print(f"Successfully saved grid image to {output_file}")


def rescale_pad_to_square(image: np.ndarray) -> np.ndarray:
    """Rescale image to a square by repeating the image in the
    dimension with the smallest size. (so not necessarily perfect square)

    Args:
        image (np.ndarray): 2D Image with shape(height, width)

    Returns:
        np.ndarray: Rescaled and padded image
    """
    image = rescale_to_rel_square(image)
    image = pad_to_square(image)
    return image


def rescale_to_rel_square(image: np.ndarray) -> np.ndarray:
    """Rescale image to a square by repeating the image in the
    dimension with the smallest size. (so not necessarily perfect square)

    Args:
        image (np.ndarray): 2D Image with shape(height, width)

    Returns:
        np.ndarray: Rescaled image
    """
    height, width = image.shape
    ratio = height / width
    if ratio > 1:
        image = np.repeat(image, ratio // 1, axis=1)
    if ratio < 1:
        image = np.repeat(image, (1 / ratio) // 1, axis=0)
    return image


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pad image to a square by adding zeros to the smaller dimension.

    Args:
        image (np.ndarray): 2D Image with shape(height, width)

    Returns:
        np.ndarray: Padded image with shape (max(height, width), max(height, width))
    """
    height, width = image.shape
    max_dim = max(height, width)

    # Calculate padding for image
    pad_height = (max_dim - height) // 2
    pad_width = (max_dim - width) // 2
    pad_height_odd = (max_dim - height) % 2
    pad_width_odd = (max_dim - width) % 2

    # Pad image
    padded_image = np.pad(
        image,
        (
            (pad_height, pad_height + pad_height_odd),
            (pad_width, pad_width + pad_width_odd),
        ),
        mode="constant",
        constant_values=0,
    )
    return padded_image


def compute_conv_output_size(
    input_size: int, kernel_size: int, stride: int, padding: int
) -> int:
    """
    Computes the output size of a convolution operation.

    Args:
        input_size (int): The size of the input (e.g., height or width).
        kernel_size (int): The size of the convolution kernel (e.g., height or width).
        stride (int): The stride of the convolution.
        padding (int): The amount of padding added to the input.

    Returns:
        int: The size of the output after the convolution operation.
    """
    return (input_size - kernel_size + 2 * padding) // stride + 1


def get_subitems(folder: Path, level: int) -> List[Path]:
    """Retrieve subitems in a folder up to a specified directory depth.

    This function returns a sorted list of subitems (files and directories) within the specified
    folder, up to a given directory depth. If the depth level is 0, the function returns the folder itself.

    Args:
        folder (Path): The path to the root folder from which subitems are to be retrieved.
        level (int): The depth level up to which subitems should be retrieved.

    Returns:
        List[Path]: A sorted list of Paths representing the subitems in the folder up to the specified depth level.

    Example:
        >>> from pathlib import Path
        >>> folder = Path('/path/to/folder')
        >>> level = 1
        >>> get_subitems(folder, level)
        [PosixPath('/path/to/folder/file1.txt'), PosixPath('/path/to/folder/file2.txt'), PosixPath('/path/to/folder/subfolder')]
    """

    if level == 0:
        return [folder]
    pattern = "/".join(["*"] * level)
    return sorted(folder.glob(pattern))


def invert_dict(d: dict[list[Any]]) -> dict[list[Any]]:
    """Inverts a dictionary where values are lists, mapping elements of those lists to their corresponding keys.

    Args:
        d (dict[list[Any]]): The input dictionary to be inverted. Keys are strings, and values are lists of elements.

    Returns:
        dict[list[Any]]: A dictionary where keys are elements from the input lists, and values are lists of keys from the input dictionary
        that correspond to those elements.

    Example:
        >>> original_dict = {'a': [1, 2], 'b': [2, 3], 'c': [1, 3]}
        >>> inverted_dict = invert_dict(original_dict)
        >>> print(inverted_dict)
        {1: ['a', 'c'], 2: ['a', 'b'], 3: ['b', 'c']}
    """
    inverted_dict = defaultdict(list)
    for key, values in d.items():
        for value in values:
            inverted_dict[value].append(key)
    return inverted_dict


def get_clean_dataclass_dict(data: dataclasses) -> dict:
    datadict = deepcopy(data.__dict__)
    popkeys = []
    for key in datadict:
        if isinstance(key, str):
            if key.startswith("__") and key.endswith("__"):
                popkeys.append(key)
    for key in popkeys:
        datadict.pop(key)
    return datadict


def merge_dict_lists_on_indices(
    init_dict: list[dict], update_dict: list[dict], indices: list[Hashable]
) -> list[dict]:
    merged_dicts = []
    for i in range(len(init_dict)):
        merged_dict = init_dict[i].copy()
        extended = False
        for j in range(len(update_dict)):
            accept = True
            for index in indices:
                if merged_dict[index] != update_dict[j][index]:
                    accept = False
            if accept:
                merged_dict.update(update_dict[j])
                extended = True
                break
        if not extended:
            raise ValueError("One dictionary in the list does not have a partner.")
        else:
            merged_dicts.append(merged_dict)
    return merged_dicts

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Hashable, List

from PIL import Image
from pydantic import dataclasses


def stitch_images(
    image_dir: Path,
    output_file: Path | str,
    columns: int = 3,
    image_padding: int = 0,
    background_color: tuple[int, int, int] = (255, 255, 255),
):
    # Get all image files from the folder
    image_files = [
        f.name
        for f in image_dir.iterdir()
        if f.is_file() and f.name.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif"))
    ]
    image_files.sort()
    print(image_files)

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

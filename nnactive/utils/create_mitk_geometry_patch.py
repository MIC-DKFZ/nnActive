import os.path
from pathlib import Path
from typing import Tuple

# TODO: this script currently only works for 3D images. Needs adjustments for 2D


def set_mitk_geometry_size(
    size: Tuple[int, int, int],
    origin: Tuple[int, int, int] = None,
    direction: Tuple[int, int, int, int, int, int, int, int, int] = None,
):
    """
    Replace the information in a xml template with the given geometry information
    Args:
        size (Tuple[int, int, int]): The size of the bounding box
        origin (Tuple[int, int, int]): The origin of the bounding box
        direction (direction: Tuple[int, int, int, int, int, int, int, int, int]): The direction of the bounding box

    Returns:
        str: The modified xml string containing the geometry information

    """
    script_dir = Path(__file__).parent.absolute()
    with open(script_dir / "mitk_geometry.xml", "r") as f:
        mitk_geometry_xml = f.read()
    mitk_geometry_xml = mitk_geometry_xml.replace(
        '<Max type="Vector3D" x="0" y="0" z="0"/>',
        f'<Max type="Vector3D" x="{size[0]}" y="{size[1]}" z="{size[2]}"/>',
    )
    if origin:
        mitk_geometry_xml = mitk_geometry_xml.replace(
            '<Offset type="Vector3D" x="0" y="0" z="0"/>',
            f'<Offset type="Vector3D" x="{origin[0]}" y="{origin[1]}" z="{origin[2]}"/>',
        )
    if direction:
        mitk_geometry_xml = mitk_geometry_xml.replace(
            '<IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0" m_0_2="0" '
            'm_1_0="0" m_1_1="1" m_1_2="0" '
            'm_2_0="0" m_2_1="0" m_2_2="1"/>',
            f'<IndexToWorld type="Matrix3x3" m_0_0="{direction[0]}" m_0_1="{direction[1]}" m_0_2="{direction[2]}" '
            f'm_1_0="{direction[3]}" m_1_1="{direction[4]}" m_1_2="{direction[5]}" '
            f'm_2_0="{direction[6]}" m_2_1="{direction[7]}" m_2_2="{direction[8]}"/>',
        )
    return mitk_geometry_xml


def save_mitk_geometry_file(mitk_geometry_xml: str, save_path: Path):
    """
    Save the .mitkgeometry file that can then be displayed as bounding box in MITK
    Args:
        mitk_geometry_xml: The xml string
        save_path: path where to save the file
    """
    if not os.path.exists(save_path.parent):
        os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(mitk_geometry_xml)


def main(
    save_path: Path,
    size: Tuple[int, int, int],
    origin: Tuple[int, int, int] = None,
    direction: Tuple[int, int, int, int, int, int, int, int, int] = None,
):
    """
    Create an .mitkgeometry file which is a bounding box for manual patch selection in MITK
    Args:
        save_path (Path): the save path of the file
        size (Tuple[int, int, int]): The size of the bounding box
        origin (Tuple[int, int, int], optional): The origin of the bounding box. Defaults to 0, 0, 0
        direction (Tuple[int, int, int, int, int, int, int, int, int], optional): The direction of the bounding box.
                  Defaults to 1, 0, 0, 0, 1, 0, 0, 0, 1
    """
    mitk_xml = set_mitk_geometry_size(size=size, origin=origin, direction=direction)
    save_mitk_geometry_file(mitk_xml, save_path)


if __name__ == "__main__":
    main(Path("/home/kckahl/MITK_patch_test/patch.mitkgeometry"), (10, 10, 10))

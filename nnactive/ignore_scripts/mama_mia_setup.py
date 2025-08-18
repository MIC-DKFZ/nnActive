import shutil
from pathlib import Path

import SimpleITK as sitk

from nnactive.utils.io import load_json, save_json

# Execute this script before resampling of the dataset

# Input and output folders
input_dir = Path("/path/to/nnActive_raw/nnUNet_raw/Dataset094_MAMA_MIA")
target = Path("/path/to/nnActive_raw/nnUNet_raw/Dataset095_MAMA_MIA_subtracted")
target.mkdir(parents=True, exist_ok=True)

changedirs = []

for subpath in input_dir.iterdir():
    if subpath.is_dir() and subpath.name.startswith("images"):
        print(f"Subtracting images inside {subpath.name}")
        changedirs.append(subpath.name)
    elif subpath.is_dir() and subpath.name.startswith("labels"):
        # Copy labels to the new directory
        print(f"Copying {subpath.name} to {target}")
        shutil.copytree(subpath, target / subpath.name)


# Load the JSON file
json_path = input_dir / "dataset.json"
d_json = load_json(json_path)
# Update the JSON file
d_json["name"] = "Dataset095_MAMA_MIA_subtracted"
# d_json["converted by"] = ""
d_json["channel_names"] = {"0": "MRI Subtraction Image"}
d_json["note"] = (
    "Obtained from original dataset by subtracting ( _0001 - _0000 ).     "
    + d_json["note"]
)
save_json(d_json, target / "dataset.json")


for changename in changedirs:
    src_dir = input_dir / changename
    target_dir = target / changename
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all _0000.nii.gz files
    img0_files = sorted(src_dir.glob("*_0000.nii.gz"))

    for img0_path in img0_files:
        # Find matching _0001.nii.gz
        img1_path = img0_path.with_name(
            img0_path.name.replace("_0000.nii.gz", "_0001.nii.gz")
        )

        if not img1_path.exists():
            print(f"Skipping {img0_path.name}: matching _0001.nii.gz not found.")
            continue

        # Load images using SimpleITK
        img0 = sitk.ReadImage(str(img0_path))
        img1 = sitk.ReadImage(str(img1_path))

        arr0 = sitk.GetArrayFromImage(img0).astype("int16")
        arr1 = sitk.GetArrayFromImage(img1).astype("int16")
        arr_sub = arr1 - arr0

        img_sub = sitk.GetImageFromArray(arr_sub)
        img_sub.SetSpacing(img0.GetSpacing())
        img_sub.SetOrigin(img0.GetOrigin())
        img_sub.SetDirection(img0.GetDirection())

        # Save with the same name as _0000 in output directory
        output_path = target_dir / img0_path.name
        sitk.WriteImage(img_sub, str(output_path))

        print(f"Saved: {output_path}")

print("Done.")

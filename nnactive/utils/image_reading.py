import json
import os

import SimpleITK as sitk


def read_images_to_numpy(dataset_json_path, images_folder, func, *args, **kwargs):
    """
    Function that iterates through a directory of images with a file ending specified in a dataset.json,
    converts the images to numpy arrays and executes a custom function using these numpy arrays

    Args:
        dataset_json_path : path to the dataset.json file that contains the information about the file ending of the images
        images_folder : folder that contains the images that should be iterated
        func : function that should be executed on the images (e.g. aggregation of values, ...)
    """
    # read the dataset json to get the information about the file ending
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
    file_ending = dataset_json["file_ending"]

    # iterate through the directory
    for image_name in os.listdir(images_folder):
        if image_name.endswith(file_ending):
            # read images and convert them to numpy arrays
            sitk_image = sitk.ReadImage(os.path.join(images_folder, image_name))
            numpy_image = sitk.GetArrayFromImage(sitk_image)
            # execute a function with the image loaded as numpy array
            func(numpy_image, image_name, images_folder, *args, **kwargs)

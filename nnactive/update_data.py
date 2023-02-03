from argparse import ArgumentParser
import os
from pathlib import Path
import json

from typing import List

from nnactive.data.prepare_starting_budget import make_patches_from_ground_truth, make_whole_from_ground_truth, make_empty_from_ground_truth, Patch

LOOP_PATTERN = "loop_"

# def label_patches(patches:List[dict], dataset_dir:Path, input_dir:Path):
    


if __name__ == "__main__":
    # 1. load all loop_XXX.json files
    parser = ArgumentParser()
    parser.add_argument("-p", "--dataset_path")
    parser.add_argument("-i", "--input_data")
    parser.add_argument("-l", "--loop", default=None, type=int)

    args = parser.parse_args()
    labeled_path = Path(args.input_data)
    data_path = Path(args.dataset_path)
    loop_val = args.loop

    loop_files = []
    for file in os.listdir(data_path):
        if file[:len(LOOP_PATTERN)] == LOOP_PATTERN:
            loop_files.append(file)

    loop_files.sort(key=lambda x: int(x.split(LOOP_PATTERN)[1].split(".json")[0]))
    
    # Take only loop_files up to a certain loop_{loop_val}.json
    if loop_val is not None:
        loop_files = [loop_files[i] for i in range(loop_val +1)]
    
    # load info 
    patches = []
    for loop_file in loop_files:
        with open(data_path/loop_file, "r") as file:
            patches_loop: List[dict] = json.load(file)["patches"]
        patches.extend(patches_loop)

    with open(data_path/"dataset.json", "r") as file:
        data_json = json.load(file)

    
    whole_label = []
    patch_label = []
    for patch in patches:
        if patch["size"] == "whole":
            whole_label.append(patch)
        else:
            patch_label.append(patch)

    labeled_files = set([patch["file"] for patch in (patch_label+whole_label)])
    empty_segs = [file for file in os.listdir(labeled_path/"labelsTr") if file.endswith(data_json["file_ending"])]
    empty_segs = [file for file in empty_segs if file not in labeled_files]

    make_whole_from_ground_truth(whole_label, labeled_path/"labelsTr", data_path/"labelsTr")

    patch_label = Patch.from_json()
    make_patches_from_ground_truth(
        patch_label, labeled_path/"labelsTr", data_path/"labelsTr", data_json, data_json["labels"]["ignore"]
    )

    # import IPython
    # IPython.embed()
    make_empty_from_ground_truth(empty_segs,labeled_path/"labelsTr", data_path/"labelsTr", data_json["labels"]["ignore"])


    


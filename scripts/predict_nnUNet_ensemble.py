import subprocess
from argparse import ArgumentParser

from nnactive.config import ActiveConfig
from nnactive.nnunet.utils import get_raw_path, get_results_path


def main():
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)
    args = parser.parse_args()
    dataset_id = args.dataset_id

    num_folds = 5
    # trainer = "nnUNetDebugTrainer"
    config = ActiveConfig.get_from_id(dataset_id)

    images_path = get_raw_path(dataset_id) / "imagesTr"
    output_path = get_results_path(dataset_id) / "predTr"

    for fold in range(num_folds):
        output_fold_path = output_path / f"fold_{fold}"
        ex_command = f"nnUNetv2_predict -d {dataset_id} -c {config.model} -i {images_path} -o {output_fold_path} -tr {config.trainer} -f {fold} --save_probabilities"
        # print(ex_command)
        subprocess.call(ex_command, shell=True)


if __name__ == "__main__":
    main()

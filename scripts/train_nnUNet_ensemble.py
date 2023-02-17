import subprocess
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)
    args = parser.parse_args()
    dataset_id = args.dataset_id

    num_folds = 5
    trainer = "nnUNetDebugTrainer"

    for fold in range(num_folds):
        ex_command = f"nnUNetv2_train {dataset_id} 3d_fullres {fold} -tr {trainer}"
        # print(ex_command)
        subprocess.call(ex_command, shell=True)


if __name__ == "__main__":
    main()

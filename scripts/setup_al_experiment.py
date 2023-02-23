import os
from argparse import ArgumentParser

from nnactive.config.hippcampus import get_test_hippocampus_config
from nnactive.nnunet.utils import convert_id_to_dataset_name
from nnactive.paths import get_nnActive_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=int)

    args = parser.parse_args()

    dataset_id: int = args.dataset

    dataset_name: str = convert_id_to_dataset_name(dataset_id)

    results_path = get_nnActive_results()

    save_path = results_path / dataset_name

    config = get_test_hippocampus_config()

    os.makedirs(save_path, exist_ok=True)

    config.save_id(dataset_id)

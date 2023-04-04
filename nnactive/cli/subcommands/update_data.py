from argparse import ArgumentParser

from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.results.state import State
from nnactive.update_data import update_data


def main():
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)

    args = parser.parse_args()
    dataset_id = args.dataset_id

    update_step(dataset_id)


def update_step(dataset_id, num_folds=5, loop_val=None):
    data_path = get_raw_path(dataset_id)
    save_splits_file = get_preprocessed_path(dataset_id) / "splits_final.json"
    target_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(dataset_id)
    ignore_label = dataset_json["labels"]["ignore"]
    file_ending = dataset_json["file_ending"]

    base_dir = get_raw_path(dataset_json["annotated_id"]) / "labelsTr"

    update_data(
        data_path,
        save_splits_file,
        ignore_label,
        file_ending,
        base_dir,
        target_dir,
        loop_val=loop_val,
        num_folds=num_folds,
    )
    state = State.get_id_state(dataset_id)
    state.update_data = True
    state.new_loop()
    state.save_state()


if __name__ == "__main__":
    main()

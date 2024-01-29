from argparse import ArgumentParser, Namespace
from copy import deepcopy
from itertools import product

from nnactive.config import ActiveConfig
from nnactive.loops.loading import get_loop_patches, get_sorted_loop_files, save_loop
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.strategies import init_strategy
from nnactive.utils.io import save_json

parser = ArgumentParser()
parser.add_argument("-d", "--dataset_id", type=int, required=True)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--query_strategy", default=None, type=str)


def main(args: Namespace):
    dataset_id = args.dataset_id
    verbose = args.verbose
    config = ActiveConfig.get_from_id(dataset_id)

    raw_dataset_path = get_raw_path(dataset_id)
    loop_val = len(get_sorted_loop_files(raw_dataset_path))
    seed = config.seed + loop_val

    value_dict = {
        "query_strategy": [
            (
                args.query_strategy
                if args.query_strategy is not None
                else config.uncertainty
            )
        ],
        "use_mirroring": [False, True],
        "use_gaussian": [False, True],
        "tile_step_size": [
            1.0,  # CUDA oom error
            # 0.75,
            # 0.5,
        ],
        "_n_patch_per_image": [
            config.query_size,
            config.query_size // 2,
            config.query_size // 4,
        ],
    }

    value_product = list(product(*list(value_dict.values())))
    dict_list = [dict(zip(value_dict.keys(), values)) for values in value_product]

    queries = []
    strategies = []
    for dictionary in dict_list:
        for key, val in dictionary.items():
            setattr(config, key, val)
        print("Config:")
        print(config)
        strategy = init_strategy(
            config.uncertainty,
            dataset_id,
            config.query_size,
            patch_size=config.patch_size,
            trials_per_img=6000,
            seed=seed,
            n_patch_per_image=config.n_patch_per_image,
            verbose=verbose,
            agg_stride=config.agg_stride,
            use_mirroring=config.use_mirroring,
            tile_step_size=config.tile_step_size,
        )
        query = strategy.query()
        strategies.append(strategy)
        queries.append(query)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

from pathlib import Path

from nnactive.config import ActiveConfig
from nnactive.loops.loading import get_loop_patches, get_sorted_loop_files, save_loop
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.strategies import get_strategy
from nnactive.utils.io import save_json


def query_pool(dataset_id: int):
    state = State.get_id_state(dataset_id)
    config = ActiveConfig.get_from_id(dataset_id)

    raw_dataset_path = get_raw_path(dataset_id)
    loop_val = len(get_sorted_loop_files(raw_dataset_path))
    seed = config.seed + loop_val
    strategy = get_strategy(config.uncertainty, dataset_id, seed=seed)
    query = strategy.query()

    top_patches_fn = f"{config.uncertainty}_{loop_val:03d}.json"
    save_json(strategy.top_patches, raw_dataset_path / top_patches_fn)

    loop_json = {"patches": query}
    save_loop(raw_dataset_path, loop_json, loop_val)

    #
    state.pred_tr = True
    state.query = True
    state.save_state()

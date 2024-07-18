from pathlib import Path

import nnunetv2.paths

from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.loops.loading import get_loop_patches, get_sorted_loop_files, save_loop
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder
from nnactive.strategies import get_strategy
from nnactive.utils.io import save_json


def query_pool(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    force: bool = False,
    verbose: bool = False,
):
    config.set_nnunet_env()
    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id)

    raw_dataset_path = get_raw_path(state.dataset_id)
    loop_val = len(get_sorted_loop_files(raw_dataset_path))
    seed = config.seed + loop_val
    strategy = get_strategy(
        config.uncertainty,
        config,
        state.dataset_id,
        seed=seed,
        loop_val=loop_val,
        verbose=verbose,
    )
    query = strategy.query(n_gpus=runtime_config.n_gpus)

    top_patches_fn = f"{config.uncertainty}_{loop_val:03d}.json"
    save_json(strategy.top_patches, raw_dataset_path / top_patches_fn)

    loop_json = {"patches": query}
    save_loop(raw_dataset_path, loop_json, loop_val)

    #
    if not force:
        state.pred_tr = True
        state.query = True
        state.save_state()

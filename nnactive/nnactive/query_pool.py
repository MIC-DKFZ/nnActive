from pathlib import Path

import nnunetv2.paths
import torch

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
        state = State.get_id_state(continue_id, verify=not force)

    raw_dataset_path = get_raw_path(state.dataset_id)
    loop_val = state.loop
    num_loop_files = len(get_sorted_loop_files(raw_dataset_path))
    seed = config.seed + loop_val + 1
    strategy = get_strategy(
        config.uncertainty,
        config,
        state.dataset_id,
        seed=seed,
        loop_val=loop_val,
        verbose=verbose,
    )
    query = strategy.query(n_gpus=runtime_config.n_gpus)

    top_patches_fn = f"{config.uncertainty}_{num_loop_files:03d}.json"
    if (
        isinstance(strategy.top_patches, list)
        and len(strategy.top_patches) > 0
        and strategy.top_patches[0].get("repr") is not None
    ):
        repr = torch.empty(
            (len(strategy.top_patches), len(strategy.top_patches[0]["repr"]))
        )
        for i, patch in enumerate(strategy.top_patches):
            repr[i] = patch.pop("repr")
        torch.save(
            repr, raw_dataset_path / f"repr_{top_patches_fn}".replace(".json", ".pt")
        )

    save_json(strategy.top_patches, raw_dataset_path / top_patches_fn)

    loop_json = {"patches": query}
    save_loop(raw_dataset_path, loop_json, num_loop_files)

    #
    if not force:
        state.pred_tr = True
        state.query = True
        state.save_state()

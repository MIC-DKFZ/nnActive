from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.logger import monitor
from nnactive.query_pool import query_pool
from nnactive.results.state import State


@register_subcommand("query_pool")
def main(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    verbose: bool = False,
    force: bool = False,
):
    """Run Query with trained models on the dataset pool.

    Args:
        config (ActiveConfig): Carries all revelant information of experiment
        runtime_config (RuntimeConfig, optional): carries n_gpus, processes etc.. Defaults to RuntimeConfig().
        continue_id (int | None, optional): _description_. Defaults to None.
        verbose (bool, optional): Disables progress bars and get more explicit print statements.. Defaults to False.
        force (bool, optional): Set this to force using this command without taking the state.json of the dataset into account. Defaults to False.
    """
    config.set_nnunet_env()

    print(f"{continue_id=}")
    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id, verify=not force)

    with monitor.active_run(config=config.to_dict()):
        query_pool(
            config, runtime_config, state.dataset_id, force=force, verbose=verbose
        )

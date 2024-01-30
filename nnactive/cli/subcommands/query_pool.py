from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.query_pool import query_pool


@register_subcommand(
    "query_pool",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (
            ("-u", "--uncertainty_type"),
            {"type": str, "default": None},
        ),  # default="pred_entropy")
        (
            ("-n", "--num_patches"),
            {
                "type": int,
                "default": None,
                "help": "Number of Patches to be queried. Currently not implemented",
            },
        ),
        (
            ("-s", "--patch_size"),
            {
                "type": int,
                "default": None,
                "help": "Patch Size to be queried. Currently not implemented",
            },
        ),
        (
            ("--force"),
            {
                "action": "store_true",
                "required": False,
                "help": "Set this to force using this command without taking the state.json of the dataset into account! "
                "Be Careufl with this one!",
            },
        ),
        (
            ("--verbose"),
            {
                "action": "store_true",
                "help": "Disables progress bars and get more explicit print statements.",
            },
        ),
    ],
)
def main(args: Namespace):
    dataset_id = args.dataset_id
    patch_size = args.patch_size
    query_size = args.num_patches
    force = args.force
    verbose = args.verbose

    config = ActiveConfig.get_from_id(dataset_id)
    if patch_size is not None:
        config.patch_size = patch_size
    if query_size is not None:
        config.query_size = query_size
    query_pool(dataset_id, force=force, verbose=verbose)

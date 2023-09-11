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
    ],
)
def main(args: Namespace):
    dataset_id = args.dataset_id
    patch_size = args.patch_size
    query_size = args.num_patches

    config = ActiveConfig.get_from_id(dataset_id)
    if patch_size is not None:
        config.patch_size = patch_size
    if query_size is not None:
        config.query_size = query_size
    query_pool(dataset_id)

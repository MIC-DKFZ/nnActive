from nnactive.cli.registry import register_subcommand
from nnactive.experiments import list_experiments


@register_subcommand("list_experiments")
def main():
    for exp in sorted(list_experiments()):
        print(exp)

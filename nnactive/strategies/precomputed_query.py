from nnactive.data import Patch
from nnactive.loops.loading import get_loop_patches
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.registry import register_strategy


@register_strategy("precomputed-queries")
class PrecomputedQuery(AbstractQueryMethod):
    def query(self, verbose=False, n_gpus: int = 0) -> list[Patch]:
        """Loads patches from loop_XXX.json file in `self.nnunet_raw_folder_path` folder"""
        return get_loop_patches(
            data_path=get_raw_path(self.dataset_id) / "PrecomputedLoops",
            loop_val=self.loop_val + 1,
        )

import pytest

from nnactive.data import Patch
from nnactive.loops.loading import (
    get_patches_from_loop_files,
    get_sorted_loop_files,
    save_loop,
)


def test_loop(tmpdir):
    patch_per_loop = 5
    num_loops = 4
    nested_patches = [
        [Patch(f"file_{i}", [i, i, i], size=[1, 1, 1]) for i in range(patch_per_loop)]
        for k in range(num_loops)
    ]
    nested_loops = [{"patches": loop_patches} for loop_patches in nested_patches]
    for i, loop_json in enumerate(nested_loops):
        save_loop(tmpdir, loop_json, loop_val=i)
        assert len(get_sorted_loop_files(tmpdir)) == i + 1
        assert len(get_patches_from_loop_files(tmpdir)) == (i + 1) * patch_per_loop

    load_patches = get_patches_from_loop_files(tmpdir)
    unnested_patches = [p for lp in nested_patches for p in lp]
    for p_l, p_gt in zip(load_patches, unnested_patches):
        assert p_l.file == p_gt.file

from time import sleep

import numpy as np
import torch
from tqdm import tqdm

from nnactive.aggregations.convolution import ConvolveAggTorch
from nnactive.utils.torchutils import get_tensor_memory_usage

sizes = [[2**i] * 3 for i in range(11)] + [[512, 512, 512]]  # [[1024, 1024, 2048]]
sizes = [[1024, 1024, 1024]]
total_memory = torch.cuda.get_device_properties(0).total_memory / (
    1024**3
)  # Convert to gigabytes
print(f"Total GPU Memory: {total_memory:.4f} GB")

# patch_sizes = [[64, 64, 64], [128, 128, 128]]

# device = torch.device("cuda:0")
# x = torch.rand([1839, 622, 622], device=device) ** 2
# x_size = get_tensor_memory_usage(x)
# print(x_size)
# torch.cuda.synchronize()
# torch.cuda.empty_cache()

# for i in tqdm(range(1000)):
#     sleep(0.1)
# x = torch.rand([1, 1839, 622, 622], device=device)
# x = -(x * torch.log(x)).sum(dim=0)
# x.cpu()

# x = x.unsqueeze(0).unsqueeze(0)
# for patch_size in patch_sizes:
#     print(patch_size)
#     kernel = torch.ones(size=[1, 1] + patch_size, device=device) / np.prod(patch_size)
#     for i in tqdm(range(100)):
#         # sleep(0.5)
#         with torch.no_grad():
#             # agg = ConvolveAggTorch(patch_size, stride=8)
#             # out = agg.forward(x)
#             out = torch.nn.functional.conv3d(
#                 input=x,
#                 weight=kernel,
#                 stride=8,
#             )
#         if i == 0:
#             print(out.shape)
#         f = (out * 2).cpu()


for size in sizes:
    for i in range(1000):
        print(f"Size: {size}")
        x = torch.rand(size, device="cuda:0")
        x_size = get_tensor_memory_usage(x)
        print(f"\tSize of Tensor: {x_size}GB")
        x *= torch.log(x)
        x = x.nan_to_num()
        # logs = torch.log(x)
        # logs_size = get_tensor_memory_usage(logs)
        # print(f"\tSize of Logs: {logs_size}GB")

        # mask = logs == -torch.inf
        # mask_size = x_size = get_tensor_memory_usage(mask)
        # print(f"\tSize of Mask: {mask_size}GB")

        # total_size = x_size + logs_size + mask_size
        # print(f"Total Size: {total_size}GB")
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        memory_cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"\tMemory allocated: {memory_allocated}")
        print(f"\tMax Memory allocated: {max_memory_allocated}")
        print(f"\tMemory cached: {memory_cached}")
        print(f"\tMemory free: {total_memory-memory_allocated}")
        # deleting values here is crucial to get the biggest size working. The overwrite does not work as smooth as expected
        del x
        # del logs
        # del mask
        torch.cuda.reset_peak_memory_stats()

    # x = torch.rand(size, device="cuda:0") ** 2
    # for patch_size in patch_sizes:
    #     agg = ConvolveAggTorch(patch_size, stride=8)

    # also works without empty cache IF values are deleted before!
    # torch.cuda.empty_cache()

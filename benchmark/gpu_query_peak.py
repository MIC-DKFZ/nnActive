import torch

from nnactive.utils.torchutils import get_tensor_memory_usage

sizes = [[2**i] * 3 for i in range(11)] + [[512, 512, 512]]  # [[1024, 1024, 2048]]
total_memory = torch.cuda.get_device_properties(0).total_memory / (
    1024**3
)  # Convert to gigabytes
print(f"Total GPU Memory: {total_memory:.4f} GB")

for size in sizes:
    print(f"Size: {size}")
    x = torch.rand(size, device="cuda:0") ** 2
    x_size = get_tensor_memory_usage(x)
    print(f"\tSize of Tensor: {x_size}GB")

    logs = torch.log(x)
    logs_size = get_tensor_memory_usage(logs)
    print(f"\tSize of Logs: {logs_size}GB")

    mask = logs == -torch.inf
    mask_size = x_size = get_tensor_memory_usage(mask)
    print(f"\tSize of Mask: {mask_size}GB")

    total_size = x_size + logs_size + mask_size
    print(f"Total Size: {total_size}GB")
    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    memory_cached = torch.cuda.memory_reserved() / (1024**3)
    print(f"\tMemory allocated: {memory_allocated}")
    print(f"\tMax Memory allocated: {max_memory_allocated}")
    print(f"\tMemory cached: {memory_cached}")
    print(f"\tMemory free: {total_memory-memory_allocated}")
    # deleting values here is crucial to get the biggest size working. The overwrite does not work as smooth as expected
    del x
    del logs
    del mask
    torch.cuda.reset_peak_memory_stats()
    # also works without empty cache IF values are deleted before!
    # torch.cuda.empty_cache()

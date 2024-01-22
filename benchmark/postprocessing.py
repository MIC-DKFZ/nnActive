import torch
import torch.nn.functional as F

from nnactive.utils.timer import CudaTimer


def get_gpu_memory_usage(obj):
    # Use torch.cuda.memory_allocated to get the amount of memory
    # currently allocated by PyTorch on the GPU
    memory_allocated = torch.cuda.memory_allocated(obj.device)

    # Convert bytes to gigabytes
    memory_gb = memory_allocated / (1024**3)

    return memory_gb


device = torch.device("cuda:0")
if __name__ == "__main__":
    for i in range(100):
        torch.randn(100 * 100, device=device).view(100, 100) @ torch.randn(
            100 * 100, device=device
        ).view(100, 100)

    image_logs = torch.randn(4 * 500**3, device=device).view(4, 500, 500, 500)
    print(f"Memory of Image: {get_gpu_memory_usage(image_logs)}GB")
    timer = CudaTimer()
    timer.start()
    image_probs = F.softmax(image_logs, dim=0)
    image_probs.cpu()
    print(timer.stop() / 1000)

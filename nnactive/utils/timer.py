import time

import torch


def timeit(func, *args, **kwargs):
    """
    Wrapper function to measure the execution time of another function.

    Parameters:
    - func: The function to be executed and timed.
    - *args: Positional arguments to be passed to the function.
    - **kwargs: Keyword arguments to be passed to the function.

    Returns:
    - result: The result of the wrapped function.
    - elapsed_time: The time taken for the function to execute.
    """
    torch.cuda.synchronize()
    start_time = time.time()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time


class CudaTimer:
    def __init__(self):
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.begin.record()

    def stop(self):
        torch.cuda.synchronize()
        self.end.record()
        torch.cuda.synchronize()
        return self.begin.elapsed_time(self.end)


class Timer:
    """Returns the time elapsed between start and stop in seconds."""

    def __init__(self):
        self.start_time = None
        self.measured = []

    def start(self):
        self.start_time = time.time()

    def stop(self) -> float:
        """Returns time between start and stop in seconds."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.measured.append(elapsed_time)
        del self.start_time
        return elapsed_time

    def average(self) -> float | None:
        """Returns the average time between start and stop in seconds."""
        if len(self.measured) == 0:
            return None
        return sum(self.measured) / len(self.measured)

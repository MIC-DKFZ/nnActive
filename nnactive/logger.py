from __future__ import annotations

from contextlib import contextmanager
from time import time
from typing import Any, Generator, Iterable, TypeVar

from loguru import logger
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from wandb.errors import CommError

import wandb

ItemT = TypeVar("ItemT")


class _nnActiveLogger(nnUNetLogger):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def log(self, key, value, epoch: int) -> None:
        super().log(key, value, epoch)
        wandb.log({key: value, "epoch": epoch})


class nnActiveMonitor:
    def __init__(self) -> None:
        super().__init__()

        self._wandb_active = False
        self._defined_metrics = set()

    def assert_wandb_active(self) -> None:
        if not self._wandb_active:
            raise ValueError("wandb is not active")

    @contextmanager
    def active_run(
        self,
        project: str = "nnactive",
        name: str | None = None,
        config: dict | None = None,
    ) -> Generator[None, None, None]:
        if self._wandb_active:
            wandb.finish()

        try:
            wandb.init(project=project, name=name, config=config)
        except CommError:
            wandb.init(project=project, name=name, config=config, mode="offline")

        self._wandb_active = True

        def _inner():
            yield
            wandb.finish()
            self._wandb_active = False

        return _inner()

    @contextmanager
    def timer(self, name: str) -> Generator[None, None, None]:
        self.assert_wandb_active()

        if f"{name}_time" not in self._defined_metrics:
            wandb.define_metric(f"{name}_time", summary="mean")
            self._defined_metrics.add(f"{name}_time")

        start = time()
        yield
        runtime = time() - start
        wandb.log({f"{name}_time": runtime})
        logger.info(f"{name} time: {runtime}s")

    def itertimer(
        self, iterable: Iterable[ItemT], name: str | None = None
    ) -> Iterable[ItemT]:
        self.assert_wandb_active()

        if f"{name}_time" not in self._defined_metrics:
            wandb.define_metric(f"{name}_time", summary="mean")
            self._defined_metrics.add(f"{name}_time")

        times = []
        times.append(time())

        for item in iterable:
            yield item
            times.append(time())
            wandb.log({f"{name}_time": times[-1] - times[-2]})

        avg = 0
        for i in range(1, len(times)):
            avg += times[i] - times[i - 1]
        avg /= len(times)
        logger.info(f"{name} average time per iteration: {avg}s")

    def get_logger(self) -> _nnActiveLogger:
        """Create a new nnUNet logger with wandb already initialized."""
        self.assert_wandb_active()
        return _nnActiveLogger()

    def log(self, key, value, epoch: int) -> None:
        self.assert_wandb_active()
        wandb.log({key: value, "epoch": epoch})


monitor = nnActiveMonitor()

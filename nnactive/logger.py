from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from time import time
from typing import Any, Generator, Iterable, TypeVar

from loguru import logger
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from wandb.errors import CommError

import wandb
from nnactive.results.state import State

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

    def is_active(self) -> bool:
        return self._wandb_active

    @contextmanager
    def active_run(
        self,
        project: str = "nnactive",
        name: str | None = None,
        config: dict | None = None,
        force: bool = False,
        group: str | None = None,
        state: State | None = None,
        state_tag: str | None = None,
    ) -> Generator[None, None, None]:
        if self._wandb_active:
            if force:
                wandb.finish()
            else:
                logger.warning(
                    "wandb is already active, set `force` to replace active run"
                )

                def _inner():
                    yield wandb.run

                return _inner()

        try:
            run = wandb.init(project=project, name=name, config=config, group=group)
        except CommError:
            run = wandb.init(project=project, name=name, config=config, mode="offline")

        self._wandb_active = True
        if state is not None:
            # Save info about the current wandb run to the list of associated runs
            run_info = dict(
                id=run.id,
                name=run.name,
                url=run.url,
                start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            if state_tag is not None:
                run_info["tag"] = state_tag
            state.wandb_runs.append(run_info)
            state.save_state()

        try:
            yield run
        finally:
            wandb.finish()
            self._wandb_active = False

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

    def log(self, key: str, value: Any, epoch: int) -> None:
        self.assert_wandb_active()
        wandb.log({key: value, "epoch": epoch})

    def write_metric(self, value: Any, name: str, epoch: int | None = None) -> None:
        self.assert_wandb_active()
        if f"{name}" not in self._defined_metrics:
            wandb.define_metric(f"{name}", summary="mean")
            self._defined_metrics.add(f"{name}")
        log_dict = {name: value}
        if epoch is not None:
            log_dict["epoch"] = epoch
        wandb.log(log_dict)
        logger.info(f"{name}: {value}s")


monitor = nnActiveMonitor()

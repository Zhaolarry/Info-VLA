from pathlib import Path

import numpy as np

from openpi.agents.base import BaseAgent
from openpi.agents.data_consistency import DataConsistencyChecker
from openpi.agents.log import Logger

class ERAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.consistency_checker = None
        self.logger = None
        self._epoch_losses: list[float] = []
        self._epoch_index = 0

    def on_train_begin(
        self,
        config,
        repo_id: str,
        device,
        loader,
        local_rank: int,
        is_main: bool,
        continue_same_task: bool,
        global_step: int,
    ) -> None:
        del config, repo_id, device, continue_same_task
        world_size = 1
        if self.stat[0]:
            import torch

            world_size = torch.distributed.get_world_size()
        self.consistency_checker = DataConsistencyChecker(local_rank, world_size)
        self._epoch_losses = []
        self._epoch_index = global_step // max(1, len(loader))
        if is_main:
            log_dir = Path(__file__).resolve().parent / "logs"
            self.logger = Logger(log_dir=log_dir, name="ER", rank=local_rank)

    def on_step_end(self, config, repo_id: str, device, global_step: int, loss, observation, actions, loader) -> None:
        del config, repo_id, device, observation, actions
        self._epoch_losses.append(float(loss.item()))
        epoch_steps = max(1, len(loader))
        if self.logger is not None and global_step % epoch_steps == 0 and self._epoch_losses:
            self._epoch_index += 1
            variance = float(np.var(self._epoch_losses, ddof=1)) if len(self._epoch_losses) > 1 else 0.0
            self.logger.write(
                f"epoch: {self._epoch_index} loss: {float(np.mean(self._epoch_losses))} variance: {variance}"
            )
            self._epoch_losses = []

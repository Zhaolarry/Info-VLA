from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


class DataConsistencyChecker:
    def __init__(self, local_rank: int = 0, world_size: int = 1):
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_ddp = torch.distributed.is_initialized() and world_size > 1

    def compute_data_hash(self, data: Any) -> str:
        if isinstance(data, torch.Tensor):
            data_bytes = data.detach().cpu().numpy().tobytes()
        elif isinstance(data, (list, tuple)):
            data_bytes = str([self.compute_data_hash(item) for item in data]).encode("utf-8")
        else:
            data_bytes = str(data).encode("utf-8")
        return hashlib.md5(data_bytes).hexdigest()

    def verify_data_distribution(self, data_loader, expected_samples_per_gpu: int) -> dict[str, Any]:
        if not self.use_ddp:
            return {"status": "single_gpu", "samples": len(data_loader)}

        local_samples = len(data_loader)
        local_tensor = torch.tensor([local_samples], dtype=torch.int32, device=f"cuda:{self.local_rank}")
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        total_samples = local_tensor.item()
        expected_total = expected_samples_per_gpu * self.world_size
        is_correct = total_samples == expected_total

        if not is_correct:
            logging.error("Data distribution mismatch: expected %s, got %s", expected_total, total_samples)

        return {
            "status": "verified" if is_correct else "error",
            "local_samples": local_samples,
            "total_samples": total_samples,
            "expected_total": expected_total,
            "world_size": self.world_size,
            "is_correct": is_correct,
        }

    def sync_random_state(self, rng_state: np.random.RandomState) -> np.random.RandomState:
        if not self.use_ddp:
            return rng_state

        state_dict = rng_state.get_state()
        if self.local_rank == 0:
            dist.broadcast_object_list([state_dict], src=0)
        else:
            received_state = [None]
            dist.broadcast_object_list(received_state, src=0)
            state_dict = received_state[0]

        synced_rng = np.random.RandomState()
        synced_rng.set_state(state_dict)
        return synced_rng


def create_consistency_checker(local_rank: int = 0, world_size: int = 1) -> DataConsistencyChecker:
    return DataConsistencyChecker(local_rank, world_size)

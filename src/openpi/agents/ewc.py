from __future__ import annotations

import logging

import jax
import torch
import tqdm

from openpi.agents.base import BaseAgent


class EWCAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.e_lambda = 50_000.0
        self.fisher: dict[str, torch.Tensor] | None = None
        self.prev_params: dict[str, torch.Tensor] | None = None

    def _save_prev_params(self):
        self.prev_params = {
            name: param.detach().clone().cpu()
            for name, param in self.unwrap_model().named_parameters()
            if param.requires_grad
        }

    def _zero_fisher(self):
        self.fisher = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self.unwrap_model().named_parameters()
            if param.requires_grad
        }

    def ewc_penalty(self, device):
        if self.prev_params is None or self.fisher is None:
            return torch.zeros((), device=device)

        penalty = torch.zeros((), device=device)
        for name, param in self.unwrap_model().named_parameters():
            if not param.requires_grad:
                continue
            penalty = penalty + (self.fisher[name].to(device) * (param - self.prev_params[name].to(device)).pow(2)).sum()
        return penalty

    def compute_loss(self, observation, actions, device) -> torch.Tensor:
        loss = super().compute_loss(observation, actions, device)
        if self._active_task_index > 0 and self.prev_params is not None and self.fisher is not None:
            loss = loss + self.e_lambda * self.ewc_penalty(device)
        return loss

    def on_train_end(self, config, repo_id: str, device):
        self._estimate_fisher(config, repo_id, device)

    def _estimate_fisher(self, config, repo_id: str, device):
        self._zero_fisher()
        loader = self.build_data_loader(config, repo_id=repo_id, framework="pytorch", shuffle=True)

        params = [(name, param) for name, param in self.unwrap_model().named_parameters() if param.requires_grad]
        if not params:
            return

        self.model.train()
        steps = 0
        for observation, actions in tqdm.tqdm(loader, desc="Estimate Fisher", disable=not self.stat[3]):
            if steps >= 50:
                break
            steps += 1

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)
            loss = super().compute_loss(observation, actions, device)
            grads = torch.autograd.grad(
                loss,
                [param for _, param in params],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            for (name, _), grad in zip(params, grads, strict=True):
                if grad is not None:
                    self.fisher[name] += grad.detach().cpu().pow(2)

        if steps == 0:
            logging.warning("Skipping Fisher estimation for %s because no batches were available", repo_id)
            return

        for name in self.fisher:
            self.fisher[name] /= steps
        self._save_prev_params()

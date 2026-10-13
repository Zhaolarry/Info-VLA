from __future__ import annotations

import copy
import logging

import safetensors.torch
import torch
import torch.nn.functional as F

from openpi.agents.base import BaseAgent


class LwfAgent(BaseAgent):
    """Learning without Forgetting (LwF) continual-learning agent.

    On task boundaries the agent snapshots the previous-task model, freezes it,
    and adds a distillation loss that encourages the current model to produce
    similar action-expert representations on the *new* task's data.  This keeps
    old knowledge alive without replaying old data.

    Key parameters
    --------------
    lwf_lambda : float
        Weight of the distillation term (default 1.0).  Higher values preserve
        previous tasks more aggressively.
    lwf_temperature : float
        Temperature for softening the representation before computing MSE
        (default 2.0).  Larger values produce softer distillation targets.
    """

    def __init__(self, config):
        super().__init__(config)
        self.lwf_lambda = 1.0
        self.lwf_temperature = 2.0
        self.model_old = None

    # ------------------------------------------------------------------
    # Old-model management
    # ------------------------------------------------------------------

    def _prepare_old_model(self, device):
        """Load the previous-task checkpoint into a frozen teacher model."""
        if self.previous_task_checkpoint_dir is None or not self._has_checkpoints(
            self.previous_task_checkpoint_dir
        ):
            self.model_old = None
            logging.warning(
                "LwF: no previous checkpoint found at %s – skipping distillation.",
                self.previous_task_checkpoint_dir,
            )
            return

        teacher = copy.deepcopy(self.unwrap_model())
        latest_step = self.get_latest_checkpoint_step(self.previous_task_checkpoint_dir)
        ckpt_dir = self.previous_task_checkpoint_dir / str(latest_step)
        safetensors.torch.load_model(teacher, ckpt_dir / "model.safetensors", device=str(device))
        teacher.to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.model_old = teacher
        logging.info(
            "LwF: loaded frozen teacher from %s (step=%s)", ckpt_dir, latest_step
        )

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

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
        del config, repo_id, loader, local_rank, is_main, global_step
        if not continue_same_task and self._active_task_index > 0:
            self._prepare_old_model(device)
        else:
            self.model_old = None

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def compute_loss(self, observation, actions, device) -> torch.Tensor:
        """Task loss + LwF distillation loss (when a teacher is active)."""
        # -- 1. Sample noise / time once so both models see the same inputs --
        raw_model = self.unwrap_model()
        noise = raw_model.sample_noise(actions.shape, device)
        time = raw_model.sample_time(actions.shape[0], device)

        # -- 2. Current model forward (task loss) --
        suffix_out, losses, lang_emb, prefix_out = self.model(
            observation, actions, noise=noise, time=time, dill=True
        )
        # Normalise losses (follows base class convention)
        if isinstance(losses, (list, tuple)):
            losses = torch.stack(
                [
                    (
                        loss
                        if isinstance(loss, torch.Tensor)
                        else torch.as_tensor(loss, device=device, dtype=torch.float32)
                    )
                    for loss in losses
                ]
            )
        elif not isinstance(losses, torch.Tensor):
            losses = torch.as_tensor(losses, device=device, dtype=torch.float32)
        task_loss = losses.mean()

        # -- 3. Early exit when no teacher --
        if self.model_old is None or self._active_task_index == 0:
            return task_loss

        # -- 4. Old model forward (no grad) --
        with torch.no_grad():
            old_suffix_out, _, _, _ = self.model_old(
                observation, actions, noise=noise, time=time, dill=True
            )

        # -- 5. Distillation loss (MSE in representation space) --
        # Apply temperature scaling for softer targets
        student_rep = suffix_out / self.lwf_temperature
        teacher_rep = old_suffix_out / self.lwf_temperature

        distill_loss = F.mse_loss(
            student_rep.to(torch.float32), teacher_rep.to(torch.float32)
        )

        total_loss = task_loss + self.lwf_lambda * distill_loss

        # Log periodically (only on main process via a simple counter)
        if not hasattr(self, "_lwf_step_counter"):
            self._lwf_step_counter = 0
        self._lwf_step_counter += 1
        if self._lwf_step_counter % 100 == 1 and self.stat[3]:
            logging.info(
                "LwF step=%d task_loss=%.4f distill_loss=%.4f total=%.4f lambda=%.2f T=%.1f",
                self._lwf_step_counter,
                task_loss.item(),
                distill_loss.item(),
                total_loss.item(),
                self.lwf_lambda,
                self.lwf_temperature,
            )

        return total_loss

    # ------------------------------------------------------------------
    # Checkpoint state (LwF has no extra state beyond the model itself)
    # ------------------------------------------------------------------

    def get_extra_checkpoint_state(self) -> dict:
        return {}

    def load_extra_checkpoint_state(self, state: dict, device) -> None:
        del state, device

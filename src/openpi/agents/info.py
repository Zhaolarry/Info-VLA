from __future__ import annotations

import copy
from itertools import chain
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn

from openpi.agents.base import BaseAgent
from openpi.agents.data_consistency import DataConsistencyChecker
from openpi.agents.log import Logger


class InfoAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        model = self.unwrap_model()
        expert_width = model.action_out_proj.in_features
        language_width = model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.shape[1]

        self.temperature = 0.07
        self.model_old = None
        self.consistency_checker = None
        self.logger = None
        self._epoch_losses: list[float] = []
        self._epoch_index = 0

        self.U = torch.empty((0, 0), dtype=torch.long)
        self.current_task_prompts = torch.empty((0, 0), dtype=torch.long)

        self.lang_proj = nn.Linear(language_width, expert_width, bias=False).to(torch.bfloat16)
        self.mlp_head = nn.Sequential(
            nn.Linear(expert_width * 2, expert_width * 4),
            nn.ReLU(inplace=True),
            nn.Linear(expert_width * 4, expert_width * 2),
        )

    def trainable_parameters(self):
        return list(chain(self.model.parameters(), self.lang_proj.parameters(), self.mlp_head.parameters()))

    def create_optimizer(self, config):
        return torch.optim.AdamW(
            self.trainable_parameters(),
            lr=config.lr_schedule.peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

    def get_extra_checkpoint_state(self) -> dict:
        return {
            "U": self.U.cpu(),
            "current_task_prompts": self.current_task_prompts.cpu(),
            "lang_proj": self.lang_proj.state_dict(),
            "mlp_head": self.mlp_head.state_dict(),
        }

    def load_extra_checkpoint_state(self, state: dict, device) -> None:
        if "U" in state:
            self.U = state["U"].cpu()
        if "current_task_prompts" in state:
            self.current_task_prompts = state["current_task_prompts"].cpu()
        if "lang_proj" in state:
            self.lang_proj.load_state_dict(state["lang_proj"])
        if "mlp_head" in state:
            self.mlp_head.load_state_dict(state["mlp_head"])
        self.lang_proj = self.lang_proj.to(device)
        self.mlp_head = self.mlp_head.to(device)

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
        del config, repo_id
        world_size = 1
        if self.stat[0]:
            world_size = torch.distributed.get_world_size()
        self.consistency_checker = DataConsistencyChecker(local_rank, world_size)
        self.lang_proj = self.lang_proj.to(device)
        self.mlp_head = self.mlp_head.to(device)
        self._epoch_losses = []
        self._epoch_index = global_step // max(1, len(loader))

        if not continue_same_task and self.current_task_prompts.numel() > 0:
            self.U = self._merge_prompt_bank(self.U, self.current_task_prompts)
            self.current_task_prompts = torch.empty((0, self.U.shape[1]), dtype=torch.long)

        if self._active_task_index > 0:
            self.prepare_for_task(device)
        else:
            self.model_old = None

        if is_main:
            log_dir = Path(__file__).resolve().parent / "logs"
            self.logger = Logger(log_dir=log_dir, name="INFO", rank=local_rank)

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

    def on_train_end(self, config, repo_id: str, device):
        del config, repo_id, device
        self.U = self._merge_prompt_bank(self.U, self.current_task_prompts)
        self.current_task_prompts = torch.empty((0, self.U.shape[1] if self.U.ndim == 2 else 0), dtype=torch.long)

    def _merge_prompt_bank(self, bank_a: torch.Tensor, bank_b: torch.Tensor) -> torch.Tensor:
        if bank_a.numel() == 0:
            return bank_b.detach().cpu().clone()
        if bank_b.numel() == 0:
            return bank_a.detach().cpu().clone()
        return torch.unique(torch.cat([bank_a.detach().cpu(), bank_b.detach().cpu()], dim=0), dim=0, sorted=False)

    def _extract_prompt_tokens(self, observation) -> torch.Tensor:
        prompts = observation.to_dict()["tokenized_prompt"][:, :12].detach()
        if prompts.ndim == 1:
            prompts = prompts.unsqueeze(0)
        return prompts

    def _update_prompt_bank(self, observation) -> None:
        prompts = self._extract_prompt_tokens(observation).cpu()
        self.current_task_prompts = self._merge_prompt_bank(self.current_task_prompts, prompts)

    def _compute_prompt_mask(self, observation, device):
        if self.U.numel() == 0:
            return None
        prompts = self._extract_prompt_tokens(observation).to(device)
        memory_prompts = self.U.to(device)
        return (memory_prompts.unsqueeze(1) == prompts.unsqueeze(0)).all(-1).any(0)

    def prepare_for_task(self, device):
        if self.previous_task_checkpoint_dir is None or not self._has_checkpoints(self.previous_task_checkpoint_dir):
            self.model_old = None
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

    def _normalize_losses(self, losses, device):
        if isinstance(losses, (list, tuple)):
            losses = torch.stack(
                [
                    loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=device, dtype=torch.float32)
                    for loss in losses
                ]
            )
        elif not isinstance(losses, torch.Tensor):
            losses = torch.as_tensor(losses, device=device, dtype=torch.float32)
        return losses

    def _pool_vision_prefix(self, prefix_out: torch.Tensor, observation) -> torch.Tensor:
        num_cameras = len(observation.images) if hasattr(observation, "images") else 3
        language_tokens = observation.tokenized_prompt.shape[1] if hasattr(observation, "tokenized_prompt") else 0
        total_vision_tokens = max(0, prefix_out.shape[1] - language_tokens)
        if total_vision_tokens == 0:
            return F.normalize(prefix_out.mean(dim=1).to(torch.float32), dim=-1)

        tokens_per_camera = max(1, total_vision_tokens // max(1, num_cameras))
        vision = prefix_out[:, : tokens_per_camera * num_cameras, :]
        if num_cameras > 1 and vision.shape[1] >= num_cameras:
            vision = vision.view(vision.shape[0], num_cameras, tokens_per_camera, vision.shape[-1]).max(dim=1).values
        return F.normalize(vision.mean(dim=1).to(torch.float32), dim=-1)

    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1).mean(dim=0)
        probs = probs.clamp_min(torch.finfo(probs.dtype).eps)
        return -(probs * probs.log()).sum()

    def _scaled_auxiliary(self, primary_loss: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(aux_loss):
            return torch.zeros_like(primary_loss)
        ratio = (primary_loss.detach() / aux_loss.detach().clamp_min(1e-6)).clamp(max=10.0)
        return aux_loss * ratio

    def _combine_distill_losses(
        self,
        primary_loss: torch.Tensor,
        distill_lambda: float,
        loss_ce: torch.Tensor,
        dl_loss: torch.Tensor,
        en_loss: torch.Tensor,
    ) -> torch.Tensor:
        if distill_lambda <= 0.0:
            return primary_loss

        scaled_ce = self._scaled_auxiliary(primary_loss, loss_ce)
        scaled_dl = self._scaled_auxiliary(primary_loss, dl_loss)
        scaled_en = self._scaled_auxiliary(primary_loss, en_loss)

        mix = {
            0.02: scaled_ce,
            0.04: scaled_ce + scaled_dl,
            0.08: scaled_dl + scaled_en,
            0.10: scaled_ce + scaled_dl + scaled_en,
        }.get(round(distill_lambda, 2), scaled_ce + scaled_dl + scaled_en)

        scale = distill_lambda if round(distill_lambda, 2) == 0.10 else 0.1
        return primary_loss + mix * scale

    def compute_loss(self, observation, actions, device) -> torch.Tensor:
        self._update_prompt_bank(observation)
        mask = self._compute_prompt_mask(observation, device) if self._active_task_index > 0 else None
        distill_lambda = self.delta_t if self._active_task_index > 0 else 0.0
        return self.forward_with_distill(observation, actions, distill_lambda=distill_lambda, mask=mask)

    def forward_with_distill(self, obs, actions, distill_lambda=0.0, mask=None, output_attentions=False):
        result = self.model(obs, actions, dill=True, output_attentions=output_attentions)
        if output_attentions and isinstance(result, tuple) and len(result) == 5:
            output, losses, lang_emb_tokens, prefix_out, attentions = result
        else:
            output, losses, lang_emb_tokens, prefix_out = result
            attentions = None

        device = actions.device
        losses = self._normalize_losses(losses, device)
        loss = losses.mean()

        if self.model_old is None or distill_lambda <= 0.0 or mask is None or not torch.any(mask):
            if output_attentions and attentions is not None:
                return loss, attentions
            return loss

        mask = mask.bool()
        z_s = F.normalize(output.mean(dim=1).to(torch.float32), dim=-1)
        lang_emb = self.lang_proj(lang_emb_tokens.mean(dim=1).to(self.lang_proj.weight.dtype))
        lang_emb = F.normalize(lang_emb.to(torch.float32), dim=-1)
        pooled_prefix = self._pool_vision_prefix(prefix_out, obs)

        with torch.no_grad():
            old_output, _, old_lang_emb_tokens, prefix_out_old = self.model_old(obs, actions, dill=True)
            z_t = F.normalize(old_output.mean(dim=1).to(torch.float32), dim=-1)
            lang_emb_old = self.lang_proj(old_lang_emb_tokens.mean(dim=1).to(self.lang_proj.weight.dtype))
            lang_emb_old = F.normalize(lang_emb_old.to(torch.float32), dim=-1)
            pooled_prefix_old = self._pool_vision_prefix(prefix_out_old, obs)

        student_masked = pooled_prefix[mask]
        teacher_masked = pooled_prefix_old[mask]
        if teacher_masked.shape[0] == 0:
            if output_attentions and attentions is not None:
                return loss, attentions
            return loss

        logits_ce = teacher_masked @ student_masked.T / self.temperature
        labels = torch.arange(teacher_masked.shape[0], device=device)
        loss_ce = F.cross_entropy(logits_ce, labels)

        logits_s = z_s @ lang_emb.T / self.temperature
        logits_t = z_t @ lang_emb_old.T / self.temperature

        fused = self.mlp_head(torch.cat((z_s, z_t), dim=1))
        fused = F.normalize(fused.to(torch.float32), dim=-1)
        lang_pair = F.normalize(torch.cat((lang_emb, lang_emb_old), dim=1), dim=-1)
        logits_st = fused @ lang_pair.T / self.temperature

        dl_loss = F.l1_loss(torch.cat((logits_s, logits_t), dim=0), torch.cat((logits_st, logits_st), dim=0))
        en_loss = self._entropy_loss(logits_st)
        loss = self._combine_distill_losses(loss, distill_lambda, loss_ce, dl_loss, en_loss)

        if output_attentions and attentions is not None:
            return loss, attentions
        return loss


class CPCProjection(nn.Module):
    def __init__(self, dim, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)

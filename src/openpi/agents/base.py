from __future__ import annotations

import dataclasses
import gc
import logging
import os
import platform
import re
import shutil
import time
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import tqdm

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.action_safety as _action_safety
import openpi.training.config as _config
import openpi.training.data_loader as _data


class BaseAgent:
    """Stable continual-training agent aligned with scripts/train_pytorch.py."""

    def __init__(self, config: _config.TrainConfig):
        self.check = bool(getattr(config, "check", False))
        self.delta_t = float(getattr(config, "delta_t", 0.0))
        self.checkpoint_dir: Path | None = None
        self.preview_checkpoint_dir: Path | None = None
        self.previous_task_checkpoint_dir: Path | None = None
        self.checkpoint_final_dir: Path | None = None
        self.stop_step = 0
        self.optim: torch.optim.Optimizer | None = None
        self._active_repo_id: str | None = None
        self._active_task_index = 0
        ddp_state = self.setup_ddp()
        _, local_rank, _ = ddp_state
        self.set_seed(config.seed, local_rank)
        self.model, self.stat = self.build_models(config, ddp_state=ddp_state)

    def setup_ddp(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_ddp = world_size > 1
        if use_ddp and not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return use_ddp, local_rank, device

    def cleanup_ddp(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    def set_seed(self, seed: int, local_rank: int):
        torch.manual_seed(seed + local_rank)
        np.random.seed(seed + local_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + local_rank)

    def unwrap_model(self, model=None):
        model = self.model if model is None else model
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    def _task_index(self, repo_id: str) -> int:
        match = re.search(r"(\d+)$", repo_id)
        return int(match.group(1)) if match else 0

    def _replace_task_index(self, repo_id: str, task_idx: int) -> str:
        if re.search(r"\d+$", repo_id):
            return re.sub(r"\d+$", str(task_idx), repo_id)
        return repo_id

    def _checkpoint_root(self, config: _config.TrainConfig) -> Path:
        checkpoint_root = config.checkpoint_dir
        if self.delta_t != 0.0:
            checkpoint_root = checkpoint_root / str(self.delta_t)
        return checkpoint_root

    def _has_checkpoints(self, checkpoint_dir: Path | None) -> bool:
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return False
        return any(path.is_dir() and path.name.isdigit() for path in checkpoint_dir.iterdir())

    def _prepare_task_dirs(self, config: _config.TrainConfig, repo_id: str):
        task_idx = self._task_index(repo_id)
        checkpoint_root = self._checkpoint_root(config)

        self._active_repo_id = repo_id
        self._active_task_index = task_idx
        self.checkpoint_dir = checkpoint_root / repo_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if task_idx > 0:
            prev_repo_id = self._replace_task_index(repo_id, task_idx - 1)
            self.previous_task_checkpoint_dir = checkpoint_root / prev_repo_id
        else:
            self.previous_task_checkpoint_dir = None

        self.preview_checkpoint_dir = self.previous_task_checkpoint_dir
        self.checkpoint_final_dir = None

    def _resolve_resume_checkpoint_dir(self, resuming: bool) -> tuple[Path | None, bool]:
        if self._has_checkpoints(self.checkpoint_dir) and (self.check or resuming):
            return self.checkpoint_dir, True
        if self._active_task_index > 0 and self._has_checkpoints(self.previous_task_checkpoint_dir):
            return self.previous_task_checkpoint_dir, False
        return None, False

    def create_optimizer(self, config: _config.TrainConfig):
        trainable_params = self.trainable_parameters()
        return torch.optim.AdamW(
            trainable_params,
            lr=config.lr_schedule.peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

    def trainable_parameters(self):
        return list(self.model.parameters())

    def get_extra_checkpoint_state(self) -> dict:
        return {}

    def load_extra_checkpoint_state(self, state: dict, device) -> None:
        del state, device

    def on_train_begin(
        self,
        config: _config.TrainConfig,
        repo_id: str,
        device,
        loader,
        local_rank: int,
        is_main: bool,
        continue_same_task: bool,
        global_step: int,
    ) -> None:
        del config, repo_id, device, loader, local_rank, is_main, continue_same_task, global_step

    def on_step_end(
        self,
        config: _config.TrainConfig,
        repo_id: str,
        device,
        global_step: int,
        loss: torch.Tensor,
        observation,
        actions,
        loader,
    ) -> None:
        del config, repo_id, device, global_step, loss, observation, actions, loader

    def create_lr_schedule(self, config: _config.TrainConfig, max_steps: int | None = None):
        configured_warmup_steps = int(getattr(config.lr_schedule, "warmup_steps", 0))
        peak_lr = config.lr_schedule.peak_lr
        configured_decay_steps = int(config.lr_schedule.decay_steps)
        end_lr = config.lr_schedule.decay_lr
        if max_steps is None:
            warmup_steps = configured_warmup_steps
            decay_steps = configured_decay_steps
        else:
            warmup_ratio = float(os.environ.get("OPENPI_WARMUP_RATIO", "0.25"))
            warmup_cap = max(1, int(max_steps * warmup_ratio)) if configured_warmup_steps > 0 else 0
            warmup_steps = min(configured_warmup_steps, warmup_cap)
            decay_steps = min(configured_decay_steps, max_steps)

        def lr_schedule(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                init_lr = peak_lr / (warmup_steps + 1)
                return init_lr + (peak_lr - init_lr) * step / warmup_steps
            progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps)))
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return end_lr + (peak_lr - end_lr) * cosine

        return lr_schedule, warmup_steps, decay_steps

    def save_checkpoint(self, model, optimizer, global_step, config, max_steps, is_main, data_config):
        if not is_main or self.checkpoint_dir is None:
            return

        should_save = (global_step % config.save_interval == 0 and global_step > 0) or global_step >= max_steps
        if not should_save:
            return

        final_ckpt_dir = self.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = self.checkpoint_dir / f"tmp_{global_step}"
        self.checkpoint_final_dir = final_ckpt_dir

        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        safetensors.torch.save_model(self.unwrap_model(model), tmp_ckpt_dir / "model.safetensors")
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
        torch.save(
            {
                "global_step": global_step,
                "config": dataclasses.asdict(config),
                "timestamp": time.time(),
            },
            tmp_ckpt_dir / "metadata.pt",
        )
        extra_state = self.get_extra_checkpoint_state()
        if extra_state:
            torch.save(extra_state, tmp_ckpt_dir / "extra_state.pt")

        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)
        logging.info("Saved checkpoint at step %s -> %s", global_step, final_ckpt_dir)

    def load_checkpoint(self, model, optimizer, checkpoint_dir, device, *, load_optimizer: bool = True):
        checkpoint_steps = [
            int(path.name)
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and not path.name.startswith("tmp_")
        ]
        if not checkpoint_steps:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest_step = max(checkpoint_steps)
        ckpt_dir = checkpoint_dir / f"{latest_step}"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage(device, latest_step, "before_loading_checkpoint")

        safetensors_path = ckpt_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        safetensors.torch.load_model(self.unwrap_model(model), safetensors_path, device=str(device))

        optimizer_path = ckpt_dir / "optimizer.pt"
        if load_optimizer and optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except ValueError as exc:
                logging.warning(
                    "Failed to load optimizer state from %s: %s. Continuing with a freshly initialized optimizer.",
                    optimizer_path,
                    exc,
                )
        elif not load_optimizer:
            logging.info("Skipping optimizer state load from %s for model-only checkpoint initialization", ckpt_dir)

        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        extra_state_path = ckpt_dir / "extra_state.pt"
        if extra_state_path.exists():
            extra_state = torch.load(extra_state_path, map_location="cpu", weights_only=False)
            self.load_extra_checkpoint_state(extra_state, device)

        logging.info("Loaded checkpoint from %s (step=%s)", ckpt_dir, global_step)
        return global_step

    def get_latest_checkpoint_step(self, checkpoint_dir):
        checkpoint_steps = [
            int(path.name)
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and not path.name.startswith("tmp_")
        ]
        return max(checkpoint_steps) if checkpoint_steps else None

    def log_memory_usage(self, device, step, phase="unknown"):
        if not torch.cuda.is_available():
            return

        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        memory_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9
        memory_stats = torch.cuda.memory_stats(device)
        max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
        max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

        ddp_info = ""
        if dist.is_initialized():
            ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

        logging.info(
            "Step %s (%s): GPU memory - allocated: %.2fGB, reserved: %.2fGB, free: %.2fGB, peak_allocated: %.2fGB, peak_reserved: %.2fGB%s",
            step,
            phase,
            memory_allocated,
            memory_reserved,
            memory_free,
            max_memory_allocated,
            max_memory_reserved,
            ddp_info,
        )

    def build_models(self, config, ddp_state=None):
        if ddp_state is None:
            ddp_state = self.setup_ddp()
        use_ddp, local_rank, device = ddp_state
        is_main = (not use_ddp) or (dist.get_rank() == 0)
        world_size = torch.distributed.get_world_size() if use_ddp else 1

        if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
            model_cfg = openpi.models.pi0_config.Pi0Config(
                dtype=config.pytorch_training_precision,
                action_dim=config.model.action_dim,
                action_horizon=config.model.action_horizon,
                max_token_len=config.model.max_token_len,
                paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
                action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
                pi05=getattr(config.model, "pi05", False),
            )
        else:
            model_cfg = config.model
            object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

        if hasattr(model, "gradient_checkpointing_enable"):
            enable_gradient_checkpointing = True
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing for memory optimization")
        else:
            enable_gradient_checkpointing = False
            logging.info("Gradient checkpointing is not supported for this model")

        if is_main and torch.cuda.is_available():
            self.log_memory_usage(device, 0, "after_model_creation")

        if world_size >= 8:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                static_graph=world_size >= 8,
            )

        if config.pytorch_weight_path is not None:
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            safetensors.torch.load_model(self.unwrap_model(model), model_path, strict=False)
            logging.info("Loaded PyTorch weights from %s", config.pytorch_weight_path)

        return model, (use_ddp, local_rank, device, is_main, model_cfg, enable_gradient_checkpointing)

    def build_data_loader(self, config, repo_id, shuffle=True, framework="pytorch"):
        return _data.create_data_loader(config, framework=framework, shuffle=shuffle, repo_id=repo_id)

    def max_steps_for_task(self, config: _config.TrainConfig, task_idx: int) -> int:
        # max_steps = int(config.num_train_steps) if task_idx > 0 else int(config.num_train_steps) *
        return int(config.num_train_steps)

    def compute_loss(self, observation, actions, device) -> torch.Tensor:
        # print(observation)
        # print(actions[0])
        # print(actions[0][0])
        # import time
        # time.sleep(100)
        losses = self.model(observation, actions)
        if isinstance(losses, (list, tuple)):
            losses = torch.stack(
                [
                    loss if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=device, dtype=torch.float32)
                    for loss in losses
                ]
            )
        elif not isinstance(losses, torch.Tensor):
            losses = torch.as_tensor(losses, device=device, dtype=torch.float32)
        return losses.mean()

    def on_train_end(self, config: _config.TrainConfig, repo_id: str, device):
        """Hook for continual-learning agents."""

    def train(self, config, repo_id, resuming):
        self._prepare_task_dirs(config, repo_id)
        
        use_ddp, local_rank, device, is_main, model_cfg, enable_gradient_checkpointing = self.stat
        self.set_seed(config.seed, local_rank)

        world_size = torch.distributed.get_world_size() if use_ddp else 1
        effective_batch_size = config.batch_size // world_size

        data_loader = self.build_data_loader(config, repo_id=repo_id, framework="pytorch", shuffle=True)
        loader, data_config = data_loader, data_loader.data_config()
        _action_safety.log_safe_quantile_dims(
            data_config,
            is_main=is_main,
            prefix=f"Piper safe quantile({repo_id})",
        )

        self.optim = self.create_optimizer(config)
        max_steps = self.max_steps_for_task(config, self._active_task_index)
        global_step = 0

        checkpoint_dir_to_load, continue_same_task = self._resolve_resume_checkpoint_dir(resuming)
        if checkpoint_dir_to_load is not None:
            loaded_step = self.load_checkpoint(
                self.model,
                self.optim,
                checkpoint_dir_to_load,
                device,
                load_optimizer=continue_same_task,
            )
            if continue_same_task:
                global_step = loaded_step
            else:
                self.optim = self.create_optimizer(config)
                global_step = 0
                logging.info("Initialized %s from previous task checkpoint %s", repo_id, checkpoint_dir_to_load)

        self.on_train_begin(
            config,
            repo_id,
            device,
            loader,
            local_rank,
            is_main,
            continue_same_task,
            global_step,
        )

        lr_schedule, warmup_steps, decay_steps = self.create_lr_schedule(config, max_steps=max_steps)
        self.model.train()
        start_time = time.time()
        infos = []

        if is_main:
            logging.info(
                "Running on: %s | world_size=%s",
                platform.node(),
                torch.distributed.get_world_size() if use_ddp else 1,
            )
            logging.info(
                "Training config: batch_size=%s, effective_batch_size=%s, num_train_steps=%s",
                config.batch_size,
                effective_batch_size,
                max_steps,
            )
            logging.info("Memory optimizations: gradient_checkpointing=%s", enable_gradient_checkpointing)
            logging.info(
                "LR schedule: effective_warmup=%s, peak_lr=%.2e, effective_decay_steps=%s, end_lr=%.2e",
                warmup_steps,
                config.lr_schedule.peak_lr,
                decay_steps,
                config.lr_schedule.decay_lr,
            )
            logging.info(
                "Optimizer: %s, weight_decay=%s, clip_norm=%s",
                type(config.optimizer).__name__,
                config.optimizer.weight_decay,
                config.optimizer.clip_gradient_norm,
            )
            logging.info("Training precision: %s", model_cfg.dtype)

        pbar = (
            tqdm.tqdm(total=max_steps, initial=global_step, desc="Training", disable=not is_main)
            if is_main
            else None
        )

        while global_step < max_steps:
            if use_ddp and hasattr(loader, "set_epoch"):
                loader.set_epoch(global_step // max(1, len(loader)))

            for observation, actions in loader:
                if global_step >= max_steps:
                    break

                observation = jax.tree.map(lambda x: x.to(device), observation)
                actions = actions.to(torch.float32).to(device)
                actions = _action_safety.clamp_and_log_piper_actions(
                    actions,
                    config=config,
                    global_step=global_step,
                    is_main=is_main,
                )

                for param_group in self.optim.param_groups:
                    param_group["lr"] = lr_schedule(global_step)

                self.optim.zero_grad(set_to_none=True)
                loss = self.compute_loss(observation, actions, device)
                
                if not torch.isfinite(loss):
                    logging.warning("Non-finite loss at step %s for repo %s: %s", global_step, repo_id, loss.item())
                    continue

                loss.backward()

                if global_step < 5 and is_main and torch.cuda.is_available():
                    self.log_memory_usage(device, global_step, "after_backward")

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters(),
                    max_norm=config.optimizer.clip_gradient_norm,
                )
                self.optim.step()

                for param in self.trainable_parameters():
                    if param.grad is not None:
                        # DDP may expose grads as bucket views, so avoid detach_().
                        param.grad = None

                if is_main:
                    infos.append(
                        {
                            "loss": float(loss.item()),
                            "learning_rate": float(self.optim.param_groups[0]["lr"]),
                            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        }
                    )

                global_step += 1
                self.save_checkpoint(self.model, self.optim, global_step, config, max_steps, is_main, data_config)
                self.on_step_end(config, repo_id, device, global_step, loss, observation, actions, loader)

                if is_main and infos and (global_step % config.log_interval == 0 or global_step == 1):
                    elapsed = time.time() - start_time
                    avg_loss = sum(info["loss"] for info in infos) / len(infos)
                    avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                    avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)
                    logging.info(
                        "step=%s loss=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                        global_step,
                        avg_loss,
                        avg_lr,
                        avg_grad_norm,
                        elapsed,
                    )
                    infos = []
                    start_time = time.time()

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{self.optim.param_groups[0]['lr']:.2e}",
                            "step": global_step,
                        }
                    )

        if self.checkpoint_final_dir is None and self._has_checkpoints(self.checkpoint_dir):
            latest_step = self.get_latest_checkpoint_step(self.checkpoint_dir)
            if latest_step is not None:
                self.checkpoint_final_dir = self.checkpoint_dir / str(latest_step)

        self.preview_checkpoint_dir = self.checkpoint_dir
        self.on_train_end(config, repo_id, device)

        if pbar is not None:
            pbar.close()

    def eval(self, args):
        raise NotImplementedError

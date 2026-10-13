from __future__ import annotations

import logging

import numpy as np
import torch


def is_piper_run(config) -> bool:
    name = str(getattr(config, "name", "")).lower()
    data_name = str(getattr(config, "data_name", "")).lower()
    return "piper" in name or "piper" in data_name or data_name.startswith("real")


def log_safe_quantile_dims(
    data_config,
    *,
    is_main: bool,
    prefix: str = "Piper safe quantile",
    monitor_dims: int = 14,
) -> None:
    """Log which action dimensions will be zero-filled by safe quantile normalization."""
    if not is_main:
        return
    if not getattr(data_config, "use_safe_quantile_norm", False):
        return
    if data_config.norm_stats is None or "actions" not in data_config.norm_stats:
        return

    action_stats = data_config.norm_stats["actions"]
    if action_stats.q01 is None or action_stats.q99 is None:
        return

    q01 = np.asarray(action_stats.q01).reshape(-1)
    q99 = np.asarray(action_stats.q99).reshape(-1)
    dims = min(monitor_dims, q01.shape[0], q99.shape[0])
    width = q99[:dims] - q01[:dims]
    threshold = float(getattr(data_config, "safe_quantile_min_width", 1e-3))
    safe_dims = np.where(np.abs(width) < threshold)[0].tolist()

    if not safe_dims:
        logging.info(
            "%s enabled, but no action dims fall below min_width=%.6f in first %d dims",
            prefix,
            threshold,
            dims,
        )
        return

    details = ", ".join(
        f"d{idx}(width={width[idx]:.6g},q01={q01[idx]:.6g},q99={q99[idx]:.6g})"
        for idx in safe_dims
    )
    logging.warning(
        "%s will zero-fill near-constant action dims at startup: %s | min_width=%.6f",
        prefix,
        details,
        threshold,
    )


def clamp_and_log_piper_actions(
    actions: torch.Tensor,
    *,
    config,
    global_step: int,
    is_main: bool,
    clamp_value: float = 3.0,
    monitor_dims: int = 14,
    log_interval: int = 100,
) -> torch.Tensor:
    """Clamp Piper actions and log dimensions that most often exceed the clamp range."""
    if not is_piper_run(config) or actions.ndim < 2:
        return actions

    dims = min(monitor_dims, actions.shape[-1])
    tracked = actions[..., :dims]
    overflow_mask = tracked.abs() > clamp_value

    if is_main and torch.any(overflow_mask) and (global_step < 10 or global_step % log_interval == 0):
        flat_mask = overflow_mask.reshape(-1, dims)
        flat_abs = tracked.abs().reshape(-1, dims)
        counts = flat_mask.sum(dim=0)
        total = max(flat_mask.shape[0], 1)
        max_abs = flat_abs.max(dim=0).values
        mean_abs = flat_abs.mean(dim=0)

        ranked = sorted(
            (
                (
                    int(idx),
                    int(counts[idx].item()),
                    float(counts[idx].item() / total),
                    float(max_abs[idx].item()),
                    float(mean_abs[idx].item()),
                )
                for idx in range(dims)
                if counts[idx].item() > 0
            ),
            key=lambda item: (item[1], item[3]),
            reverse=True,
        )

        summary = ", ".join(
            f"d{idx}:count={count},ratio={ratio:.3f},max={mx:.2f},mean={mean:.2f}"
            for idx, count, ratio, mx, mean in ranked[: min(8, len(ranked))]
        )
        logging.warning(
            "Piper action clamp triggered at step %s with clamp=%.2f | top exploding dims: %s",
            global_step,
            clamp_value,
            summary,
        )

    clamped = actions.clone()
    clamped[..., :dims] = tracked.clamp(min=-clamp_value, max=clamp_value)
    return clamped

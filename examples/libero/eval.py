#!/usr/bin/env python3
"""
离线 LIBERO 评测脚本
- 不启动 WebSocket server
- 不依赖 host / port
- 直接加载本地 checkpoint 推理
用法示例：
python /zhaolibang/openpi/examples/libero/eval.py\
    --checkpoint_dir /zhaolibang/openpi/checkpoints/pi0_libero/BASE/libero/9999 \
    --config_name pi0_libero \
    --task_suite_name libero_object \
    --num_trials_per_task 5
"""
import collections
import dataclasses
import enum
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import sys, pathlib
_repo_root = pathlib.Path(__file__).resolve().parents[2]  # 指向 openpi/
_pkg_root  = _repo_root / "src"    
print(_pkg_root)                       # 指向 openpi/src
sys.path.insert(0, str(_pkg_root))
from openpi_client import image_tools
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# ---------- 与 serve_policy 共用部分 ----------
class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    # ---- 模型 ----
    checkpoint_dir: str  # 本地 checkpoint 路径
    config_name: str = "pi0_libero"  # 训练 config 名

    # ---- LIBERO 任务参数 ----
    task_suite_name: str = "libero_object"  # libero_spatial / object / goal / 10 / 90
    num_trials_per_task: int = 10
    num_steps_wait: int = 10
    seed: int = 7

    # ---- 输出 ----
    video_out_path: str = "data/libero/videos_offline"
    resize_size: int = 224
    replan_steps: int = 5


# ---------- 环境初始化工具 ----------
def _get_libero_env(task, resolution, seed):
    task_description = task.language
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl_path,
                             camera_heights=resolution,
                             camera_widths=resolution)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    # 摘自 robosuite
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------- 主逻辑 ----------
def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        datefmt="%m-%d %H:%M:%S")

    # 1. 加载策略（本地，无需 server）
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint_dir)
    logging.info("Policy loaded from %s", args.checkpoint_dir)

    # 2. 初始化任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = task_suite.n_tasks
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # 3. 按任务循环
    total_ep, total_succ = 0, 0
    for task_id in tqdm.trange(n_tasks, desc="Task"):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_desc = _get_libero_env(task, 256, args.seed)

        task_ep, task_succ = 0, 0
        for trial in tqdm.trange(args.num_trials_per_task, desc="Trial", leave=False):
            env.reset()
            obs = env.set_init_state(init_states[trial])
            replay_images, t = [], 0
            action_plan = collections.deque()

            while t < 520 + args.num_steps_wait:  # 520 足够覆盖 object/spatial
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step([0.0]*6 + [-1.0])
                    t += 1
                    continue

                # 预处理图像
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                replay_images.append(img)

                # 需要重新规划
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])),
                        "prompt": str(task_desc),
                    }
                    chunk = policy.infer(element)["actions"]
                    assert len(chunk) >= args.replan_steps
                    action_plan.extend(chunk[:args.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                t += 1
                if done:
                    task_succ += 1
                    break

            # 记录结果 & 存视频
            suffix = "success" if done else "failure"
            video_name = f"{args.task_suite_name}_task{task_id}_trial{trial}_{suffix}.mp4"
            imageio.mimwrite(pathlib.Path(args.video_out_path) / video_name,
                             [np.asarray(x) for x in replay_images], fps=10)
            task_ep += 1
            total_ep += 1
            total_succ += int(done)

        logging.info("Task %d / %d  success  %.1f %%",
                     task_id, n_tasks, 100 * task_succ / task_ep)

    logging.info("All done! Total success rate %.1f %% (%d / %d)",
                 100 * total_succ / total_ep, total_succ, total_ep)


if __name__ == "__main__":
    tyro.cli(main)
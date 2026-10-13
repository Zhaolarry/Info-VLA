# OpenPI-INFO

基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 的机器人基础模型 **π₀ 持续学习（Continual Learning）训练框架**（PyTorch 版）。

在依次到来的多个任务上微调 π₀ 时，模型会遗忘旧任务。本框架实现了 5 种持续学习策略：

| 方法 | `exp_name` | 说明 |
|---|---|---|
| BASE | `BASE` | 基线：直接顺序微调，无防遗忘措施 |
| ER | `ER` | Experience Replay，经验回放 |
| EWC | `EWC` | Elastic Weight Consolidation，弹性权重巩固 |
| LWF | `LWF` | Learning without Forgetting，知识蒸馏 |
| **INFO** | `INFO` | **本仓库主线方法**：语言-图像对齐 + 任务提示库（prompt bank） |

方法注册表位于 `src/openpi/agents/config.py`，实现在 `src/openpi/agents/`。

## 目录结构

```
src/openpi/
├── agents/            # 持续学习策略（BASE/ER/EWC/LWF/INFO）
├── models/            # 上游 JAX 模型定义（transforms/training 的依赖）
├── models_pytorch/    # PyTorch 版 π₀ / Gemma / SigLIP 模型
├── training/          # 训练配置（TrainConfig）、数据加载、权重加载
├── policies/          # 策略封装（libero / aloha / droid / piper）
├── serving/           # gRPC 服务
└── shared/            # 归一化、下载等工具
scripts/
├── train_pytorch_continual.py   # 持续学习训练入口
├── serve_policy.py              # 策略 gRPC 服务（libero 评测用）
├── serve_policy_piper.py        # 策略 gRPC 服务（piper 真机用）
├── compute_norm_stats.py        # 计算数据归一化统计
└── docker/                      # 部署镜像
old/                   # 历史归档（旧命名 GEN、SNN/MAR/Stellar 实验、日志、输出、assets、docs、third_party），不参与版本管理
```

> `assets/`（norm_stats）、`docs/`、`third_party/`（aloha/libero 仿真环境）未包含在仓库中，需要的部分请从上游 [openpi 官方仓库](https://github.com/Physical-Intelligence/openpi) 获取。

## 环境安装

```bash
# 安装 uv（https://docs.astral.sh/uv/）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（Python >= 3.10, < 3.13）
uv sync
```

GPU 环境请按官方指引安装对应版本的 PyTorch（CUDA）。如访问 HuggingFace 不稳定，可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 模型权重下载

训练基于 π₀（pi05）预训练权重，需要先准备 PyTorch 格式权重：

1. **官方权重转换**（推荐）：从 openpi 官方资产下载 `pi05_base`，再用仓库脚本转换：

   ```bash
   # 下载 JAX 权重（gs://openpi-assets/checkpoints/pi05_base，或 HuggingFace 镜像）
   python examples/convert_jax_model_to_pytorch.py --input_dir <jax权重目录> --output_dir <输出目录>
   ```

2. 得到 PyTorch 权重目录后，修改训练配置中的路径。`pi05_libero` 配置（`src/openpi/training/config.py`）里默认写的是服务器路径，请改为你的本地路径：

   ```python
   pytorch_weight_path="<你的路径>/checkpoints/pi05_base"
   ```

## 数据准备

数据为 HuggingFace LeRobot 格式，按任务分段存放，仓库 ID 规则为 `{用户}/{data_name}_task{序号}`（默认值见代码，可自行修改）：

- **LIBERO 仿真**：默认任务序列如 `libero/libero1010er_task0` ~ `task5`（数据准备/转换脚本见 `examples/libero/`）
- **Piper 真机**：任务序列如 `piper/real_mix_task0`、`piper/real_5_task1` 等

训练时会自动从 HuggingFace 拉取对应任务数据。首次训练前需要计算归一化统计（norm_stats）：

```bash
uv run scripts/compute_norm_stats.py <config_name> --repo_ids <任务仓库ID>
```

## 训练

单任务（普通微调）：

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch_continual.py pi05_libero \
    --exp_name INFO --total_task_num 1 --data_name libero1010er
```

持续学习（依次训练 6 个任务）：

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch_continual.py pi05_libero \
    --exp_name INFO --total_task_num 6 --data_name libero1010er \
    --num_train_steps 1500 --delta_t 0.03 --batch_size 256 --port 30888
```

常用参数（定义见 `src/openpi/training/config.py` 的 `TrainConfig`）：

| 参数 | 说明 |
|---|---|
| `exp_name` | 方法名：`BASE` / `ER` / `EWC` / `LWF` / `INFO` |
| `total_task_num` | 任务总数 |
| `data_name` | 任务序列前缀，如 `libero1010er` |
| `num_train_steps` | 每任务训练步数 |
| `delta_t` | 实验超参（标签/蒸馏温度系数），不同取值用于多组对照实验 |
| `batch_size` | 批大小（多卡时按卡数均分） |
| `save_interval` | 保存间隔（步） |
| `--resume` | 断点续训（配合 `--resume_num N` 从第 N 个任务继续） |
| `--num N` | 只训练第 N 个任务 |
| `--check` | 启动前检查已有检查点，存在则自动续训 |
| `port` | 训练端口 |

检查点保存路径规则：`checkpoints/<配置名>/<exp_name>/[delta_t值/]<用户>/<data_name>_task<序号>/`。训练日志默认写入 wandb（`--wandb-enabled false` 可关闭），同时在 `src/openpi/agents/logs/` 下有按方法命名的文本日志。

## 评测

训练完成后，用 `serve_policy.py` 把某个任务的检查点启动为 gRPC 策略服务，再跑评测客户端：

```bash
uv run scripts/serve_policy.py --name pi05_libero --exp_name INFO --port 702 \
    --current 5 policy:checkpoint \
    --policy.config=pi0_libero \
    --policy.dir=<检查点目录>
```

- `--current N`：评测到第 N 个任务为止的模型（即第 N 个任务训练完成后的检查点）
- `--record`：记录策略输出到 `policy_records/` 便于调试
- 仿真评测客户端见 `examples/`，远程部署文档见上游 [openpi 官方仓库](https://github.com/Physical-Intelligence/openpi) 的 `docs/`

## 添加新方法

1. 在 `src/openpi/agents/` 新建 `my_method.py`，继承 `BaseAgent`，实现 `trainable_parameters` / `compute_loss` / 任务切换钩子（参考 `info.py`）
2. 在 `src/openpi/agents/config.py` 的注册表中添加 `"MYMETHOD": MyMethodAgent`
3. 训练时 `--exp_name MYMETHOD` 即可使用

## 备注

- `scripts/train.py`（JAX 版训练）与 `scripts/train_pytorch.py` 为上游原始脚本，持续学习请使用 `train_pytorch_continual.py`
- `old/` 为历史归档：GEN → INFO 改名前的旧模块、SNN / MAR / Stellar 实验代码、训练日志与运行输出，以及本地上传用的 `assets/`、`docs/`、`third_party/`。该目录已被 `.gitignore` 排除
- 上游 README 与更多示例见 [openpi 官方仓库](https://github.com/Physical-Intelligence/openpi)

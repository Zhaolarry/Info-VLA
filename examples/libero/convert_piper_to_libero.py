import json
import os
import shutil
import numpy as np
import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
from tqdm import tqdm
from pathlib import Path
from typing import List

def main(
    data_dir: str,
    repo_name: str,                # 新增命令行参数
    raw_dataset_names: List[str],  # 外部传入原始数据集列表
    *,
    push_to_hub: bool = False
):
    # 输出目录
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="piper",
        fps=20,
        features={
            "front_head": {"dtype": "image", "shape": (240, 424, 3), "names": ["height", "width", "channel"]},
            "right_hand": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
            "left_hand": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
        },
        image_writer_threads=20,  # 增加线程数以加速图像写入
        image_writer_processes=10,  # 增加进程数以加速图像写入
    )

    # 遍历原始数据集
    for raw_dataset_name in raw_dataset_names:
        print("raw dataset ", raw_dataset_name,"data_dir",data_dir)
        raw_dataset_path = os.path.join(data_dir, raw_dataset_name)
        json_files_path = []

        for root, _, files in os.walk(raw_dataset_path):
            json_files_path.extend(
                os.path.join(root, file) 
                for file in files 
                if "data.json" in file
            )

        print('json files path: ', json_files_path)

        for json_file_path in tqdm(json_files_path, desc=f"Processing files in {raw_dataset_name}", unit="file", leave=False):
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            base_dir = os.path.dirname(json_file_path)
            num_frames = len(data) - 1
            
            # 优化1: 预先计算所有图像路径，减少循环中的字符串拼接
            image_paths = []
            for index in range(num_frames):
                step = data[index]
                image_paths.append((
                    os.path.join(base_dir, step['front']),
                    os.path.join(base_dir, step['left']),
                    os.path.join(base_dir, step['right'])
                ))
            
            # 优化2: 移除嵌套tqdm，减少显示开销
            for index in range(num_frames):
                step = data[index]
                state = np.array(step['joint'], dtype=np.float32)
                actions = np.array(data[index + 1]['joint'], dtype=np.float32)
                
                # 优化3: 使用预先计算的路径
                front_path, left_path, right_path = image_paths[index]
                
                # 针对PNG格式优化：使用IMREAD_COLOR标志，跳过格式检测和alpha通道
                front_head = cv2.cvtColor(cv2.imread(front_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                left_hand = cv2.cvtColor(cv2.imread(left_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                right_hand = cv2.cvtColor(cv2.imread(right_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

                dataset.add_frame({
                    'front_head': front_head,
                    'left_hand': left_hand,
                    'right_hand': right_hand,
                    'actions': actions,
                    'state': state,
                    'task': step['task'][0]
                })
            dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)



'''
HF_LEROBOT_HOME=/zhaolibang/datasets \
uv run /zhaolibang/openpi/examples/libero/convert_piper_to_libero.py \
--data_dir=/zhaolibang/datasets/piper \
--repo_name=piper/real_new_task0 \
--raw_dataset_names 2025-12-23_banana_base

export PATH=$PATH:/root/miniconda3/bin 
cd /zhaolibang/openpi
source activate pi0
export HF_ENDPOINT=https://hf-mirror.com
export HF_LEROBOT_HOME="/zhaolibang/datasets"
export OPENPI_DATA_HOME="/zhaolibang/models/openpi"
uv run /zhaolibang/openpi/examples/libero/convert_piper_to_libero.py \
--data_dir=/zhaolibang/datasets \
--repo_name=piper/real_5_task2 \
--raw_dataset_names 2025-12-23_Sachet_base

--data_dir：原始数据目录
--repo_name：最终生成的数据集名称
--raw_dataset_names：可以指定多个原始数据目录 中间用空格隔开
''' 

# -rw-rw-r--. 1 root root     0 Jan 20 03:55 /liujinxin/dataset/piper/cloth_new/2026-01-20_demo_clothes/episode_113/data.json
# -rw-rw-r--. 1 root root     0 Jan 22 07:53 /liujinxin/dataset/piper/cloth_new/2026-01-22_demo_clothes/episode_261/data.json

"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot_new.py --data_dir /zhaolibang/dataset

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets -i https://pypi.tuna.tsinghua.edu.cn/simple

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.

"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "piper/libero18"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset
CONTINUAL_TASK_GROUPINGS = {
        "libero18er_task0": [
             "put both moka pots on the stove" ,
             "put the black bowl in the bottom drawer of the cabinet and close it" ,
             "put the bowl on the stove" ,
             "put the cream cheese in the bowl",
             "pick up the tomato sauce and place it in the basket",
             "pick up the alphabet soup and place it in the basket",
             "pick up the black bowl on the wooden cabinet and place it on the plate",
             "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate" ,
             "pick up the black bowl next to the plate and place it on the plate",
             "pick up the cream cheese and place it in the basket"

        ],
        "libero18er_task4": [
            "put both the alphabet soup and the cream cheese box in the basket" ,
            "put both the cream cheese box and the butter in the basket" 
        ],
        "libero18er_task3": [
            "open the top drawer and put the bowl inside",
            "open the middle drawer of the cabinet" ,
        ],
        "libero18er_task1": [
            "pick up the milk and place it in the basket" ,
            "pick up the butter and place it in the basket",
        ],
        "libero18er_task2": [
            "pick up the black bowl on the stove and place it on the plate" ,
            "pick up the black bowl from table center and place it on the plate",
        ]
}

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    for task_group_name in CONTINUAL_TASK_GROUPINGS:
        output_path = HF_LEROBOT_HOME / f"libero/{task_group_name}"
        if output_path.exists():
            shutil.rmtree(output_path)
        dataset = LeRobotDataset.create(
            repo_id=f"libero/{task_group_name}",
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

        # Loop over raw Libero datasets and write episodes to the LeRobot dataset
        # You can modify this for your own data format
        for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode in raw_dataset:
                episode_task = None
                for step in episode["steps"].as_numpy_iterator():
                    episode_task = step["language_instruction"].decode()
                    break
                    
                if episode_task in CONTINUAL_TASK_GROUPINGS[task_group_name]:
                
                    for step in episode["steps"].as_numpy_iterator():
                        dataset.add_frame(
                            {
                                "image": step["observation"]["image"],
                                "wrist_image": step["observation"]["wrist_image"],
                                "state": step["observation"]["state"],
                                "actions": step["action"],
                                "task": step["language_instruction"].decode(),
                            }
                        )
                    dataset.save_episode()

        # Optionally push to the Hugging Face Hub
        if push_to_hub:
            dataset.push_to_hub(
                tags=["libero", "panda", "rlds"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    tyro.cli(main)
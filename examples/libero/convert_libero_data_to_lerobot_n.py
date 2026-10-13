"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot_n.py --data_dir /zhaolibang/datasets

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

REPO_NAME = "libero/libero_spatial"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    # "libero_10_no_noops",
    # "libero_goal_no_noops",
    # "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

        # "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        # "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        # "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        # "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        # "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        # "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        # "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        # "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        # "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        # "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",

CONTINUAL_TASK_GROUPINGS = {


    "libero_spatial_task0": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    ],
    "libero_spatial_task1": [
    "pick up the black bowl next to the ramekin and place it on the plate",
    ],
    "libero_spatial_task2": [
    "pick up the black bowl from table center and place it on the plate",
    ],
    "libero_spatial_task3": [
    "pick up the black bowl on the cookie box and place it on the plate",
    ],
    "libero_spatial_task4": [
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    ],
#      "libero10er_task0": [
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',

#         ],
# "libero10_task0":['put both the alphabet soup and the tomato sauce in the basket'],
# "libero10_task1":['put both the cream cheese box and the butter in the basket'],
# "libero10_task2":['turn on the stove and put the moka pot on it'],
# "libero10_task3":['put the black bowl in the bottom drawer of the cabinet and close it'],
# "libero10_task4":['put the white mug on the plate and put the chocolate pudding to the right of the plate'],
# "libero10_task5":['pick up the book and place it in the back compartment of the caddy'],
# "libero10_task6":['put the white mug on the left plate and put the yellow and white mug on the right plate'],
# "libero10_task7":['put both the alphabet soup and the cream cheese box in the basket'],
# "libero10_task8":['put both moka pots on the stove'],
# "libero10_task9":['put the yellow and white mug in the microwave and close it'],


# "libero10er_task0":['put both the alphabet soup and the tomato sauce in the basket'],
# "libero10er_task1":['put both the cream cheese box and the butter in the basket'],
# "libero10er_task2":['turn on the stove and put the moka pot on it'],
# "libero10er_task3":['put the black bowl in the bottom drawer of the cabinet and close it'],
# "libero10er_task4":['put the white mug on the plate and put the chocolate pudding to the right of the plate'],
# "libero10er_task5":['pick up the book and place it in the back compartment of the caddy'],
# "libero10er_task6":['put the white mug on the left plate and put the yellow and white mug on the right plate'],
# "libero10er_task7":['put both the alphabet soup and the cream cheese box in the basket'],
# "libero10er_task8":['put both moka pots on the stove'],
# "libero10er_task9":['put the yellow and white mug in the microwave and close it']
 

#         "liberogoaler_task0": [
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
#         ],
#         "libero1030er_task1": [
# 'put the yellow and white mug in the microwave and close it',
#         ],
#         "libero1030er_task3": [
# 'put the black bowl in the bottom drawer of the cabinet and close it',
#         ],
#         "libero1030er_task5": [
# 'put both the cream cheese box and the butter in the basket', 
#         ],
#         "libero1030er_task2": [
# 'put both the alphabet soup and the tomato sauce in the basket',           
#         ],
#         "libero1030er_task4": [
# 'put both the alphabet soup and the cream cheese box in the basket',            
#         ],


# "liberogoal_task0": [
# "put the bowl on the plate"
# ],
# "liberogoal_task1": [
# "put the wine bottle on the rack"
# ],
# "liberogoal_task2": [
# "open the top drawer and put the bowl inside"
# ],
# "liberogoal_task3": [
# "put the cream cheese in the bowl"
# ],
# "liberogoal_task4": [
# "put the wine bottle on top of the cabinet"
# ],


}


CONTINUAL_TASK = {
    "libero_spatial_task0": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    ],
    "libero_spatial_task1": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    ],
    "libero_spatial_task2": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",

    ],
    "libero_spatial_task3": [

    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    ],
    "libero_spatial_task4": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    ],

# "libero10er_task1":[
# 'put both the alphabet soup and the tomato sauce in the basket',],
# "libero10er_task2":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',],
# "libero10er_task3":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',],
# "libero10er_task4":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',
# 'put the black bowl in the bottom drawer of the cabinet and close it',],
# "libero10er_task5":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',
# 'put the black bowl in the bottom drawer of the cabinet and close it',
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate', ],
# "libero10er_task6":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',
# 'put the black bowl in the bottom drawer of the cabinet and close it',
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy', ],
# "libero10er_task7":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',
# 'put the black bowl in the bottom drawer of the cabinet and close it',
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',],
# "libero10er_task8":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',
# 'put the black bowl in the bottom drawer of the cabinet and close it',
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'put both the alphabet soup and the cream cheese box in the basket', ],
# "libero10er_task9":[
# 'put both the alphabet soup and the tomato sauce in the basket',
# 'put both the cream cheese box and the butter in the basket',
# 'turn on the stove and put the moka pot on it',
# 'put the black bowl in the bottom drawer of the cabinet and close it',
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'put both the alphabet soup and the cream cheese box in the basket', 
# 'put both moka pots on the stove',]

# "liberogoaler_task1": [
# "put the bowl on the plate"
# ],
# "liberogoaler_task2": [
# "put the bowl on the plate",
# "put the wine bottle on the rack",
# ],
# "liberogoaler_task3": [
#       "put the bowl on the plate",
# "put the wine bottle on the rack",
# "open the top drawer and put the bowl inside"
# ],
# "liberogoaler_task4": [
# "put the bowl on the plate",
# "put the wine bottle on the rack",
# "open the top drawer and put the bowl inside",
# "put the cream cheese in the bowl"
# ],

#         "libero10er_task0": [
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',

#         ],
#         "libero1030er_task2": [
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the yellow and white mug in the microwave and close it',
#         ],
#         "libero1030er_task3": [
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the yellow and white mug in the microwave and close it',
# 'put both the alphabet soup and the tomato sauce in the basket',     
#         ],
#         "libero1030er_task4": [
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the yellow and white mug in the microwave and close it',
# 'put both the alphabet soup and the tomato sauce in the basket',     
# 'put the black bowl in the bottom drawer of the cabinet and close it',
#         ],
#         "libero1030er_task5":[
# 'put the white mug on the left plate and put the yellow and white mug on the right plate',
# 'turn on the stove and put the moka pot on it',
# 'put both moka pots on the stove', 
# 'put the white mug on the plate and put the chocolate pudding to the right of the plate',
# 'pick up the book and place it in the back compartment of the caddy',
# 'put the yellow and white mug in the microwave and close it',
# 'put both the alphabet soup and the tomato sauce in the basket',     
# 'put the black bowl in the bottom drawer of the cabinet and close it',
# 'put both the alphabet soup and the cream cheese box in the basket',   
#         ]

}

def load_all_episodes(dataset, task_group_name,data_dir):
    
    d = {}
    for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode in raw_dataset:
                episode_task = None
                for step in episode["steps"].as_numpy_iterator():
                    episode_task = step["language_instruction"].decode()
                    break
                    
                if episode_task in CONTINUAL_TASK[task_group_name] and d.get(episode_task, 0) < 1:
                    print(episode_task,d.get(episode_task, 0))
                    d[episode_task] = d.get(episode_task, 0) + 1
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
    return dataset


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
        # load old data
        # if 'task0' not  in task_group_name:
        dataset = load_all_episodes(dataset, task_group_name,data_dir)
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
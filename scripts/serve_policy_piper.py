from flask import Flask, request, Response
import argparse
import os
import json
import sys
import torch
import time
import numpy as np
import jax

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
openpi_client_path = os.path.join(root_path,'packages/openpi-client/src/')
openpi_path = os.path.join(root_path,'src/')
sys.path.append(root_path)
sys.path.append(openpi_path)
sys.path.append(openpi_client_path)

from openpi.training import config as _config
from openpi.shared import download
from openpi.policies import policy_config as _policy_config


class LLMRobotServer:
    def __init__(self):
        
        # config = _config.get_config("pi05_piper_packages_demo_v1")
        # checkpoint_dir = download.maybe_download("/home/ace/models/checkpoints/pi05_piper_packages_demo_v1/30000_torch")

        # config = _config.get_config("pi05_piper_flatten_fold_v2")
        # checkpoint_dir = download.maybe_download("/home/ace/models/checkpoints/pi05_piper_flatten_fold_v2/39999_torch") 

        # config = _config.get_config("pi05_piper_flatten_fold_final")
        # checkpoint_dir = download.maybe_download("/home/ace/models/checkpoints/pi05_piper_flatten_fold_final/37000_torch") 
        
        # config = _config.get_config("pi05_piper_base_fold_sweater")
        # checkpoint_dir = download.maybe_download("/home/ace/models/checkpoints/pi05_piper_base_fold_sweater/16000_torch")

        config = _config.get_config("pi05_piper")
        checkpoint_dir = download.maybe_download("/zhaolibang/openpi/checkpoints/pi05_piper/INFO/0.1/piper/real_mix_task0/299")
        
        config = _config.get_config("pi05_piper")
        checkpoint_dir = download.maybe_download("/zhaolibang/openpi/checkpoints/pi05_piper/INFO/0.1/piper/real_task0/299")

        config = _config.get_config("pi05_piper")
        checkpoint_dir = download.maybe_download("/zhaolibang/openpi/checkpoints/pi05_piper/ER/piper/real_mix_task0/299")

        config = _config.get_config("pi05_piper")
        checkpoint_dir = download.maybe_download("/zhaolibang/openpi/checkpoints/pi05_piper/ER/piper/real_task0/299")

        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=5009)
    args = parser.parse_args()

    flask_app = Flask(__name__)
    llm_robot = LLMRobotServer()

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        # added for printing model inference time
        start_time = time.perf_counter_ns()
        
        front_head = torch.frombuffer(request.files["front_head"].read(), dtype=torch.uint8)
        front_head = front_head.reshape((240, 424, 3))
        left_hand = torch.frombuffer(request.files["left_hand"].read(), dtype=torch.uint8)
        left_hand = left_hand.reshape((240, 320, 3))
        right_hand = torch.frombuffer(request.files["right_hand"].read(), dtype=torch.uint8)
        right_hand = right_hand.reshape((240, 320, 3))

        

        state = torch.frombuffer(request.files["state"].read(), dtype=torch.float32)
        state = state.reshape((14))

        # 语言部分
        content = request.files["json"].read()
        content = json.loads(content)
        instruction = content["instruction"]
        
        data = {
            "prompt": f"{instruction}",
            "state": state,
            "front_head": front_head,
            "left_hand": left_hand,
            "right_hand": right_hand,
        }

        result = llm_robot.policy.infer(data, blocking=True)
        # 推理动作
        action = result["actions"][None]
        action_bytes = action.tobytes()
        
        end_time = time.perf_counter_ns()
        print(
            f"total inference time: {(end_time - start_time) / 1e6} ms; "
            f"""model inference time: {result["policy_timing"]["infer_ms"]} ms"""
        )

        return Response(action_bytes, content_type='application/octet-stream')

    flask_app.run(host="0.0.0.0", port=args.port)

    # CUDA_VISIBLE_DEVICES=0 python flask_server_piper_v1_torch_5009.py
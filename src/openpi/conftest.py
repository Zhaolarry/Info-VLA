import os

import pynvml
import pytest


# def set_jax_cpu_backend_if_no_gpu() -> None:
#     try:
#         pynvml.nvmlInit()
#         pynvml.nvmlShutdown()
#     except pynvml.NVMLError:
#         # No GPU found.
#         os.environ["JAX_PLATFORMS"] = "cpu"


# def pytest_configure(config: pytest.Config) -> None:
#     set_jax_cpu_backend_if_no_gpu()



from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi05_libero")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
# example = {
#     "observation/exterior_image_1_left": ...,
#     "observation/wrist_image_left": ...,
#     ...
#     "prompt": "pick up the fork"
# }
# action_chunk = policy.infer(example)["actions"]
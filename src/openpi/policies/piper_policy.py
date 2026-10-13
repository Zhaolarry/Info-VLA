import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:

        # # First, concatenate the joints and gripper into the state vector.
        # # Pad to the expected input dimensionality of the model (same as action_dim).
        # #state = np.concatenate([data["joints"], data["gripper"]])
        # state = transforms.pad_to_dim(data['state'], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        # print("data:",data)
        front_head = _parse_image(data["front_head"])
        left_hand = _parse_image(data["left_hand"])
        right_hand = _parse_image(data["right_hand"])

        # Create inputs dict.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": front_head,
                "left_wrist_0_rgb": left_hand,
                "right_wrist_0_rgb": right_hand,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        # print(inputs)
        # Pad actions to the model action dimension.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        
        # For Piper, we only return the first 14 actions (since the rest is padding).
        return {"actions": np.asarray(data["actions"][:, :14])}

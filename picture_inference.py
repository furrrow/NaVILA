import argparse
from collections import deque
import re
import numpy as np
from typing import List
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_image,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model


def navila_command_to_waypoints(
    command: str,
    num_steps: int = 8,
) -> List[np.ndarray]:
    """
    Convert a NaVILA verbal navigation command into relative robot-frame
    waypoints compatible with pd_controller().

    Coordinate convention:
        +x = forward
        +y = left

    Waypoint formats:
        translation: [dx, dy]
        rotation:    [0, 0, hx, hy]

    Examples:
        "move forward 50 cm"
            -> five [0.1, 0.0] waypoints

        "turn left 30 degree"
            -> three +10 degree heading waypoints

        "turn right 20 degree"
            -> two -10 degree heading waypoints

        "stop"
            -> []
    """

    text = command.lower().strip()

    if "stop" in text:
        return []

    forward_match = re.search(r"(?:move\s+)?forward.*?([0-9]+(?:\.[0-9]+)?)\s*(cm|m)", text)
    if forward_match:
        distance = float(forward_match.group(1))
        unit = forward_match.group(2)

        if unit == "cm":
            distance /= 100.0

        return _split_forward_motion(
            distance_m=distance,
            n_steps=num_steps,
        )

    turn_match = re.search(r"turn\s+(left|right).*?([0-9]+(?:\.[0-9]+)?)\s*(?:degree|degrees|deg)",text)
    if turn_match:
        direction = turn_match.group(1)
        angle_deg = float(turn_match.group(2))

        if direction == "right":
            angle_deg = -angle_deg
        return _split_rotation(angle_deg=angle_deg, n_steps=num_steps,)

    raise ValueError(f"Could not parse NaVILA command: {command!r}")


def _split_forward_motion(distance_m: float, n_steps: float) -> List[np.ndarray]:

    fwd_distances = np.linspace(0, distance_m, n_steps)
    side_distances = np.zeros_like(fwd_distances)
    waypoints = np.stack((fwd_distances, side_distances)).T

    return waypoints


def _split_rotation(angle_deg: float, n_steps: float) -> List[np.ndarray]:

    degree_array = np.linspace(0, angle_deg, n_steps)
    print(degree_array)
    radian_array = np.deg2rad(degree_array)
    hx_array = np.cos(radian_array)
    hy_array = np.sin(radian_array)
    zero_array = np.zeros_like(hx_array)
    waypoints = np.stack((zero_array, zero_array, hx_array, hy_array)).T
    return waypoints

class NavilaPolicy:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_frames: int = 8,
        conv_mode: str = "llama_3",
    ):
        self.device = device
        self.num_frames = num_frames
        self.conv_mode = conv_mode

        model_name = get_model_name_from_path(model_path)

        (
            self.tokenizer,
            self.model,
            self.image_processor,
            _,
        ) = load_pretrained_model(
            model_path,
            model_name,
            None,
        )

        self.model = self.model.to(device)
        self.model.eval()

        self.frames = deque(maxlen=num_frames)

    def add_frame(self, image: Image.Image):
        self.frames.append(image.convert("RGB"))

    def _get_frame_history(self):
        if not self.frames:
            raise RuntimeError("No camera image has been added.")

        frames = list(self.frames)

        # Pad startup history to num_frames.
        while len(frames) < self.num_frames:
            frames.insert(0, frames[0])

        return frames[-self.num_frames:]

    def _process_images(self, frames):
        tensors = []

        # NaVILA's process_image expects image_processor through config.
        self.model.config.image_processor = self.image_processor

        for image in frames:
            image_tensor = process_image(
                image,
                self.model.config,
                None,
            )
            tensors.append(image_tensor)

        image_tensor = torch.stack(tensors, dim=0)

        return image_tensor.to(
            self.device,
            dtype=torch.float16,
        )

    @torch.inference_mode()
    def predict(self, instruction: str):
        frames = self._get_frame_history()
        image_tensor = self._process_images(frames)

        image_token = "<image>\n"

        prompt_text = (
            f"Imagine you are a robot programmed for navigation tasks. "
            f"You have been given a video of historical observations "
            f"{image_token * (self.num_frames - 1)}"
            f"and current observation <image>\n. "
            f'Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, "
            f"which could be turning left or right by a specific degree, "
            f"moving forward a certain distance, or stop if the task is completed."
        )

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        stop_str = (
            conv.sep
            if conv.sep_style != SeparatorStyle.TWO
            else conv.sep2
        )

        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str],
            self.tokenizer,
            input_ids,
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
            device=input_ids.device,
        )

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        output = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()

        return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", default="./checkpoints/navila-llama3-8b-8f")
    parser.add_argument("--image_path", default="/home/jim/Projects/steernav/assets/iribe_corridoor_93.png")
    parser.add_argument("--instruction", default="Go to the chair")

    args = parser.parse_args()

    policy = NavilaPolicy(args.model_path)

    image = Image.open(args.image_path).convert("RGB")

    # For a single-image smoke test, filling the rolling history with
    # the same image is preferable to changing the model interface.
    for _ in range(policy.num_frames):
        policy.add_frame(image)

    output = policy.predict(args.instruction)
    print(output)
    waypoints = navila_command_to_waypoints(output, num_steps=4)
    # print(waypoints)


if __name__ == "__main__":
    main()
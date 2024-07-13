"""VIP."""

import json
import re
from typing import Generator

import cv2
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
from models import ActionModel, StyleModel
import vip
from vlms import GPT4V


def make_prompt(description, top_n=3):
  return f"""
INSTRUCTIONS:
You are tasked to locate an object, region, or point in space in the given annotated image according to a description.
The image is annoated with numbered circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what the description is describing in the image.
You are a five-time world champion in this game. 
Give a one sentence analysis of why you chose those points.
Provide your answer at the end in a valid JSON of this format:

{{"points": []}}

DESCRIPTION: {description}
IMAGE:
""".strip()


def extract_json(response, key):
  json_part = re.search(r"\{.*\}", response, re.DOTALL)
  parsed_json = {}
  if json_part:
    json_data = json_part.group()
    # Parse the JSON data
    parsed_json = json.loads(json_data)
  else:
    print("No JSON data found ******\n", response)
  return parsed_json[key]


def vip_perform_selection(prompter: vip.VisualIterativePrompter, vlm, im, desc, arm_coord, samples, top_n):
  """Perform one selection pass given samples."""
  print("I was run!!\n\n\n")
  image_circles_np = prompter.add_arrow_overlay_plt(
      image=im, samples=samples, arm_xy=arm_coord
  )
  _, encoded_image_circles = cv2.imencode(".png", image_circles_np)
  

  prompt_seq = [make_prompt(desc, top_n=top_n), encoded_image_circles]
  response = vlm.query(prompt_seq)

  try:
    arrow_ids = extract_json(response, "points")
  except Exception as e:
    print(e)
    arrow_ids = []
  # for arrow_id in arrow_ids:
  #   if arrow_id >= len(samples):
  #     print(arrow_id)
  #     print(response)
  return arrow_ids, image_circles_np


async def vip_runner(
    vlm: GPT4V,
    im: np.ndarray,
    desc: str,
    style: StyleModel,
    action_spec: ActionModel,
    n_samples_init=25,
    n_samples_opt=10,
    n_iters=3,
    n_parallel_trials=1,
) -> tuple[int, int]:
  """Queries the VLM with PVIOT arrows
  Args:
    - vlm: a VLM wrapper to query GPT
    - im: input (unmarked) image
    - desc: description of task
    - style: info about how the script should be run. See pydantic model for format
    - action_spec: information for where arrows should be placed, etc.
    - n_saples_init: how many initial points are sampled for the first PIVOT iteration.
    - n_samples_opt: how many points are sampled for subsequent iterations
    - n_iters: I'm assuming it's how many passes of the VLM it goes through
    - n_parallel_trials: How many times to repeat this experiment (to average out?)
  Returns:
    - [], error if the query failed
  Yields:
    - (list[annotated_images], feedback string)
  """

  prompter = vip.VisualIterativePrompter(
      style, action_spec, vip.SupportedEmbodiments.HF_DEMO
  )

  output_ims = []
  arm_coord = action_spec["arm_coord"]

  new_samples = []
  center_mean = action_spec["loc"]
  for i in range(n_parallel_trials):
    center_mean = action_spec["loc"]
    center_std = action_spec["scale"]
    for itr in trange(n_iters):
      if itr == 0:
        style["num_samples"] = n_samples_init
      else:
        style["num_samples"] = n_samples_opt
      samples = prompter.sample_actions(im, arm_coord, center_mean, center_std)
      arrow_ids, image_circles_np = vip_perform_selection(
          prompter, vlm, im, desc, arm_coord, samples, top_n=3
      )

      # plot sampled circles as red
      selected_samples = []
      for selected_id in arrow_ids:
        sample = samples[selected_id]
        sample.coord.color = (255, 0, 0)
        selected_samples.append(sample)
      image_circles_marked_np = prompter.add_arrow_overlay_plt(
          image_circles_np, selected_samples, arm_coord
      )
      output_ims.append(image_circles_marked_np)
      # yield output_ims, f"Image generated for parallel sample {i+1}/{n_parallel_trials} iteration {itr+1}/{n_iters}. Still working..."

      # if at last iteration, pick one answer out of the selected ones
      if itr == n_iters - 1:
        arrow_ids, _ = vip_perform_selection(
            prompter, vlm, im, desc, arm_coord, selected_samples, top_n=1
        )

        selected_samples = []
        for selected_id in arrow_ids:
          sample = samples[selected_id]
          sample.coord.color = (255, 0, 0)
          selected_samples.append(sample)
        image_circles_marked_np = prompter.add_arrow_overlay_plt(
            im, selected_samples, arm_coord
        )
        output_ims.append(image_circles_marked_np)
        new_samples += selected_samples
        # yield output_ims, f"Image generated for parallel sample {i+1}/{n_parallel_trials} last iteration. Still working..."
      center_mean, center_std = prompter.fit(arrow_ids, samples)

  if n_parallel_trials > 1:
    # adjust sample label to avoid duplications
    for sample_id in range(len(new_samples)):
      new_samples[sample_id].label = str(sample_id)
    arrow_ids, _ = vip_perform_selection(
        prompter, vlm, im, desc, arm_coord, new_samples, top_n=1
    )

    selected_samples = []
    for selected_id in arrow_ids:
      sample = new_samples[selected_id]
      sample.coord.color = (255, 0, 0)
      selected_samples.append(sample)
    image_circles_marked_np = prompter.add_arrow_overlay_plt(
        im, selected_samples, arm_coord
    )
    output_ims.append(image_circles_marked_np)
    center_mean, _ = prompter.fit(arrow_ids, new_samples)

  return np.round(prompter.action_to_coord(center_mean, im, arm_coord).xy, decimals=0)

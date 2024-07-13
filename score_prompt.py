import asyncio
import glob
import json
from math import atan
import math
import os
from typing import Any, Coroutine
from dotenv import load_dotenv
from matplotlib.image import imread, imsave
import numpy as np

from planner import PivotPlanner
from vlms import GPT4V
from PIL import Image


with open("fix_prompt.txt", "r") as file:
    FIX_PROMPT = "\n".join(file.readlines())

async def get_score_of_image(image_filepath: str, metadata: dict[str, Any], planner: PivotPlanner, gpt: GPT4V) -> float:
    
    image = imread(image_filepath)#np.asarray(Image.open(image_filepath))
    # print(image_filepath)
    # print(image.shape)
    # print(image_filepath)
    # print(np.amin(image), np.amax(image))
    # if np.amax(image) < 2:
    #     image = np.round(255 * image)
    # minValue = np.amin(image)
    # print(minValue, maxValue)

    # print(image[0][0])
    image_name = os.path.basename(image_filepath)
    block_pos: tuple[int, int] = (metadata["block_center"][0], metadata["block_center"][1])
    results = await planner.get_arrow_corrections(image, flip_image=False, origin_point=block_pos)

    
    final_coords: tuple[int, int] = results[1]  #type: ignore
    # block_pos = (int(block_pos[0] * 1024 // image.shape[1]), int((image.shape[0] - block_pos[1]) * 1024 // image.shape[1]))
    # print(final_coords, block_pos)
    direction = math.atan2(block_pos[1]-final_coords[1], final_coords[0]-block_pos[0])
    # print(metadata.keys())
    target_dir = metadata["target_direction"]

    # print(f"Error for {image_name}. Target dir: {target_dir}, curr dir: {direction}. Error={abs(target_dir-direction)}")
    if not os.path.exists(f"out_images/{image_name}/"):
        os.mkdir(f"out_images/{image_name}")
    # imsave(f"out_images/{image_name}/scoring.jpg", results[0][-1])
    out = abs(((target_dir - direction + math.pi / 2) % math.pi) - math.pi/2)
    print(out)
    return out


async def main_loop(tasks: list[Coroutine]):
    res = await asyncio.gather(*tasks)
    return sum(res)

if __name__ == "__main__":
    load_dotenv()
    images = glob.glob("imgs/*.jpg")
    with open("imgs/metadata.json", "r") as metadata_file:
        metadata = json.load(metadata_file)
    skipped = 0
    pivot = PivotPlanner()
    gpt = GPT4V(os.environ["OPENAI_API_KEY"])
    tasks = []
    # images = images[:15]
    count = 0
    for i, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        if not(image_name in metadata.keys()):
            print(f"Skipping {image_name}")
            skipped += 1
            continue
        image_info: dict[str, Any] = metadata[image_name]
        if "to_delete" in image_info.keys() or image_info["is_correct"] or image_info["view"] != "wrist":
            continue
        tasks.append(get_score_of_image(image_path, image_info, pivot, gpt))
        count += 1
        if count == 15:
            break
        # print(f"Current error: {loss / (i+1)}")
    loss = asyncio.run(main_loop(tasks))
    loss /= 15#len(images)
    print("FINAL ANSWERS:")
    print(loss)
    print(skipped)
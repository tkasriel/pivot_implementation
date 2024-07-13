import asyncio
import glob
import json
import os
import random
from typing import Any
from matplotlib.image import imread, imsave
from dotenv import load_dotenv
from app import run_vip
from gpt_utils import request_gpt
from PIL import Image


ROTATE_PROMPT = """You are a skilled robot planner capable of aligning blocks. Is the top block correctly rotated in such a way to align with the block below it? Describe your reasoning in detail, and then give a number in degrees rotation on the last line (only the number, without the degrees symbol) If the block is already correctly rotated, your answer should end in 0.
You should also consider that we will need to perform a translation later. We are only concerned with the orientation of the two blocks for now."""
with open("fix_prompt.txt", "r") as file:
    FIX_PROMPT = "\n".join(file.readlines())

def run_on_image(image_filepath: str, metadata: dict[str, Any]) -> None:
    image = imread(image_filepath)
    image_name = os.path.basename(image_filepath)
    block_pos = (metadata["block_center"][0], metadata["block_center"][1])
    results = run_vip(image, FIX_PROMPT, 25, 15, os.environ["OPENAI_API_KEY"], block_pos, n_iters=5)
    if not os.path.exists(f"out_images/{image_name}/"):
        os.mkdir(f"out_images/{image_name}")
        for i, re in enumerate(results):
            imsave(f"out_images/{image_name}/{i}.jpg", re[0][-1])
            # print(re[1])
    
async def rotate_image(image_filepath: str, metadata: dict[str, Any]) -> None:
    image = Image.open(image_filepath)
    image_name = os.path.basename(image_filepath)
    # block_pos = (metadata["block_center"][0], metadata["block_center"][1])
    response = await request_gpt(ROTATE_PROMPT, image)
    with open(f"out/{image_name}.txt", "w") as file:
        file.write(response)

async def main_loop(tasks):
    await asyncio.gather(*tasks)
    return None

if __name__ == "__main__":
    load_dotenv()
    images = glob.glob("imgs/*.jpg")
    with open("imgs/metadata.json", "r") as metadata_file:
        metadata = json.load(metadata_file)
    # images = ["images/side_9.5.JPG"]
    skipped = 0
    tasks = []
    for image_path in images:
        image_name = os.path.basename(image_path)
        if not(image_name in metadata.keys()):
            skipped += 1
            print (f"skipping {image_name}")
            continue
        image_info = metadata[image_name]
        if "to_delete" in image_info.keys():
            continue
        if not image_info["rotate"]:
            if random.random() < 0.9:
                continue
        tasks.append(rotate_image(image_path, image_info))
    asyncio.run(main_loop(tasks))
    print(skipped)
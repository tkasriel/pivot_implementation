import glob
import json
import os
from typing import Any
from matplotlib.image import imread, imsave
from dotenv import load_dotenv
from app import run_vip

FIX_PROMPT = """Which arrow best represents the direction which the top should be moved to cover the block underneath it?
"""

def run_on_image(image_filepath: str, metadata: dict[str, Any]) -> None:
    image = imread(image_filepath)
    image_name = os.path.basename(image_filepath)
    block_pos = (metadata["block_center"]["x"], metadata["block_center"]["y"])
    results = run_vip(image, FIX_PROMPT, 25, 15, os.environ["OPENAI_API_KEY"], block_pos)
    if not os.path.exists(f"out_images/{image_name}/"):
        os.mkdir(f"out_images/{image_name}")
        for i, re in enumerate(results):
            imsave(f"out_images/{image_name}/{i}.jpg", re[0][-1])
            print(re[1])
    

if __name__ == "__main__":
    load_dotenv()
    images = glob.glob("images/*.JPG")
    with open("images/metadata.json", "r") as metadata_file:
        metadata = json.load(metadata_file)
    for image_path in images:
        image_name = os.path.basename(image_path)
        image_info = metadata[image_name]
        run_on_image(image_path, image_info)
"""

"""
import os
import base64
import re
import time
from typing import Any, Tuple
import aiohttp, requests
from io import BytesIO
from aiohttp_retry import RetryClient
# import cv2


from numpy.typing import NDArray
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import asyncio

STACKING_ORDER_PROMPT = """You are a skilled robot planner controlling a robot with suction. Your task is to stack toy blocks to create a stable tower.

Instructions:
1. Refer to objects by their labeled number, writing them in brackets, e.g., [1], [2].
2. Do not use ranges; if you need to refer to multiple items, write them in a list format, e.g., [1], [2], [3].
3. Note that not all labels refer to manipulable objects, and some objects may have multiple labels referring to different parts.
4. Some labels have been placed on the table. Do not attempt to grasp the table.
5. Identify only the labels corresponding to manipulable blocks. The labels are where the robot will grasp the object, if a label is near, but not on an object then it IS NOT referring to that object. 
6. Provide the best order to stack these blocks in a list format, with the first item in the list being the bottommost block in the stack. This means that the blocks should be in order from largest to smallest in size.

Please list the blocks in the most stable order for stacking. Do not include objects which are not stackable blocks or which can't be grasped by a suction cup.
On the first line, you should only include the list of numbers, without any other text.
On the next N lines, explain why each object is suitable for stacking, and why it should appear in that location.

Example answer:
[1], [2], [3], [4], [5]"""

GOTO_ORDER_PROMPT = """You are a skilled robot controlling a robot with suction. Your task is to stack the blocks seen in image 1 to obtain the stack found in image 2
Instructions:
1. Refer to objects by their labeled number, writing them in brackets, e.g., [1], [2].
2. Do not use ranges; if you need to refer to multiple items, write them in a list format, e.g., [1], [2], [3].
3. Note that not all labels refer to manipulable objects, and some objects may have multiple labels referring to different parts.
4. Identify only the labels corresponding to manipulable blocks.
5. Provide the best order to stack these blocks in a list format, with the first item in the list being the bottommost block in the stack.

Please list the blocks that should be manipulated in order to achieve the task. You may reason as much as you want beforehand, listing the blocks and their properties.
On the last line, write a list of the labels to stack in order as your final answer.
"""

# STACKING_ORDER_PROMPT = """
#     This is a top-down view of a tabletop with objects. Select the IDs of the toy blocks to stack. Do not select objects which are not toy blocks. Provide the ID's as a list with numbers between brackets , e.g., [1], [2]...
#     List the blocks in order they should be stacked based on their sizes and geometries.
#     Example answer:
#     [7], [2], [5], [3], [9], [12]
# """

# Get OpenAI API Key from environment variable
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']


# TODO(kuanfang): Maybe also support free form responses.
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

DEFAULT_LLM_MODEL_NAME = 'gpt-4'
DEFAULT_VLM_MODEL_NAME = 'gpt-4o'
# DEFAULT_VLM_MODEL_NAME = 'gpt-4-vision-preview'


def encode_image_from_file(image_path):
    # Function to encode the image
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def prepare_inputs(messages : list[str] | str,
                   images : list[str] | list[Image.Image],
                   meta_prompt: str,
                   model_name: str,
                   local_image: bool) -> dict[str, Any]:

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content: dict[str, Any] = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}',
                "detail": "low"
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload


async def request_gpt(message,
                images,
                meta_prompt='',
                model_name=None,
                local_image=False) -> str:

    # TODO(kuan): A better interface should allow interleaving images and
    # messages in the inputs.

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME
    
    # import numpy as np
    # import matplotlib.pyplot as plt
    # try:
    #     rand = np.random.rand(1)
    #     plt.imsave(f"/home/apgoldberg/autolab/moka/moka_pre_release/image_{str(rand)[2:]}.jpg", np.array(images[0]))
    #     if len(images) > 1:
    #         print("multipel image prompt")
    #         breakpoint()
    #     with open(f"/home/apgoldberg/autolab/moka/moka_pre_release/prompt_{str(rand)[2:]}.txt", "w") as f:
    #         f.write(str(message) + "SYSTEM: " +  meta_prompt)
    # except:
    #     print("saving failure")
    #     breakpoint()
    #     print("after break")

    payload = prepare_inputs(message,
                             images,
                             meta_prompt=meta_prompt,
                             model_name=model_name,
                             local_image=local_image)
    # start_time = time.time()
    async with aiohttp.ClientSession() as client:
        # print(f"Setting up session took {time.time()-start_time} secs")
        # start_time = time.time()
        retry_client = RetryClient(client_session=client)
        async with retry_client.post('https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload) as response:
            # print(f"Sending to gpt took {time.time() - start_time} secs")
            try:
                res = (await response.json())['choices'][0]['message']['content']
                # print(f"GPT request took {time.time()-start_time} secs")
            except Exception:
                print('\nInvalid response: ')
                print(response)
                print('\nInvalid response: ')
                print(response.json())
                exit()
            return res
    # with open(f"/home/apgoldberg/autolab/moka/moka_pre_release/prompt_{str(rand)[2:]}.txt", "a") as f:
    #     f.write(res)

    


def prepare_inputs_incontext(
        messages,
        images,
        meta_prompt,
        model_name,
        local_image,
        example_images,
        example_responses,
):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for example_image, example_response in zip(
            example_images, example_responses):
        if local_image:
            base64_image = encode_image_from_file(example_image)
        else:
            base64_image = encode_image_from_pil(example_image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

        content = {
            'type': 'text',
            'text': example_response,
        }
        user_content.append(content)

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload


def request_gpt_incontext(
        message,
        images,
        meta_prompt='',
        example_images=None,
        example_responses=None,
        model_name=None,
        local_image=False):

    # TODO(kuan): A better interface should allow interleaving images and
    # messages in the inputs.

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs_incontext(
        message,
        images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
        example_images=example_images,
        example_responses=example_responses)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception:
        print('\nInvalid response: ')
        print(response)
        print('\nInvalid response: ')
        print(response.json())
        exit()

    return res

def get_block_stacking_order(image: Image.Image | NDArray, model_name: str | None = None, debug: bool = False) -> list[int]:
    """Give the VLM's result for the best stacking order of blocks.
    Args:
        - image: Annotated image with blocks 
        - model_name: name of model (leave None for gpt-4o)
    Returns:
        - list of blocks to stack in order
    """
    if type(image) is np.ndarray:
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    if debug:
        image.save("debug_annotated_image.jpg")
    gpt_output = asyncio.run(request_gpt(STACKING_ORDER_PROMPT, [image], model_name=model_name))
    with open("gpt_result.txt", "w", encoding="utf-8") as file:
        file.write(gpt_output)
    pattern = r'\[(\d+)\]'
    first_line = gpt_output.split("\n")[0]
    block_order = re.findall(pattern, first_line)
    out = list(map(int, block_order))
    print(f"Got plan! I will stack (in order): {out}")
    return out

def make_target_image_order(current: Image.Image | NDArray, target: Image.Image | NDArray | str, model_name: str | None = None, debug: bool = False) -> list[int]:
    """Give the VLM's result for the best stacking order of blocks.
    Args:
        - image: Annotated image with blocks 
        - target: unannotated target image (PIL Imge, NDArray or file path)
    """
    if type(current) is np.ndarray:
        current = Image.fromarray(current.astype('uint8'), 'RGB')
    if type(target) is np.ndarray:
        target = Image.fromarray(target.astype('uint8'), 'RGB')
    elif type(target) is str:
        target = Image.open(target)
    
    comb_img = _combine_images(current, target)
    if debug:
        comb_img.save("debug_annotated_image.jpg")
    gpt_output = request_gpt(GOTO_ORDER_PROMPT, [comb_img], model_name=model_name)
    with open("gpt_result_target.txt", "w", encoding="utf-8") as file:
        file.write(gpt_output)
    last_line = gpt_output.split("\n")[-1]
    pattern = r'\[(\d+)\]'
    block_order = re.findall(pattern, last_line)
    out = list(map(int, block_order))
    print(f"Got plan! I will stack (in order): {out}")
    return out
    

def _combine_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
    # Get the dimensions of the images
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a new image with a width that is the sum of both images' widths and the height of the tallest image
    combined_width = width1 + width2
    combined_height = max(height1, height2)
    combined_img = Image.new('RGB', (combined_width, combined_height))

    # Paste the images into the new image
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (width1, 0))

    # Load a font
    # font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    # font = ImageFont.truetype(font_path, size=128)
    font = ImageFont.load_default(128)

    # try:
    #     font = ImageFont.truetype("arial.ttf", 300)
    # except IOError:
    #     font = ImageFont.load_default(size=80)

    # Create a draw object
    draw = ImageDraw.Draw(combined_img)

    # Draw the labels on the images with a background rectangle for visibility
    def draw_label(draw: ImageDraw.ImageDraw, position: Tuple[int, int], text: str) -> None:
        x, y = position
        text_bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(text_bbox, fill="white")
        draw.text((x, y), text, font=font, fill="black")

    draw_label(draw, (10, 0), "1")
    draw_label(draw, (width1 + 10, 0), "2")
    return combined_img

if __name__ == "__main__":
    # debug testing

    curr = Image.open("current_image_example.jpg").convert("RGB")
    target = Image.open("target_image_example.JPG").convert("RGB")
    print(make_target_image_order(curr, target, debug=True))

    
    # print(get_block_stacking_order(np.array(img), debug=True))
    # from raftstereo.zed_stereo import Zed
    
    # while True:
    #     rgb_l,rgb_r,depth = robot.take_img()
    #     rgb_l = cv2.cvtColor(rgb_l,cv2.COLOR_BGR2RGB)
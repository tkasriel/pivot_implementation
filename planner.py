import asyncio
import json
import re
import time
from typing import Any
from matplotlib.pyplot import imread, imsave
import numpy as np
from tqdm import trange
import numpy as np
import sys
from gpt_utils import request_gpt
from PIL import Image
import vip

#TODO: Prompt engineering :)
CORRECTION_PROMPT = """Where should I go to stack the block in the suction such that it's aligned vertically with the tower?"""

class PivotPlanner:
    """VLM planner based on PIVOT (https://arxiv.org/abs/2402.07872)
        Note: most of this code is taken from the huggingface demo: https://huggingface.co/spaces/pivot-prompt/pivot-prompt-demo/tree/main"""
    def __init__(self, origin_point: tuple[int, int] | None = None, model: str = "gpt-4o") -> None:
        """Args:
            - orgin point (x, y): start point of PIVOT arrows
            - model : model to use (gpt-4o)"""
        self.origin_point = origin_point
        self.gpt = None # TODO
    
    def sync_get_arrow_corrections(self, image: np.ndarray, flip_image: bool = True, n_iters: int = 3, n_parallel_trials: int = 1, origin_point: tuple[int, int] | None = None, debug: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
        """Get PIVOT arrow corrections synchronously.
        Warning: only use in a synchronous context. If calling from an async function, use get_arrow_corrections instead
        Args:
            - image: numpy RGB image to annotate
            - flip_image: wehether to flip the image vertically (if you are using a wrist view)
            - n_iters: PIVOT iterations
            - n_parallel_trials: number of times to repeat experiment
            - origin_point: (x,y) coord to start PIVOT arrows
            - debug: whether to save images for debug purposes
        Returns:
            - ((origin_x, origin_y), (end_x, end_y))"""
        return asyncio.run(self.get_arrow_corrections(image, flip_image, n_iters, n_parallel_trials, origin_point, debug))
    
    async def get_arrow_corrections(self, image: np.ndarray, flip_image: bool = True, n_iters: int = 3, n_parallel_trials: int = 1, origin_point: tuple[int, int] | None = None, debug: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
        """Get PIVOT arrow corrections.
        Args:
            - image: numpy RGB image to annotate
            - flip_image: wehether to flip the image vertically (if you are using a wrist view)
            - n_iters: PIVOT iterations
            - n_parallel_trials: number of times to repeat experiment (TODO: with random crops)
            - origin_point: (x,y) coord to start PIVOT arrows
            - debug: whether to save images for debug purposes
        Returns:
            - ((origin_x, origin_y), (end_x, end_y))"""
        if flip_image:
            image = np.flip(image, axis=0)
        if origin_point is None:
            if self.origin_point is None:
                raise ValueError("Origin point must be set for pivot planner, either during instantiation or during the arrow correction call.")
            origin_point = self.origin_point
        img_size = np.min(image.shape[:2])
        radius_per_pixel = 0.05
        style = {
            'num_samples': 12,
            'circle_alpha': 0.6,
            'alpha': 0.8,
            'arrow_alpha': 0.8,
            'radius': int(img_size * radius_per_pixel),
            'thickness': 1,
            'fontsize': int(img_size * radius_per_pixel),
            'rgb_scale': 255,
            'focal_offset': 1,  # camera distance / std of action in z
        }

        action_spec = {
            'loc': [0, 0, 0],
            'scale': [50.0, 100, 100],
            'min_scale': [0.0, 30, 30],
            'min': [0, -300.0, -300],
            'max': [0, 300, 300],
            'action_to_coord': 250,
            'robot': None,
            "arm_coord": origin_point
        }

        return await self._run_vip_parallel(image, CORRECTION_PROMPT, style, action_spec, n_iters=n_iters, n_parallel_trials=n_parallel_trials, debug=debug)

    async def _run_vip_parallel(self,
            im: np.ndarray,
            desc: str,
            style: dict[str, Any],
            action_spec: dict[str, Any],
            n_samples_init=25,
            n_samples_opt=10,
            n_iters=3,
            n_parallel_trials=1,
            debug: bool = False
        ) -> tuple[np.ndarray, tuple[int, int]]: # TODO: Async-ify it. Currently takes no advantage of async
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

        arm_coord = action_spec["arm_coord"]

        todos = []
        for i in range(n_parallel_trials):
            todos.append(self._run_vip_one(im, prompter, desc, style, action_spec, i, n_samples_init, n_samples_opt, n_iters, debug))
        res = await asyncio.gather(*todos)
        center_mean = res[0][0]
        image_circles_marked_np: np.ndarray = res[0][1]
        new_samples = []
        for x in res:
            new_samples += x[1]

        if n_parallel_trials > 1:
            # adjust sample label to avoid duplications
            for sample_id in range(len(new_samples)):
                new_samples[sample_id].label = str(sample_id)
            arrow_ids, _ = await self._vip_perform_selection(
                prompter, im, desc, arm_coord, new_samples, top_n=1
            )

            selected_samples = []
            for selected_id in arrow_ids:
                sample = new_samples[selected_id]
                sample.coord.color = (255, 0, 0)
                selected_samples.append(sample)
            image_circles_marked_np = prompter.add_arrow_overlay_plt(
                im, selected_samples, arm_coord
            )
            if debug:
                imsave(f"/home/gogs/block_stacking/test_outs/pivot/final.jpg", image_circles_marked_np)
            center_mean, _ = prompter.fit(arrow_ids, new_samples)

        
        return (image_circles_marked_np, tuple(np.round(prompter.action_to_coord(center_mean, im, arm_coord).xy, decimals=0)))
    
    async def _run_vip_one(self, 
            im: np.ndarray,
            prompter: vip.VisualIterativePrompter,
            desc: str,
            style: dict[str, Any],
            action_spec: dict[str, Any],
            parallel_ind: int,
            n_samples_init=25,
            n_samples_opt=10,
            n_iters=3,
            debug: bool = False,
        ) -> tuple[list[float], list[vip.Sample]]:
        arm_coord = action_spec["arm_coord"]
        center_mean = action_spec["loc"]
        center_std = action_spec["scale"]
        for itr in trange(n_iters):
            if itr == 0:
                style["num_samples"] = n_samples_init
            else:
                style["num_samples"] = n_samples_opt
            samples = prompter.sample_actions(im, arm_coord, center_mean, center_std)
            arrow_ids, image_circles_np = await self._vip_perform_selection(
                prompter, im, desc, arm_coord, samples, top_n=3
            )

            # plot sampled circles as red
            selected_samples = []
            for selected_id in arrow_ids:

                sample = samples[selected_id]
                sample.coord.color = (255, 0, 0) # type: ignore
                selected_samples.append(sample)
            image_circles_marked_np = prompter.add_arrow_overlay_plt(
                image_circles_np, selected_samples, arm_coord
            )
            if debug:
                # print(f"{parallel_ind}{itr}")
                imsave(f"/home/gogs/block_stacking/test_outs/pivot/{parallel_ind}_{itr}.jpg", image_circles_marked_np)

            # if at last iteration, pick one answer out of the selected ones
            if itr == n_iters - 1:
                arrow_ids, _ = await self._vip_perform_selection(
                    prompter, im, desc, arm_coord, selected_samples, top_n=1
                )
                

                selected_samples = []
                for selected_id in arrow_ids:
                    found_index = 0
                    for i, s in enumerate(samples):
                        if int(s.label) == selected_id:
                            found_index = i
                    sample = samples[found_index]
                    sample.coord.color = (255, 0, 0) # type: ignore
                    selected_samples.append(sample)
                image_circles_marked_np = prompter.add_arrow_overlay_plt(
                    im, selected_samples, arm_coord
                )
                if debug:
                    # print(f"{parallel_ind}_final")
                    imsave(f"/home/gogs/block_stacking/test_outs/pivot/{parallel_ind}_final.jpg", image_circles_marked_np)
            center_mean, center_std = prompter.fit(arrow_ids, samples)
        # print(center_mean, selected_samples)
        return (center_mean, selected_samples)

    async def _vip_perform_selection(self, prompter: vip.VisualIterativePrompter, im: np.ndarray, desc: str, arm_coord: tuple[int, int], samples: list[vip.Sample], top_n: int) -> tuple[list[int], np.ndarray]:
        """Perform one selection pass given samples."""
        # start_time = time.time()
        image_circles_np = Image.fromarray(prompter.add_arrow_overlay_plt(
            image=im, samples=samples, arm_xy=arm_coord
        
        ))
        # print(f"Arrow overlay took: {time.time() - start_time} seconds")

        # _, encoded_image_circles = cv2.imencode(".png", image_circles_np)

        # start_time = time.time()
        response = await request_gpt(self._make_prompt(desc, top_n=top_n), image_circles_np)
        # print(f"GPT took: {time.time() - start_time} seconds")
        
        try:
            arrow_ids = list(map(int, self._extract_json(response, "points")))
        except Exception as e:
            print(e)
            arrow_ids = []
        for arrow_id in arrow_ids:
            if arrow_id >= len(samples):
                pass
            #     print(arrow_ids)
            #     with open("test.txt", "w") as file:
            #         file.write(response + "\n" + str(arrow_ids))
            #     sys.exit(0)
        return arrow_ids, np.array(image_circles_np)

    def _extract_json(self, response: str, key: str) -> list[int]:
        json_part = re.search(r"\{.*\}", response, re.DOTALL)
        parsed_json = {}
        if json_part:
            json_data = json_part.group()
            # Parse the JSON data
            parsed_json = json.loads(json_data)
        else:
            print("No JSON data found ******\n", response)
        return parsed_json[key]

    def _make_prompt(self, message: str, top_n: int) -> str:
        return f"""
INSTRUCTIONS:
You are tasked to locate an object, region, or point in space in the given annotated image according to a description.
The image is annoated with numbered circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what the description is describing in the image.
You are a five-time world champion in this game. 
Give a one sentence analysis of why you chose those points.
Provide your answer at the end in a valid JSON of this format:

{{"points": []}}

DESCRIPTION: {message}
IMAGE:
""".strip()

if __name__ == "__main__":
    # Debugging
    img_file = "/home/gogs/block_stacking/test_outs/annotated.jpg" # currently an image of a pear
    CORRECTION_PROMPT = """I want to move the pear to the left."""    
    pp = PivotPlanner((500, 500))
    image = imread(img_file)
    print(np.amax(image))

    print(asyncio.run(pp.get_arrow_corrections(image, n_iters=2, n_parallel_trials=3, debug=True)))

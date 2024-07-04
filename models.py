from typing import TypedDict
from pydantic import BaseModel


class StyleModel(TypedDict):
    num_samples: int = 12
    circle_alpha: int = 0.6
    alpha: int = 0.8
    arrow_alpha: int = 0.0
    radius: int
    thickness: int = 2
    fontsize: int
    rgb_scale: int = 255
    focal_offset: int = 1

class ActionModel(TypedDict):
    loc: tuple[int, int, int]
    scale: tuple[float, float, float]
    min_scale: tuple[float, float, float]
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    arm_coord: tuple[int, int]
    action_to_coord: int
    robot: None
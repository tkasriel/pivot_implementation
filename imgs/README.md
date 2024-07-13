# Tower Dataset

A bunch of towers for testing.
Each metadata entry follows one of the two following pydantic validators:
```py
class ImageMeta(BaseModel):
	block_center: list[int]
	is_correct: bool
	target_direction: Optional[float] = None

	has_suction: bool
	suction_center: Optional[list[int]] = None

	rotate: bool
	view: Literal["top", "side", "wrist"]

class DeleteImage(BaseModel):
	to_delete: Literal[True]
```

You can ignore those who have "to_delete". They were not uploaded to the drive.
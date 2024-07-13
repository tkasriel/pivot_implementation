import glob
import math
import os, sys
from typing import Any, Literal, Optional
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from pydantic import BaseModel, ValidationError

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


# goal json:
"""
image_filename: {
	block_center: [x, y]
	
	is_correct: bool
	target_direction: float if is_correct

	has_suction: bool
	suction_center: [x, y] if not has_suction

	rotate: bool
	view: Enum["top", "side", "wrist"]
}

"""

VALUE_COUNT = 8 # view, block_center_x, block_center_y, suction_x, suction_y, target_x, target_y, rotate

# print(glob.glob("imgs/*.JPG"))
imgs = map(lambda x: os.path.basename(x), glob.glob("imgs/*.JPG") + glob.glob("imgs/*.jpg") + glob.glob("imgs/*.png"))
fig = plt.figure()
coords: list[Any] = []
file = open("out.csv", "w")

img_file = ""
ignore_files = []

img_file = next(imgs)
print("Current file: " + img_file)
img = mpimg.imread(f"imgs/{img_file}")
imgplot = plt.imshow(img)
imgplot.set_extent((0, img.shape[1], 0, img.shape[0]))

def load_json() -> dict[str, Any]:
	with open("imgs/metadata.json", "r") as file:
		dict_json: dict[str, Any] = json.load(file)
	for key in list(dict_json):
		try:
			ImageMeta.model_validate(dict_json[key])
		except ValidationError as e:
			# print(e)
			try: 
				DeleteImage.model_validate(dict_json[key])
				ignore_files.append(key)
			except ValidationError:
				dict_json.pop(key, None)
				print(f"ignoring {key} from loading")
	return dict_json

out_dict: dict[str, Any] = load_json()

def onclick(event):
	global coords, img_file
	ix, iy = int(event.xdata), img.shape[0] - int(event.ydata)
	if len(coords) == 1 or len(coords) == 3 or len(coords) == 5:
		coords.extend([ix, iy])
	check_next_img()

def on_press(event):
	global coords, img_file
	if event.key == " " and len(coords) == 3 or len(coords) == 5:
			coords.extend([None, None])
	elif event.key == "t" and len(coords) == 0:
		coords.append("top")
	elif event.key == "i" and len(coords) == 0:
		coords.append("side")
	elif event.key == "w" and len(coords) == 0:
		coords.extend(["wrist", 686, 269, 652, 224])
	elif event.key == "y" and len(coords) == 7:
		coords.append(True)
	elif event.key == "n" and len(coords) == 7:
		coords.append(False)
	elif event.key == "d":
		print("Image marked as to be deleted")
		out_dict[img_file] = {"to_delete": True}
		next_img()
		return
	else:
		print(f"Error. got incorrect key: {event.key} w/ length {len(coords)}")
	check_next_img()

def check_next_img():
	if len(coords) == 1:
		print("Select block pos")
	elif len(coords) == 3:
		print("Select suction coords. If there is no suction, press space")
	elif len(coords) == 5:
		print("Select target coords. If the block is correct, press space")
	elif len(coords) == 7:
		print("Does this image require rotation? (y)es, (n)o")
	elif len(coords) == 8:
		end_img()
	else:
		print(f"got weird number of vals: {len(coords)}")
	# print(coords)
	# print(len(coords))

def end_img():
	global coords, img_file, img
	save_img()
	next_img()

def save_img():
	global coords, img_file
	view = coords[0]
	bcenter = (coords[1], coords[2])
	suction = (coords[3], coords[4])
	target = (coords[5], coords[6])
	rotate = coords[7]
	new_name_base = f"{view}_{"suction" if suction[0] else "no_suction"}_{"translate" if target[0] else "no_translate"}_{"rotate" if rotate else "no_rotate"}"
	new_name = new_name_base
	i = 2
	while os.path.exists(f"imgs/{new_name}.jpg"):
		new_name = new_name_base + str(i)
		i += 1
	os.rename(f"imgs/{img_file}", f"imgs/{new_name}.jpg")
	img_file = new_name + ".jpg"
	out_dict[img_file] = {
		"block_center": bcenter,
		"rotate": rotate,
		"view": view
	}
	if target[0]:
		out_dict[img_file]["is_correct"] = False
		out_dict[img_file]["target_direction"] = math.atan2(target[1]-bcenter[1], target[0]-bcenter[0])
	else:
		out_dict[img_file]["is_correct"] = True
	if coords[4]:
		out_dict[img_file]["has_suction"] = True
		out_dict[img_file]["suction_center"] = suction
	else:
		out_dict[img_file]["has_suction"] = False
	print("image saved")

def next_img():
	global img_file, out_dict, img, coords
	print("Choose view: (t)op, s(i)de, (w)rist")
	
	with open("imgs/metadata.json", "w") as file:
		json.dump(out_dict, file)
	while (img_file in out_dict.keys() or img_file in ignore_files) and img_file:
		img_file = next(imgs, None)
	if not img_file:
		sys.exit(0)
	print(f"Current image: {img_file}")
	img = mpimg.imread(f"imgs/{img_file}")
	imgplot.set_extent((0, img.shape[1], 0, img.shape[0]))
	imgplot.set_data(img)
	coords = []
	plt.draw()


# file.write("img_filename,blockx,blocky,targetx,targety,suctionx,suctiony\n")
# next_img()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_press)
print("Choose view: (t)op, s(i)de, (w)rist")
plt.show()


# with open("out.csv", "w") as file:
# 	for img_file in imgs:
# 		img = mpimg.imread(f"imgs/{img_file}")
# 		imgplot = plt.imshow(img)
		
# 		bx, by, sx, sy = input("bx by sx sy: ").split()
# 		file.write(",".join((img_file, bx, by, sx, sy)))


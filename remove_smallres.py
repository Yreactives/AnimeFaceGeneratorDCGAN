from PIL import Image
import os

directory = "venv/images"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        with Image.open(f) as im:
            x, y = im.size
        if x < 64 or y < 64:
            os.remove(f)


from pathlib import Path
from typing import List


path = Path("./images")
old_images = [] #type:List[Path]
for image in path.glob("*"):
    if image.suffix != "jpg":
        old_images.append(image.absolute())


for image in old_images:
    image.rename(image.with_suffix(".jpg"))

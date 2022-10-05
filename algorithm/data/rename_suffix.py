from pathlib import Path
from typing import List


path = Path("./FaceMaskDataset/train")
old_images = [] #type:List[Path]
for image in path.glob("*"):
    if image.suffix == ".png":
        old_images.append(image.absolute())


for image in old_images:
    image.rename(image.with_suffix(".jpg"))

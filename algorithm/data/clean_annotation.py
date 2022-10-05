from pathlib import Path
from PIL import Image
from os import remove
from algorithm.utils.annotation_utils import VocAnnotationReader
from xml.etree import ElementTree as ET

reader = VocAnnotationReader({"face": 0, "face_mask": 1}, True)
path = Path("./algorithm/data/FaceMaskDataset/train/Annotations")
empties = []

for anno_path in path.glob("*"):
    tree = ET.parse(anno_path)
    root = tree.getroot()
    img_size = root.find('size')
    image_path = anno_path.parent.parent / "JPEGImages"/(anno_path.stem+".jpg")

    if not img_size:
        image = Image.open(image_path)
        size = ET.Element("size")
        width = ET.Element("width")
        width.text = str(image.width)
        height = ET.Element("height")
        height.text = str(image.height)
        size.append(width)
        size.append(height)
        root.append(size)
        tree.write(anno_path, encoding="utf-8")
    else:
        w = int(img_size.find("width").text)
        h = int(img_size.find("height").text)
        if not w or not h:
            image = Image.open(image_path)
            img_size.find("width").text = str(image.width)
            img_size.find("height").text = str(image.height)
            tree.write(anno_path, encoding="utf-8")

    target = reader.read(anno_path)
    if not target:
        empties.append((anno_path, image_path))


# 删除空数据
for anno_path, image_path in empties:
    # remove(anno_path)
    # remove(image_path)
    print(anno_path, image_path)

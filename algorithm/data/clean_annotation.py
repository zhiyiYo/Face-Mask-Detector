from pathlib import Path
from PIL import Image
from xml.etree import ElementTree as ET


path = Path("./algorithm/data/FaceMaskDataset/val/Annotations")
for anno_path in path.glob("*"):
    tree = ET.parse(anno_path)
    root = tree.getroot()
    img_size = root.find('size')
    if img_size:
        continue

    image_path = anno_path.parent.parent/"JPEGImages"/(anno_path.stem+".jpg")
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

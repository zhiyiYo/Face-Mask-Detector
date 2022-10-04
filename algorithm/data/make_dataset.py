from pathlib import Path


image_names = []
for image_path in Path("images").glob("*.jpg"):
    image_names.append(image_path.stem+"\n")

with open("all.txt", "w", encoding="utf-8") as f:
    f.writelines(image_names)
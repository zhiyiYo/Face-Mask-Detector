from pathlib import Path

root = Path("./FaceMaskDataset/train/Annotations")
names = []
for path in root.glob("*"):
    names.append(path.stem+"\n")

with open(root.parent/"ImageSets/Main/all.txt", "w", encoding="utf-8") as f:
    f.writelines(names)

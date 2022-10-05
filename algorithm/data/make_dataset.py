from pathlib import Path
import shutil


files = list(Path("./FaceMaskDataset/val").glob("*"))
for i, path in enumerate(files):
    print(f"\r当前进度：{i}/{len(files)}", end="")
    if path.suffix == ".jpg":
        shutil.move(str(path.absolute()), path.parent/"JPEGImages"/path.name)
    elif path.suffix == ".xml":
        shutil.move(str(path.absolute()), path.parent/"Annotations"/path.name)

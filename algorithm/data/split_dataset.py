# coding: utf-8
from pathlib import Path
from sklearn.model_selection import train_test_split


def write_data(file_path: str, data: list):
    with open(file_path, 'w') as f:
        for i in data:
            f.write(i+'\n')


root = Path('./images')
images = []

for image in root.glob('*.jpg'):
    images.append(image.stem)

# train:val:test = 6:2:2
image_trainval, image_test = train_test_split(images, test_size=0.2)
image_train, image_val = train_test_split(image_trainval, test_size=0.25)

# write_data('all.txt', images)
write_data('trainval.txt', image_trainval)
write_data('train.txt', image_train)
write_data('val.txt', image_val)
write_data('test.txt', image_test)

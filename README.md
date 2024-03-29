# Face-Mask-Detector
A face mask detector based on STM32F103ZET6 and Yolov4.


## Interface
![app](./doc/image/screenshot.png)


## Compile and Load
You should install `arm-none-eabi-gcc` to compile this project.

```sh
cd stm32
make update
```


## Build Environment
1. Create virtual environment:

    ```shell
    conda create -n Face_Mask_Detector python=3.8
    conda activate Face_Mask_Detector
    pip install -r requirements.txt
    ```

2. Install [PyTorch](https://pytorch.org/), refer to the [blog](https://blog.csdn.net/qq_23013309/article/details/103965619) for details.


## Train
1. Download face mask dataset from [kaggle](https://www.kaggle.com/datasets/zhiyiyo/face-mask-dataset) and unzip it.

2. Download pre-trained `CSPDarknet53.pth` model from [Google Drive](https://drive.google.com/file/d/12oV8QL937S1JWFQhzLNPoqyYc_bi0lWT/view?usp=sharing).

3. Modify the value of `root` in `train.py`, please ensure that the directory structure of the `root` folder is as follows:

    ```txt
    root
    ├───Annotations
    ├───ImageSets
    │   ├───Layout
    │   ├───Main
    │   └───Segmentation
    ├───JPEGImages
    ├───SegmentationClass
    └───SegmentationObject
    ```

4. start training:

    ```sh
    conda activate Face_Mask_Detector
    python train.py
    ```

## Evaluation
### one model
1. Modify the value of `root` and `model_path` in `eval.py`.
2. Calculate mAP:

    ```sh
    conda activate Face_Mask_Detector
    python eval.py
    ```

### multi models
1. Modify the value of `root` and `model_dir` in `evals.py`.
2. Calculate and plot mAP:

    ```shell
    conda activate Face_Mask_Detector
    python evals.py
    ```

### mAP curve
![map curve](./doc/image/mAP_%E6%9B%B2%E7%BA%BF.png)


## Detection
1. Modify the `model_path` and `image_path` in `demo.py`.

2. Display detection results:

    ```shell
    conda activate Face_Mask_Detector
    python demo.py
    ```



## License
Face-Mask-Detector is licensed under [GPLv3](./LICENSE).

Copyright © 2021 by zhiyiYo.
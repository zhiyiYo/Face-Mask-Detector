# coding:utf-8
import glob
from xml.etree import ElementTree as ET
from random import choice

import numpy as np

from utils.box_utils import jaccard_overlap_numpy as iou


class AnchorKmeans:
    """ 先验框聚类 """

    def __init__(self, annotation_dir: str):
        self.annotation_dir = annotation_dir
        self.bbox = self.get_bbox()

    def get_bbox(self) -> np.ndarray:
        """ 获取所有的边界框 """
        bbox = []

        for path in glob.glob(f'{self.annotation_dir}/*xml'):
            root = ET.parse(path).getroot()

            # 图像的宽度和高度
            w = int(root.find('size/width').text)
            h = int(root.find('size/height').text)

            if w==0:
                print(path)

            # 获取所有边界框
            for obj in root.iter('object'):
                box = obj.find('bndbox')

                # 归一化坐标
                xmin = int(box.find('xmin').text)/w
                ymin = int(box.find('ymin').text)/h
                xmax = int(box.find('xmax').text)/w
                ymax = int(box.find('ymax').text)/h

                bbox.append([0, 0, xmax-xmin, ymax-ymin])

        return np.array(bbox)

    def get_cluster(self, n_clusters=9, metric=np.median):
        """ 获取聚类结果

        Parameters
        ----------
        n_clusters: int
            聚类数

        metric: callable
            选取聚类中心点的方式
        """
        rows = self.bbox.shape[0]

        if rows < n_clusters:
            raise ValueError("n_clusters 不能大于边界框样本数")

        last_clusters = np.zeros(rows)
        clusters = np.ones((n_clusters, 2))
        distances = np.zeros((rows, n_clusters))  # type:np.ndarray

        # 随机选取出几个点作为聚类中心
        np.random.seed(1)
        clusters = self.bbox[np.random.choice(rows, n_clusters, replace=False)]

        # 开始聚类
        while True:
            # 计算距离
            distances = 1-self.iou(clusters)

            # 将每一个边界框划到一个聚类中
            nearest_clusters = distances.argmin(axis=1)

            # 如果聚类中心不再变化就退出
            if np.array_equal(nearest_clusters, last_clusters):
                break

            # 重新选取聚类中心
            for i in range(n_clusters):
                clusters[i] = metric(self.bbox[nearest_clusters == i], axis=0)

            last_clusters = nearest_clusters

        return clusters[:, 2:]

    def average_iou(self, clusters: np.ndarray):
        """ 计算 IOU 均值

        Parameters
        ----------
        clusters: `~np.ndarray` of shape `(n_clusters, 2)`
            聚类中心
        """
        clusters = np.hstack((np.zeros((clusters.shape[0], 2)), clusters))
        return np.mean([np.max(iou(bbox, clusters)) for bbox in self.bbox])

    def iou(self, clusters: np.ndarray):
        """ 计算所有边界框和所有聚类中心的交并比

        Parameters
        ----------
        clusters: `~np.ndarray` of shape `(n_clusters, 4)`
            聚类中心

        Returns
        -------
        iou: `~np.ndarray` of shape `(n_bbox, n_clusters)`
            交并比
        """
        bbox = self.bbox
        A = self.bbox.shape[0]
        B = clusters.shape[0]

        xy_max = np.minimum(bbox[:, np.newaxis, 2:].repeat(B, axis=1),
                            np.broadcast_to(clusters[:, 2:], (A, B, 2)))
        xy_min = np.maximum(bbox[:, np.newaxis, :2].repeat(B, axis=1),
                            np.broadcast_to(clusters[:, :2], (A, B, 2)))

        # 计算交集面积
        inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
        inter = inter[:, :, 0]*inter[:, :, 1]

        # 计算每个矩阵的面积
        area_bbox = ((bbox[:, 2]-bbox[:, 0])*(bbox[:, 3] -
                     bbox[:, 1]))[:, np.newaxis].repeat(B, axis=1)
        area_clusters = ((clusters[:, 2] - clusters[:, 0])*(
            clusters[:, 3] - clusters[:, 1]))[np.newaxis, :].repeat(A, axis=0)

        return inter/(area_bbox+area_clusters-inter)


if __name__ == '__main__':
    root = 'data/FaceMaskDataset/train/Annotations'
    model = AnchorKmeans(root)
    clusters = model.get_cluster(9)
    clusters = np.array(sorted(clusters.tolist(), key=lambda i: i[0]*i[1], reverse=True))

    # 将先验框还原为原本的大小
    print('聚类结果:\n', (clusters*416).astype(int))
    print('平均 IOU:', model.average_iou(clusters))

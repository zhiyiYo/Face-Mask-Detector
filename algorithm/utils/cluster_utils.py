# coding:utf-8
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def iou(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和其他边界框的交并比
    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框
    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    # 计算 iou
    iou = inter/(area_box+area_boxes-inter)  # type: np.ndarray
    return iou


class AnchorKmeans:
    """ 先验框聚类 """

    def __init__(self, annotation_dir: str):
        self.annotation_dir = Path(annotation_dir)
        if not self.annotation_dir.exists():
            raise ValueError(f'标签文件夹 `{annotation_dir}` 不存在')

        self.bbox = self.get_bbox()

    def get_bbox(self) -> np.ndarray:
        """ 获取所有的边界框 """
        bbox = []

        for path in self.annotation_dir.glob('*.txt'):
            lines = path.read_text("utf-8").split("\n")
            for line in lines:
                if not line.strip():
                    continue

                c, cx, cy, w, h = line.strip().split()
                bbox.append([0, 0, float(w), float(h)])

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
    # 标签文件夹
    root = 'data/labels'
    model = AnchorKmeans(root)
    clusters = model.get_cluster(9)
    clusters = np.array(sorted(clusters.tolist(), key=lambda i: i[0]*i[1]))

    # 将先验框还原为原本的大小
    print('聚类结果:\n', (clusters*416).astype(int))
    print('平均 IOU:', model.average_iou(clusters))

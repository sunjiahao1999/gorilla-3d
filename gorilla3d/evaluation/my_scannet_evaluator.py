# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import os.path as osp
from typing import List, Union

import numpy as np

import gorilla
from gorilla.evaluation import DatasetEvaluators

from .pattern import SemanticEvaluator, InstanceEvaluator

CLASS_LABELS = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]
CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]


class MyScanNetSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 class_labels: List[str] = CLASS_LABELS,
                 class_ids: Union[np.ndarray, List[int]] = CLASS_IDS,
                 **kwargs):
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         **kwargs)

    def reset(self):
        max_id = self.class_ids.max() + 1
        self.confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.int64)
        self.miou = np.array([0.],dtype=np.float32)

    def process(self, labels, outputs):
        """
        Args:
            labels: batches semantic gt (B*N)
            outputs: batches semantic pred (B*N)
        """
        if not isinstance(labels, List):
            labels = [labels]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for label, output in zip(labels, outputs):
            semantic_gt = label.cpu().clone().numpy()
            semantic_gt = self.class_ids[semantic_gt]
            semantic_pred = output.cpu().clone().numpy()
            semantic_pred = self.class_ids[semantic_pred]
            self.fill_confusion(semantic_pred, semantic_gt)

    def print_result(self):
        # calculate ious
        tp, fp, fn, intersection, union = self.prase_iou()
        ious = (intersection / union) * 100

        # build IoU table
        haeders = ["class", "IoU", "TP/(TP+FP+FN)"]
        results = []
        self.logger.info("Evaluation results for semantic segmentation:")

        max_length = max(15, max(map(lambda x: len(x), self.class_labels)))
        for class_id in self.include:
            results.append(
                (self.id_to_label[class_id].ljust(max_length,
                                                  " "), ious[class_id],
                 f"({intersection[class_id]:>6d}/{union[class_id]:<6d})"))

        acc_table = gorilla.table(results, headers=haeders, stralign="left")
        for line in acc_table.split("\n"):
            self.logger.info(line)
        self.logger.info(f"mean: {np.nanmean(ious[self.include]):.1f}")
        self.logger.info("")
        return np.nanmean(ious[self.include])

    def evaluate(self):
        # print semantic segmentation result(IoU)
        self.miou = self.print_result()

        # return miou
        return self.miou

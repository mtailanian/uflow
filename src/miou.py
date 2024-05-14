from abc import ABC
import numpy as np
from typing import Any, Callable, Optional, Union, List

import torch
from torch import Tensor
from torchmetrics import Metric

EPS = np.finfo(float).eps


class mIoU(Metric, ABC):
    """
    Computes intersection over union metric (or Jaccard Index), for different thresholds
    J(A, B) = (A \cap B) / (A \cup B), for each image independently, and then takes the
    average for all images for each threshold
    """
    full_state_update: bool = False

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        thresholds: Optional[Union[float, List]] = None
    ):
        super(mIoU, self).__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )
        if type(thresholds) is not list:
            thresholds = [thresholds]
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)

        self.add_state("intersection", default=[])
        self.add_state("union", default=[])

    def update(self, preds: Tensor, target: Tensor):
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)

        preds, target = preds.view(preds.shape[0], -1), target.view(target.shape[0], -1)

        intersections, unions = self.compute_intersection_and_union(preds, target)
        self.intersection.append(intersections)
        self.union.append(unions)

    def compute_intersection_and_union(self, detection: np.array, labels: np.array):
        intersections, unions = [], []
        labels_thr = (torch.max(labels) - torch.min(labels)) / 2
        for thr in self.thresholds:
            pred, target = detection > thr, labels > labels_thr
            intersections.append(torch.sum(pred & target, dim=1))
            unions.append(torch.sum(pred | target, dim=1))
        return torch.stack(intersections).T, torch.stack(unions).T

    def compute(self):
        intersection = torch.cat(self.intersection)
        union = torch.cat(self.union)
        return torch.mean((intersection + EPS) / (union + EPS), dim=0)

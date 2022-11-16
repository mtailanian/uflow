from abc import ABC
import numpy as np
from typing import Any, Callable, Optional, Union, List

import torch
from torch import Tensor
from torchmetrics import Metric

EPS = np.finfo(float).eps


class IoU(Metric, ABC):
    """
    Computes intersection over union metric (or Jaccard Index), for different thresholds
    J(A, B) = (A \cap B) / (A \cup B)
    """
    full_state_update: bool = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        thresholds: Optional[Union[float, List]] = None
    ):
        super(IoU, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if type(thresholds) is not list:
            thresholds = [thresholds]
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)

        self.add_state("intersection", default=[])
        self.add_state("union", default=[])

    def update(self, preds: Tensor, target: Tensor):
        preds, target = preds.flatten(), target.flatten()

        intersections, unions = self.compute_intersection_and_union(preds, target)
        self.intersection.append(intersections)
        self.union.append(unions)

    def compute_intersection_and_union(self, detection: np.array, labels: np.array):
        intersections, unions = [], []
        labels_thr = (torch.max(labels) - torch.min(labels)) / 2
        for thr in self.thresholds:
            pred, target = detection > thr, labels > labels_thr
            intersections.append(torch.sum(pred & target))
            unions.append(torch.sum(pred | target))
        return torch.tensor(intersections), torch.tensor(unions)

    def compute(self):
        self.intersection = torch.stack(self.intersection)
        self.union = torch.stack(self.union)
        return torch.mean((self.intersection + EPS) / (self.union + EPS), dim=0)

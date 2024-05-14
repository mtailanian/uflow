"""
This implementation is taken from Anomalib: https://github.com/openvinotoolkit/anomalib

Implementation of AUPRO score based on TorchMetrics.

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""

from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import cv2
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import roc
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat
import torch.nn.functional as F


class AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric."""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        fpr_limit: float = 0.3,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("target", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def _compute(self) -> Tuple[Tensor, Tensor]:
        """Compute the pro/fpr value-pairs until the fpr specified by self.fpr_limit.

        It leverages the fact that the overlap corresponds to the tpr, and thus computes the overall
        PRO curve by aggregating per-region tpr/fpr values produced by ROC-construction.

        Raises:
            ValueError: ValueError is raised if self.target doesn't conform with requirements imposed by kornia for
                        connected component analysis.

        Returns:
            Tuple[Tensor, Tensor]: tuple containing final fpr and tpr values.
        """
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                (
                    f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                    f"interval was [{target.min()}, {target.max()}]."
                )
            )
        target = target.unsqueeze(1)  # kornia expects N1HW format
        target = target.type(torch.float)  # kornia expects FloatTensor
        if target.is_cuda:
            cca = connected_components_gpu(target)
        else:
            cca = connected_components_cpu(target)

        preds = preds.flatten()
        cca = cca.flatten()
        target = target.flatten()

        # compute the global fpr-size
        fpr: Tensor = roc(preds, target.type(torch.int), "binary")[0]  # only need fpr
        output_size = torch.where(fpr <= self.fpr_limit)[0].size(0)

        # compute the PRO curve by aggregating per-region tpr/fpr curves/values.
        tpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        fpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        new_idx = torch.arange(0, output_size, device=preds.device, dtype=torch.float)

        # Loop over the labels, computing per-region tpr/fpr curves, and aggregating them.
        # Note that, since the groundtruth is different for every all to `roc`, we also get
        # different/unique tpr/fpr curves (i.e. len(_fpr_idx) is different for every call).
        # We therefore need to resample per-region curves to a fixed sampling ratio (defined above).
        labels = cca.unique()[1:]  # 0 is background
        background = cca == 0
        _fpr: Tensor
        _tpr: Tensor
        for label in labels:
            interp: bool = False
            new_idx[-1] = output_size - 1
            mask = cca == label
            # Need to calculate label-wise roc on union of background & mask, as otherwise we wrongly consider other
            # label in labels as FPs. We also don't need to return the thresholds
            _fpr, _tpr = roc(preds[background | mask], mask[background | mask], "binary")[:-1]

            # catch edge-case where ROC only has fpr vals > self.fpr_limit
            if _fpr[_fpr <= self.fpr_limit].max() == 0:
                _fpr_limit = _fpr[_fpr > self.fpr_limit].min()
            else:
                _fpr_limit = self.fpr_limit

            _fpr_idx = torch.where(_fpr <= _fpr_limit)[0]
            # if computed roc curve is not specified sufficiently close to self.fpr_limit,
            # we include the closest higher tpr/fpr pair and linearly interpolate the tpr/fpr point at self.fpr_limit
            if not torch.allclose(_fpr[_fpr_idx].max(), self.fpr_limit):
                _tmp_idx = torch.searchsorted(_fpr, self.fpr_limit)
                _fpr_idx = torch.cat([_fpr_idx, _tmp_idx.unsqueeze_(0)])
                _slope = 1 - ((_fpr[_tmp_idx] - self.fpr_limit) / (_fpr[_tmp_idx] - _fpr[_tmp_idx - 1]))
                interp = True

            _fpr = _fpr[_fpr_idx]
            _tpr = _tpr[_fpr_idx]

            _fpr_idx = _fpr_idx.float()
            _fpr_idx /= _fpr_idx.max()
            _fpr_idx *= new_idx.max()

            if interp:
                # last point will be sampled at self.fpr_limit
                new_idx[-1] = _fpr_idx[-2] + ((_fpr_idx[-1] - _fpr_idx[-2]) * _slope)

            _tpr = self.interp1d(_fpr_idx, _tpr, new_idx)
            _fpr = self.interp1d(_fpr_idx, _fpr, new_idx)
            tpr += _tpr
            fpr += _fpr

        # Actually perform the averaging
        tpr /= labels.size(0)
        fpr /= labels.size(0)
        return fpr, tpr

    def compute(self) -> Tensor:
        """Fist compute PRO curve, then compute and scale area under the curve.

        Returns:
            Tensor: Value of the AUPRO metric
        """
        fpr, tpr = self._compute()

        aupro = auc(fpr, tpr)
        aupro = aupro / fpr[-1]  # normalize the area

        return aupro

    def generate_figure(self) -> Tuple[Figure, str]:
        """Generate a figure containing the PRO curve and the AUPRO.

        Returns:
            Tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """
        fpr, tpr = self._compute()
        aupro = self.compute()

        xlim = (0.0, self.fpr_limit.detach_().cpu().numpy())
        ylim = (0.0, 1.0)
        xlabel = "Global FPR"
        ylabel = "Averaged Per-Region TPR"
        loc = "lower right"
        title = "PRO"

        fig, _axis = plot_figure(fpr, tpr, aupro, xlim, ylim, xlabel, ylabel, loc, title)

        return fig, "PRO"

    @staticmethod
    def interp1d(old_x: Tensor, old_y: Tensor, new_x: Tensor) -> Tensor:
        """Function to interpolate a 1D signal linearly to new sampling points.

        Args:
            old_x (Tensor): original 1-D x values (same size as y)
            old_y (Tensor): original 1-D y values (same size as x)
            new_x (Tensor): x-values where y should be interpolated at

        Returns:
            Tensor: y-values at corresponding new_x values.
        """

        # Compute slope
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

        # Prepare idx for linear interpolation
        idx = torch.searchsorted(old_x, new_x)

        # searchsorted looks for the index where the values must be inserted
        # to preserve order, but we actually want the preceeding index.
        idx -= 1
        # we clamp the index, because the number of intervals = old_x.size(0) -1,
        # and the left neighbour should hence be at most number of intervals -1, i.e. old_x.size(0) - 2
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)

        # perform actual linear interpolation
        y_new = old_y[idx] + slope[idx] * (new_x - old_x[idx])

        return y_new


def connected_components_gpu(binary_input: Tensor, num_iterations: int = 1000) -> Tensor:
    """Perform connected component labeling on GPU and remap the labels from 0 to N.

    Args:
        binary_input (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)
        num_iterations (int): Number of iterations used in the connected component computation.

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = connected_components(binary_input, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()


def connected_components_cpu(image: Tensor) -> Tensor:
    """Connected component labeling on CPU.

    Args:
        image (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = torch.zeros_like(image)
    label_idx = 1
    for i, mask in enumerate(image):
        mask = mask.squeeze().numpy().astype(np.uint8)
        _, comps = cv2.connectedComponents(mask)
        # remap component values to make sure every component has a unique value when outputs are concatenated
        for label in np.unique(comps)[1:]:
            components[i, 0, ...][np.where(comps == label)] = label_idx
            label_idx += 1
    return components.int()



def connected_components(image: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. warning::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       connected_components.html>`__.

    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _ = image_view.shape
    out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out.view_as(image)


def plot_figure(
    x_vals: Tensor,
    y_vals: Tensor,
    auc: Tensor,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    loc: str,
    title: str,
    sample_points: int = 1000,
) -> Tuple[Figure, Axis]:
    """Generate a simple, ROC-style plot, where x_vals is plotted against y_vals.

    Note that a subsampling is applied if > sample_points are present in x/y, as matplotlib plotting draws
    every single plot which takes very long, especially for high-resolution segmentations.

    Args:
        x_vals (Tensor): x values to plot
        y_vals (Tensor): y values to plot
        auc (Tensor): normalized area under the curve spanned by x_vals, y_vals
        xlim (Tuple[float, float]): displayed range for x-axis
        ylim (Tuple[float, float]): displayed range for y-axis
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        loc (str): string-based legend location, for details see
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): title of the plot
        sample_points (int): number of sampling points to subsample x_vals/y_vals with

    Returns:
        Tuple[Figure, Axis]: Figure and the contained Axis
    """
    fig, axis = plt.subplots()

    x_vals = x_vals.detach().cpu()
    y_vals = y_vals.detach().cpu()

    if sample_points < x_vals.size(0):
        possible_idx = range(x_vals.size(0))
        interval = len(possible_idx) // sample_points

        idx = [0]  # make sure to start at first point
        idx.extend(possible_idx[::interval])
        idx.append(possible_idx[-1])  # also include last point

        idx = torch.tensor(
            idx,
            device=x_vals.device,
        )
        x_vals = torch.index_select(x_vals, 0, idx)
        y_vals = torch.index_select(y_vals, 0, idx)

    axis.plot(
        x_vals,
        y_vals,
        color="darkorange",
        figure=fig,
        lw=2,
        label=f"AUC: {auc.detach().cpu():0.2f}",
    )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc=loc)
    axis.set_title(title)
    return fig, axis

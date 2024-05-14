import warnings
from pathlib import Path
from typing import Union, List
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ignite.contrib import metrics
from scipy import interpolate
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_curve
from tqdm import tqdm

from src.model import UFlow
from src.datamodule import MVTecLightningDatamodule
from src.miou import mIoU
from src.aupro import AUPRO
from src.nfa_block import compute_log_nfa_anomaly_score
from src.nfa_tree import compute_nfa_anomaly_score_tree

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 256
TARGET_FPR = 1. / TARGET_SIZE / TARGET_SIZE
ALL_CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw",
    "toothbrush", "transistor", "zipper"
]


def reproduce_results(args):

    categories = args.categories
    if categories is None:
        categories = ALL_CATEGORIES
    elif isinstance(categories, str):
        categories = [categories]
    assert all([c in ALL_CATEGORIES for c in categories]), f"Categories must be inside the set {ALL_CATEGORIES}"

    for category in categories:

        print(f"\n{category.upper()}")

        config = yaml.safe_load(open(Path("configs") / f"{category}.yaml", "r"))

        # Data
        datamodule = MVTecLightningDatamodule(
            data_dir=args.data,
            category=category,
            input_size=config['model']['input_size'],
            batch_train=10,
            batch_test=10,
            shuffle_test=False
        )

        # Load model
        print("\tLoading model...", end=" ")
        uflow = UFlow(**config['model'])
        print("Done!")

        # Auroc
        uflow.from_pretrained(Path(args.models_dir) / "auc" / f"{category}.ckpt")
        uflow.eval()
        eval_auroc(uflow, datamodule.val_dataloader(), TARGET_SIZE)

        # Aupro
        uflow.from_pretrained(Path(args.models_dir) / "pro" / f"{category}.ckpt")
        uflow.eval()
        eval_aupro(uflow, datamodule.val_dataloader(), TARGET_SIZE)

        # mIoU
        uflow.from_pretrained(Path(args.models_dir) / "miou" / f"{category}.ckpt")
        uflow.eval()
        eval_miou(
            uflow,
            datamodule,
            target_size=TARGET_SIZE
        )


def eval_auroc(model, dataloader, target_size: Union[None, int] = None):

    if target_size is None:
        target_size = model.input_size

    model = model.to(DEVICE)

    auroc = metrics.ROC_AUC()
    progress_bar = tqdm(dataloader)
    progress_bar.set_description("\tComputing AUROC")
    for images, targets, img_paths in progress_bar:
        with torch.no_grad():
            z, _ = model.forward(images.to(DEVICE))

        anomaly_score = 1 - model.get_probability(z, target_size)

        if targets.shape[-1] != target_size:
            targets = F.interpolate(targets, size=[target_size, target_size], mode="bilinear", align_corners=False)
        targets = 1 * (targets > 0.5)

        auroc.update((anomaly_score.ravel(), targets.ravel()))

    print(f"\t\tAUROC: {auroc.compute()}")


def eval_aupro(model, dataloader, target_size: Union[None, int] = None):

    if target_size is None:
        target_size = model.input_size

    model = model.to(DEVICE)

    aupro = AUPRO()

    progress_bar = tqdm(dataloader)
    progress_bar.set_description("\tComputing AuPRO")
    for images, targets, img_paths in progress_bar:
        with torch.no_grad():
            z, _ = model.forward(images.to(DEVICE))

        anomaly_score = 1 - model.get_probability(z, target_size)

        if targets.shape[-1] != target_size:
            targets = F.interpolate(targets, size=[target_size, target_size], mode="bilinear", align_corners=False)
        targets = 1 * (targets > 0.5)

        aupro.update(anomaly_score, targets.to(DEVICE))

    print(f"\t\tAUPRO: {aupro.compute()}")


def eval_miou(model, datamodule, target_size: Union[None, int] = None):

    if target_size is None:
        target_size = model.input_size

    model = model.to(DEVICE)

    # This would be the code for computing the fair threshold for the case when we do not have an automatic threshold
    # fair_likelihood_thr = get_fair_threshold(model, datamodule.train_dataloader(), TARGET_FPR, target_size)

    nfa_thresholds = list(np.arange(-400, 1001, 1))
    miou_metric = mIoU(thresholds=nfa_thresholds)

    progress_bar = tqdm(datamodule.val_dataloader())
    progress_bar.set_description("\tComputing mIoU")
    for image, target, _ in progress_bar:
        image, targets = image.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            z, _ = model(image)

        anomaly_score = compute_nfa_anomaly_score_tree(z, target_size=target_size)

        # Alternative old computation -------------------------------------------
        block_nfa = False
        if block_nfa:
            anomaly_score = compute_log_nfa_anomaly_score(z, high_precision=True)
        # -----------------------------------------------------------------------

        if targets.shape[-1] != target_size:
            targets = F.interpolate(targets, size=[target_size, target_size], mode="bilinear", align_corners=False)
            targets = 1 * (targets > 0.5)

        miou_metric.update(anomaly_score.detach().cpu(), targets.cpu())

    mious = miou_metric.compute().numpy()

    print(f"\t\tmIoU @ log(NFA)=0: {mious[list(np.around(nfa_thresholds, 2)).index(0)]}")
    print(f"\t\tmIoU @ oracle-thr: {np.max(mious)}")


def get_fair_threshold(model, dataloader, target_fpr=0.01, target_size=None):
    """
    This is the code used for computing the fair threshold over the likelihoods for the case when we do not have an
    automatic thresholds (i.e. we do not have the NFA). This method was used for computing the fair threshold for all
    competitors in the paper. It mimics the same rationale of the NFA, allowing at most one false positive per image on
    average. as explained in the paper, or this computation only the anomaly free images are used.
    Parameters
    ----------
    model:
    dataloader
    target_fpr
    target_size

    Returns
    -------
    The fair threshold, i.e. the threshold that allows at most one false positive per image on average.
    """
    if target_size is None:
        target_size = model.input_size

    model = model.to(DEVICE)
    auroc = metrics.ROC_AUC()

    progress_bar = tqdm(dataloader)
    progress_bar.set_description("\tFinding fair threshold")
    for img in progress_bar:
        with torch.no_grad():
            z, _ = model(img.to(DEVICE))

        anomaly_score = 1 - model.get_probability(z, target_size)
        target = torch.zeros(*anomaly_score.shape, dtype=torch.bool)

        auroc.update((anomaly_score.cpu().ravel(), target.cpu().ravel()))

    targets = torch.cat(auroc._targets).clone()
    predictions = torch.cat(auroc._predictions)
    fpr, tpr, thresholds = roc_curve(targets, predictions)

    thr_interp = interpolate.interp1d(fpr, thresholds)

    threshold = thr_interp(target_fpr)
    return float(threshold)


if __name__ == "__main__":
    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("-cat", "--categories", default=None, type=str, nargs='+')
    p.add_argument("-data", "--data", default="data", type=str)
    p.add_argument("-models_dir", "--models_dir", default="models", type=str)
    # p.add_argument("-hp", "--high-precision", default=False, type=bool)
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    reproduce_results(cmd_args)

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
from src.iou import IoU
from src.nfa import compute_log_nfa_anomaly_score

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
        print("\tLoading model...")
        flow_model = UFlow(**config['model'])

        # Auroc
        flow_model.from_pretrained(Path("models") / "auc" / f"{category}.ckpt")
        flow_model.eval()
        eval_auroc(flow_model, datamodule.val_dataloader(), TARGET_SIZE)

        # IoU
        flow_model.from_pretrained(Path("models") / "iou" / f"{category}.ckpt")
        flow_model.eval()
        eval_iou(
            flow_model,
            datamodule,
            target_size=TARGET_SIZE,
            high_precision=args.high_precision
        )


def eval_auroc(model, dataloader, target_size: Union[None, int] = None):

    if target_size is None:
        target_size = model.input_size

    model = model.to(DEVICE)

    auroc = metrics.ROC_AUC()
    progress_bar = tqdm(dataloader)
    progress_bar.set_description("\tComputing AuROC")
    for images, targets, img_paths in progress_bar:
        with torch.no_grad():
            z, _ = model.forward(images.to(DEVICE))

        anomaly_score = 1 - model.get_probability(z, target_size)

        if targets.shape[-1] != target_size:
            targets = F.interpolate(targets, size=[target_size, target_size], mode="bilinear", align_corners=False)
        targets = 1 * (targets > 0.5)

        auroc.update((anomaly_score.ravel(), targets.ravel()))

    print(f"\t\tAUROC: {auroc.compute()}")


def eval_iou(model, datamodule, target_size: Union[None, int] = None, high_precision: bool = False):

    if target_size is None:
        target_size = model.input_size

    model = model.to(DEVICE)

    fair_likelihood_thr = get_fair_threshold(model, datamodule.train_dataloader(), TARGET_FPR, target_size)
    nfa_thresholds = list(np.arange(-2, 2, 0.05))

    iou_likelihood = IoU(thresholds=[fair_likelihood_thr])
    iou_nfa = IoU(thresholds=nfa_thresholds)

    progress_bar = tqdm(datamodule.val_dataloader())
    progress_bar.set_description("\tComputing IoU")
    for image, target, _ in progress_bar:
        image, targets = image.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            z, _ = model(image)

        anomaly_score_likelihood = 1 - model.get_probability(z, target_size)
        anomaly_score_nfa = compute_log_nfa_anomaly_score(
            z, win_size=5, binomial_probability_thr=0.9, high_precision=high_precision
        )

        if targets.shape[-1] != target_size:
            targets = F.interpolate(targets, size=[target_size, target_size], mode="bilinear", align_corners=False)
            targets = 1 * (targets > 0.5)

        iou_likelihood.update(anomaly_score_likelihood.detach().cpu(), targets.cpu())
        iou_nfa.update(anomaly_score_nfa.detach().cpu(), targets.cpu())

    iou_fair = iou_likelihood.compute().numpy()
    iou_nfas = iou_nfa.compute().numpy()

    print(f"\t\tIoU @ log(NFA)=0: {iou_nfas[list(np.around(nfa_thresholds, 2)).index(0)]}")
    print(f"\t\tIoU @ oracle-thr: {np.max(iou_nfas)}")
    print(f"\t\tIoU @ fair-thr  : {iou_fair}")


def get_fair_threshold(model, dataloader, target_fpr=0.01, target_size=None):

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
    p.add_argument("-hp", "--high-precision", default=False, type=bool)
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    reproduce_results(cmd_args)

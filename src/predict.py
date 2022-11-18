import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.nfa import compute_log_nfa_anomaly_score
from src.datamodule import MVTecLightningDatamodule, mvtec_un_normalize
from src.model import UFlow

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 256


def predict(args):

    config = yaml.safe_load(open(Path("configs") / f"{args.category}.yaml", "r"))

    # Data
    datamodule = MVTecLightningDatamodule(
        data_dir=args.data,
        category=args.category,
        input_size=config['model']['input_size'],
        batch_train=1,
        batch_test=10,
        shuffle_test=False
    )

    progress_bar = tqdm(datamodule.val_dataloader())
    progress_bar.set_description(f"{args.category.upper()}")

    # Load model
    flow_model = UFlow(**config['model'])
    flow_model.from_pretrained(Path("models") / "auc" / f"{args.category}.ckpt")
    flow_model.eval()
    model = flow_model.to(DEVICE)

    all_images, all_targets, all_scores, all_lnfas = [], [], [], []
    for images, targets, img_paths in progress_bar:
        with torch.no_grad():
            z, _ = model.forward(images.to(DEVICE))

        all_scores.append(1 - model.get_probability(z, TARGET_SIZE))
        all_lnfas.append(compute_log_nfa_anomaly_score(z, 5, 0.9, TARGET_SIZE, False))
        all_images.append(np.clip(mvtec_un_normalize(
            F.interpolate(images, [TARGET_SIZE, TARGET_SIZE], mode="bilinear", align_corners=False)), 0, 1))
        all_targets.append(F.interpolate(targets, [TARGET_SIZE, TARGET_SIZE], mode="bilinear", align_corners=False))

    all_scores = torch.cat(all_scores, dim=0)
    all_lnfas = torch.cat(all_lnfas, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    score_min, score_max = np.percentile(all_scores.cpu(), .1), np.percentile(all_scores.cpu(), 99.9)
    lnfa_min, lnfa_max = np.percentile(all_lnfas, .1), np.percentile(all_lnfas, 99.)

    for img, target, score, lnfa in zip(all_images, all_targets, all_scores, all_lnfas):
        plt.figure(1)
        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
        heatmap = np.clip((score[0].detach().cpu().numpy() - score_min) / (score_max - score_min), 0, 1)
        plt.imshow(heatmap, alpha=0.4, cmap='turbo')
        plt.contour(target[0].detach().cpu().numpy(), [0.5])
        plt.title('Likelihood')
        plt.axis('off')
        plt.tight_layout()

        plt.figure(2)
        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
        heatmap = np.clip((lnfa[0].detach().cpu().numpy() - lnfa_min) / (lnfa_max - lnfa_min), 0, 1)
        plt.imshow(heatmap, alpha=0.4, cmap='turbo')
        plt.contour(target[0].detach().cpu().numpy(), [0.5])
        plt.title('Log(NFA)')
        plt.axis('off')
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("-cat", "--category", default="carpet", type=str)
    p.add_argument("-data", "--data", default="data", type=str)
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    predict(cmd_args)

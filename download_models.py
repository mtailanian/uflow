import argparse
from pathlib import Path

import gdown

urls = {
    'bottle': {
        'auc': '1IFXHmeFvpaXguJVVuLr9Gla7IQMInT9c',
        'iou': '1JumC6R8OmKykjU1lFUCWp51wtpIn7pYL'
    },
    'cable': {
        'auc': '1YuWM90eRsbKMw_c3OSlorPatVvXOBnni',
        'iou': '1-vWmAhrl5Bq7nsfBV5vlMVc-xsyypytp'
    },
    'capsule': {
        'auc': '1iXQU-bAt0fxf9HgvBdH3FcGoDCuEvfBB',
        'iou': '1Tq3VE2J3kjdvbz0wznFjYCkF6Yd2BAB0'
    },
    'carpet': {
        'auc': '1YnGs9Y4dsHsDky3kx7k4xdD0d_lVMLnY',
        'iou': '12ZgoyzBWoip1FfmuEiQc0NDweXJr5uNd'
    },
    'grid': {
        'auc': '1veUuiCoqu0BM1vXd0lKmQT8ehEg2peO0',
        'iou': '1EabUFvtSaVMGdwNWMCT0YhZg9EKhm-tc'
    },
    'hazelnut': {
        'auc': '1oeJYAYZTGf4g1lV4fGe3S91BqtjDE13u',
        'iou': '1DSgXF6PE-Yqjs_norDblFZw2JGv88EsS'
    },
    'leather': {
        'auc': '1wHK3VpV_6_z1_YvEpoe_-h1tRN9rBmwc',
        'iou': '1n5JmX7L_H3XJ9i2FXKA9rlPoj7ag2bsE'
    },
    'metal_nut': {
        'auc': '1-l2_ivF_-0n6Jb2qMGCePYbeVSb_EWQ1',
        'iou': '17GoHuUcQxSiHNEgHSsApSOru9H80IYQb'
    },
    'pill': {
        'auc': '1atPD3w94pH704YajhdtpMSmmw9kCK1ER',
        'iou': '1I4Tic_Vc3LF_G4akHdihEdF_2N5HMytL'
    },
    'screw': {
        'auc': '1Foe4OEb6PMEe1y4PG1lZZBlg4HW-k6Z-',
        'iou': '1DZSRPcq6UrAWNDu0dLGEWqdGak0t2PKo'
    },
    'tile': {
        'auc': '13xrYYa3b7B1HfaEwVYr2-qgCwzYSrOQx',
        'iou': '1VbaAIw_6NSKIv1fmrylBmJ9nFhi-rmHq'
    },
    'toothbrush': {
        'auc': '1SeBT18OXfewtsoEPz6OA9Jn5OLKSGdQz',
        'iou': '1VnDh96vUtYjQh35lF0AHyRXJaiTKSHh7'
    },
    'transistor': {
        'auc': '1jpDF_iwemef8t31G33VpH8vox39vFBH1',
        'iou': '1xiRYOcbaZQhhh4N19l5m7PCdWHA7PKNk'
    },
    'wood': {
        'auc': '14YSNRn0O16L4xUf-L0uo3aK79PMX_fn3',
        'iou': '1aa3ZjUT1vBhsH5nRd_-TyJdZzYBfA_Ks'
    },
    'zipper': {
        'auc': '10C7I9ehU28Vqj5-mLXKK81_p6L1g4bbc',
        'iou': '1QpPF8oPUtL1UnQVrRoCW-T3i2pAZCWcF'
    }
}


def download_models(args):
    if args.categories is None:
        categories = list(urls.keys())
    else:
        categories = args.categories

    assert all([c in urls.keys() for c in categories]), \
        f"categories must be None (no argument) or a subset of {urls.keys()}"

    print(
        "Starting download. If any some error happens during the process, you can always download models by hand from "
        "here: https://drive.google.com/drive/folders/1DNnOzVysOJS2hLuUd7xS1t9QR0IZqrVK?usp=sharing\nIf downloading by "
        "hand please remember to put all downloaded models in the same folder structure inside `<uflow-root>/models` "
        "directory\n"
    )
    for cat in categories:
        print(f"\nCategory: {cat}")
        download_category_models(cat, args.force_overwrite)


def download_category_models(category, overwrite):
    for ckpt in ['auc', 'iou']:
        print(f"Pre-trained checkpoint for {ckpt}:")
        Path(f"models/{ckpt}").mkdir(exist_ok=True, parents=True)
        url = f"https://drive.google.com/u/1/uc?id={urls[category][ckpt]}&export=download"
        out_path = f"models/{ckpt}/{category}.ckpt"
        if Path(out_path).exists():
            print("The selected model is already downloaded. ", end="")
            if overwrite:
                print("Force overwrite option selected. Downloading it again...")
            else:
                print("Skipping...")
                continue
        gdown.download(url, out_path)


if __name__ == "__main__":
    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("-cat", "--categories", default=None, type=str, nargs='+')
    p.add_argument("-overwrite", "--force-overwrite", default=False, type=bool)
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    download_models(cmd_args)

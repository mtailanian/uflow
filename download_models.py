import argparse
from pathlib import Path

import gdown

urls = {
    'bottle': {
        'auc': '1yBElaUsDq9TVxlbLixIQimvpPmXszArz',
        'iou': '1Sa63skrhYWvzHA0G61dJcBG_F9m66ts8'
    },
    'cable': {
        'auc': '1oOM1WUmI0SrcKWz_S5WgX9eBj62gY4v3',
        'iou': '1_2rcYyPT9PGnaUke12vTk0eBdBtNvPsZ'
    },
    'capsule': {
        'auc': '1IVAzH8NbLRA2h6fvjrgRM1QiQx1wrtns',
        'iou': '1tyUYTjh9pPxL60TWv6HjVPABwpqzw-eH'
    },
    'carpet': {
        'auc': '1fvT-X4FWx8XWowekFj7bVoCdkAE7MGbD',
        'iou': '1tAolF8Krtw6y-EASgLrrX6MBDckPblwC'
    },
    'grid': {
        'auc': '1sQCm2sxYaooYh2jBQw3EZw8HgaDmvKlp',
        'iou': '1MlrtwTJ91kFrOoLWW6ZaqNaBgVJEN0fd'
    },
    'hazelnut': {
        'auc': '1pqHLwZ2MRLEBjMCNRPy3nOLhS0YZ_1lu',
        'iou': '1RDnsHtVmrP6Xw6soXFYnKJmoeRi2iBsq'
    },
    'leather': {
        'auc': '1yjh_7vSC44bc9fxbUmWKRTlKeipfeb6r',
        'iou': '1RDnsHtVmrP6Xw6soXFYnKJmoeRi2iBsq'
    },
    'metal_nut': {
        'auc': '1IKXy7WpeMnV4gfK4GZIYVww_RVHYLV_x',
        'iou': '1BpkFO7PPq6xYGbxJS2Rs6N-sPXYX-lTA'
    },
    'pill': {
        'auc': '1B9UxInnrv2OsToUW94ocvhZr8ZbLJcO-',
        'iou': '1PeAcmhXnUsRf2ALAoSqxQQobbDB2aRWf'
    },
    'screw': {
        'auc': '1h2uGLVFkXMJETsmVqfReooml6nKOZHMJ',
        'iou': '1z02h-hmUe3mgHWuSApj9mWIizMZuxp1f'
    },
    'tile': {
        'auc': '1QWSvhtVfk1b_pOhaIFKE6rKSnAMhhG9H',
        'iou': '1rs8Agv6CM2uzm7M7i96EQgAKwQW9QnDX'
    },
    'toothbrush': {
        'auc': '1bCIKvCeU45dinzvGZPOpn7Jgf3LVkkwS',
        'iou': '1YxQG1oHhhAsZuRgTxCqjZ6ELNFpjRpnN'
    },
    'transistor': {
        'auc': '1I8yeuTpUJPNHy0ToY8ZM3qv-Y4VFKOr2',
        'iou': '11IxXg95xB_b5UAVKdFs6cO0kDJNK6i8T'
    },
    'wood': {
        'auc': '10W39Zk6c2MKb4CLgx9ksco1GfqC4m8JS',
        'iou': '158HrCMoQi8Z7lTxpOY3HzLRzfB3FzieS'
    },
    'zipper': {
        'auc': '1zbISk4lD5FUr0B5hE1zOw9_GwzaGofp5',
        'iou': '1lR3GiO7IrkaP8JL2DoXIbUHjaaOugqrA'
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

import argparse
from pathlib import Path

import gdown

urls = {
    'bottle': {
        'auc': '1yBElaUsDq9TVxlbLixIQimvpPmXszArz',
        'iou': '1Sa63skrhYWvzHA0G61dJcBG_F9m66ts8',
        'pro': '1m2Kq3A03ClyqZ3So8c4uMHORSo1oxRf1'
    },
    'cable': {
        'auc': '1oOM1WUmI0SrcKWz_S5WgX9eBj62gY4v3',
        'iou': '1_2rcYyPT9PGnaUke12vTk0eBdBtNvPsZ',
        'pro': '1iiqviSrZrN7YsIkdiIfFLTivtuDjaBfj'
    },
    'capsule': {
        'auc': '1IVAzH8NbLRA2h6fvjrgRM1QiQx1wrtns',
        'iou': '1tyUYTjh9pPxL60TWv6HjVPABwpqzw-eH',
        'pro': '1uA81ONAbdIN_tr0p0aqY9YipqvMW8LxE'
    },
    'carpet': {
        'auc': '1fvT-X4FWx8XWowekFj7bVoCdkAE7MGbD',
        'iou': '1tAolF8Krtw6y-EASgLrrX6MBDckPblwC',
        'pro': '1UFCcHnbvsXrL-6xFHVB13tCyeGuSiFip'
    },
    'grid': {
        'auc': '1sQCm2sxYaooYh2jBQw3EZw8HgaDmvKlp',
        'iou': '1MlrtwTJ91kFrOoLWW6ZaqNaBgVJEN0fd',
        'pro': '1c-pp8y71sSfLAyHS_Yzz-lfFBJM_3Wg8'
    },
    'hazelnut': {
        'auc': '1pqHLwZ2MRLEBjMCNRPy3nOLhS0YZ_1lu',
        'iou': '1RDnsHtVmrP6Xw6soXFYnKJmoeRi2iBsq',
        'pro': '1W7c4qd8Tnkops3QIxiSd5RYYPqWHsg8V'
    },
    'leather': {
        'auc': '1yjh_7vSC44bc9fxbUmWKRTlKeipfeb6r',
        'iou': '1RDnsHtVmrP6Xw6soXFYnKJmoeRi2iBsq',
        'pro': '14a8sCu-BjsCV-i4ZiaZ1OcdPGNUppg9L'
    },
    'metal_nut': {
        'auc': '1IKXy7WpeMnV4gfK4GZIYVww_RVHYLV_x',
        'iou': '1BpkFO7PPq6xYGbxJS2Rs6N-sPXYX-lTA',
        'pro': '1AeMdEbE6Rfqs6d9liEfQEGogz5jYz7On'
    },
    'pill': {
        'auc': '1B9UxInnrv2OsToUW94ocvhZr8ZbLJcO-',
        'iou': '1PeAcmhXnUsRf2ALAoSqxQQobbDB2aRWf',
        'pro': '1I3UlQ4yvlF0n8i9o2ehN0VM0UytTE1ms'
    },
    'screw': {
        'auc': '1h2uGLVFkXMJETsmVqfReooml6nKOZHMJ',
        'iou': '1z02h-hmUe3mgHWuSApj9mWIizMZuxp1f',
        'pro': '1R4rqWd9p75OEFy97nVaWD67EDW4wX-ye'
    },
    'tile': {
        'auc': '1QWSvhtVfk1b_pOhaIFKE6rKSnAMhhG9H',
        'iou': '1rs8Agv6CM2uzm7M7i96EQgAKwQW9QnDX',
        'pro': '1jeeF-IqzPrbViJlMtt6YlrqkR94uamMZ'
    },
    'toothbrush': {
        'auc': '1bCIKvCeU45dinzvGZPOpn7Jgf3LVkkwS',
        'iou': '1YxQG1oHhhAsZuRgTxCqjZ6ELNFpjRpnN',
        'pro': '1Pss1slIfDaxmycr9FOXRT-QyDLeLLhye'
    },
    'transistor': {
        'auc': '1I8yeuTpUJPNHy0ToY8ZM3qv-Y4VFKOr2',
        'iou': '11IxXg95xB_b5UAVKdFs6cO0kDJNK6i8T',
        'pro': '1UlRHOqe7yQJjHVDxa60WuET6eZ9Junzm'
    },
    'wood': {
        'auc': '10W39Zk6c2MKb4CLgx9ksco1GfqC4m8JS',
        'iou': '158HrCMoQi8Z7lTxpOY3HzLRzfB3FzieS',
        'pro': '1USqJsJmaSZQDPb7qrXVuuRMScSJY4oTK'
    },
    'zipper': {
        'auc': '1zbISk4lD5FUr0B5hE1zOw9_GwzaGofp5',
        'iou': '1lR3GiO7IrkaP8JL2DoXIbUHjaaOugqrA',
        'pro': '1UpKGT6JB-zk1NAfJFhD337Y6E0LgId7x'
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
    for ckpt in ['auc', 'iou', 'pro']:
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

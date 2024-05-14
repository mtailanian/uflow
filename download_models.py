import argparse
from pathlib import Path

import gdown

urls = {
    'bottle': {
        'auc': '1yBElaUsDq9TVxlbLixIQimvpPmXszArz',
        'miou': '1mf9_AggYGHVypWa5Tx3K1_oISn-0jP1w',
        'pro': '1m2Kq3A03ClyqZ3So8c4uMHORSo1oxRf1'
    },
    'cable': {
        'auc': '1oOM1WUmI0SrcKWz_S5WgX9eBj62gY4v3',
        'miou': '166Q7pHqUWw5Pi3q93llFpPt7vyLI7z2f',
        'pro': '1iiqviSrZrN7YsIkdiIfFLTivtuDjaBfj'
    },
    'capsule': {
        'auc': '1IVAzH8NbLRA2h6fvjrgRM1QiQx1wrtns',
        'miou': '1H3Mg0--ZvD6ifbVf5wWdnAI6tvxAyTgI',
        'pro': '1uA81ONAbdIN_tr0p0aqY9YipqvMW8LxE'
    },
    'carpet': {
        'auc': '1fvT-X4FWx8XWowekFj7bVoCdkAE7MGbD',
        'miou': '15Mjno37Wojm0SX6kxbVFuQk9HHo3vatT',
        'pro': '1UFCcHnbvsXrL-6xFHVB13tCyeGuSiFip'
    },
    'grid': {
        'auc': '1sQCm2sxYaooYh2jBQw3EZw8HgaDmvKlp',
        'miou': '1-Cx1q0YQ8VtS42CivAjxFR_cL9CkjiAF',
        'pro': '1c-pp8y71sSfLAyHS_Yzz-lfFBJM_3Wg8'
    },
    'hazelnut': {
        'auc': '1pqHLwZ2MRLEBjMCNRPy3nOLhS0YZ_1lu',
        'miou': '1xgQsGew5gOK-YjSH77dxW6V_2Uv9MyrC',
        'pro': '1W7c4qd8Tnkops3QIxiSd5RYYPqWHsg8V'
    },
    'leather': {
        'auc': '1yjh_7vSC44bc9fxbUmWKRTlKeipfeb6r',
        'miou': '1XVqgFD1DBgFLgAYAlvHMNXuReDHYQqdp',
        'pro': '14a8sCu-BjsCV-i4ZiaZ1OcdPGNUppg9L'
    },
    'metal_nut': {
        'auc': '1IKXy7WpeMnV4gfK4GZIYVww_RVHYLV_x',
        'miou': '1cl7_N9EagjdX7Nc_vQCrHwYfvBWhx5-L',
        'pro': '1AeMdEbE6Rfqs6d9liEfQEGogz5jYz7On'
    },
    'pill': {
        'auc': '1B9UxInnrv2OsToUW94ocvhZr8ZbLJcO-',
        'miou': '1qGEza9BLN8kX-Ec1l5054SnUMjvECcTE',
        'pro': '1I3UlQ4yvlF0n8i9o2ehN0VM0UytTE1ms'
    },
    'screw': {
        'auc': '1h2uGLVFkXMJETsmVqfReooml6nKOZHMJ',
        'miou': '16BoX81QCBpZs3AmTa0TYXO0uKzfTFFM1',
        'pro': '1R4rqWd9p75OEFy97nVaWD67EDW4wX-ye'
    },
    'tile': {
        'auc': '1QWSvhtVfk1b_pOhaIFKE6rKSnAMhhG9H',
        'miou': '1ehcF-66zfi8Z0q9dYxnlGC4HS2U8xm9c',
        'pro': '1jeeF-IqzPrbViJlMtt6YlrqkR94uamMZ'
    },
    'toothbrush': {
        'auc': '1bCIKvCeU45dinzvGZPOpn7Jgf3LVkkwS',
        'miou': '1K8na1UvWYrKje262cBpS-cfspa1GzWPX',
        'pro': '1Pss1slIfDaxmycr9FOXRT-QyDLeLLhye'
    },
    'transistor': {
        'auc': '1I8yeuTpUJPNHy0ToY8ZM3qv-Y4VFKOr2',
        'miou': '1LgW47tSttt3qcAnJNTJwp76G3HhUrCOB',
        'pro': '1UlRHOqe7yQJjHVDxa60WuET6eZ9Junzm'
    },
    'wood': {
        'auc': '10W39Zk6c2MKb4CLgx9ksco1GfqC4m8JS',
        'miou': '1j_fJhkmPu8JBTYX9-u1JfcSdt8HKisst',
        'pro': '1USqJsJmaSZQDPb7qrXVuuRMScSJY4oTK'
    },
    'zipper': {
        'auc': '1zbISk4lD5FUr0B5hE1zOw9_GwzaGofp5',
        'miou': '1n_oflm6b3MiF0mGysO9Ejs4S7D8nowqX',
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
        "directory\n\n"
        "Another option is to download all checkpoints form this Release in GitHub: "
        "https://github.com/mtailanian/uflow/releases/tag/trained-models-for-all-mvtec-categories\n\n"
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

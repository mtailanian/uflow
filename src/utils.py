from pathlib import Path


def get_training_dir(base_dir, prefix="exp_"):
    out_path = Path(base_dir)
    previous_experiments = [int(l.stem.split('_')[1]) for l in out_path.glob(f'{prefix}*')]
    last_experiment = max(previous_experiments) if len(previous_experiments) > 0 else 0
    out_path = out_path / f"{prefix}{last_experiment + 1:04d}"
    out_path.mkdir(exist_ok=True, parents=True)
    return out_path

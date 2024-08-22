from pathlib import Path

import numpy as np

data_dir = Path(__file__).parent
csv_dir = data_dir / "csv"


def load_csv(subdir, name):
    return np.loadtxt(csv_dir / subdir / f"{name}.csv")

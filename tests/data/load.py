from pathlib import Path

import numpy as np
import jax.numpy as jnp

data_dir = Path(__file__).parent
csv_dir = data_dir / "csv"


def load_csv(name, params, subdir):
    return jnp.array(np.loadtxt(csv_dir / subdir / f"{name}:{params}.csv"))

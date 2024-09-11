from functools import partial

import jax.numpy as jnp

from optispec.models.base import Spectrum

from .load import load_csv

load_csv = partial(
    load_csv,
    subdir="two_state",
    params="points=0,20k,2001;temp=300;broad=200;t=100;e=8k;bs=20;50,f=1200,100;c=0p7,2",
)

points = jnp.linspace(0, 20_000, 2001)

OLD_PACKAGE_ABSORPTION = Spectrum(points, load_csv("absorption-intensities"))

FORTRAN_CODE_ABSORPTION = Spectrum(points, load_csv("absorption-intensities-fortran"))

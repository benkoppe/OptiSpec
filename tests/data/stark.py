from functools import partial

import jax.numpy as jnp

from optispec.models.base import Spectrum

from .load import load_csv

load_csv = partial(
    load_csv,
    subdir="stark",
    params="model=twostate;points=0,20k,2001;temp=300;broad=200;t=100;e=8k;bs=20,200;f=1200,100;c=0p7,2;field=0p01;percent=0p5;delta_d=38;delta_p=1000",
)

OLD_PACKAGE_ABSORPTION = Spectrum(
    jnp.linspace(0, 20_000, 2001), load_csv("absorption-intensities")
)

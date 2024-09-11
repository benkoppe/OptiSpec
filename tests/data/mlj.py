from functools import partial

from .load import load_csv

load_csv = partial(
    load_csv,
    subdir="mlj",
    params="points=0,20k,2001;temp=300;e=8k;d=0;bs=20,f=1200,100;c=0p7,2",
)

OLD_PACKAGE_ABSORPTION = load_csv("absorption-intensities")

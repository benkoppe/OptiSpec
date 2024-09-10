from functools import partial

from .load import load_csv

load_csv = partial(
    load_csv,
    subdir="mlj",
    params="points=0,20k,2001;temp=300;broad=200;t=100;e=8k;bs=20;50,f=1200,100;c=0p7,2",
)

OLD_PACKAGE_ABSORPTION = load_csv("absorption-intensities")

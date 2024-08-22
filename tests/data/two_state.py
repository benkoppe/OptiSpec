from functools import partial

from .load import load_csv

load_csv = partial(
    load_csv, subdir="two_state", params="t=100;e=0,8000;bs=9,10;f=1200,100;c=0p7,2"
)

OLD_PACKAGE_ABSORPTION = load_csv("absorption")

from functools import partial

from .load import load_csv

load_csv = partial(
    load_csv, subdir="hamiltonian", params="t=100;e=0,8000;bs=9,10;f=1200,100;c=0p7,2"
)

OLD_PACKAGE_MATRIX = load_csv("matrix")
OLD_PACKAGE_EIGENVALUES = load_csv("vals")
OLD_PACKAGE_EIGENVECTORS = load_csv("vects")

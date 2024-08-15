import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent

_file_postfix = "t=100;e=0,8000;bs=9,10;f=1200,100;c=0p7,2.csv"

OLD_PACKAGE_MATRIX = np.loadtxt(data_dir / f"matrix:{_file_postfix}")
OLD_PACKAGE_EIGENVALUES = np.loadtxt(data_dir / f"vals:{_file_postfix}")
OLD_PACKAGE_EIGENVECTORS = np.loadtxt(data_dir / f"vects:{_file_postfix}")

from .load import load_csv

subdir = "hamiltonian"
_file_postfix = "t=100;e=0,8000;bs=9,10;f=1200,100;c=0p7,2"

OLD_PACKAGE_MATRIX = load_csv(subdir, f"matrix:{_file_postfix}")
OLD_PACKAGE_EIGENVALUES = load_csv(subdir, f"vals:{_file_postfix}")
OLD_PACKAGE_EIGENVECTORS = load_csv(subdir, f"vects:{_file_postfix}")

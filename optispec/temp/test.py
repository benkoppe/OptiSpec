import jax
import jax.numpy as jnp
import numpy as np
import random
import time

from optispec.models import two_state as ts

# display sample usage of two-state model

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def time_function():
    p = ts.Params(transfer_integral=random.randint(90, 110), mode_basis_sets=(40, 200))

    start = time.time()

    diag = ts.diagonalize(p)

    peaks = ts._peaks(diag, p.transfer_integral, p.temperature_kelvin)

    end = time.time()

    return end - start, peaks


def time_matrix():
    p = ts.Params(transfer_integral=random.randint(90, 110))

    start = time.time()

    ts.diagonalize(p)

    end = time.time()

    return end - start


N = 10
runtimes = []

for _ in range(N):
    runtime = time_matrix()
    print(runtime)
    runtimes.append(runtime)


print(f"Average runtime: {np.mean(runtimes):.2f} seconds")
print(f"Average runtime excluding first: {np.mean(runtimes[1:]):.2f} seconds")

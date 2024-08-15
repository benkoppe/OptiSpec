import jax
import jax.numpy as jnp
import numpy as np
import random
import time

from optispec.models import two_state as ts

# display sample usage of two-state model

jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_enable_x64", True)


def time_absorption():
    p = ts.Params()

    start = time.time()

    spec = ts.absorption(p)

    end = time.time()

    return end - start, spec


def time_matrix():
    p = ts.Params(
        transfer_integral=random.randint(90, 110),
        temperature_kelvin=random.choice([0.0, 300.0]),
    )

    start = time.time()

    ts.diagonalize(p)

    end = time.time()

    return end - start


N = 10
runtimes = []

for _ in range(N):
    runtime, _ = time_absorption()
    print(runtime)
    runtimes.append(runtime)


print(f"Average runtime: {np.mean(runtimes):.2f} seconds")
print(f"Average runtime excluding first: {np.mean(runtimes[1:]):.2f} seconds")

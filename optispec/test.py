import jax
import jax.numpy as jnp
import random
import time

from optispec.models import two_state as ts

# display sample usage of two-state model

jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_enable_x64", True)


def time_function():
    p = ts.Params(transfer_integral=random.randint(90, 110))

    start = time.time()

    diag = ts.diagonalize(p)

    peaks = ts._peaks(diag, p.transfer_integral, p.temperature_kelvin)

    end = time.time()

    return end - start, peaks


N = 10
runtimes = []

for _ in range(N):
    runtime, peaks = time_function()
    print(runtime)
    # print(peaks)
    runtimes.append(runtime)

print(f"Average runtime: {sum(runtimes[1:]) / (N - 1):.2f} seconds")

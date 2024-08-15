import jax
import jax.numpy as jnp
import random
import time

from optispec import hamiltonian as h

jax.config.update("jax_platform_name", "gpu")


def time_diagonalization():
    p = h.Params(
        transfer_integrals=100,
        state_energies=jnp.array([0, random.randint(9_000, 15_000)]),
        mode_basis_sets=(20, 200),
        mode_localities=[True, True],
        mode_frequencies=jnp.array([1400, 100]),
        mode_state_couplings=jnp.array(
            [[0.0, 0.0], [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)]]
        ),
    )

    start = time.time()

    h.diagonalize(p)

    end = time.time()

    return end - start


for _ in range(10):
    runtime = time_diagonalization()
    print(runtime)

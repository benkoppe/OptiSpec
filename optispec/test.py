import jax
import jax.numpy as jnp
import random
import time

from optispec import hamiltonian as h

# display sample usage of hamiltonian

jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_enable_x64", True)


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

    # h.hamiltonian(p)

    end = time.time()

    return end - start


N = 10
runtimes = []

for _ in range(N):
    runtime = time_diagonalization()
    print(runtime)
    runtimes.append(runtime)

print(f"Average runtime: {sum(runtimes[1:]) / N:.2f} seconds")

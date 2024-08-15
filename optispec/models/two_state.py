from jaxtyping import Array, Float, Int
import jax_dataclasses as jdc
import jax.numpy as jnp

from optispec.models.base import CommonParams, Spectrum
from optispec import hamiltonian as h


@jdc.pytree_dataclass
class Params(CommonParams):
    broadening: float = 200.0
    temperature_kelvin: float = 300.0

    transfer_integral: float = 100.0
    energy_gap: float = 8_000.0

    # mode arguments
    mode_frequencies: Float[Array, "num_modes"] = jnp.array([1200.0, 100.0])
    mode_couplings: Float[Array, "num_modes"] = jnp.array([0.7, 2.0])

    # static mode arguments
    mode_basis_sets: jdc.Static[tuple[int, ...]] = (20, 200)


@jdc.jit
def diagonalize(params: Params) -> h.Diagonalization:
    return h.diagonalize(_hamiltonian_params(params))


@jdc.jit
def hamiltonian(params: Params) -> h.Matrix:
    return h.hamiltonian(_hamiltonian_params(params))


@jdc.jit
def _hamiltonian_params(params: Params) -> h.Params:
    return h.Params(
        transfer_integrals=params.transfer_integral,
        state_energies=jnp.array([0.0, params.energy_gap]),
        mode_basis_sets=params.mode_basis_sets,
        mode_localities=(True, True),
        mode_frequencies=params.mode_frequencies,
        mode_state_couplings=jnp.array([[0.0, 0.0], params.mode_couplings]),
    )


# broaden peaks

# compute peaks

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from optispec import hamiltonian as h
from optispec.models.base import CommonParams, Spectrum
from optispec.utils import kelvin_to_wavenumbers


@jdc.pytree_dataclass
class Params(CommonParams):
    temperature_kelvin: jdc.Static[float] = 300.0
    broadening: float = 200.0

    ct_energy_gap: float = 10_745.0
    le_energy_gap: float = 12_358.0

    gs_ct_coupling: float = 100.0
    ct_le_coupling: float = 850.0

    # mode arguments
    mode_frequencies: Float[Array, " num_modes"] = jdc.field(
        default_factory=lambda: jnp.array([1200.0, 100.0])
    )
    ct_mode_couplings: Float[Array, " num_modes"] = jdc.field(
        default_factory=lambda: jnp.array([0.7, 2.0])
    )
    le_mode_couplings: Float[Array, " num_modes"] = jdc.field(
        default_factory=lambda: jnp.array([-0.85, -2.8])
    )

    # static mode arguments
    mode_basis_sets: jdc.Static[tuple[int, ...]] = (20, 200)


def absorption(params: Params) -> Spectrum:
    diagonalization = diagonalize(params)
    # energies, intensities = _peaks(
    #     diagonalization, params.coupling, params.temperature_kelvin
    # )

    sample_points = jnp.linspace(
        params.start_energy, params.end_energy, params.num_points
    )

    # broadened_spectrum = _broaden_peaks(
    #     sample_points, energies, intensities, params.broadening
    # )

    return Spectrum(sample_points, sample_points)


@jdc.jit
def diagonalize(params: Params) -> h.Diagonalization:
    return h.diagonalize(_hamiltonian_params(params))


@jdc.jit
def hamiltonian(params: Params) -> h.Matrix:
    return h.hamiltonian(_hamiltonian_params(params))


def _hamiltonian_params(params: Params) -> h.Params:
    return h.Params(
        transfer_integrals=jnp.array(
            [params.gs_ct_coupling, 0.0, params.ct_le_coupling]
        ),
        state_energies=jnp.array([0.0, params.ct_energy_gap, params.le_energy_gap]),
        mode_basis_sets=params.mode_basis_sets,
        mode_localities=(True, True, True),
        mode_frequencies=params.mode_frequencies,
        mode_state_couplings=jnp.array(
            [[0.0, 0.0], params.ct_mode_couplings, params.le_mode_couplings]
        ),
    )

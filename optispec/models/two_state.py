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

    energy_gap: float = 8_000.0
    coupling: float = 100.0

    # mode arguments
    mode_frequencies: Float[Array, " num_modes"] = jdc.field(
        default_factory=lambda: jnp.array([1200.0, 100.0])
    )
    mode_couplings: Float[Array, " num_modes"] = jdc.field(
        default_factory=lambda: jnp.array([0.7, 2.0])
    )

    # static mode arguments
    mode_basis_sets: jdc.Static[tuple[int, ...]] = (20, 200)

    def apply_electric_field(self, field_energy_change: float):
        return jdc.replace(self, energy_gap=self.energy_gap + field_energy_change)


def absorption(params: Params) -> Spectrum:
    diagonalization = diagonalize(params)
    energies, intensities = _peaks(
        diagonalization, params.coupling, params.temperature_kelvin
    )

    sample_points = jnp.linspace(
        params.start_energy, params.end_energy, params.num_points
    )

    broadened_spectrum = _broaden_peaks(
        sample_points, energies, intensities, params.broadening
    )

    return Spectrum(sample_points, broadened_spectrum)


@jdc.jit
def diagonalize(params: Params) -> h.Diagonalization:
    return h.diagonalize(_hamiltonian_params(params))


@jdc.jit
def hamiltonian(params: Params) -> h.Matrix:
    return h.hamiltonian(_hamiltonian_params(params))


@jdc.jit
def _hamiltonian_params(params: Params) -> h.Params:
    return h.Params(
        transfer_integrals=params.coupling,
        state_energies=jnp.array([0.0, params.energy_gap]),
        mode_basis_sets=params.mode_basis_sets,
        mode_localities=(True, True),
        mode_frequencies=params.mode_frequencies,
        mode_state_couplings=jnp.array([[0.0, 0.0], params.mode_couplings]),
    )


PeakEnergies = Float[Array, " num_peaks"]  # array of selected peak energies
PeakIntensities = Float[Array, " num_peaks"]  # array of selected peak intensities

# matrix of all considered transitions
# first dimension sliced in half because we only consider transitions from ground to excited states
AllTransitionsMatrix = Float[Array, " matrix_size/2 matrix_size"]


def _broaden_peaks(
    sample_points: Float[Array, " num_points"],
    energies: PeakEnergies,
    intensities: PeakIntensities,
    broadening: jdc.Static[float],
) -> Float[Array, " num_points"]:
    return jnp.sum(
        _apply_gaussians(sample_points, energies, intensities, broadening), axis=0
    )


@jax.jit
def _apply_gaussian(
    sample_points: Float[Array, " num_points"],
    peak_energy: Float[Array, "1"],
    peak_intensity: Float[Array, "1"],
    broadening: jdc.Static[float],
) -> Float[Array, " num_points"]:
    return (
        peak_intensity
        * (1.0 / (jnp.sqrt(jnp.pi) * broadening))
        * jnp.exp(-jnp.power((sample_points - peak_energy) / broadening, 2))
    )


_apply_gaussians = jax.vmap(_apply_gaussian, in_axes=(None, 0, 0, None))


def _peaks(
    diagonalization: h.Diagonalization,
    coupling: float,
    temperature_kelvin: jdc.Static[float],
) -> tuple[PeakEnergies, PeakIntensities]:
    energies = _peak_energies(diagonalization)
    intensities = _peak_intensities(diagonalization, coupling)

    temperature_wavenumbers = kelvin_to_wavenumbers(temperature_kelvin)

    # we must select the number of ground state transitions to consider based on the temperature
    # if temperature is 0, we select 1, otherwise we select the first 50 transitions
    ground_state_range = jax.lax.cond(
        (temperature_wavenumbers == 0.0),
        lambda: 1,
        lambda: min(50, len(diagonalization.eigenvalues)),
    )

    # if temperature is not zero, compute probability scalars and apply them to intensities
    scaled_intensities = jax.lax.cond(
        (temperature_wavenumbers == 0.0),
        lambda: intensities,
        lambda: intensities
        * _peak_probability_scalars(diagonalization, temperature_wavenumbers)[:, None],
    )

    # select pairs of energies/intensities to include in the broadened spectrum based on the eigenvector range
    return _select_peaks(energies, scaled_intensities, ground_state_range)


@jdc.jit
def _peak_energies(
    diagonalization: h.Diagonalization,
) -> AllTransitionsMatrix:
    # index [i,j] represents eigenvalue j - eigenvalue i (energy difference between i and j)

    eigenvalues = diagonalization.eigenvalues
    half_size = len(eigenvalues) // 2

    eigenvalues_col = eigenvalues[:, jnp.newaxis]
    differences_matrix = eigenvalues - eigenvalues_col

    return differences_matrix[:half_size]


@jdc.jit
def _peak_intensities(
    diagonalization: h.Diagonalization, coupling: float
) -> AllTransitionsMatrix:
    # index [i,j] represents intensity of transition from i to j

    eigenvectors = diagonalization.eigenvectors
    half_size = len(eigenvectors) // 2

    vector_slices_1 = eigenvectors[:half_size, :]
    vector_slices_2 = jax.lax.cond(
        (coupling == 0.0),
        lambda: eigenvectors[half_size:, :],
        lambda: vector_slices_1,
    )

    intensities_matrix = jnp.dot(vector_slices_1.T, vector_slices_2) ** 2

    return intensities_matrix[:half_size]


@jdc.jit
def _peak_probability_scalars(
    diagonalization: h.Diagonalization, temperature_wavenumbers: jdc.Static[float]
) -> Float[Array, " matrix_size/2"]:
    # each ground state has a probability scalar associated with it
    # when temperature is not 0, intensities are scaled by the scalar associated with the ground state of the transition

    eigenvalues = diagonalization.eigenvalues
    half_size = len(eigenvalues) // 2

    differences = (
        eigenvalues[:half_size] - eigenvalues[0]
    )  # difference between first ground state and all other ground states
    exponentials = jnp.exp(-differences / temperature_wavenumbers)

    return exponentials / jnp.sum(exponentials)  # boltzmann population


def _select_peaks(
    energies: AllTransitionsMatrix,
    intensities: AllTransitionsMatrix,
    ground_state_range: jdc.Static[int],
) -> tuple[PeakEnergies, PeakIntensities]:
    triu_indices = jnp.triu_indices(ground_state_range, k=1, m=len(energies))

    mask = ((intensities >= 0) & (energies >= 0))[triu_indices]

    selected_energies = energies[triu_indices][mask]
    selected_intensities = intensities[triu_indices][mask]

    return selected_energies, selected_intensities

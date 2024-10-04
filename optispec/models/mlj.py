import math

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int

from optispec.models.base import CommonParams, Spectrum
from optispec.utils import kelvin_to_wavenumbers


@jdc.pytree_dataclass
class Params(CommonParams):
    # static arguments
    basis_size: jdc.Static[int] = 10
    temperature_kelvin: jdc.Static[float] = 300.0

    # non-static arguments
    energy_gap: float = 8_000.0
    disorder_meV: float = 0.0

    # mode arguments
    mode_frequencies: Float[Array, "2"] = jdc.field(
        default_factory=lambda: jnp.array([1200.0, 100.0])
    )
    mode_couplings: Float[Array, "2"] = jdc.field(
        default_factory=lambda: jnp.array([0.7, 2.0])
    )

    def apply_electric_field(self, field_energy_change: float) -> "Params":
        return jdc.replace(self, energy_gap=self.energy_gap + field_energy_change)


def absorption(params: Params) -> Spectrum:
    sorted_freq_indices = jnp.argsort(params.mode_frequencies)
    low_freq_index = sorted_freq_indices[0]
    high_freq_index = sorted_freq_indices[1]

    constants = _MLJCalculationConstants(
        temperature_kbT=kelvin_to_wavenumbers(params.temperature_kelvin),
        factorials=jnp.array([math.factorial(n) for n in range(params.basis_size + 1)]),
        disorder_wavenumbers=params.disorder_meV * 8061 * 0.001,
        low_freq_relaxation_energy=params.mode_couplings[low_freq_index].item() ** 2
        * params.mode_frequencies[low_freq_index],
        high_freq_huang_rhys_factor=params.mode_couplings[high_freq_index].item() ** 2,
        high_frequency=params.mode_frequencies[high_freq_index].item(),
        energy_gap=params.energy_gap,
    )

    sample_points = jnp.linspace(
        params.start_energy, params.end_energy, params.num_points
    )

    spectrum = _mlj_spectrum(constants, sample_points)

    return Spectrum(sample_points, spectrum)


@jdc.pytree_dataclass
class _MLJCalculationConstants:
    # static arguments
    temperature_kbT: jdc.Static[float]
    factorials: jdc.Static[Int[Array, " basis_size + 1"]]

    # non-staic arguments
    disorder_wavenumbers: float
    low_freq_relaxation_energy: float
    high_freq_huang_rhys_factor: float

    # ordered original arguments
    high_frequency: float
    energy_gap: float


def _mlj_spectrum(
    constants: _MLJCalculationConstants, sample_points: Float[Array, " num_points"]
):
    spectrum = jax.vmap(lambda point: _mlj_single_intensity(constants, point))(
        sample_points
    )

    return spectrum / jnp.max(spectrum)


def _mlj_single_intensity(
    constants: _MLJCalculationConstants, curr_energy: jdc.Static[float]
):
    indices = jnp.arange(jnp.size(constants.factorials) + 1)
    intensity_components = jax.vmap(
        lambda idx: _mlj_single_intensity_component(constants, curr_energy, idx)
    )(indices)
    return jnp.sum(intensity_components)


def _mlj_single_intensity_component(
    constants: _MLJCalculationConstants,
    curr_energy: jdc.Static[float],
    index: jdc.Static[int],
):
    part1 = (
        jnp.exp(-constants.high_freq_huang_rhys_factor)
        * (constants.high_freq_huang_rhys_factor**index)
        / constants.factorials[index]
    )

    part2 = (
        index * constants.high_frequency
        + constants.energy_gap
        + constants.low_freq_relaxation_energy
        - curr_energy
    ) ** 2

    part3 = (
        4 * constants.low_freq_relaxation_energy * constants.temperature_kbT
        + 2 * constants.disorder_wavenumbers**2
    )

    part2_3 = jnp.exp(-part2 / part3)
    part1_2_3 = part1 * part2_3

    return part1_2_3

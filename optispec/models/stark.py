from types import ModuleType
from typing import TypeVar, Generic

import jax.numpy as jnp
import jax_dataclasses as jdc

from optispec.models import two_state
from optispec.models.base import CommonParams, Spectrum

P = TypeVar("P", bound=CommonParams)


@jdc.pytree_dataclass
class Params(Generic[P]):
    model: jdc.Static[ModuleType] = jdc.field(default=two_state)
    neutral_params: jdc.Static[P] = jdc.field(default=two_state.Params())

    field_strength: float = 0.01
    positive_field_contribution_ratio: float = 0.5

    field_delta_dipole: float = 10.0
    field_delta_polarizability: float = 100.0


def absorption(params: Params) -> Spectrum:
    neutral_spectrum = _compute_neutral_spectrum(params)
    positive_spectrum = _compute_charged_spectrum(params, 1.0)
    negative_spectrum = _compute_charged_spectrum(params, -1.0)

    positive_spectrum *= params.positive_field_contribution_ratio
    negative_spectrum *= 1 - params.positive_field_contribution_ratio

    charged_sum = jnp.sum(
        jnp.array([positive_spectrum.intensities, negative_spectrum.intensities]),
        axis=0,
    )

    spectrum = charged_sum - neutral_spectrum.intensities

    return Spectrum(energies=neutral_spectrum.energies, intensities=spectrum)


def _compute_neutral_spectrum(params: Params) -> Spectrum:
    return params.model.absorption(params.neutral_params)


def _compute_charged_spectrum(params: Params, field_multiplier: float) -> Spectrum:
    field_strength = params.field_strength * field_multiplier
    dipole_energy_change = params.field_delta_dipole * field_strength * 1679.0870295
    polarizability_energy_change = (
        0.5 * (field_strength**2) * params.field_delta_polarizability * 559.91
    )
    field_energy_change = -1 * (dipole_energy_change + polarizability_energy_change)

    charged_params = params.neutral_params.apply_electric_field(field_energy_change)

    return params.model.absorption(charged_params)

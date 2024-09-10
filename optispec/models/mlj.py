import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from optispec.models.base import CommonParams, Spectrum


@jdc.pytree_dataclass
class Params(CommonParams):
    # static arguments
    basis_size: jdc.Static[int] = 20
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


def absorption(params: Params) -> Spectrum:
    pass


def _compute_spectrum(
    params: Params, low_freq_index: jdc.Static[int]
) -> Float[Array, " num_points"]:
    # compute necessary values
    disorder_wavenumbers = params.disorder_meV * 8061 * 0.001
    # low_freq_relaxation_energy = params.mode_couplings[low_freq_index] ** 2 * opa

from jaxtyping import Array, Float, Int
import jax_dataclasses as jdc

from optispec.models.base import CommonParams, Spectrum


@jdc.pytree_dataclass
class Params(CommonParams):
    broadening: float
    temperature_kelvin: float

    transfer_integral: float
    energy_gap: float

    mode_basis_sets: Int[Array, "num_modes"]
    mode_frequencies: Float[Array, "num_modes"]
    mode_couplings: Float[Array, "num_modes"]


def absorption(params: Params) -> Spectrum:
    pass

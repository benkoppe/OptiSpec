from jaxtyping import Array, Float, Int

from optispec.models.base import CommonParams, Spectrum


class Params(CommonParams):
    broadening: float
    temperature_kelvin: float

    transfer_integral: float
    energy_gap: float

    mode_basis_sets: Int[Array, "num_modes"]
    mode_frequencies: Float[Array, "num_modes"]
    mode_couplings: Float[Array, "num_modes"]


def run(params: Params) -> Spectrum:
    pass

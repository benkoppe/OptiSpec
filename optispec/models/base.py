import jax_dataclasses as jdc
from jaxtyping import Array, Float


@jdc.pytree_dataclass
class CommonParams:
    start_energy: jdc.Static[float] = 0.0
    end_energy: jdc.Static[float] = 20_000.0
    num_points: jdc.Static[int] = 2_001


@jdc.pytree_dataclass
class Spectrum:
    energies: Float[Array, " num_points"]
    intensities: Float[Array, " num_points"]

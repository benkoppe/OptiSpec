import jax.numpy as jnp
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

    def energies_equal(self, other: "Spectrum") -> bool:
        return jnp.array_equal(self.energies, other.energies).item()

    def intensities_similar(self, other: "Spectrum", rtol=1e-05, atol=1e-08) -> bool:
        return jnp.allclose(self.intensities, other.intensities, rtol=rtol, atol=atol).item()

    def __mul__(self, other: Float) -> "Spectrum":
        return Spectrum(self.energies, self.intensities * other)

    def match_greatest_peak_of(
        self, other_intensities: Float[Array, " num_points"]
    ) -> "Spectrum":
        max_intensity = self.intensities.max()
        other_max_intensity = other_intensities.max()

        return self * (other_max_intensity / max_intensity)

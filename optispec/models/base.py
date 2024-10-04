from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


@jdc.pytree_dataclass
class CommonParams(ABC):
    start_energy: jdc.Static[float] = 0.0
    end_energy: jdc.Static[float] = 20_000.0
    num_points: jdc.Static[int] = 2_001

    @abstractmethod
    def apply_electric_field(
        self,
        field_energy_change: float,
    ):
        raise NotImplementedError


@jdc.pytree_dataclass
class Spectrum:
    energies: Float[Array, " num_points"]
    intensities: Float[Array, " num_points"]

    def plot(self, show: bool = True, ax: Optional[Axes] = None) -> Line2D:
        if ax is None:
            ax = plt.gca()
        line = ax.plot(self.energies, self.intensities)[0]
        if show:
            plt.show()
        return line

    def save_plot(self, file_path: str):
        plt.clf()
        self.plot(show=False)
        plt.savefig(file_path)

    def save_data(self, file_path: str):
        np.savetxt(
            file_path,
            jnp.column_stack((self.energies, self.intensities)),
            delimiter=",",
            header="energies,intensities",
        )

    def energies_equal(self, other: "Spectrum") -> bool:
        return jnp.array_equal(self.energies, other.energies).item()

    def intensities_similar(self, other: "Spectrum", rtol=1e-05, atol=1e-08) -> bool:
        return jnp.allclose(
            self.intensities, other.intensities, rtol=rtol, atol=atol
        ).item()

    def assert_similarity(self, other: "Spectrum", rtol=1e-05, atol=1e-08) -> bool:
        return self.energies_equal(other) and self.intensities_similar(
            other, rtol=rtol, atol=atol
        )

    def __mul__(self, other: Float) -> "Spectrum":
        return Spectrum(self.energies, self.intensities * other)

    def match_greatest_peak_of(
        self, other_intensities: Float[Array, " num_points"]
    ) -> "Spectrum":
        max_intensity = self.intensities.max()
        other_max_intensity = other_intensities.max()

        return self * (other_max_intensity / max_intensity)

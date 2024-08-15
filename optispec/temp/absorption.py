import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from optispec.models import two_state as ts

jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_enable_x64", True)


p = ts.Params()

spec = ts.absorption(p)

plt.plot(spec.energies, spec.intensities)

plt.show()

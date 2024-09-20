# OptiSpec

Optispec is a Python package that simulates molecular absorption spectra. Currently, OptiSpec implements the following models:

- **[Two-State Vibronic Hamiltonian](#two-state)**: smaller quantum mechanical model
- **[Marcus–Levich–Jortner (MLJ)](#mlj)**: so-called 'semiclassical' computation
- **Three-State Vibronic Hamiltonian**: *work in progress (see branch)*

A general [Hamiltonian](#hamiltonian) model is also provided that is used under-the-hood by the quantum-mechanical models.

## Table of Contents

1. [Installation](#installation) from PyPI
2. [Usage](#usage) after installation
3. Description of base [Models](#models)
4. [Spectrum](#spectrum) model return object
5. Description of underlying [Hamiltonian](#hamiltonian)
6. Common [Troubleshooting](#troubleshooting)

## Installation

TODO

## Usage

After completing installation, create a Python script or start the REPL in the terminal to use the package.

1. The call to run a script or start the REPL from the terminal is simply:

    ```zsh
    python <path-to-file>
    ```

    To start the REPL, don't include a file path in the terminal call.
    
> [!NOTE]
> If `python` doesn't work on your machine, try `python3`.

2. From there, import a model. Models are located in `optispec.models`:

    ```python
    from optispec.models import two_state
    from optispec.models import mlj
    ```

3. To use a model, set parameters with an instance of the model's `Params` class:

    ```python
    import jax.numpy as jnp
    
    default_params = two_state.Params()
    
    all_set_params = two_state.Params(
        start_energy = 0.0,
        end_energy = 20_000.0,
        num_points = 2_001,
        temperature_kelvin = 300.0,
        broadening = 200.0,
        energy_gap = 8_000.0,
        coupling = 100.0,
        mode_frequencies = jnp.array([1200.0, 100.0]),
        mode_couplings = jnp.array([0.7, 2.0]),
        mode_basis_sets = (20, 200)
    )
    ```

    For more information about each model's `Params`, see the [Models](#models) section.

> [!CAUTION]
> Please be careful about types! Note in the example that `float` variables always have a decimal, arrays are always `jax.numpy.array` arrays, and tuples are wrapped in `()`. See the [Models](#models) section for more detail.

5. Finally, transform the `Params` instance into a output `Spectrum` instance with `model.absorption()`:

    ```python
    output_spectrum = two_state.absorption(default_params)
    ```

> [!IMPORTANT]
> If you run into issues here, please see [Troubleshooting](#troubleshooting).

5. `Spectrum` instance

    This output object includes useful parameters and methods to interact with the computed absorption spectrum, including:

    ```python
    s = output_spectrum
    
    s.plot() # display a plot with the spectrum
    s.save_plot(file_path) # plot and save to file_path
    s.save_data(file_path) # save energy/intensity data to file_path
    s.energies # JAX array of the energies
    s.intensities # JAX array of the intensities
    ```

    For more detail, see the [Spectrum](#spectrum) section.


## Models

### Two-State

The two-state model simulates the ground state (GS) and charge-transfer state (CT). The model creates a two-state vibronic Hamiltonian, diagonalizes, and determines the spectrum using contributions calculated from the resulting eigenenergies.

To import this model, call:

```python
from optispec.models import two_state
```

A `Params` instance has these parameters and defaults (will use if not provided):

```python
import jax.numpy as jnp

default_params = two_state.Params(
    start_energy = 0.0, # start energy of output points
    end_energy = 20_000.0, # end energy of output points
    num_points = 2_001, # number of output points
    temperature_kelvin = 300.0, # system temperature in kelvin
    broadening = 200.0, # width of gaussian peak expansion
    energy_gap = 8_000.0, # energy gap between GS and CT states
    coupling = 100.0, # coupling between GS and CT states
    mode_frequencies = jnp.array([1200.0, 100.0]), # frequencies per mode
    mode_couplings = jnp.array([0.7, 2.0]), # couplings per mode
    mode_basis_sets = (20, 200) # basis set per mode
)
```

> [!CAUTION]
> Ensure types:
> 
> - **`float`**: `start_energy`, `end_energy`, `temperature_kelvin`, `broadening`, `energy_gap`, and `coupling`
> - **`int`**: `num_points`
> - **JAX Array `jax.numpy.array()`**: `mode_frequencies` and `mode_couplings`
> - **`tuple`**: `mode_basis_sets`

To transform parameters to an absorption spectrum, call `two_state.absorption(params)`:

```python
spectrum = two_state.absorption(default_params)
```

This will return a [Spectrum](#spectrum) object.

### MLJ

The MLJ 'semiclassical' model is an approximation that provides less accurate results at a significantly faster speed than its corresponding two-state quantum-mechanical model. Its parameters represent the same kind of two-state system as the Two-State Model.

To import this model, call:

```python
from optispec.models import mlj
```

A `Params` instance has these parameters and defaults (will use if not provided):

```python
import jax.numpy as jnp

default_params = mlj.Params(
    start_energy = 0.0, # start energy of output points
    end_energy = 20_000.0, # end energy of output points
    num_points = 2_001, # number of output points
    basis_size = 10, # number of basis functions
    temperature_kelvin = 300.0, # system temperature in kelvin
    energy_gap = 8_000.0, # energy gap between GS and CT states
    disorder_meV = 0.0, # disorder in millielectronvolts
    mode_frequencies = jnp.array([1200.0, 100.0]), # frequencies per mode (expects 2 modes)
    mode_couplings = jnp.array([0.7, 2.0]), # couplings per mode (expects 2 modes)
)
```

> [!CAUTION]
> Ensure types:
> 
> - **`float`**: `start_energy`, `end_energy`, `temperature_kelvin`, `energy_gap`, and `disorder_meV`
> - **`int`**: `num_points` and `basis_size`
> - **JAX Array `jax.numpy.array()`**: `mode_frequencies` and `mode_couplings`

To transform parameters to an absorption spectrum, call `mlj.absorption(params)`:

```python
spectrum = mlj.absorption(default_params)
```

This will return a [Spectrum](#spectrum) object.

## Spectrum

TODO

## Hamiltonian

TODO

## Troubleshooting

This package uses [JAX](https://jax.readthedocs.io/en/latest/index.html) for computation, which provides high-performance array operations that can be easily run on both the CPU and GPU, multithreaded, just-in-time compiled, etc., with high abstraction. However, this can also cause some problems -- look through the sections here to see if yours is included!

### Exception: RuntimeError: Unable to initialize backend 'METAL'

In these cases, JAX is attempting to use an unsupported backend such as METAL. To fix the backend, set the environment variable in your terminal:

```zsh
export JAX_PLATFORMS=cpu # or gpu, if available
```

This should fix the problem.

### 32-bit Computation

By default, JAX uses 32-bit integers and floats. If you want full 64-bit precision, set the following environment variable:

```zsh
export JAX_ENABLE_X64=True
```

# OptiSpec

Optispec is a Python package that simulates molecular absorption spectra. Currently, OptiSpec implements the following models:

- **[Two-State Vibronic Hamiltonian](#two-state)**: quantum mechanical model with ground and charge-transfer states
- **[Marcus–Levich–Jortner (MLJ)](#mlj)**: so-called 'semiclassical' computation
- **Three-State Vibronic Hamiltonian**: *work in progress (see branch)*
- **[Stark Effect](#stark)**: computes the electric field effects on any other model

A general [Hamiltonian](#hamiltonian) model is also provided that is used under-the-hood by the quantum-mechanical models.

## Table of Contents

1. [Installation](#installation) from PyPI
2. [Usage](#usage) after installation
3. Description of base [Models](#models)
4. [Spectrum](#spectrum) model return object
5. Description of underlying [Hamiltonian](#hamiltonian)
6. How to enable [GPU computation](#gpu-computation)
7. Common [Troubleshooting](#troubleshooting)

## Installation

1. Install Python

   If you don’t have Python installed, you can download it from the official website: https://www.python.org/downloads/

> [!NOTE]
> Some Python distributions use `python3` instead of `python`.
>
> If you are using a distribution that uses `python3`, you should replace `python` with `python3` in all commands.
>
> Similarly, `pip3` may need to be used instead of `pip`.

2. Create a directory

   Create a directory to hold your OptiSpec installation. Run the following commands in a folder of your choice:

   ```zsh
   mkdir optispec # create a directory of any name
   cd optispec # move to the new directory
   ```

3. Create a virtual environment

   Inside the new directory, create a virtual environment with the following command:

   ```zsh
   python -m venv .venv
   ```

   This will create a new directory `.venv` that contains the virtual environment. Then, activate the environment with:

   ```zsh
   source .venv/bin/activate
   ```

   This will activate the virtual environment. **In future terminal sessions, you will need to re-run this command.**

4. Install from `pip`

   Finally, install OptiSpec into your venv with:

   ```zsh
   pip install optispec
   ```

   This will install the package and all dependencies.

## Usage

After completing installation, create a Python script or start the REPL in the terminal to use the package.

1. The call to run a script or start the REPL from the terminal is simply:

    ```zsh
    python <path-to-file>
    ```

    To start the REPL, don't include a file path in the terminal call.

> [!NOTE]
> If `python` doesn't work on your machine, try `python3`.

3. From there, either in your script or directly in the REPL, import a model. Models are located in `optispec.models`:

    ```python
    from optispec.models import two_state
    from optispec.models import mlj
    ```

4. To use a model, set parameters with an instance of the model's `Params` class:

    ```python
    import jax.numpy as jnp
    
    default_params = two_state.Params()
    
    manually_specified_params = two_state.Params(
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

> [!CAUTION]
> Ensure types:
> 
> - **`float`**: `start_energy`, `end_energy`, `temperature_kelvin`, `broadening`, `energy_gap`, and `coupling`
> - **`int`**: `num_points`
> - **JAX Array `jax.numpy.array()`**: `mode_frequencies` and `mode_couplings`
> - **`tuple`**: `mode_basis_sets`

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

> [!CAUTION]
> Ensure types:
> 
> - **`float`**: `start_energy`, `end_energy`, `temperature_kelvin`, `energy_gap`, and `disorder_meV`
> - **`int`**: `num_points` and `basis_size`
> - **JAX Array `jax.numpy.array()`**: `mode_frequencies` and `mode_couplings`

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

To transform parameters to an absorption spectrum, call `mlj.absorption(params)`:

```python
spectrum = mlj.absorption(default_params)
```

This will return a [Spectrum](#spectrum) object.

### Stark

The stark effect model applies a positive and negative electric field to a given model's parameters, averages them at a specified contribution ratio, and takes the difference between their average and the model's neutral (no field) output.

To import this model, call:

```python
from optispec.models import stark, two_state # rather than two_state, import whichever additional model you need
```

A `Params` instance has these parameters and defaults (will use if not provided):

> [!CAUTION]
> Ensure types:
> 
> - **`float`**: `field_strength`, `positive_field_contribution_ratio`, `field_delta_dipole`, and `field_delta_polarizability`
> - **`optispec.model`**: `model`
> - **`optispec.model.Params`**: `neutral_params`

```python
import jax.numpy as jnp

default_params = stark.Params(
    model = two_state, # model to be used
    neutral_params = two_state.Params(
        ... # neutral model parameters (no electric field)
    ),
    field_strength = 0.01, # positive strength of the electric field
    positive_field_contribution_ratio = 0.5, # ratio of positive field's contribution, always sums with negative ratio to 1
    field_delta_dipole = 38.0, # electric field's change in dipole moment
    field_delta_polarizability = 1000.0, # electric field's change in polarizability
)
```

To transform parameters to an absorption spectrum, call `stark.absorption(params)`:

```python
spectrum = stark.absorption(default_params)
```

This will return a [Spectrum](#spectrum) object.

## Spectrum

`Spectrum` objects are returned from all absorption spectrum computations. It is a `dataclass` with the following attributes:

```python
spectrum.energies # JAX array
spectrum.intensities # JAX array
```

To convert a JAX array to a `numpy` array, wrap with `np.array`: as in, `np_energies = np.array(spectrum.energies)`.

To see a summary of the most useful methods in `Spectrum`, see [Usage](#usage). In total, however, `Spectrum` objects contain the following methods:

```python
spectrum.plot(show: bool = True, ax: Optional[Axes] = None) # plot the spectrum on given axes (or plt.gca() if unprovided) and show afterwards
spectrum.save_plot(file_path: str) # plot the spectrum and save to file_path
spectrum.save_data(file_path: str) # save the spectrum's data to a file_path
spectrum.energies_equal(other: Spectrum) # ensure energies match other spectrum
spectrum.intensities_similar(other: Spectrum, rtol: float = 1e-05, atol: float = 1e-08) # ensure intensities match other spectrum with given relative and absolute tolerance
spectrum.assert_similarity(other: Spectrum, rtol: float = 1e-05, atol: float = 1e-08) # combines both `energies_equal` and `intensities_similar`
spectrum.__mul__(other: float) # allows multiplying intensities by a scalar
spectrum.match_greatest_peak_of(other_intensities: Float[Array, "num_points"]) # returns a new spectrum where intensity maximum matches given array
```

## Hamiltonian

To make the quantum mechanical models possible, OptiSpec also implements a general `Hamiltonian` model. This model supports arbitrary electronic states and vibrational modes.

To import the hamiltonian model, call:

```python
from optispec import hamiltonian
```

From there, a `Params` dataclass can be initialized in the same way as the models:

```python
import jax.numpy as jnp

example_params = hamiltonian.Params(
    transfer_integrals = 0.0, # single float to apply to all states, or an array for each combination of states
    state_energies = jnp.array([]), # energy of each energetic state
    mode_frequencies = jnp.array([]), # frequency of each vibrational mode
    mode_state_couplings = jnp.array([]), # coupling between all states and modes -- rows are states, columns are modes
    mode_basis_sets = (20, 200), # basis set for each vibrational mode
    mode_localities = (True, True), # locality of each state
)
```

> [!NOTE]
> Unlike models, default values are *not* provided.

From there, a matrix can be constructed and/or diagonalized with the following methods:

```python
diagonalization = hamiltonian.diagonalize(example_params)
matrix = hamiltonian.hamiltonian(example_params)
```

`matrix` is simply a large 2d JAX array, while `diagonalization` is a `Diagonalization` NamedTuple with two attributes:

```python
diagonalization.eigenvalues # 1d JAX array
diagonalization.eigenvectors # 2d JAX array
```

## GPU Computation

Because this package uses JAX, it comes with a capability for computation on a GPU baked in, which can bring massive speedups. With the virtual environment sourced and a GPU on-machine, you can start using JAX on the GPU with the following terminal command:

```zsh
pip install -U "jax[cuda12]"
```

This will install the JAX builds necessary to enable to GPU. To switch back to the CPU-only build, run:

```zsh
pip install -U jax
```


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

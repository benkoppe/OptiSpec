# OptiSpec

Optispec is a Python package that simulates molecular absorption spectra. Currently, OptiSpec implements the following models:

- **Two-State Vibronic Hamiltonian**: smaller quantum mechanical model
- **Marcus–Levich–Jortner (MLJ)**: so-called 'semiclassical' computation
- **Three-State Vibronic Hamiltonian**: *work in progress (see branch)*

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

    For more information about each model's `Params`, see the [MODELS] section.

> [!CAUTION]
> Please be careful about types! Note in the example that `float` variables always have a decimal, arrays are always `jax.numpy.array` arrays, and tuples are wrapped in `()`. See the [MODELS] section for more detail.

5. Finally, transform the `Params` instance into a output `Spectrum` instance with `model.absorption()`:

    ```python
    output_spectrum = two_state.absorption(default_params)
    ```

> [!IMPORTANT]
> If you run into issues here, please see [LINK TO TROUBLESHOOTING]

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


## Models

TODO

## Hamiltonian

TODO

## Troubleshooting

TODO

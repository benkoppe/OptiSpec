# OptiSpec

Optispec is a Python package that simulates molecular absorption spectra. Currently, OptiSpec implements the following models:

- **Marcus–Levich–Jortner (MLJ)**: so-called 'semiclassical' computation
- **Two-State Vibronic Hamiltonian**: smaller quantum mechanical model
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
    default_params = two_state.Params()
    all_set_params = two_state.Params()
    ```

    For more information about each model's `Params`, see the [MODELS] section.

4. Finally, transform the `Params` instance into a output `Spectrum` instance with `model.absorption()`:

    ```python
    output_spectrum = two_state.absorption(default_params)
    ```

> [!IMPORTANT]
> If you run into issues here, please see [LINK TO TROUBLESHOOTING]

5. `Spectrum` 

    This output object includes useful parameters and methods to interact with the computed absorption spectrum. This includes:
    
    TODO: Spectrum parameters/methods


## Models

TODO

## Hamiltonian

TODO

## Troubleshooting

TODO

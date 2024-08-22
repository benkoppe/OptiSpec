from jaxtyping import Float, Array


def kelvin_to_wavenumbers(temperature_kelvin: float) -> float:
    return 0.695028 * temperature_kelvin


def outer_sum(*arrays: Float[Array, "*"]) -> Float[Array, "*"]:
    """
    Computes the outer sum of multiple JAX arrays.

    This function takes multiple JAX arrays as input and computes their outer sum. It starts with the first array and
    iteratively adds each subsequent array to it in a way that's similar to computing the outer product, but with
    summation instead. This is done by reshaping the arrays for broadcasting, ensuring dimensions are aligned correctly
    for the sum.

    Parameters
    ----------
    *arrays: tuple[Float[Array, "*"]]
        Variable number of JAX array arguments.
        Each array should be compatible for broadcasting.
        There should be at least one array passed to this function.

    Returns
    -------
    Float[Array, "*"]
        A JAX array containing the outer sum of the input arrays.

    Raises
    ------
    ValueError
        If no arrays are provided as input.

    Examples
    --------
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1, 2])
        >>> b = jnp.array([3, 4])
        >>> outer_sum(a, b)
        DeviceArray([[4, 5],
                     [5, 6]], dtype=int32)

    Notes
    -----
        The function requires at least one input array and all input arrays must be compatible for broadcasting
        following the JAX rules.
    """
    # Ensure there is at least one array
    if not arrays:
        raise ValueError("At least one array is required")

    # Start with the first array, reshaping it to have a new axis for each additional array
    result = arrays[0].reshape(arrays[0].shape + (1,) * (len(arrays) - 1))

    # Iteratively add each subsequent array, reshaping appropriately for broadcasting
    for i, arr in enumerate(arrays[1:], 1):
        # The new shape has 1's in all positions except the current dimension being added
        new_shape = (1,) * i + arr.shape + (1,) * (len(arrays) - i - 1)
        result += arr.reshape(new_shape)

    return result

import jax.numpy as jnp
import jax
import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int

from typing import NamedTuple
from optispec.utils import outer_sum


@jdc.pytree_dataclass
class Params:
    # arguments
    transfer_integrals: float | Float[Array, " (num_states * (num_states - 1)) // 2"]
    state_energies: Float[Array, " num_states"]

    # mode arguments
    mode_frequencies: Float[Array, " num_modes"]
    mode_state_couplings: Float[Array, " num_states num_modes"]

    # static mode arguments, should be provided as tuples
    mode_basis_sets: jdc.Static[tuple[int, ...]]
    mode_localities: jdc.Static[tuple[bool, ...]]


class Diagonalization(NamedTuple):
    eigenvalues: Float[Array, " matrix_size"]
    eigenvectors: Float[Array, " matrix_size matrix_size"]


Matrix = Float[Array, " matrix_size matrix_size"]
MatrixBlock = Float[Array, " block_size block_size"]


@jdc.jit
def diagonalize(params: Params) -> Diagonalization:
    eigenvalues, eigenvectors = jnp.linalg.eigh(hamiltonian(params))
    return Diagonalization(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


@jdc.jit
def hamiltonian(params: Params) -> Matrix:
    num_states = _num_states(params)
    transfer_integrals_matrix = _transfer_integrals_matrix(params)

    # build the matrix, state by state
    rows = []
    for state_row in range(num_states):
        cols = []
        for state_col in range(num_states):
            state_index = max(state_row, state_col)

            if state_row == state_col:
                # calclate a local (disagonal) state block
                state = _local_block(state_index=state_index, params=params)
            else:
                # calculate a non-local (off-diagonal) state block
                state = _non_local_block(
                    state_index=state_index,
                    transfer_integral=transfer_integrals_matrix[state_row, state_col],
                    params=params,
                )
            cols.append(state)
        rows.append(jnp.hstack(cols))
    matrix = jnp.vstack(rows)

    return matrix


def _num_states(params: Params) -> jdc.Static[int]:
    return len(params.state_energies)


@jdc.jit
def _transfer_integrals_matrix(
    params: Params,
) -> Float[Array, " num_states num_states"]:
    num_states = _num_states(params)
    row_indices, col_indices = jnp.triu_indices(num_states, k=1)

    integrals_matrix = jnp.zeros((num_states, num_states))
    integrals_matrix = integrals_matrix.at[row_indices, col_indices].set(
        params.transfer_integrals
    )
    integrals_matrix = integrals_matrix.at[col_indices, row_indices].set(
        params.transfer_integrals
    )

    return integrals_matrix


@jdc.jit
def _local_block(state_index: int, params: Params) -> MatrixBlock:
    state_energy = params.state_energies[state_index]
    mode_couplings = params.mode_state_couplings[state_index]

    block = _create_block_with_modes(
        state_locality=True,
        basis_sets=params.mode_basis_sets,
        localities=params.mode_localities,
        frequencies=params.mode_frequencies,
        couplings=mode_couplings,
    )

    diagonals = _local_diagonals(
        energy=state_energy,
        basis_sets=params.mode_basis_sets,
        frequencies=params.mode_frequencies,
        couplings=mode_couplings,
    )

    return block + jnp.diag(diagonals)


@jdc.jit
def _local_diagonals(
    energy: Float[Array, "1"],
    basis_sets: jdc.Static[tuple[int, ...]],
    frequencies: Float[Array, " num_modes"],
    couplings: Float[Array, " num_modes"],
) -> Float[Array, " block_size"]:
    all_diagonal_components = [
        _mode_diagonal_components_all(basis_set, frequency, coupling)
        for basis_set, frequency, coupling in zip(basis_sets, frequencies, couplings)
    ]

    sum_contribution_combinations = outer_sum(*all_diagonal_components).flatten()

    all_diagonal_values = energy + sum_contribution_combinations

    return all_diagonal_values


@jdc.jit
def _non_local_block(
    state_index: int, transfer_integral: Float[Array, "1"], params: Params
) -> MatrixBlock:
    mode_couplings = params.mode_state_couplings[state_index]

    block = _create_block_with_modes(
        state_locality=False,
        basis_sets=params.mode_basis_sets,
        localities=params.mode_localities,
        frequencies=params.mode_frequencies,
        couplings=mode_couplings,
    )

    diagonals = jnp.repeat(transfer_integral, len(block))

    return block + jnp.diag(diagonals)


def _create_block_with_modes(
    state_locality: jdc.Static[bool],
    basis_sets: jdc.Static[tuple[int, ...]],
    localities: jdc.Static[tuple[bool, ...]],
    frequencies: Float[Array, " num_modes"],
    couplings: Float[Array, " num_modes"],
) -> MatrixBlock:
    # base case: no modes, return an empty block
    if len(basis_sets) == 0:
        return jnp.zeros((1, 1))

    # otherwise, pop the first mode's properties
    basis_set, locality, frequency, coupling = (
        basis_sets[0],
        localities[0],
        frequencies[0],
        couplings[0],
    )

    # call the function recursively with the remaining modes
    block = _create_block_with_modes(
        state_locality,
        basis_sets[1:],
        localities[1:],
        frequencies[1:],
        couplings[1:],
    )
    previous_block_size = len(block)

    # calculate the mode block's off diagonal elements
    off_diagonal_elements = jax.lax.cond(
        (locality == state_locality),
        lambda: _mode_offdiagonal_elements(
            jnp.arange(basis_set - 1), frequency, coupling
        ),
        lambda: jnp.zeros(basis_set - 1),
    )

    off_diagonal_elements = jnp.repeat(
        off_diagonal_elements, repeats=previous_block_size
    )

    # create a new block by repeating the previous block diagonally
    new_block = jax.scipy.linalg.block_diag(*[block for _ in range(basis_set)])

    new_block = (
        new_block
        + jnp.diag(off_diagonal_elements, k=previous_block_size)
        + jnp.diag(off_diagonal_elements, k=-previous_block_size)
    )

    return new_block


def _mode_diagonal_component(
    component_index: Int[Array, "1"],
    frequency: Float[Array, "1"],
    coupling: Float[Array, "1"],
) -> Float[Array, "1"]:
    return frequency * ((component_index + 0.5) + (coupling**2))


_mode_diagonal_components = jax.vmap(_mode_diagonal_component, in_axes=(0, None, None))


def _mode_diagonal_components_all(
    basis_set: jdc.Static[int],
    frequency: Float[Array, "1"],
    coupling: Float[Array, "1"],
) -> Float[Array, " basis_set"]:
    return _mode_diagonal_components(jnp.arange(basis_set), frequency, coupling)


def _mode_offdiagonal_element(
    component_index: Int[Array, "1"],
    frequency: Float[Array, "1"],
    coupling: Float[Array, "1"],
) -> Float[Array, "1"]:
    return frequency * coupling * jnp.sqrt(component_index + 1)


_mode_offdiagonal_elements = jax.vmap(
    _mode_offdiagonal_element, in_axes=(0, None, None)
)
